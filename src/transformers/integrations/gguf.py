# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""On-demand GGUF modules."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ..utils.generic import maybe_autocast
from .gguf_dequant import GGUFQuantizedTensor, dequantize_gguf_tensor
from .moe import (
    ExpertsInterface,
    _batched_linear,
    _default_apply_gate,
    _grouped_linear,
    use_experts_implementation,
)


def _dequantize_weight(weight: GGUFQuantizedTensor, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return dequantize_gguf_tensor(weight, weight.quant_type, dtype=dtype, device=device)


def _dequantize_rows(weight: GGUFQuantizedTensor, rows: torch.Tensor, dtype, device):
    payload = weight.as_subclass(torch.Tensor)
    rows = rows.to(payload.device)
    selected = payload.index_select(0, rows)
    return dequantize_gguf_tensor(selected, weight.quant_type, dtype=dtype, device=device)


def _dequantize_experts(weight: torch.Tensor, expert_indices: torch.Tensor, dtype, device):
    if not isinstance(weight, GGUFQuantizedTensor):
        return weight.index_select(0, expert_indices.to(weight.device)).to(device=device, dtype=dtype)
    return _dequantize_rows(weight, expert_indices, dtype, device)


def _get_expert_weight_state(module: nn.Module) -> str:
    states = []
    for weight in (module.gate_proj, module.up_proj, module.down_proj):
        if isinstance(weight, GGUFQuantizedTensor):
            states.append("packed")
        elif weight.is_floating_point():
            states.append("floating")
        else:
            states.append("placeholder")
    return states[0] if len(set(states)) == 1 else "mixed"


def _validate_expert_weights(module: nn.Module) -> str:
    state = _get_expert_weight_state(module)
    if state == "placeholder":
        raise RuntimeError("GGUFExperts weights have not been loaded with packed or floating-point parameters")
    if state == "mixed":
        raise RuntimeError(
            "GGUFExperts gate, up, and down projections must be all packed or all floating-point parameters; "
            "mixed packed, floating-point, and placeholder states are not supported"
        )
    return state


def _validate_expert_indices(expert_indices: torch.Tensor, num_experts: int):
    if torch.any((expert_indices < 0) | (expert_indices >= num_experts)):
        raise IndexError("GGUF experts do not support expert-parallel sentinel indices")


class _GGUFLinearFunction(torch.autograd.Function):
    """Recompute a frozen GGUF weight for input gradients instead of saving its dense form."""

    @staticmethod
    def forward(ctx, input, weight, bias, compute_dtype):
        device_type = input.device.type
        ctx.autocast_enabled = torch.is_autocast_enabled(device_type)
        ctx.autocast_dtype = torch.get_autocast_dtype(device_type)
        ctx.bias_dtype = bias.dtype if bias is not None else None
        ctx.compute_dtype = compute_dtype
        ctx.device_type = device_type
        ctx.input_dtype = input.dtype
        ctx.quant_type = weight.quant_type
        if ctx.needs_input_grad[0]:
            ctx.save_for_backward(weight.as_subclass(torch.Tensor))

        dense_weight = _dequantize_weight(weight, compute_dtype, input.device)
        return F.linear(input, dense_weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_bias = None
        if ctx.needs_input_grad[0]:
            (payload,) = ctx.saved_tensors
            with maybe_autocast(
                ctx.device_type,
                dtype=ctx.autocast_dtype,
                enabled=ctx.autocast_enabled,
            ):
                dense_weight = dequantize_gguf_tensor(
                    payload,
                    ctx.quant_type,
                    dtype=ctx.compute_dtype,
                    device=grad_output.device,
                )
                flat_grad_output = grad_output.reshape(-1, grad_output.shape[-1])
                grad_input = torch.mm(flat_grad_output, dense_weight)
                grad_input = grad_input.reshape(*grad_output.shape[:-1], dense_weight.shape[-1]).to(ctx.input_dtype)

        if ctx.needs_input_grad[2]:
            reduction_dims = tuple(range(grad_output.ndim - 1))
            grad_bias = grad_output.sum(dim=reduction_dims) if reduction_dims else grad_output
            grad_bias = grad_bias.to(ctx.bias_dtype)

        return grad_input, None, grad_bias, None


class _GGUFComputeDtypeMixin:
    """Keep compute policy separate from packed GGUF parameter storage."""

    compute_dtype: torch.dtype

    def set_compute_dtype(self, dtype: torch.dtype):
        if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
            raise TypeError(f"GGUF compute dtype must be a floating-point torch.dtype, got {dtype!r}")
        self.compute_dtype = dtype
        return self

    def _apply(self, fn, recurse=True):
        compute_probe = fn(torch.empty(0, dtype=self.compute_dtype))
        module = nn.Module._apply(self, fn, recurse=recurse)
        if compute_probe.dtype.is_floating_point:
            self.compute_dtype = compute_probe.dtype
        return module


def gguf_batched_mm_experts_forward(
    self: nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    _validate_expert_weights(self)
    input_dtype = hidden_states.dtype
    num_top_k = top_k_index.size(-1)
    num_tokens = hidden_states.size(0)
    expert_ids = top_k_index.reshape(-1)
    _validate_expert_indices(expert_ids, self.num_experts)

    selected_hidden_states = hidden_states.to(self.compute_dtype).repeat_interleave(num_top_k, dim=0)
    active_experts, local_expert_ids = torch.unique(expert_ids, sorted=True, return_inverse=True)

    gate_weights = _dequantize_experts(
        self.gate_proj, active_experts, self.compute_dtype, hidden_states.device
    ).index_select(0, local_expert_ids)
    gate = _batched_linear(selected_hidden_states, gate_weights)
    del gate_weights

    up_weights = _dequantize_experts(
        self.up_proj, active_experts, self.compute_dtype, hidden_states.device
    ).index_select(0, local_expert_ids)
    up = _batched_linear(selected_hidden_states, up_weights)
    del up_weights

    intermediate = self._apply_split_gate(gate, up)
    down_weights = _dequantize_experts(
        self.down_proj, active_experts, self.compute_dtype, hidden_states.device
    ).index_select(0, local_expert_ids)
    output = _batched_linear(intermediate, down_weights)
    output = output * top_k_weights.reshape(-1, 1).to(output.dtype)
    return output.view(num_tokens, num_top_k, self.hidden_dim).sum(dim=1).to(input_dtype)


def gguf_grouped_mm_experts_forward(
    self: nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    _validate_expert_weights(self)
    input_dtype = hidden_states.dtype
    num_top_k = top_k_index.size(-1)
    num_tokens = hidden_states.size(0)
    expert_ids = top_k_index.reshape(-1)
    _validate_expert_indices(expert_ids, self.num_experts)

    sorted_expert_ids, permutation = torch.sort(expert_ids)
    selected_hidden_states = hidden_states.to(self.compute_dtype)[permutation // num_top_k]
    selected_routing_weights = top_k_weights.reshape(-1)[permutation]
    active_experts, expert_counts = torch.unique_consecutive(sorted_expert_ids, return_counts=True)
    offsets = expert_counts.cumsum(0, dtype=torch.int32)

    gate_weights = _dequantize_experts(self.gate_proj, active_experts, self.compute_dtype, hidden_states.device)
    gate = _grouped_linear(selected_hidden_states, gate_weights, offsets)
    del gate_weights

    up_weights = _dequantize_experts(self.up_proj, active_experts, self.compute_dtype, hidden_states.device)
    up = _grouped_linear(selected_hidden_states, up_weights, offsets)
    del up_weights

    intermediate = self._apply_split_gate(gate, up)
    down_weights = _dequantize_experts(self.down_proj, active_experts, self.compute_dtype, hidden_states.device)
    output = _grouped_linear(intermediate, down_weights, offsets)
    output = output * selected_routing_weights.unsqueeze(-1).to(output.dtype)

    inverse_permutation = torch.empty_like(permutation)
    inverse_permutation[permutation] = torch.arange(permutation.size(0), device=permutation.device)
    output = output[inverse_permutation]
    return output.view(num_tokens, num_top_k, self.hidden_dim).sum(dim=1).to(input_dtype)


class GGUFExpertsInterface(ExpertsInterface):
    """Switchable MoE implementations that understand compressed GGUF expert parameters."""

    _global_mapping = {
        "grouped_mm": gguf_grouped_mm_experts_forward,
        "batched_mm": gguf_batched_mm_experts_forward,
    }

    def supported_implementations(self) -> tuple[str, ...]:
        return ("eager", *self.valid_keys())

    def validate_implementation(self, experts_implementation: str | None) -> str | None:
        if experts_implementation is not None and experts_implementation not in self.supported_implementations():
            raise ValueError(
                f"GGUF experts do not support {experts_implementation!r}; use 'eager', 'grouped_mm', or 'batched_mm'."
            )
        return experts_implementation

    def get_interface(self, experts_implementation, default):
        self.validate_implementation(experts_implementation)
        return super().get_interface(experts_implementation, default)


ALL_GGUF_EXPERTS_FUNCTIONS = GGUFExpertsInterface()


@use_experts_implementation(
    experts_interface=ALL_GGUF_EXPERTS_FUNCTIONS,
    is_concatenated=None,
    projection_layout="split_gate_up",
)
class GGUFExperts(_GGUFComputeDtypeMixin, nn.Module):
    """Routed experts backed by separate compressed GGUF gate, up, and down payloads."""

    supported_experts_implementations = ALL_GGUF_EXPERTS_FUNCTIONS.supported_implementations()
    experts_implementation_switchable = True

    def __init__(
        self,
        config,
        device=None,
        compute_dtype=None,
        *,
        num_experts=None,
        hidden_dim=None,
        intermediate_dim=None,
        act_fn=None,
    ):
        super().__init__()
        from ..activations import ACT2FN

        self.num_experts = num_experts if num_experts is not None else config.num_experts
        self.hidden_dim = hidden_dim if hidden_dim is not None else config.hidden_size
        self.intermediate_dim = intermediate_dim if intermediate_dim is not None else config.moe_intermediate_size
        self.act_fn = act_fn if act_fn is not None else ACT2FN[config.hidden_act]
        self.set_compute_dtype(compute_dtype or torch.get_default_dtype())
        self.gate_proj = nn.Parameter(
            torch.empty((self.num_experts, self.intermediate_dim, self.hidden_dim), dtype=torch.uint8, device=device),
            requires_grad=False,
        )
        self.up_proj = nn.Parameter(
            torch.empty((self.num_experts, self.intermediate_dim, self.hidden_dim), dtype=torch.uint8, device=device),
            requires_grad=False,
        )
        self.down_proj = nn.Parameter(
            torch.empty((self.num_experts, self.hidden_dim, self.intermediate_dim), dtype=torch.uint8, device=device),
            requires_grad=False,
        )

    @classmethod
    def _validate_supported_experts_implementation(cls, experts_implementation: str | None) -> str | None:
        return ALL_GGUF_EXPERTS_FUNCTIONS.validate_implementation(experts_implementation)

    @classmethod
    def _source_module_contract(cls, module: nn.Module):
        if not hasattr(module, "config"):
            raise ValueError("GGUF expert replacement requires a source module with a config")
        if not getattr(module, "has_gate", True):
            raise ValueError("GGUF expert replacement does not yet support ungated expert modules")
        if getattr(module, "has_bias", False) or any(
            getattr(module, name, None) is not None for name in ("gate_up_proj_bias", "down_proj_bias")
        ):
            raise ValueError("GGUF expert replacement does not yet support expert projection bias")
        if getattr(module, "is_transposed", False):
            raise ValueError("GGUF expert replacement does not yet support transposed expert projections")

        projection_layout = getattr(module, "projection_layout", None)
        if projection_layout is None:
            is_concatenated = getattr(module, "is_concatenated", True)
            projection_layout = "concatenated_gate_up" if is_concatenated is True else "interleaved_gate_up"
        if projection_layout != "concatenated_gate_up":
            raise ValueError(
                f"GGUF expert replacement does not yet support source projection layout {projection_layout!r}"
            )

        gate_implementation = getattr(module, "gate_implementation", None)
        if gate_implementation is None:
            source_apply_gate = getattr(type(module), "_apply_gate", None)
            gate_implementation = "default" if source_apply_gate in (None, _default_apply_gate) else "custom"
        if gate_implementation != "default":
            raise ValueError("GGUF expert replacement does not yet support custom gate behavior")

        gate_up_proj = getattr(module, "gate_up_proj", None)
        down_proj = getattr(module, "down_proj", None)
        if not isinstance(gate_up_proj, torch.Tensor) or not isinstance(down_proj, torch.Tensor):
            raise ValueError("GGUF expert replacement requires gate_up_proj and down_proj tensor parameters")
        if gate_up_proj.ndim != 3 or down_proj.ndim != 3:
            raise ValueError("GGUF expert replacement requires rank-3 expert projection tensors")

        num_experts, gate_up_dim, hidden_dim = gate_up_proj.shape
        if gate_up_dim % 2:
            raise ValueError("GGUF expert replacement requires an even concatenated gate/up dimension")
        intermediate_dim = gate_up_dim // 2
        expected_down_shape = (num_experts, hidden_dim, intermediate_dim)
        if tuple(down_proj.shape) != expected_down_shape:
            raise ValueError(
                f"GGUF expert down projection has shape {tuple(down_proj.shape)}, expected {expected_down_shape}"
            )

        for attribute, expected in (
            ("num_experts", num_experts),
            ("hidden_dim", hidden_dim),
            ("intermediate_dim", intermediate_dim),
        ):
            value = getattr(module, attribute, expected)
            if value != expected:
                raise ValueError(
                    f"GGUF expert source {attribute}={value} does not match projection shape value {expected}"
                )

        if gate_up_proj.device != down_proj.device:
            raise ValueError("GGUF expert replacement requires source projections on the same device")
        if gate_up_proj.dtype != down_proj.dtype or not gate_up_proj.is_floating_point():
            raise ValueError("GGUF expert replacement requires source projections with one floating-point dtype")
        if not callable(getattr(module, "act_fn", None)):
            raise ValueError("GGUF expert replacement requires a callable act_fn")

        return (
            module.config,
            num_experts,
            hidden_dim,
            intermediate_dim,
            gate_up_proj.device,
            gate_up_proj.dtype,
            module.act_fn,
        )

    @classmethod
    def from_module(cls, module: nn.Module, compute_dtype=None):
        config, num_experts, hidden_dim, intermediate_dim, device, source_dtype, act_fn = cls._source_module_contract(
            module
        )
        return cls(
            config,
            device=device,
            compute_dtype=source_dtype if compute_dtype is None else compute_dtype,
            num_experts=num_experts,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            act_fn=act_fn,
        )

    @property
    def weight_state(self) -> str:
        return _get_expert_weight_state(self)

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        _validate_expert_weights(self)
        _validate_expert_indices(top_k_index, self.num_experts)
        compute_hidden_states = hidden_states.to(self.compute_dtype)
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
            active_experts = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero(as_tuple=False).view(-1)

        for expert_idx in active_experts:
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = compute_hidden_states[token_idx]
            expert = expert_idx.reshape(1)

            gate_weight = _dequantize_experts(
                self.gate_proj, expert, self.compute_dtype, hidden_states.device
            ).squeeze(0)
            gate = F.linear(current_state, gate_weight)
            del gate_weight

            up_weight = _dequantize_experts(self.up_proj, expert, self.compute_dtype, hidden_states.device).squeeze(0)
            up = F.linear(current_state, up_weight)
            del up_weight

            intermediate = self._apply_split_gate(gate, up)
            down_weight = _dequantize_experts(
                self.down_proj, expert, self.compute_dtype, hidden_states.device
            ).squeeze(0)
            output = F.linear(intermediate, down_weight)
            output = output * top_k_weights[token_idx, top_k_pos, None].to(output.dtype)
            final_hidden_states.index_add_(0, token_idx, output.to(final_hidden_states.dtype))

        return final_hidden_states


class GGUFLinear(_GGUFComputeDtypeMixin, nn.Linear):
    """Linear layer backed by a frozen compressed GGUF payload."""

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        device=None,
        dtype=None,
        compute_dtype=None,
    ):
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.set_compute_dtype(compute_dtype or dtype or torch.get_default_dtype())
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), dtype=torch.uint8, device=device), requires_grad=False
        )

    @classmethod
    def from_linear(cls, module: nn.Linear, compute_dtype=None):
        return cls(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            device=module.weight.device,
            dtype=module.weight.dtype,
            compute_dtype=compute_dtype or module.weight.dtype,
        )

    def materialize_logical_weight(
        self,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Materialize the floating matrix represented by this module's packed or floating weight.

        Callers should keep the returned tensor scoped to one operation; packed weights remain
        frozen and are dequantized again on the next call.
        """

        dtype = self.compute_dtype if dtype is None else dtype
        if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
            raise TypeError(f"GGUF logical weights require a floating-point dtype, got {dtype!r}")
        device = self.weight.device if device is None else torch.device(device)

        if isinstance(self.weight, GGUFQuantizedTensor):
            weight = _dequantize_weight(self.weight, dtype, device)
        else:
            if not self.weight.is_floating_point():
                raise RuntimeError("GGUFLinear weight has not been loaded with a packed or floating-point parameter")
            weight = self.weight.to(device=device, dtype=dtype)
        return weight.contiguous()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not isinstance(self.weight, GGUFQuantizedTensor):
            if not self.weight.is_floating_point():
                raise RuntimeError("GGUFLinear weight has not been loaded with a packed or floating-point parameter")
            return nn.Linear.forward(self, input)
        input_dtype = input.dtype
        compute_input = input.to(self.compute_dtype)
        bias = self.bias.to(self.compute_dtype) if self.bias is not None else None
        if torch.is_grad_enabled() and compute_input.requires_grad:
            output = _GGUFLinearFunction.apply(compute_input, self.weight, bias, self.compute_dtype)
        else:
            weight = _dequantize_weight(self.weight, self.compute_dtype, input.device)
            output = F.linear(compute_input, weight, bias)
        return output.to(input_dtype)


class GGUFEmbedding(_GGUFComputeDtypeMixin, nn.Embedding):
    """Embedding layer backed by row-selective GGUF dequantization.

    Packed GGUF embeddings are frozen. ``padding_idx`` remains supported, while
    options that mutate the embedding weight or configure its gradients are not.
    """

    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        padding_idx=None,
        max_norm=None,
        norm_type=2.0,
        scale_grad_by_freq=False,
        sparse=False,
        device=None,
        dtype=None,
        compute_dtype=None,
    ):
        if max_norm is not None:
            raise ValueError("GGUFEmbedding does not support max_norm for frozen packed weights")
        if norm_type != 2.0:
            raise ValueError("GGUFEmbedding only supports the default norm_type=2.0")
        if scale_grad_by_freq:
            raise ValueError("GGUFEmbedding does not support scale_grad_by_freq for frozen packed weights")
        if sparse:
            raise ValueError("GGUFEmbedding does not support sparse gradients for frozen packed weights")

        super().__init__(
            num_embeddings,
            embedding_dim,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse,
            device=device,
            dtype=dtype,
        )
        self.set_compute_dtype(compute_dtype or dtype or torch.get_default_dtype())
        self.weight = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), dtype=torch.uint8, device=device), requires_grad=False
        )

    @classmethod
    def from_embedding(cls, module: nn.Embedding, compute_dtype=None):
        return cls(
            module.num_embeddings,
            module.embedding_dim,
            padding_idx=module.padding_idx,
            max_norm=module.max_norm,
            norm_type=module.norm_type,
            scale_grad_by_freq=module.scale_grad_by_freq,
            sparse=module.sparse,
            device=module.weight.device,
            dtype=module.weight.dtype,
            compute_dtype=compute_dtype or module.weight.dtype,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not isinstance(self.weight, GGUFQuantizedTensor):
            if not self.weight.is_floating_point():
                raise RuntimeError(
                    "GGUFEmbedding weight has not been loaded with a packed or floating-point parameter"
                )
            return nn.Embedding.forward(self, input)
        unique_rows, inverse = torch.unique(input.reshape(-1), sorted=False, return_inverse=True)
        selected = _dequantize_rows(self.weight, unique_rows, self.compute_dtype, input.device)
        output = selected.index_select(0, inverse.to(selected.device))
        return output.reshape(*input.shape, self.embedding_dim)


def replace_with_gguf_modules(model, compute_dtype=None):
    """Replace linear, embedding, and structurally compatible routed-expert modules on the meta model."""
    modules = list(model.named_modules())
    for name, module in modules:
        if not name:
            continue
        if (name == "experts" or name.endswith(".experts")) and all(
            hasattr(module, attribute) for attribute in ("config", "gate_up_proj", "down_proj")
        ):
            GGUFExperts._source_module_contract(module)

    for name, module in modules:
        if not name:
            continue
        if isinstance(module, GGUFLinear | GGUFEmbedding | GGUFExperts):
            continue
        if (name == "experts" or name.endswith(".experts")) and all(
            hasattr(module, attribute) for attribute in ("config", "gate_up_proj", "down_proj")
        ):
            replacement = GGUFExperts.from_module(module, compute_dtype=compute_dtype)
        elif isinstance(module, nn.Linear):
            replacement = GGUFLinear.from_linear(module, compute_dtype=compute_dtype)
        elif isinstance(module, nn.Embedding):
            replacement = GGUFEmbedding.from_embedding(module, compute_dtype=compute_dtype)
        else:
            continue
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        parent._modules[child_name] = replacement
    return model
