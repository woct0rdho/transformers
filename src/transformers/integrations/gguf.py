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
    _ExpertExecutionPlan,
    _ExpertRoutingPlan,
    _grouped_linear,
    batched_mm_experts_forward,
    grouped_mm_experts_forward,
    use_experts_implementation,
)


def _dequantize_weight(weight: GGUFQuantizedTensor, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return dequantize_gguf_tensor(weight, weight.quant_type, dtype=dtype, device=device)


def _dequantize_selected_payload(
    payload: torch.Tensor,
    quant_type,
    indices: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    indices = indices.to(payload.device)
    selected = payload.index_select(0, indices)
    return dequantize_gguf_tensor(selected, quant_type, dtype=dtype, device=device)


def _dequantize_rows(weight: GGUFQuantizedTensor, rows: torch.Tensor, dtype, device):
    payload = weight.as_subclass(torch.Tensor)
    return _dequantize_selected_payload(payload, weight.quant_type, rows, dtype, device)


def _dequantize_experts(weight: torch.Tensor, expert_indices: torch.Tensor, dtype, device):
    if not isinstance(weight, GGUFQuantizedTensor):
        return weight.index_select(0, expert_indices.to(weight.device)).to(device=device, dtype=dtype)
    return _dequantize_rows(weight, expert_indices, dtype, device)


def _expert_projection_tensors(module: nn.Module) -> dict[str, torch.Tensor]:
    provider = getattr(module, "_get_expert_projection_tensors", None)
    if callable(provider):
        projections = provider()
        if not isinstance(projections, dict) or not all(
            isinstance(name, str) and isinstance(weight, torch.Tensor) for name, weight in projections.items()
        ):
            raise ValueError("Expert projection providers must return a dictionary with string keys and tensor values")
        return projections

    projections = {"down": getattr(module, "down_proj", None)}
    if getattr(module, "has_gate", True):
        projections["gate_up"] = getattr(module, "gate_up_proj", None)
    else:
        projections["up"] = getattr(module, "up_proj", None)
    return projections


def _get_expert_weight_state(module: nn.Module) -> str:
    states = []
    for weight in _expert_projection_tensors(module).values():
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


def _save_gguf_autocast_state(ctx, input, compute_dtype):
    device_type = input.device.type
    ctx.autocast_enabled = torch.is_autocast_enabled(device_type)
    ctx.autocast_dtype = torch.get_autocast_dtype(device_type)
    ctx.compute_dtype = compute_dtype
    ctx.device_type = device_type
    ctx.input_dtype = input.dtype


def _linear_input_gradient(grad_output, weight, input_dtype):
    flat_grad_output = grad_output.reshape(-1, grad_output.shape[-1])
    grad_input = torch.mm(flat_grad_output, weight)
    return grad_input.reshape(*grad_output.shape[:-1], weight.shape[-1]).to(input_dtype)


def _run_expert_projection(input, weight, implementation, route_indices=None, offsets=None):
    if implementation == "eager":
        return F.linear(input, weight.squeeze(0))
    if implementation == "grouped_mm":
        return _grouped_linear(input, weight, offsets)
    if implementation == "batched_mm":
        route_weight = weight.index_select(0, route_indices)
        return _batched_linear(input, route_weight)
    raise ValueError(f"Unknown GGUF expert projection implementation {implementation!r}")


def _run_expert_input_gradient(grad_output, weight, implementation, input_dtype, route_indices=None, offsets=None):
    if implementation == "eager":
        return _linear_input_gradient(grad_output, weight.squeeze(0), input_dtype)
    if implementation == "grouped_mm":
        grad_input = _grouped_linear(grad_output, weight, offsets, is_transposed=True)
    elif implementation == "batched_mm":
        route_weight = weight.index_select(0, route_indices)
        grad_input = _batched_linear(grad_output, route_weight, is_transposed=True)
    else:
        raise ValueError(f"Unknown GGUF expert projection implementation {implementation!r}")
    return grad_input.to(input_dtype)


class _GGUFExpertProjectionFunction(torch.autograd.Function):
    """Recompute selected frozen GGUF expert weights for input gradients."""

    @staticmethod
    def forward(ctx, input, weight, expert_indices, route_indices, offsets, compute_dtype, implementation):
        _save_gguf_autocast_state(ctx, input, compute_dtype)
        ctx.implementation = implementation
        ctx.quant_type = weight.quant_type
        ctx.has_route_indices = route_indices is not None
        ctx.has_offsets = offsets is not None
        if ctx.needs_input_grad[0]:
            saved_tensors = [weight.as_subclass(torch.Tensor), expert_indices]
            if route_indices is not None:
                saved_tensors.append(route_indices)
            if offsets is not None:
                saved_tensors.append(offsets)
            ctx.save_for_backward(*saved_tensors)

        dense_weight = _dequantize_experts(weight, expert_indices, compute_dtype, input.device)
        return _run_expert_projection(input, dense_weight, implementation, route_indices, offsets)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        if ctx.needs_input_grad[0]:
            saved_tensors = iter(ctx.saved_tensors)
            payload = next(saved_tensors)
            expert_indices = next(saved_tensors)
            route_indices = next(saved_tensors) if ctx.has_route_indices else None
            offsets = next(saved_tensors) if ctx.has_offsets else None
            with maybe_autocast(
                ctx.device_type,
                dtype=ctx.autocast_dtype,
                enabled=ctx.autocast_enabled,
            ):
                dense_weight = _dequantize_selected_payload(
                    payload,
                    ctx.quant_type,
                    expert_indices,
                    ctx.compute_dtype,
                    grad_output.device,
                )
                grad_input = _run_expert_input_gradient(
                    grad_output,
                    dense_weight,
                    ctx.implementation,
                    ctx.input_dtype,
                    route_indices,
                    offsets,
                )

        return grad_input, None, None, None, None, None, None


def _expert_projection(
    input,
    weight,
    expert_indices,
    compute_dtype,
    implementation,
    route_indices=None,
    offsets=None,
):
    if isinstance(weight, GGUFQuantizedTensor) and torch.is_grad_enabled() and input.requires_grad:
        return _GGUFExpertProjectionFunction.apply(
            input,
            weight,
            expert_indices,
            route_indices,
            offsets,
            compute_dtype,
            implementation,
        )
    dense_weight = _dequantize_experts(weight, expert_indices, compute_dtype, input.device)
    return _run_expert_projection(input, dense_weight, implementation, route_indices, offsets)


class _GGUFLinearFunction(torch.autograd.Function):
    """Recompute a frozen GGUF weight for input gradients instead of saving its dense form."""

    @staticmethod
    def forward(ctx, input, weight, bias, compute_dtype):
        _save_gguf_autocast_state(ctx, input, compute_dtype)
        ctx.bias_dtype = bias.dtype if bias is not None else None
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
                grad_input = _linear_input_gradient(grad_output, dense_weight, ctx.input_dtype)

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


class GGUFExpertsInterface(ExpertsInterface):
    """Switchable MoE implementations that understand compressed GGUF expert parameters."""

    display_name = "GGUFExpertsInterface"
    _global_mapping = {
        "grouped_mm": grouped_mm_experts_forward,
        "batched_mm": batched_mm_experts_forward,
    }


ALL_GGUF_EXPERTS_FUNCTIONS = GGUFExpertsInterface()


@use_experts_implementation(
    experts_interface=ALL_GGUF_EXPERTS_FUNCTIONS,
    is_concatenated=None,
    projection_layout="split_gate_up",
)
class GGUFExperts(_GGUFComputeDtypeMixin, nn.Module):
    """Routed experts backed by separate compressed GGUF gate, up, and down payloads."""

    def _prepare_expert_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states.to(self.compute_dtype)

    def _prepare_expert_execution(
        self,
        routing_plan: _ExpertRoutingPlan,
        implementation: str,
    ) -> _ExpertExecutionPlan:
        _validate_expert_weights(self)
        _validate_expert_indices(routing_plan.expert_indices, self.num_experts)
        if implementation == "batched_mm":
            active_experts, local_expert_ids = torch.unique(
                routing_plan.expert_indices,
                sorted=True,
                return_inverse=True,
            )
            return _ExpertExecutionPlan(active_experts, local_expert_ids, None, None, None)
        if implementation == "grouped_mm":
            active_experts, expert_counts = torch.unique_consecutive(
                routing_plan.expert_indices,
                return_counts=True,
            )
            offsets = expert_counts.cumsum(0, dtype=torch.int32)
            return _ExpertExecutionPlan(active_experts, None, offsets, None, None)
        raise ValueError(f"Unknown experts implementation {implementation!r}")

    def _get_expert_projection_tensors(self) -> dict[str, torch.Tensor]:
        return {"gate": self.gate_proj, "up": self.up_proj, "down": self.down_proj}

    def _project_expert_weight(
        self,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        execution_plan: _ExpertExecutionPlan,
        implementation: str,
    ) -> torch.Tensor:
        return _expert_projection(
            hidden_states,
            weight,
            execution_plan.expert_indices,
            self.compute_dtype,
            implementation,
            route_indices=execution_plan.route_indices,
            offsets=execution_plan.offsets,
        )

    def _project_expert_up(
        self,
        hidden_states: torch.Tensor,
        execution_plan: _ExpertExecutionPlan,
        implementation: str,
    ) -> torch.Tensor:
        gate = self._project_expert_weight(hidden_states, self.gate_proj, execution_plan, implementation)
        up = self._project_expert_weight(hidden_states, self.up_proj, execution_plan, implementation)
        return self._apply_split_gate(gate, up)

    def _project_expert_down(
        self,
        hidden_states: torch.Tensor,
        execution_plan: _ExpertExecutionPlan,
        implementation: str,
    ) -> torch.Tensor:
        return self._project_expert_weight(hidden_states, self.down_proj, execution_plan, implementation)

    def _cast_expert_routing_weights(
        self,
        routing_weights: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        return routing_weights.to(output.dtype)

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

        projections = _expert_projection_tensors(module)
        gate_up_proj = projections.get("gate_up")
        down_proj = projections.get("down")
        if not isinstance(gate_up_proj, torch.Tensor) or not isinstance(down_proj, torch.Tensor):
            raise ValueError("GGUF expert replacement requires gate_up and down projection tensors from its provider")
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
        validator = getattr(cls, "_validate_supported_experts_implementation")
        validator(getattr(config, "_experts_implementation", None))
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

            gate = _expert_projection(
                current_state,
                self.gate_proj,
                expert,
                self.compute_dtype,
                "eager",
            )
            up = _expert_projection(
                current_state,
                self.up_proj,
                expert,
                self.compute_dtype,
                "eager",
            )

            intermediate = self._apply_split_gate(gate, up)
            output = _expert_projection(
                intermediate,
                self.down_proj,
                expert,
                self.compute_dtype,
                "eager",
            )
            output = output * top_k_weights[token_idx, top_k_pos, None].to(output.dtype)
            final_hidden_states.index_add_(0, token_idx, output.to(final_hidden_states.dtype))

        return final_hidden_states


class GGUFLinear(_GGUFComputeDtypeMixin, nn.Linear):
    """Linear layer backed by a frozen compressed GGUF payload.

    Optional index permutations describe a GGUF physical row or column layout while keeping the
    module's public inputs and outputs in the model's canonical logical layout.
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        device=None,
        dtype=None,
        compute_dtype=None,
        input_permutation=None,
        input_permutation_offset=0,
        output_permutation=None,
        output_permutation_offset=0,
        floating_weight=False,
    ):
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.set_compute_dtype(compute_dtype or dtype or torch.get_default_dtype())
        weight_dtype = dtype if floating_weight else torch.uint8
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), dtype=weight_dtype, device=device), requires_grad=False
        )
        self.input_permutation_offset = input_permutation_offset
        self.output_permutation_offset = output_permutation_offset
        self.input_permutation = (
            None if input_permutation is None else tuple(int(index) for index in input_permutation)
        )
        self.output_permutation = (
            None if output_permutation is None else tuple(int(index) for index in output_permutation)
        )
        self._layout_permutation_cache = {}

    @classmethod
    def from_linear(
        cls,
        module: nn.Linear,
        compute_dtype=None,
        input_permutation=None,
        input_permutation_offset=0,
        output_permutation=None,
        output_permutation_offset=0,
        floating_weight=False,
    ):
        return cls(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            device=module.weight.device,
            dtype=module.weight.dtype,
            compute_dtype=compute_dtype or module.weight.dtype,
            input_permutation=input_permutation,
            input_permutation_offset=input_permutation_offset,
            output_permutation=output_permutation,
            output_permutation_offset=output_permutation_offset,
            floating_weight=floating_weight,
        )

    def _permute_segment(self, input, permutation, offset, cache_key):
        if permutation is None:
            return input
        device_key = (cache_key, input.device.type, input.device.index)
        permutation_tensor = self._layout_permutation_cache.get(device_key)
        if permutation_tensor is None:
            permutation_tensor = torch.tensor(permutation, dtype=torch.long, device=input.device)
            self._layout_permutation_cache[device_key] = permutation_tensor
        size = len(permutation)
        if input.shape[-1] < offset + size:
            raise RuntimeError(
                f"GGUFLinear layout permutation requires dimension {offset + size}, got {input.shape[-1]}"
            )
        permuted = input[..., offset : offset + size].index_select(-1, permutation_tensor)
        pieces = []
        if offset:
            pieces.append(input[..., :offset])
        pieces.append(permuted)
        if offset + size < input.shape[-1]:
            pieces.append(input[..., offset + size :])
        return torch.cat(pieces, dim=-1) if len(pieces) > 1 else pieces[0]

    def _apply(self, fn, recurse=True):
        self._layout_permutation_cache.clear()
        return super()._apply(fn, recurse=recurse)

    def materialize_logical_weight(
        self,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Materialize the floating weight represented by this module's public logical layout.

        GGUF payloads and Qwen3.5 recurrent projections can use a physical layout that external
        consumers must not infer from ``weight``. This method provides the matrix equivalent to
        calling ``forward`` without a bias, including input and output layout permutations.
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

        if self.input_permutation is not None:
            inverse_input_permutation = tuple(
                torch.argsort(torch.tensor(self.input_permutation, dtype=torch.long)).tolist()
            )
            weight = self._permute_segment(
                weight,
                inverse_input_permutation,
                self.input_permutation_offset,
                "logical_weight_input",
            )
        if self.output_permutation is not None:
            weight = self._permute_segment(
                weight.transpose(0, 1),
                self.output_permutation,
                self.output_permutation_offset,
                "logical_weight_output",
            ).transpose(0, 1)
        return weight.contiguous()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = self._permute_segment(input, self.input_permutation, self.input_permutation_offset, "input")
        if not isinstance(self.weight, GGUFQuantizedTensor):
            if not self.weight.is_floating_point():
                raise RuntimeError("GGUFLinear weight has not been loaded with a packed or floating-point parameter")
            output = nn.Linear.forward(self, input)
        else:
            input_dtype = input.dtype
            compute_input = input.to(self.compute_dtype)
            bias = self.bias.to(self.compute_dtype) if self.bias is not None else None
            if torch.is_grad_enabled() and compute_input.requires_grad:
                output = _GGUFLinearFunction.apply(compute_input, self.weight, bias, self.compute_dtype)
            else:
                weight = _dequantize_weight(self.weight, self.compute_dtype, input.device)
                output = F.linear(compute_input, weight, bias)
            output = output.to(input_dtype)
        return self._permute_segment(output, self.output_permutation, self.output_permutation_offset, "output")


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
        floating_weight=False,
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
        weight_dtype = dtype if floating_weight else torch.uint8
        self.weight = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), dtype=weight_dtype, device=device), requires_grad=False
        )

    @classmethod
    def from_embedding(cls, module: nn.Embedding, compute_dtype=None, floating_weight=False):
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
            floating_weight=floating_weight,
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


def _is_expert_module_candidate(name: str, module: nn.Module) -> bool:
    if name != "experts" and not name.endswith(".experts"):
        return False
    has_projection_provider = callable(getattr(module, "_get_expert_projection_tensors", None))
    has_legacy_projections = all(hasattr(module, attribute) for attribute in ("gate_up_proj", "down_proj"))
    return hasattr(module, "config") and (has_projection_provider or has_legacy_projections)


def _validate_qwen35_gated_delta_net(name: str, module: nn.Module):
    required_dimensions = (
        "hidden_size",
        "num_k_heads",
        "num_v_heads",
        "head_k_dim",
        "head_v_dim",
        "key_dim",
        "value_dim",
        "conv_dim",
        "conv_kernel_size",
    )
    missing_dimensions = [attribute for attribute in required_dimensions if not hasattr(module, attribute)]
    if missing_dimensions:
        raise ValueError(f"Qwen3.5 GGUF recurrent module {name!r} is missing dimensions {missing_dimensions}")
    if module.num_v_heads % module.num_k_heads:
        raise ValueError(f"Qwen3.5 GGUF recurrent module {name!r} requires num_v_heads to be divisible by num_k_heads")

    projection_shapes = {
        "in_proj_qkv": (module.conv_dim, module.hidden_size),
        "in_proj_z": (module.value_dim, module.hidden_size),
        "in_proj_a": (module.num_v_heads, module.hidden_size),
        "in_proj_b": (module.num_v_heads, module.hidden_size),
        "out_proj": (module.hidden_size, module.value_dim),
    }
    for projection_name, expected_shape in projection_shapes.items():
        projection = getattr(module, projection_name, None)
        if not isinstance(projection, nn.Linear) or tuple(projection.weight.shape) != expected_shape:
            actual_shape = tuple(projection.weight.shape) if isinstance(projection, nn.Linear) else None
            raise ValueError(
                f"Qwen3.5 GGUF recurrent projection {name}.{projection_name} has shape {actual_shape}, "
                f"expected {expected_shape}"
            )
        if projection.bias is not None:
            raise ValueError(f"Qwen3.5 GGUF recurrent projection {name}.{projection_name} must not have a bias")

    conv = getattr(module, "conv1d", None)
    expected_conv_shape = (module.conv_dim, 1, module.conv_kernel_size)
    if (
        not isinstance(conv, nn.Conv1d)
        or tuple(conv.weight.shape) != expected_conv_shape
        or conv.groups != module.conv_dim
        or conv.bias is not None
    ):
        raise ValueError(
            f"Qwen3.5 GGUF recurrent convolution {name}.conv1d must be bias-free depthwise convolution "
            f"with shape {expected_conv_shape}"
        )
    for parameter_name, expected_shape in (
        ("A_log", (module.num_v_heads,)),
        ("dt_bias", (module.num_v_heads,)),
    ):
        parameter = getattr(module, parameter_name, None)
        if not isinstance(parameter, torch.Tensor) or tuple(parameter.shape) != expected_shape:
            raise ValueError(
                f"Qwen3.5 GGUF recurrent parameter {name}.{parameter_name} must have shape {expected_shape}"
            )
    norm_weight = getattr(getattr(module, "norm", None), "weight", None)
    if not isinstance(norm_weight, torch.Tensor) or tuple(norm_weight.shape) != (module.head_v_dim,):
        raise ValueError(f"Qwen3.5 GGUF recurrent norm {name}.norm must have weight shape {(module.head_v_dim,)}")


def _qwen35_value_head_orders(module: nn.Module, head_dim: int):
    value_heads_per_key = module.num_v_heads // module.num_k_heads
    physical_order = (
        torch.arange(module.num_v_heads * head_dim, device="cpu")
        .reshape(module.num_k_heads, value_heads_per_key, head_dim)
        .transpose(0, 1)
        .reshape(-1)
    )
    return physical_order, torch.argsort(physical_order)


_QWEN35_TEXT_MODEL_TYPES = {"qwen3_5_text", "qwen3_5_moe_text"}


def _qwen35_linear_layout(model: nn.Module, name: str):
    if getattr(model.config, "model_type", None) not in _QWEN35_TEXT_MODEL_TYPES or ".linear_attn." not in name:
        return {}
    parent_name, _, projection_name = name.rpartition(".")
    module = model.get_submodule(parent_name)
    if projection_name == "out_proj":
        physical_order, _ = _qwen35_value_head_orders(module, module.head_v_dim)
        return {"input_permutation": physical_order}
    if projection_name in ("in_proj_qkv", "in_proj_z"):
        _, canonical_order = _qwen35_value_head_orders(module, module.head_v_dim)
        return {
            "output_permutation": canonical_order,
            "output_permutation_offset": 2 * module.key_dim if projection_name == "in_proj_qkv" else 0,
        }
    if projection_name in ("in_proj_a", "in_proj_b"):
        _, canonical_order = _qwen35_value_head_orders(module, 1)
        return {"output_permutation": canonical_order}
    return {}


def replace_with_gguf_modules(model, compute_dtype=None, floating_checkpoint_params=None):
    """Replace linear, embedding, and structurally compatible routed-expert modules on the meta model."""
    floating_checkpoint_params = set(floating_checkpoint_params or ())
    modules = list(model.named_modules())
    for name, module in modules:
        if not name:
            continue
        if getattr(getattr(model, "config", None), "model_type", None) in _QWEN35_TEXT_MODEL_TYPES and name.endswith(
            ".linear_attn"
        ):
            _validate_qwen35_gated_delta_net(name, module)
        if isinstance(module, GGUFExperts):
            module._validate_supported_experts_implementation(getattr(module.config, "_experts_implementation", None))
        elif _is_expert_module_candidate(name, module):
            GGUFExperts._source_module_contract(module)
            GGUFExperts._validate_supported_experts_implementation(
                getattr(module.config, "_experts_implementation", None)
            )

    for name, module in modules:
        if not name:
            continue
        if isinstance(module, GGUFLinear | GGUFEmbedding | GGUFExperts):
            continue
        if _is_expert_module_candidate(name, module):
            replacement = GGUFExperts.from_module(module, compute_dtype=compute_dtype)
        elif isinstance(module, nn.Linear):
            replacement = GGUFLinear.from_linear(
                module,
                compute_dtype=compute_dtype,
                floating_weight=f"{name}.weight" in floating_checkpoint_params,
                **_qwen35_linear_layout(model, name),
            )
        elif isinstance(module, nn.Embedding):
            replacement = GGUFEmbedding.from_embedding(
                module,
                compute_dtype=compute_dtype,
                floating_weight=f"{name}.weight" in floating_checkpoint_params,
            )
        else:
            continue
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        parent._modules[child_name] = replacement
    return model
