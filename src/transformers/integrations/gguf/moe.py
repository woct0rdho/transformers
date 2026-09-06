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
"""MoE modules for persistent packed GGUF checkpoints."""

import torch
from torch import nn
from torch.nn import functional as F

from ..moe import (
    ExpertsInterface,
    _batched_linear,
    _ExpertExecutionPlan,
    _ExpertRoutingPlan,
    _grouped_linear,
    batched_mm_experts_forward,
    grouped_mm_experts_forward,
    use_experts_implementation,
)
from .gguf_quantized_parameter import GgufQuantizedParameter


def _linear_input_gradient(grad_output, weight, input_dtype):
    flat = grad_output.reshape(-1, grad_output.shape[-1])
    return torch.mm(flat, weight).reshape(*grad_output.shape[:-1], weight.shape[-1]).to(input_dtype)


def _dequantize_selected_payload(payload, quant_type, indices, dtype, device, logical_shape):
    indices = indices.to(payload.device)
    selected = payload.index_select(0, indices)
    selected = GgufQuantizedParameter(
        selected,
        quant_type=quant_type,
        logical_shape=(indices.numel(), *logical_shape[1:]),
    )
    return selected.dequantize(dtype=dtype, device=device)


def _dequantize_experts(weight, expert_indices, dtype, device):
    if not isinstance(weight, GgufQuantizedParameter):
        return weight.index_select(0, expert_indices.to(weight.device)).to(device=device, dtype=dtype)
    return _dequantize_selected_payload(
        weight.as_subclass(torch.Tensor),
        weight.quant_type,
        expert_indices,
        dtype,
        device,
        weight.logical_shape,
    )


def _expert_projection_tensors(module):
    provider = getattr(module, "_get_expert_projection_tensors", None)
    if callable(provider):
        projections = provider()
        if not isinstance(projections, dict) or not all(
            isinstance(name, str) and isinstance(weight, torch.Tensor) for name, weight in projections.items()
        ):
            raise ValueError("Expert projection providers must return string keys and tensor values")
        return projections
    projections = {"down": getattr(module, "down_proj", None)}
    projections["gate_up" if getattr(module, "has_gate", True) else "up"] = getattr(
        module, "gate_up_proj" if getattr(module, "has_gate", True) else "up_proj", None
    )
    return projections


def _validate_expert_weights(module):
    states = []
    for weight in _expert_projection_tensors(module).values():
        if isinstance(weight, GgufQuantizedParameter):
            states.append("packed")
        elif isinstance(weight, torch.Tensor) and weight.is_floating_point():
            states.append("floating")
        else:
            states.append("placeholder")
    state = states[0] if len(set(states)) == 1 else "mixed"
    if state == "placeholder":
        raise RuntimeError("GgufExperts weights have not been loaded")
    if state == "mixed":
        raise RuntimeError("GgufExperts projections must be all packed or all floating-point parameters")
    return state


def _validate_expert_indices(expert_indices, num_experts):
    if torch.any(expert_indices < 0):
        raise IndexError("GGUF expert indices cannot be negative")


class _GgufExpertProjectionFunction(torch.autograd.Function):
    """Recompute selected frozen GGUF expert weights for activation gradients."""

    @staticmethod
    def forward(ctx, input, weight, expert_indices, route_indices, offsets, compute_dtype, implementation):
        ctx.quant_type = weight.quant_type
        ctx.logical_shape = weight.logical_shape
        ctx.compute_dtype = compute_dtype
        ctx.implementation = implementation
        ctx.route_indices = route_indices
        ctx.offsets = offsets
        ctx.input_dtype = input.dtype
        if ctx.needs_input_grad[0]:
            ctx.save_for_backward(weight.as_subclass(torch.Tensor), expert_indices)
        dense_weight = _dequantize_experts(weight, expert_indices, compute_dtype, input.device)
        if implementation == "eager":
            return F.linear(input, dense_weight.squeeze(0))
        if implementation == "grouped_mm":
            return _grouped_linear(input, dense_weight, offsets)
        route_weight = dense_weight.index_select(0, route_indices)
        return _batched_linear(input, route_weight)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        if ctx.needs_input_grad[0]:
            payload, expert_indices = ctx.saved_tensors
            dense_weight = _dequantize_selected_payload(
                payload,
                ctx.quant_type,
                expert_indices,
                ctx.compute_dtype,
                grad_output.device,
                ctx.logical_shape,
            )
            if ctx.implementation == "eager":
                grad_input = _linear_input_gradient(grad_output, dense_weight.squeeze(0), ctx.input_dtype)
            elif ctx.implementation == "grouped_mm":
                grad_input = _grouped_linear(grad_output, dense_weight, ctx.offsets, is_transposed=True)
            else:
                route_weight = dense_weight.index_select(0, ctx.route_indices)
                grad_input = _batched_linear(grad_output, route_weight, is_transposed=True).to(ctx.input_dtype)
        return grad_input, None, None, None, None, None, None


def _project_expert(input, weight, expert_indices, compute_dtype, implementation, route_indices=None, offsets=None):
    if isinstance(weight, GgufQuantizedParameter) and torch.is_grad_enabled() and input.requires_grad:
        return _GgufExpertProjectionFunction.apply(
            input, weight, expert_indices, route_indices, offsets, compute_dtype, implementation
        )
    dense_weight = _dequantize_experts(weight, expert_indices, compute_dtype, input.device)
    if implementation == "eager":
        return F.linear(input, dense_weight.squeeze(0))
    if implementation == "grouped_mm":
        return _grouped_linear(input, dense_weight, offsets)
    return _batched_linear(input, dense_weight.index_select(0, route_indices))


class GgufExpertsInterface(ExpertsInterface):
    """Experts dispatch interface for packed GGUF projections."""

    display_name = "GgufExpertsInterface"
    _global_mapping = {"batched_mm": batched_mm_experts_forward, "grouped_mm": grouped_mm_experts_forward}


ALL_GGUF_EXPERTS_FUNCTIONS = GgufExpertsInterface()


@use_experts_implementation(
    experts_interface=ALL_GGUF_EXPERTS_FUNCTIONS,
    is_concatenated=None,
    projection_layout="split_gate_up",
)
class GgufExperts(nn.Module):
    """Routed experts backed by separate compressed GGUF gate, up, and down payloads."""

    _supported_source_gate_implementations = frozenset({"default"})

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
        from ...activations import ACT2FN

        self.num_experts = num_experts if num_experts is not None else config.num_experts
        self.hidden_dim = hidden_dim if hidden_dim is not None else config.hidden_size
        self.intermediate_dim = intermediate_dim if intermediate_dim is not None else config.moe_intermediate_size
        self.act_fn = act_fn if act_fn is not None else ACT2FN[config.hidden_act]
        self.compute_dtype = compute_dtype or torch.get_default_dtype()
        self.gate_proj = nn.Parameter(
            torch.empty(self.num_experts, self.intermediate_dim, self.hidden_dim, dtype=torch.uint8, device=device),
            requires_grad=False,
        )
        self.up_proj = nn.Parameter(
            torch.empty(self.num_experts, self.intermediate_dim, self.hidden_dim, dtype=torch.uint8, device=device),
            requires_grad=False,
        )
        self.down_proj = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim, dtype=torch.uint8, device=device),
            requires_grad=False,
        )

    def _get_expert_projection_tensors(self):
        return {"gate": self.gate_proj, "up": self.up_proj, "down": self.down_proj}

    def _forward_eager(self, hidden_states, top_k_index, top_k_weights):
        _validate_expert_weights(self)
        _validate_expert_indices(top_k_index, self.num_experts)
        output = torch.zeros_like(hidden_states)
        compute_hidden_states = hidden_states.to(self.compute_dtype)
        with torch.no_grad():
            valid = top_k_index < self.num_experts
            safe_indices = top_k_index.clamp(max=self.num_experts - 1)
            mask = F.one_hot(safe_indices, num_classes=self.num_experts).permute(2, 1, 0)
            mask &= valid.permute(1, 0).unsqueeze(0)
            active = mask.sum(dim=(-1, -2)).nonzero(as_tuple=False).flatten()
        for expert_idx in active:
            top_k_pos, token_idx = torch.where(mask[expert_idx])
            expert = expert_idx.reshape(1)
            gate = _project_expert(
                compute_hidden_states[token_idx], self.gate_proj, expert, self.compute_dtype, "eager"
            )
            up = _project_expert(compute_hidden_states[token_idx], self.up_proj, expert, self.compute_dtype, "eager")
            intermediate = self._apply_split_gate(gate, up)
            current = _project_expert(intermediate, self.down_proj, expert, self.compute_dtype, "eager")
            current = current * top_k_weights[token_idx, top_k_pos, None].to(current.dtype)
            output.index_add_(0, token_idx, current.to(output.dtype))
        return output

    def forward(self, hidden_states, top_k_index, top_k_weights):
        return self._forward_eager(hidden_states, top_k_index, top_k_weights)

    def _prepare_expert_hidden_states(self, hidden_states):
        return hidden_states.to(self.compute_dtype)

    def _prepare_expert_execution(self, routing_plan: _ExpertRoutingPlan, implementation: str):
        _validate_expert_weights(self)
        _validate_expert_indices(routing_plan.expert_ids, self.num_experts)
        expert_ids = routing_plan.expert_ids
        if implementation == "grouped_mm":
            valid = expert_ids < self.num_experts
            valid_ids = expert_ids[valid]
            if valid_ids.numel() == 0:
                active = expert_ids.new_zeros(1)
                offsets = torch.zeros(1, dtype=torch.int32, device=expert_ids.device)
            else:
                active, counts = torch.unique_consecutive(valid_ids, return_counts=True)
                offsets = counts.cumsum(0, dtype=torch.int32)
            sentinel_mask = (~valid).unsqueeze(-1)
            return _ExpertExecutionPlan(active, None, offsets, sentinel_mask, sentinel_mask)
        valid = expert_ids < self.num_experts
        safe_ids = expert_ids.clamp(max=self.num_experts - 1)
        active, inverse = torch.unique(safe_ids, sorted=True, return_inverse=True)
        sentinel_mask = (~valid).unsqueeze(-1)
        return _ExpertExecutionPlan(active, inverse, None, sentinel_mask, sentinel_mask)

    def _project_expert_up(self, hidden_states, execution_plan, implementation):
        gate = _project_expert(
            hidden_states,
            self.gate_proj,
            execution_plan.expert_ids,
            self.compute_dtype,
            implementation,
            execution_plan.route_indices,
            execution_plan.offsets,
        )
        up = _project_expert(
            hidden_states,
            self.up_proj,
            execution_plan.expert_ids,
            self.compute_dtype,
            implementation,
            execution_plan.route_indices,
            execution_plan.offsets,
        )
        return self._apply_split_gate(gate, up)

    def _project_expert_down(self, hidden_states, execution_plan, implementation):
        return _project_expert(
            hidden_states,
            self.down_proj,
            execution_plan.expert_ids,
            self.compute_dtype,
            implementation,
            execution_plan.route_indices,
            execution_plan.offsets,
        )

    def _cast_expert_routing_weights(self, sample_weights, proj_out):
        return sample_weights.to(proj_out.dtype)

    @classmethod
    def _source_module_contract(cls, module):
        if not hasattr(module, "config") or not getattr(module, "has_gate", True):
            raise ValueError("GGUF expert replacement requires a gated source module with a config")
        if getattr(module, "has_bias", False) or getattr(module, "is_transposed", False):
            raise ValueError("GGUF expert replacement does not support expert bias or transposed projections")
        layout = getattr(module, "projection_layout", "concatenated_gate_up")
        if layout != "concatenated_gate_up":
            raise ValueError(f"GGUF expert replacement does not support source projection layout {layout!r}")
        if getattr(module, "gate_implementation", "default") not in cls._supported_source_gate_implementations:
            raise ValueError("GGUF expert replacement does not support custom gate behavior")
        projections = _expert_projection_tensors(module)
        gate_up, down = projections.get("gate_up"), projections.get("down")
        if (
            not isinstance(gate_up, torch.Tensor)
            or not isinstance(down, torch.Tensor)
            or gate_up.ndim != 3
            or down.ndim != 3
        ):
            raise ValueError("GGUF expert replacement requires rank-3 expert projection tensors")
        num_experts, gate_up_dim, hidden_dim = gate_up.shape
        if gate_up_dim % 2 or tuple(down.shape) != (num_experts, hidden_dim, gate_up_dim // 2):
            raise ValueError("GGUF expert projection dimensions are incompatible")
        if gate_up.device != down.device or gate_up.dtype != down.dtype or not gate_up.is_floating_point():
            raise ValueError("GGUF expert projections must share one floating-point device and dtype")
        return module.config, num_experts, hidden_dim, gate_up_dim // 2, gate_up.device, gate_up.dtype, module.act_fn

    @classmethod
    def from_module(cls, module, compute_dtype=None):
        config, n, hidden, intermediate, device, source_dtype, act_fn = cls._source_module_contract(module)
        cls._validate_supported_experts_implementation(getattr(config, "_experts_implementation", None))
        return cls(
            config,
            device=device,
            compute_dtype=compute_dtype or source_dtype,
            num_experts=n,
            hidden_dim=hidden,
            intermediate_dim=intermediate,
            act_fn=act_fn,
        )

    @property
    def weight_state(self):
        return _validate_expert_weights(self)


class DeepseekV4GgufExperts(GgufExperts):
    """DeepSeek V4 packed experts with clamped split SwiGLU gating."""

    _supported_source_gate_implementations = frozenset({"custom"})

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.limit = config.swiglu_limit

    def _apply_split_gate(self, gate, up):
        return self.act_fn(gate.clamp(max=self.limit)) * up.clamp(min=-self.limit, max=self.limit)
