# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Persistent linear and embedding modules for GGUF checkpoints."""

import torch
from torch import nn
from torch.nn import functional as F

from .gguf_quantized_parameter import GgufQuantizedParameter


class _ComputeDtypeMixin:
    def set_compute_dtype(self, dtype):
        if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
            raise TypeError(f"GGUF compute dtype must be a floating-point torch.dtype, got {dtype!r}")
        self.compute_dtype = dtype
        return self

    def _apply(self, fn, recurse=True):
        probe = fn(torch.empty(0, dtype=self.compute_dtype))
        module = nn.Module._apply(self, fn, recurse=recurse)
        if probe.dtype.is_floating_point:
            self.compute_dtype = probe.dtype
        return module


def _linear_input_gradient(grad_output, weight, input_dtype):
    flat = grad_output.reshape(-1, grad_output.shape[-1])
    return torch.mm(flat, weight).reshape(*grad_output.shape[:-1], weight.shape[-1]).to(input_dtype)


class _GgufLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, compute_dtype):
        ctx.quant_type = weight.quant_type
        ctx.logical_shape = weight.logical_shape
        ctx.compute_dtype = compute_dtype
        ctx.input_dtype = input.dtype
        ctx.bias_dtype = bias.dtype if bias is not None else None
        if ctx.needs_input_grad[0]:
            ctx.save_for_backward(weight.as_subclass(torch.Tensor))
        dense = weight.dequantize(dtype=compute_dtype, device=input.device)
        return F.linear(input, dense, bias)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_bias = None
        if ctx.needs_input_grad[0]:
            (payload,) = ctx.saved_tensors
            payload = GgufQuantizedParameter(payload, ctx.quant_type, ctx.logical_shape)
            dense = payload.dequantize(dtype=ctx.compute_dtype, device=grad_output.device)
            grad_input = _linear_input_gradient(grad_output, dense, ctx.input_dtype)
        if ctx.needs_input_grad[2]:
            dims = tuple(range(grad_output.ndim - 1))
            grad_bias = grad_output.sum(dim=dims) if dims else grad_output
            grad_bias = grad_bias.to(ctx.bias_dtype)
        return grad_input, None, grad_bias, None


class GgufLinear(_ComputeDtypeMixin, nn.Linear):
    """Linear layer with frozen raw GGUF storage and logical dense dimensions."""

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
        storage_dtype = dtype if floating_weight else torch.uint8
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), dtype=storage_dtype, device=device), requires_grad=False
        )
        self.input_permutation_offset = input_permutation_offset
        self.output_permutation_offset = output_permutation_offset
        self.input_permutation = input_permutation
        self.output_permutation = output_permutation
        self._permutation_cache = {}

    @classmethod
    def from_linear(cls, module, **kwargs):
        return cls(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            device=module.weight.device,
            dtype=module.weight.dtype,
            **kwargs,
        )

    def _permute_segment(self, value, permutation, offset, cache_key):
        if permutation is None:
            return value
        key = (cache_key, value.device.type, value.device.index)
        indices = self._permutation_cache.get(key)
        if indices is None:
            indices = torch.as_tensor(permutation, dtype=torch.long, device=value.device)
            self._permutation_cache[key] = indices
        size = indices.numel()
        if value.shape[-1] < offset + size:
            raise RuntimeError(f"GGUF layout permutation requires dimension {offset + size}, got {value.shape[-1]}")
        selected = value[..., offset : offset + size].index_select(-1, indices)
        return torch.cat((value[..., :offset], selected, value[..., offset + size :]), dim=-1)

    def _apply(self, fn, recurse=True):
        self._permutation_cache.clear()
        return super()._apply(fn, recurse=recurse)

    def materialize_logical_weight(self, *, dtype=None, device=None):
        dtype = self.compute_dtype if dtype is None else dtype
        if not dtype.is_floating_point:
            raise TypeError(f"GGUF logical weights require a floating-point dtype, got {dtype}")
        device = self.weight.device if device is None else torch.device(device)
        if isinstance(self.weight, GgufQuantizedParameter):
            weight = self.weight.dequantize(dtype=dtype, device=device)
        elif self.weight.is_floating_point():
            weight = self.weight.to(device=device, dtype=dtype)
        else:
            raise RuntimeError("GgufLinear weight has not been loaded")
        if self.input_permutation is not None:
            inverse = torch.argsort(torch.as_tensor(self.input_permutation, device=device))
            weight = self._permute_segment(weight, inverse, self.input_permutation_offset, "logical_input")
        if self.output_permutation is not None:
            transposed = self._permute_segment(
                weight.transpose(0, 1), self.output_permutation, self.output_permutation_offset, "logical_output"
            )
            weight = transposed.transpose(0, 1)
        return weight.contiguous()

    def forward(self, input):
        input = self._permute_segment(input, self.input_permutation, self.input_permutation_offset, "input")
        if isinstance(self.weight, GgufQuantizedParameter):
            compute_input = input.to(self.compute_dtype)
            bias = self.bias.to(self.compute_dtype) if self.bias is not None else None
            if torch.is_grad_enabled() and compute_input.requires_grad:
                output = _GgufLinearFunction.apply(compute_input, self.weight, bias, self.compute_dtype)
            else:
                weight = self.weight.dequantize(dtype=self.compute_dtype, device=input.device)
                output = F.linear(compute_input, weight, bias)
            output = output.to(input.dtype)
        elif self.weight.is_floating_point():
            output = F.linear(input, self.weight, self.bias)
        else:
            raise RuntimeError("GgufLinear weight has not been loaded")
        return self._permute_segment(output, self.output_permutation, self.output_permutation_offset, "output")


class _GgufGroupedLinearFunction(torch.autograd.Function):
    """Recompute a packed grouped weight for activation gradients."""

    @staticmethod
    def forward(ctx, input, weight, compute_dtype, n_groups):
        ctx.quant_type = weight.quant_type
        ctx.logical_shape = weight.logical_shape
        ctx.compute_dtype = compute_dtype
        ctx.input_dtype = input.dtype
        ctx.input_shape = tuple(input.shape)
        ctx.n_groups = n_groups
        if ctx.needs_input_grad[0]:
            ctx.save_for_backward(weight.as_subclass(torch.Tensor))
        dense_weight = weight.dequantize(dtype=compute_dtype, device=input.device)
        grouped_weight = dense_weight.view(n_groups, -1, dense_weight.shape[-1])
        flat_input = input.to(compute_dtype).reshape(-1, n_groups, input.shape[-1]).transpose(0, 1)
        output = torch.bmm(flat_input, grouped_weight.transpose(1, 2)).transpose(0, 1)
        return output.reshape(*input.shape[:-2], n_groups, -1).to(input.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        if ctx.needs_input_grad[0]:
            (payload,) = ctx.saved_tensors
            packed = GgufQuantizedParameter(payload, ctx.quant_type, ctx.logical_shape)
            dense_weight = packed.dequantize(dtype=ctx.compute_dtype, device=grad_output.device)
            grouped_weight = dense_weight.view(ctx.n_groups, -1, dense_weight.shape[-1])
            flat_grad = (
                grad_output.to(ctx.compute_dtype).reshape(-1, ctx.n_groups, grad_output.shape[-1]).transpose(0, 1)
            )
            grad_input = (
                torch.bmm(flat_grad, grouped_weight).transpose(0, 1).reshape(ctx.input_shape).to(ctx.input_dtype)
            )
        return grad_input, None, None, None


class GgufGroupedLinear(GgufLinear):
    """Packed block-diagonal linear layer."""

    def __init__(self, in_features, out_features, n_groups, **kwargs):
        if not isinstance(n_groups, int) or n_groups <= 0 or out_features % n_groups:
            raise ValueError("GGUF grouped linear requires a positive group count dividing out_features")
        super().__init__(in_features, out_features, bias=False, **kwargs)
        self.n_groups = n_groups

    @classmethod
    def from_grouped_linear(cls, module, **kwargs):
        if module.bias is not None:
            raise ValueError("GGUF grouped linear replacement does not support bias")
        return cls(
            module.in_features,
            module.out_features,
            module.n_groups,
            device=module.weight.device,
            dtype=module.weight.dtype,
            **kwargs,
        )

    def forward(self, input):
        if isinstance(self.weight, GgufQuantizedParameter):
            if torch.is_grad_enabled() and input.requires_grad:
                return _GgufGroupedLinearFunction.apply(input, self.weight, self.compute_dtype, self.n_groups)
            weight = self.weight.dequantize(dtype=self.compute_dtype, device=input.device)
            compute_input = input.to(self.compute_dtype)
        elif self.weight.is_floating_point():
            weight = self.weight.to(dtype=self.compute_dtype)
            compute_input = input
        else:
            raise RuntimeError("GgufGroupedLinear weight has not been loaded")
        input_shape = compute_input.shape[:-2]
        grouped_weight = weight.view(self.n_groups, -1, weight.shape[-1])
        flat_input = compute_input.reshape(-1, self.n_groups, compute_input.shape[-1]).transpose(0, 1)
        output = torch.bmm(flat_input, grouped_weight.transpose(1, 2)).transpose(0, 1)
        return output.reshape(*input_shape, self.n_groups, -1).to(input.dtype)


class GgufEmbedding(_ComputeDtypeMixin, nn.Embedding):
    """Embedding layer backed by row-selective GGUF dequantization."""

    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        padding_idx=None,
        device=None,
        dtype=None,
        compute_dtype=None,
        floating_weight=False,
        **kwargs,
    ):
        if (
            kwargs.get("max_norm") is not None
            or kwargs.get("norm_type", 2.0) != 2.0
            or kwargs.get("scale_grad_by_freq", False)
            or kwargs.get("sparse", False)
        ):
            raise ValueError("GgufEmbedding does not support mutable or sparse embedding options")
        super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx, device=device, dtype=dtype)
        self.set_compute_dtype(compute_dtype or dtype or torch.get_default_dtype())
        storage_dtype = dtype if floating_weight else torch.uint8
        self.weight = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), dtype=storage_dtype, device=device), requires_grad=False
        )

    @classmethod
    def from_embedding(cls, module, **kwargs):
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
            **kwargs,
        )

    def forward(self, input):
        if isinstance(self.weight, GgufQuantizedParameter):
            rows, inverse = torch.unique(input.reshape(-1), sorted=False, return_inverse=True)
            payload = self.weight.as_subclass(torch.Tensor).index_select(0, rows)
            payload = GgufQuantizedParameter(
                payload,
                quant_type=self.weight.quant_type,
                logical_shape=(rows.numel(), self.embedding_dim),
            )
            selected = payload.dequantize(dtype=self.compute_dtype, device=input.device)
            return selected.index_select(0, inverse.to(selected.device)).reshape(*input.shape, self.embedding_dim)
        if self.weight.is_floating_point():
            return F.embedding(
                input,
                self.weight,
                self.padding_idx,
                self.max_norm,
                self.norm_type,
                self.scale_grad_by_freq,
                self.sparse,
            )
        raise RuntimeError("GgufEmbedding weight has not been loaded")


class GgufQwen4ExpIndexerLinear(nn.Module):
    """Persistent Qwen4-Exp indexer projection backed by Q and K payloads."""

    def __init__(self, in_features, q_out_features, k_out_features, **kwargs):
        super().__init__()
        kwargs.pop("bias", None)
        self.q_proj = GgufLinear(in_features, q_out_features, bias=False, **kwargs)
        self.k_proj = GgufLinear(in_features, k_out_features, bias=False, **kwargs)

    @classmethod
    def from_linear(cls, module, q_out_features, k_out_features, **kwargs):
        if module.bias is not None or module.out_features != q_out_features + k_out_features:
            raise ValueError("Qwen4-Exp indexer projection must be bias-free with matching Q and K output sizes")
        return cls(
            module.in_features,
            q_out_features,
            k_out_features,
            device=module.weight.device,
            dtype=module.weight.dtype,
            **kwargs,
        )

    def forward(self, input):
        return torch.cat((self.q_proj(input), self.k_proj(input)), dim=-1)

    def materialize_logical_weight(self, **kwargs):
        return torch.cat(
            (self.q_proj.materialize_logical_weight(**kwargs), self.k_proj.materialize_logical_weight(**kwargs)), dim=0
        )


def _is_expert_candidate(name, module):
    is_experts_name = name == "experts" or name.endswith(".experts")
    has_provider = callable(getattr(module, "_get_expert_projection_tensors", None))
    has_legacy_projections = hasattr(module, "down_proj") and hasattr(module, "gate_up_proj")
    return is_experts_name and hasattr(module, "config") and (has_provider or has_legacy_projections)


def replace_with_gguf_modules(model, compute_dtype=None, floating_checkpoint_params=None, packed_parameter_names=None):
    """Replace model modules before loading so GGUF parameters retain their physical storage."""
    selective = floating_checkpoint_params is not None or packed_parameter_names is not None
    floating_checkpoint_params = set(floating_checkpoint_params or ())
    packed_parameter_names = set(packed_parameter_names or ())
    # A tied output projection has no separate GGUF source key, but it must still use a packed-aware
    # module before the loader calls `tie_weights()` and aliases its parameter to the embedding.
    for target, source in (getattr(model, "_tied_weights_keys", None) or {}).items():
        if source in packed_parameter_names:
            packed_parameter_names.add(target)
        elif source in floating_checkpoint_params:
            floating_checkpoint_params.add(target)
    modules = list(model.named_modules())
    config = getattr(model, "config", None)
    text_config = config.get_text_config() if config is not None else None
    model_type = getattr(text_config, "model_type", getattr(config, "model_type", None))
    if model_type == "deepseek_v4":
        from .moe import DeepseekV4GgufExperts as experts_class
    else:
        from .moe import GgufExperts as experts_class

    for name, module in modules:
        if name and _is_expert_candidate(name, module):
            experts_class._source_module_contract(module)

    replacements = {}
    for name, module in modules:
        if not name or isinstance(module, (GgufLinear, GgufEmbedding, GgufGroupedLinear, GgufQwen4ExpIndexerLinear)):
            continue
        parameter_name = f"{name}.weight"
        if _is_expert_candidate(name, module):
            expert_names = {
                f"{name}.gate_proj",
                f"{name}.up_proj",
                f"{name}.down_proj",
            }
            if selective and not expert_names & (packed_parameter_names | floating_checkpoint_params):
                continue
            replacement = experts_class.from_module(module, compute_dtype=compute_dtype)
        elif model_type == "qwen4_exp_text" and name.endswith(".index_qk_proj") and isinstance(module, nn.Linear):
            if selective and not any(
                f"{name}.{part}.weight" in packed_parameter_names | floating_checkpoint_params
                for part in ("q_proj", "k_proj")
            ):
                continue
            replacement = GgufQwen4ExpIndexerLinear.from_linear(
                module,
                q_out_features=text_config.indexer_n_heads * text_config.indexer_head_dim,
                k_out_features=text_config.indexer_kv_heads * text_config.indexer_head_dim,
                compute_dtype=compute_dtype,
                floating_weight=all(
                    f"{name}.{part}.weight" in floating_checkpoint_params for part in ("q_proj", "k_proj")
                ),
            )
        elif isinstance(module, nn.Linear) and hasattr(module, "n_groups"):
            if (
                selective
                and parameter_name not in packed_parameter_names
                and parameter_name not in floating_checkpoint_params
            ):
                continue
            replacement = GgufGroupedLinear.from_grouped_linear(
                module, compute_dtype=compute_dtype, floating_weight=parameter_name in floating_checkpoint_params
            )
        elif isinstance(module, nn.Linear):
            if (
                selective
                and parameter_name not in packed_parameter_names
                and parameter_name not in floating_checkpoint_params
            ):
                continue
            replacement = GgufLinear.from_linear(
                module, compute_dtype=compute_dtype, floating_weight=parameter_name in floating_checkpoint_params
            )
        elif isinstance(module, nn.Embedding):
            if (
                selective
                and parameter_name not in packed_parameter_names
                and parameter_name not in floating_checkpoint_params
            ):
                continue
            replacement = GgufEmbedding.from_embedding(
                module, compute_dtype=compute_dtype, floating_weight=parameter_name in floating_checkpoint_params
            )
        else:
            continue
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        parent._modules[child_name] = replacement
        if _is_expert_candidate(name, module):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                replacements[f"{name}.{projection}"] = replacement
        elif isinstance(replacement, GgufQwen4ExpIndexerLinear):
            replacements[f"{name}.q_proj"] = replacement.q_proj
            replacements[f"{name}.k_proj"] = replacement.k_proj
        else:
            replacements[parameter_name] = replacement
    return replacements
