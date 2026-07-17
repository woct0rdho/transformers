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


def _dequantize_weight(weight: GGUFQuantizedTensor, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return dequantize_gguf_tensor(weight, weight.quant_type, dtype=dtype, device=device)


def _dequantize_rows(weight: GGUFQuantizedTensor, rows: torch.Tensor, dtype, device):
    payload = weight.as_subclass(torch.Tensor)
    rows = rows.to(payload.device)
    selected = payload.index_select(0, rows)
    return dequantize_gguf_tensor(selected, weight.quant_type, dtype=dtype, device=device)


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
    """Replace dense Qwen3 linear and embedding modules in a meta-initialized model."""
    for name, module in list(model.named_modules()):
        if not name:
            continue
        if isinstance(module, GGUFLinear | GGUFEmbedding):
            continue
        if isinstance(module, nn.Linear):
            replacement = GGUFLinear.from_linear(module, compute_dtype=compute_dtype)
        elif isinstance(module, nn.Embedding):
            replacement = GGUFEmbedding.from_embedding(module, compute_dtype=compute_dtype)
        else:
            continue
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        parent._modules[child_name] = replacement
    return model
