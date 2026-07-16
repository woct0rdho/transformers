# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""GGUF-specific :class:`ConversionOps` for persistent metadata and compatibility dequantization."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .core_model_loading import ConversionOps


if TYPE_CHECKING:
    import torch


def _single_input_target(input_dict, source_patterns, target_patterns):
    """Return the output key for a single-input conversion op."""
    if len(input_dict) != 1:
        raise ValueError("Undefined Operation encountered!")
    if len(target_patterns) > 1:
        if len(source_patterns) == 1:
            return source_patterns[0]
        raise ValueError("Undefined Operation encountered!")
    target = target_patterns[0]
    if r"\1" in target:
        return next(iter(input_dict))
    return target


class GGUFDequantize(ConversionOps):
    """First op in every GGUF ``WeightConverter`` chain.

    Reads ``quant_type`` from each input :class:`GGUFQuantizedTensor` and
    dequantizes the raw uint8 bytes to a floating-point ``torch.Tensor`` using
    the pure-torch kernels in ``integrations/gguf_dequant.py`` (city96-style,
    same kernels diffusers uses). The dequant runs on whatever device the
    input tensor is already on, so the loader's ``.to(device)`` upstream means
    MPS / CUDA dequant happens on-device.
    """

    def convert(
        self,
        input_dict,
        source_patterns,
        target_patterns,
        **kwargs,
    ):
        from .integrations.gguf_dequant import GGUFQuantizedTensor, dequantize_gguf_tensor

        out = {}
        for key, tensors in input_dict.items():
            tensors_list = tensors if isinstance(tensors, list) else [tensors]
            dequantized = [
                dequantize_gguf_tensor(t, t.quant_type, device=t.device) if isinstance(t, GGUFQuantizedTensor) else t
                for t in tensors_list
            ]
            out[key] = dequantized if isinstance(tensors, list) else dequantized[0]
        return out

    @property
    def reverse_op(self):
        raise NotImplementedError("GGUFDequantize is one-way")


class GGUFSetMetadata(ConversionOps):
    """Keep a GGUF parameter compressed while resolving its converted target key."""

    def convert(self, input_dict, source_patterns, target_patterns, **kwargs):
        target_pattern = _single_input_target(input_dict, source_patterns, target_patterns)
        return {target_pattern: next(iter(input_dict.values()))}

    @property
    def reverse_op(self):
        raise NotImplementedError("GGUF metadata assignment is one-way")


class Unsqueeze(ConversionOps):
    """Unsqueeze a tensor along ``dim``."""

    def __init__(self, dim: int = 0):
        self.dim = dim

    def convert(
        self,
        input_dict: dict[str, torch.Tensor],
        source_patterns: list[str],
        target_patterns: list[str],
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        target_pattern = _single_input_target(input_dict, source_patterns, target_patterns)
        tensors = next(iter(input_dict.values()))
        tensor = tensors[0] if isinstance(tensors, list) else tensors
        return {target_pattern: tensor.unsqueeze(self.dim)}

    @property
    def reverse_op(self) -> ConversionOps:
        return Squeeze(self.dim)


class Squeeze(ConversionOps):
    """Squeeze a tensor along ``dim``. Inverse of :class:`Unsqueeze`."""

    def __init__(self, dim: int = 0):
        self.dim = dim

    def convert(
        self,
        input_dict: dict[str, torch.Tensor],
        source_patterns: list[str],
        target_patterns: list[str],
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        target_pattern = _single_input_target(input_dict, source_patterns, target_patterns)
        tensors = next(iter(input_dict.values()))
        tensor = tensors[0] if isinstance(tensors, list) else tensors
        return {target_pattern: tensor.squeeze(self.dim)}

    @property
    def reverse_op(self) -> ConversionOps:
        return Unsqueeze(self.dim)


class SubtractOne(ConversionOps):
    """Subtract 1 from a tensor (used for GGUF norm weight de-offset)."""

    def convert(
        self,
        input_dict: dict[str, torch.Tensor],
        source_patterns: list[str],
        target_patterns: list[str],
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        target_pattern = _single_input_target(input_dict, source_patterns, target_patterns)
        tensors = next(iter(input_dict.values()))
        tensor = tensors[0] if isinstance(tensors, list) else tensors
        return {target_pattern: tensor - 1}

    @property
    def reverse_op(self) -> ConversionOps:
        return AddOne()


class AddOne(ConversionOps):
    """Add 1 to a tensor. Inverse of :class:`SubtractOne`."""

    def convert(
        self,
        input_dict: dict[str, torch.Tensor],
        source_patterns: list[str],
        target_patterns: list[str],
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        target_pattern = _single_input_target(input_dict, source_patterns, target_patterns)
        tensors = next(iter(input_dict.values()))
        tensor = tensors[0] if isinstance(tensors, list) else tensors
        return {target_pattern: tensor + 1}

    @property
    def reverse_op(self) -> ConversionOps:
        return SubtractOne()


class Qwen3_5ReorderValueHeads(ConversionOps):
    """Restore canonical Qwen3.5 value-head order for small floating GGUF tensors.

    llama.cpp tiles value heads by their position within each key-head group. Transformers groups
    all value heads belonging to a key head contiguously. Large projection matrices retain the GGUF
    layout and adapt activations at runtime; this operation is only for small state and convolution
    tensors that are stored unquantized.
    """

    def __init__(self, dim: int = 0, head_dim: int | None = 1, value_offset: str | None = None):
        self.dim = dim
        self.head_dim = head_dim
        self.value_offset = value_offset

    def convert(
        self,
        input_dict: dict[str, torch.Tensor],
        source_patterns: list[str],
        target_patterns: list[str],
        config: Any = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        import torch

        from .integrations.gguf_dequant import GGUFQuantizedTensor

        target_pattern = _single_input_target(input_dict, source_patterns, target_patterns)
        tensors = next(iter(input_dict.values()))
        tensor = tensors[0] if isinstance(tensors, list) else tensors
        if isinstance(tensor, GGUFQuantizedTensor):
            raise ValueError("Qwen3.5 value-head loading conversion requires a floating-point GGUF tensor")
        if not tensor.is_floating_point():
            raise ValueError("Qwen3.5 value-head loading conversion requires a floating-point tensor")

        num_key_heads = config.linear_num_key_heads
        num_value_heads = config.linear_num_value_heads
        if num_value_heads % num_key_heads:
            raise ValueError(
                "Qwen3.5 linear_num_value_heads must be divisible by linear_num_key_heads, got "
                f"{num_value_heads} and {num_key_heads}"
            )
        value_heads_per_key = num_value_heads // num_key_heads
        head_dim = config.linear_value_head_dim if self.head_dim is None else self.head_dim
        value_size = num_value_heads * head_dim
        value_offset = 0
        if self.value_offset == "qkv":
            value_offset = 2 * config.linear_num_key_heads * config.linear_key_head_dim
        elif self.value_offset is not None:
            raise ValueError(f"Unknown Qwen3.5 value-head offset {self.value_offset!r}")

        dim = self.dim % tensor.ndim
        if tensor.shape[dim] < value_offset + value_size:
            raise ValueError(
                f"Qwen3.5 value-head dimension {tensor.shape[dim]} is smaller than the required "
                f"offset plus value size {value_offset + value_size}"
            )
        physical_order = (
            torch.arange(value_size, device=tensor.device)
            .reshape(num_key_heads, value_heads_per_key, head_dim)
            .transpose(0, 1)
            .reshape(-1)
        )
        canonical_order = torch.argsort(physical_order)
        values = tensor.narrow(dim, value_offset, value_size).index_select(dim, canonical_order)
        pieces = []
        if value_offset:
            pieces.append(tensor.narrow(dim, 0, value_offset))
        pieces.append(values)
        suffix_offset = value_offset + value_size
        if suffix_offset < tensor.shape[dim]:
            pieces.append(tensor.narrow(dim, suffix_offset, tensor.shape[dim] - suffix_offset))
        return {target_pattern: torch.cat(pieces, dim=dim) if len(pieces) > 1 else pieces[0]}

    @property
    def reverse_op(self) -> ConversionOps:
        raise NotImplementedError("Qwen3_5ReorderValueHeads is one-way")


class LogNegate(ConversionOps):
    """Apply ``log(-tensor)`` (used for GGUF Mamba SSM-A de-transform)."""

    def convert(
        self,
        input_dict: dict[str, torch.Tensor],
        source_patterns: list[str],
        target_patterns: list[str],
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        import torch

        target_pattern = _single_input_target(input_dict, source_patterns, target_patterns)
        tensors = next(iter(input_dict.values()))
        tensor = tensors[0] if isinstance(tensors, list) else tensors
        return {target_pattern: torch.log(-tensor)}

    @property
    def reverse_op(self) -> ConversionOps:
        raise NotImplementedError("LogNegate is not easily reversible")


class ReversePermuteAttnQ(ConversionOps):
    """Reverse Q-projection GGUF permutation. Reads ``config.num_attention_heads`` at convert time."""

    def convert(
        self,
        input_dict: dict[str, torch.Tensor],
        source_patterns: list[str],
        target_patterns: list[str],
        config: Any = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        num_heads = config.num_attention_heads
        target_pattern = _single_input_target(input_dict, source_patterns, target_patterns)
        tensors = next(iter(input_dict.values()))
        tensor = tensors[0] if isinstance(tensors, list) else tensors
        dim = tensor.shape[0] // num_heads // 2
        return {
            target_pattern: tensor.reshape(num_heads, dim, 2, *tensor.shape[1:]).swapaxes(2, 1).reshape(tensor.shape)
        }

    @property
    def reverse_op(self) -> ConversionOps:
        raise NotImplementedError("ReversePermuteAttnQ is one-way")


class ReversePermuteAttnK(ConversionOps):
    """Reverse K-projection GGUF permutation. Reads ``config.num_attention_heads`` and
    ``config.num_key_value_heads`` at convert time."""

    def convert(
        self,
        input_dict: dict[str, torch.Tensor],
        source_patterns: list[str],
        target_patterns: list[str],
        config: Any = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        num_kv_heads = config.num_key_value_heads
        target_pattern = _single_input_target(input_dict, source_patterns, target_patterns)
        tensors = next(iter(input_dict.values()))
        tensor = tensors[0] if isinstance(tensors, list) else tensors
        dim = tensor.shape[0] // num_kv_heads // 2
        return {
            target_pattern: tensor.reshape(num_kv_heads, dim, 2, *tensor.shape[1:])
            .swapaxes(2, 1)
            .reshape(tensor.shape)
        }

    @property
    def reverse_op(self) -> ConversionOps:
        raise NotImplementedError("ReversePermuteAttnK is one-way")


class BloomReshapeQKVWeight(ConversionOps):
    """Reverse Bloom QKV weight reshape. Reads ``config.n_head`` and ``config.hidden_size`` at convert time."""

    def convert(
        self,
        input_dict: dict[str, torch.Tensor],
        source_patterns: list[str],
        target_patterns: list[str],
        config: Any = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        import torch

        n_head = config.n_head
        n_embed = config.hidden_size
        target_pattern = _single_input_target(input_dict, source_patterns, target_patterns)
        tensors = next(iter(input_dict.values()))
        w = tensors[0] if isinstance(tensors, list) else tensors
        q, k, v = torch.chunk(w, 3, dim=0)
        q = q.reshape(n_head, n_embed // n_head, n_embed)
        k = k.reshape(n_head, n_embed // n_head, n_embed)
        v = v.reshape(n_head, n_embed // n_head, n_embed)
        qkv = torch.stack([q, k, v], dim=1)
        return {target_pattern: qkv.reshape(n_head * 3 * (n_embed // n_head), n_embed)}

    @property
    def reverse_op(self) -> ConversionOps:
        raise NotImplementedError("BloomReshapeQKVWeight is one-way")


class BloomReshapeQKVBias(ConversionOps):
    """Reverse Bloom QKV bias reshape. Reads ``config.n_head`` and ``config.hidden_size`` at convert time."""

    def convert(
        self,
        input_dict: dict[str, torch.Tensor],
        source_patterns: list[str],
        target_patterns: list[str],
        config: Any = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        import torch

        n_head = config.n_head
        n_embed = config.hidden_size
        target_pattern = _single_input_target(input_dict, source_patterns, target_patterns)
        tensors = next(iter(input_dict.values()))
        w = tensors[0] if isinstance(tensors, list) else tensors
        q, k, v = torch.chunk(w, 3)
        q = q.reshape(n_head, n_embed // n_head)
        k = k.reshape(n_head, n_embed // n_head)
        v = v.reshape(n_head, n_embed // n_head)
        return {target_pattern: torch.stack([q, k, v], dim=1).flatten()}

    @property
    def reverse_op(self) -> ConversionOps:
        raise NotImplementedError("BloomReshapeQKVBias is one-way")
