# Copyright 2026 The HuggingFace Team. All rights reserved.
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
"""Persistent GGUF parameter storage and materialization."""

import copy
import math
import operator

import torch

from .dequant import GGML_BLOCK, dequantize


class GgufQuantizedParameter(torch.nn.Parameter):
    """Frozen parameter containing GGUF bytes in their canonical packed shape."""

    _logical_shape: tuple[int, ...]
    _quant_type: int

    @staticmethod
    def __new__(cls, data, quant_type, logical_shape, requires_grad=False):
        if requires_grad:
            raise ValueError("Packed GGUF parameters cannot require gradients")
        if not isinstance(data, torch.Tensor):
            raise TypeError(f"Packed GGUF parameters must use a torch.Tensor, got {type(data).__name__}")
        if data.dtype != torch.uint8:
            raise TypeError(f"Packed GGUF parameters must use torch.uint8 storage, got {data.dtype}")
        if data.device.type == "meta":
            raise ValueError("Packed GGUF parameters require materialized storage and cannot use the meta device")
        logical_shape = tuple(operator.index(dim) for dim in logical_shape)
        if not logical_shape or any(dim < 0 for dim in logical_shape):
            raise ValueError(f"Packed GGUF logical_shape must contain non-negative dimensions, got {logical_shape}")
        quant_type = operator.index(quant_type)
        if quant_type not in GGML_BLOCK:
            raise ValueError(f"Packed GGUF tensor has unsupported quantization type {quant_type!r}")

        block_elements, block_bytes = GGML_BLOCK[quant_type]
        if logical_shape[-1] % block_elements:
            raise ValueError(
                f"Packed GGUF logical shape {logical_shape} does not contain a whole number of quantization blocks "
                "in its last dimension"
            )
        expected_shape = (*logical_shape[:-1], logical_shape[-1] // block_elements * block_bytes)
        expected_storage = math.prod(expected_shape)
        if data.numel() != expected_storage:
            raise ValueError(
                f"Packed GGUF payload has {data.numel()} bytes, expected {expected_storage} for logical shape "
                f"{logical_shape} and quantization type {quant_type}"
            )
        if tuple(data.shape) != expected_shape:
            raise ValueError(
                f"Packed GGUF payload has shape {tuple(data.shape)}, expected canonical shape {expected_shape} for "
                f"logical shape {logical_shape} and quantization type {quant_type}"
            )

        instance = torch.Tensor._make_subclass(cls, data, require_grad=False)
        instance._logical_shape = logical_shape
        instance._quant_type = quant_type
        return instance

    def __init__(self, data, quant_type, logical_shape, requires_grad=False):
        # Tensor subclass construction happens in __new__; this signature is needed by static type checkers.
        pass

    @property
    def logical_shape(self):
        return self._logical_shape

    @property
    def quant_type(self):
        return self._quant_type

    @property
    def logical_numel(self):
        return math.prod(self.logical_shape)

    def _new_with_payload(self, payload):
        return type(self)(payload, quant_type=self.quant_type, logical_shape=self.logical_shape)

    def requires_grad_(self, requires_grad=True):
        if requires_grad:
            raise ValueError("Packed GGUF parameters are frozen; attach trainable adapters instead")
        return self

    def dequantize(self, dtype=None, device=None):
        if dtype is None:
            dtype = torch.float32
        payload = self.as_subclass(torch.Tensor).contiguous()
        if device is not None and payload.device != torch.device(device):
            payload = payload.to(device, non_blocking=True)
        values = dequantize(payload, self.quant_type, dtype=dtype)
        return values.reshape(self.logical_shape)

    def __getitem__(self, key):
        # The checkpoint materialization protocol reads every source with `source[...]`.
        if key is Ellipsis:
            return self
        return self.as_subclass(torch.Tensor).__getitem__(key)

    def to(self, *args, **kwargs):
        kwargs = dict(kwargs)
        copy_storage = kwargs.pop("copy", False)
        device, _, non_blocking, memory_format = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None and device.type == "meta":
            raise ValueError("Packed GGUF parameters cannot be moved to the meta device without reloading")
        payload = self.as_subclass(torch.Tensor)
        to_kwargs = {
            "device": device,
            "dtype": self.dtype,
            "non_blocking": non_blocking,
            "copy": copy_storage,
        }
        if memory_format is not None:
            to_kwargs["memory_format"] = memory_format
        return self._new_with_payload(payload.to(**to_kwargs))

    def type(self, dtype=None, non_blocking=False, **kwargs):
        if dtype is None:
            return self.as_subclass(torch.Tensor).type()
        # Module.type() bypasses to(); use an empty tensor to resolve legacy tensor types without casting the payload.
        target = torch.empty(0, device=self.device).type(dtype, non_blocking=non_blocking, **kwargs)
        return self.to(device=target.device, non_blocking=non_blocking)

    def cpu(self, memory_format=torch.preserve_format):
        return self.to(device="cpu", memory_format=memory_format)

    def cuda(self, device=None, non_blocking=False, memory_format=torch.preserve_format):
        return self.to(
            device="cuda" if device is None else device,
            non_blocking=non_blocking,
            memory_format=memory_format,
        )

    def xpu(self, device=None, non_blocking=False, memory_format=torch.preserve_format):
        return self.to(
            device="xpu" if device is None else device,
            non_blocking=non_blocking,
            memory_format=memory_format,
        )

    def __copy__(self):
        return self._new_with_payload(self.as_subclass(torch.Tensor))

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        result = self._new_with_payload(copy.deepcopy(self.as_subclass(torch.Tensor), memo))
        memo[id(self)] = result
        return result
