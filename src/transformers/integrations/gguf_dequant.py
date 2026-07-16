# Copyright 2025 The HuggingFace Inc. team and City96. All rights reserved.
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
"""Pure-PyTorch GGUF dequantization.

Port of city96 / ComfyUI-GGUF (also used by diffusers' GGUF quantizer):
https://github.com/city96/ComfyUI-GGUF/blob/main/dequant.py
https://github.com/huggingface/diffusers/blob/main/src/diffusers/quantizers/gguf/utils.py

The reference dequant in ``gguf-py`` is pure NumPy and operates on grouped
row slices, which is ~5–20x slower than the same logic expressed as
``torch`` ops on a ``uint8`` view of the raw bytes (and also avoids the
``__array_finalize__`` overhead from memmap slicing). The ops here run on
CPU or GPU unchanged — pass an already-on-device tensor as input.

``GGUFQuantizedTensor`` is a frozen ``torch.nn.Parameter`` subclass that carries
intrinsic ``quant_type`` and ``logical_shape`` metadata alongside raw uint8
bytes. The standard loader can move it to the target device with ``.to(device)``;
persistent GGUF modules retain it as a packed parameter, while
``GGUFDequantize`` handles compatibility loading.
"""

from __future__ import annotations

import copy
import math

import torch


class GGUFQuantizedTensor(torch.nn.Parameter):
    """Frozen model parameter containing a raw GGUF payload.

    The historical name is kept for compatibility, but packed GGUF weights use
    parameter semantics like bitsandbytes' ``Params4bit`` and ``Int8Params``.
    ``logical_shape`` describes the dequantized weight while the underlying
    tensor shape and dtype describe its physical checkpoint storage.
    """

    logical_shape = None
    quant_type = None

    @staticmethod
    def __new__(cls, data, quant_type=None, logical_shape=None, requires_grad=False):
        if requires_grad:
            raise ValueError("Packed GGUF parameters cannot require gradients")
        data = data if data is not None else torch.empty(0, dtype=torch.uint8)
        if data.dtype != torch.uint8:
            raise TypeError(f"Packed GGUF parameters must use torch.uint8 storage, got {data.dtype}")
        instance = torch.Tensor._make_subclass(cls, data, require_grad=False)
        instance.logical_shape = tuple(logical_shape) if logical_shape is not None else tuple(data.shape)
        instance.quant_type = quant_type
        return instance

    def __init__(self, data, quant_type=None, logical_shape=None, requires_grad=False):
        pass

    @property
    def gguf_metadata(self):
        return {"logical_shape": self.logical_shape, "quant_type": self.quant_type}

    @property
    def logical_numel(self):
        return math.prod(self.logical_shape)

    @property
    def storage_nbytes(self):
        return self.numel() * self.element_size()

    def __getitem__(self, key):
        if key is Ellipsis:
            return self
        return self.as_subclass(torch.Tensor).__getitem__(key)

    def to(self, *args, **kwargs):
        kwargs = dict(kwargs)
        copy_storage = kwargs.pop("copy", False)
        device, _, non_blocking, memory_format = torch._C._nn._parse_to(*args, **kwargs)
        payload = self.as_subclass(torch.Tensor)
        to_kwargs = {
            "device": device,
            "dtype": self.dtype,
            "non_blocking": non_blocking,
            "copy": copy_storage,
        }
        if memory_format is not None:
            to_kwargs["memory_format"] = memory_format
        moved = payload.to(**to_kwargs)
        return type(self)(moved, **self.gguf_metadata)

    def cpu(self):
        return self.to(device="cpu")

    def cuda(self, device=None, non_blocking=False):
        return self.to(device="cuda" if device is None else device, non_blocking=non_blocking)

    def xpu(self, device=None, non_blocking=False):
        return self.to(device="xpu" if device is None else device, non_blocking=non_blocking)

    def __copy__(self):
        return type(self)(self.as_subclass(torch.Tensor), **self.gguf_metadata)

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        result = type(self)(copy.deepcopy(self.as_subclass(torch.Tensor), memo), **self.gguf_metadata)
        memo[id(self)] = result
        return result


def dequantize_gguf_tensor(data, quant_type, dtype=None, device=None) -> torch.Tensor:
    """Dequantize a GGUF payload on the requested device.

    Args:
        data: A GGUF reader NumPy array or a tensor containing the raw payload.
        quant_type: The corresponding ``gguf.GGMLQuantizationType`` value.
        dtype: Floating-point output dtype, defaulting to ``torch.float32``.
        device: Device on which to run the dequantization kernel.
    """
    from .gguf_dequant_kernels import dequantize

    if dtype is None:
        dtype = torch.float32
    target_device = torch.device(device) if device is not None else None

    if isinstance(data, GGUFQuantizedTensor):
        payload = data.as_subclass(torch.Tensor).contiguous()
    elif isinstance(data, torch.Tensor):
        payload = data.contiguous()
    else:
        import warnings

        import numpy as np

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
            payload = torch.from_numpy(np.ascontiguousarray(data))

    if target_device is not None and payload.device != target_device:
        payload = payload.to(target_device, non_blocking=True)

    return dequantize(payload, quant_type, dtype=dtype)
