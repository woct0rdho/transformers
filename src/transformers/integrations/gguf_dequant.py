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

``GGUFQuantizedTensor`` is a ``torch.Tensor`` subclass that carries the
``quant_type`` metadata alongside the raw uint8 bytes, so the standard
loader can move it to the target device with ``.to(device)`` and the
``GGUFDequantize`` op in the weight-conversion chain can dequant on-device
without any GGUF-specific hook in ``core_model_loading``.
"""

from __future__ import annotations

import torch


class GGUFQuantizedTensor(torch.Tensor):
    """``torch.Tensor`` subclass that carries the GGUF ``quant_type`` alongside
    raw uint8 bytes.

    Three small affordances make the hot path through ``spawn_materialize`` fast
    *without* touching ``core_model_loading``:

    * ``__getitem__(Ellipsis)`` short-circuits — ``tensor[...]`` is a no-op for
      already-loaded torch bytes; the default goes through ``__torch_function__``
      dispatch for no reason.
    * ``.to(...)`` defaults ``non_blocking=True``, letting the loader queue the
      next transfer / dequant on the same MPS/CUDA stream while the previous
      copy is still in flight.
    * ``__torch_function__`` only re-wraps on ``Tensor.to`` — the one place we
      need ``quant_type`` to survive (so the :class:`GGUFDequantize` op in the
      conversion chain can read it on the device side). All other ops return
      plain tensors, which avoids the per-op wrap overhead.

    Inspired by ``GGUFParameter`` in diffusers.
    """

    # Class-level default so subclass instances spawned by torch's default
    # ``__torch_function__`` path (which doesn't invoke our ``__new__``) still
    # have the attribute defined.
    quant_type = None

    @staticmethod
    def __new__(cls, data, quant_type=None):
        data = data if data is not None else torch.empty(0)
        instance = torch.Tensor._make_subclass(cls, data, require_grad=False)
        instance.quant_type = quant_type
        return instance

    def __getitem__(self, key):
        # ``_materialize_copy`` does ``tensor = tensor[...]`` to pull a memmap
        # safetensors slice into RAM. Our bytes are already a torch.Tensor view,
        # so this is a no-op; short-circuit before torch dispatches via
        # ``__torch_function__`` (which would re-wrap into a fresh subclass).
        if key is Ellipsis:
            return self
        return super().__getitem__(key)

    def to(self, *args, **kwargs):
        # The loader queues a dequant op on the destination device right after
        # this transfer (same MPS/CUDA stream), so ``non_blocking=True`` overlaps
        # the bytes copy with the next ``.to(device)`` call from another worker.
        kwargs.setdefault("non_blocking", True)
        return super().to(*args, **kwargs)

    @staticmethod
    def _extract_quant_type(args):
        for arg in args:
            if isinstance(arg, GGUFQuantizedTensor) and arg.quant_type is not None:
                return arg.quant_type
            if isinstance(arg, (list, tuple)) and arg and isinstance(arg[0], GGUFQuantizedTensor):
                if arg[0].quant_type is not None:
                    return arg[0].quant_type
        return None

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        result = super().__torch_function__(func, types, args, kwargs)
        # Only re-wrap on ``Tensor.to`` — the GGUFDequantize op in the conversion
        # chain reads ``quant_type`` off the post-transfer tensor. Other ops in
        # the path don't care, so skipping the wrap saves Python overhead per call.
        if func is not torch.Tensor.to:
            return result
        quant_type = cls._extract_quant_type(args)
        if quant_type is None:
            return result
        if isinstance(result, cls):
            result.quant_type = quant_type
            return result
        if isinstance(result, torch.Tensor):
            return cls(result, quant_type=quant_type)
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
