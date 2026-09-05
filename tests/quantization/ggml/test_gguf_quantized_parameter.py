# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import copy
import unittest

import torch
from torch import nn

from transformers.integrations.gguf.dequant import GGML_BLOCK, GGML_Q8_0
from transformers.integrations.gguf.gguf_quantized_parameter import GgufQuantizedParameter
from transformers.modeling_utils import ModuleUtilsMixin


class _TinyModel(ModuleUtilsMixin, nn.Module):
    pass


class GgufQuantizedParameterTests(unittest.TestCase):
    @staticmethod
    def packed(rows=2):
        return GgufQuantizedParameter(torch.zeros(rows, 34, dtype=torch.uint8), GGML_Q8_0, (rows, 32))

    def test_dequantization_preserves_logical_shape(self):
        for quant_type, (block_elements, block_bytes) in GGML_BLOCK.items():
            logical_shape = (2, block_elements)
            packed = GgufQuantizedParameter(
                torch.zeros(2, block_bytes, dtype=torch.uint8), quant_type=quant_type, logical_shape=logical_shape
            )
            result = packed.dequantize()
            self.assertEqual(tuple(result.shape), logical_shape)
            self.assertEqual(result.dtype, torch.float32)
            self.assertFalse(packed.requires_grad)
            self.assertEqual(packed.logical_numel, block_elements * 2)

    def test_rejects_invalid_shape_or_payload(self):
        with self.assertRaisesRegex(ValueError, "whole number of quantization blocks"):
            GgufQuantizedParameter(torch.zeros(2, 17, dtype=torch.uint8), GGML_Q8_0, (2, 16))
        with self.assertRaisesRegex(ValueError, "expected 34"):
            GgufQuantizedParameter(torch.zeros(32, dtype=torch.uint8), GGML_Q8_0, (32,))
        with self.assertRaisesRegex(ValueError, "expected canonical shape"):
            GgufQuantizedParameter(torch.zeros(68, dtype=torch.uint8), GGML_Q8_0, (2, 32))
        with self.assertRaisesRegex(TypeError, "torch.uint8"):
            GgufQuantizedParameter(torch.zeros(34), GGML_Q8_0, (32,))

    def test_rejects_invalid_metadata(self):
        with self.assertRaises(TypeError):
            GgufQuantizedParameter(torch.zeros(1, 34, dtype=torch.uint8), GGML_Q8_0, (1, 32.0))
        with self.assertRaisesRegex(ValueError, "non-negative dimensions"):
            GgufQuantizedParameter(torch.zeros(1, 34, dtype=torch.uint8), GGML_Q8_0, (1, -32))
        with self.assertRaises(TypeError):
            GgufQuantizedParameter(torch.zeros(1, 34, dtype=torch.uint8), [], (1, 32))
        with self.assertRaisesRegex(ValueError, "unsupported quantization type"):
            GgufQuantizedParameter(torch.zeros(1, 34, dtype=torch.uint8), 999, (1, 32))
        with self.assertRaisesRegex(TypeError, "must use a torch.Tensor"):
            GgufQuantizedParameter(None, GGML_Q8_0, (1, 32))
        with self.assertRaisesRegex(ValueError, "cannot use the meta device"):
            GgufQuantizedParameter(torch.empty(1, 34, dtype=torch.uint8, device="meta"), GGML_Q8_0, (1, 32))

    def test_metadata_is_read_only_and_parameter_is_frozen(self):
        packed = self.packed()
        with self.assertRaises(AttributeError):
            packed.logical_shape = (1, 64)
        with self.assertRaises(AttributeError):
            packed.quant_type = 999
        with self.assertRaisesRegex(ValueError, "cannot require gradients"):
            GgufQuantizedParameter(torch.zeros(1, 34, dtype=torch.uint8), GGML_Q8_0, (1, 32), requires_grad=True)
        with self.assertRaisesRegex(ValueError, "attach trainable adapters"):
            packed.requires_grad_(True)
        self.assertIs(packed.requires_grad_(False), packed)

    def test_to_preserves_storage_and_metadata(self):
        packed = self.packed(rows=1)
        moved = packed.to(dtype=torch.float32, copy=True)
        self.assertIsInstance(moved, GgufQuantizedParameter)
        self.assertEqual(moved.dtype, torch.uint8)
        self.assertEqual(moved.logical_shape, (1, 32))
        self.assertEqual(moved.quant_type, GGML_Q8_0)
        self.assertFalse(moved.requires_grad)
        self.assertNotEqual(moved.data_ptr(), packed.data_ptr())

        on_cpu = moved.cpu(memory_format=torch.preserve_format)
        self.assertIsInstance(on_cpu, GgufQuantizedParameter)
        self.assertEqual(on_cpu.logical_shape, moved.logical_shape)
        with self.assertRaisesRegex(ValueError, "cannot be moved to the meta device"):
            packed.to("meta")

    def test_module_type_preserves_packed_storage(self):
        model = _TinyModel()
        model.weight = self.packed(rows=1)
        model.tied_weight = model.weight
        model.adapter = nn.Parameter(torch.ones(3))

        model.type(torch.float16)
        self.assertIs(model.weight, model.tied_weight)
        self.assertIsInstance(model.weight, GgufQuantizedParameter)
        self.assertEqual(model.weight.dtype, torch.uint8)
        self.assertEqual(model.weight.logical_shape, (1, 32))
        self.assertEqual(model.weight.quant_type, GGML_Q8_0)
        self.assertEqual(model.weight.type(), "torch.ByteTensor")
        self.assertEqual(model.adapter.dtype, torch.float16)

        model.type("torch.FloatTensor")
        self.assertIs(model.weight, model.tied_weight)
        self.assertIsInstance(model.weight, GgufQuantizedParameter)
        self.assertEqual(model.weight.dtype, torch.uint8)
        self.assertEqual(model.adapter.dtype, torch.float32)

    def test_materialization_and_tensor_transforms(self):
        packed = self.packed()
        self.assertIs(packed[...], packed)

        for transformed in (packed[:], packed.clone(), packed.detach()):
            self.assertNotIsInstance(transformed, GgufQuantizedParameter)
            self.assertFalse(hasattr(transformed, "logical_shape"))
            self.assertFalse(hasattr(transformed, "quant_type"))
        self.assertEqual(packed.detach().data_ptr(), packed.data_ptr())

    def test_copy_preserves_metadata_and_copy_semantics(self):
        packed = self.packed()
        shallow = copy.copy(packed)
        deep = copy.deepcopy(packed)

        for result in (shallow, deep):
            self.assertIsInstance(result, GgufQuantizedParameter)
            self.assertEqual(result.logical_shape, packed.logical_shape)
            self.assertEqual(result.quant_type, packed.quant_type)
        self.assertEqual(shallow.data_ptr(), packed.data_ptr())
        self.assertNotEqual(deep.data_ptr(), packed.data_ptr())

    def test_module_lifecycle_uses_logical_parameter_count_and_physical_storage(self):
        model = _TinyModel()
        packed = self.packed()
        model.weight = packed
        model.tied_weight = packed
        model.adapter = nn.Parameter(torch.ones(3))

        self.assertEqual(model.num_parameters(), packed.logical_numel + model.adapter.numel())
        self.assertEqual(model.num_parameters(only_trainable=True), model.adapter.numel())
        self.assertEqual(packed.nelement() * packed.element_size(), 68)
        state_weight = model.state_dict()["weight"]
        self.assertNotIsInstance(state_weight, GgufQuantizedParameter)
        self.assertEqual(state_weight.data_ptr(), packed.data_ptr())

        model.to(dtype=torch.float16)
        self.assertIs(model.weight, model.tied_weight)
        self.assertIsInstance(model.weight, GgufQuantizedParameter)
        self.assertEqual(model.weight.dtype, torch.uint8)
        self.assertEqual(model.adapter.dtype, torch.float16)


if __name__ == "__main__":
    unittest.main()
