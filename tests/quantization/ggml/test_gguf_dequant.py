# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import unittest

import torch

from transformers.integrations.gguf.dequant import GGML_BLOCK, dequantize


class GgufDequantTests(unittest.TestCase):
    def test_all_dequantization_types_return_flat_values(self):
        for quant_type, (block_elements, block_bytes) in GGML_BLOCK.items():
            payload = torch.zeros(block_bytes, dtype=torch.uint8)
            result = dequantize(payload, quant_type)
            self.assertEqual(tuple(result.shape), (block_elements,))
            self.assertTrue(result.is_floating_point())

    def test_kernels_cast_only_after_dequantization(self):
        for quant_type, (_, block_bytes) in GGML_BLOCK.items():
            payload = torch.arange(block_bytes, dtype=torch.uint8)
            expected = dequantize(payload, quant_type, torch.float32)
            for dtype in (torch.float16, torch.bfloat16):
                actual = dequantize(payload, quant_type, dtype)
                self.assertEqual(actual.dtype, dtype)
                torch.testing.assert_close(actual.float(), expected.to(dtype).float(), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
