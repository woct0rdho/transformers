# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from transformers.integrations.gguf.dequant import GGML_BLOCK, GGML_Q4_K
from transformers.quantizers.quantizer_gguf import GgufHfQuantizer
from transformers.utils.quantization_config import GgufConfig


class GgufLoaderTests(unittest.TestCase):
    def test_persistent_quantizer_reports_packed_element_size(self):
        quantizer = GgufHfQuantizer(GgufConfig())
        quantizer.supported = True
        quantizer.quantized = {"packed.weight": GGML_Q4_K}
        quantizer.packed_modules = {"packed.weight": object()}

        packed = torch.empty(2, 256, dtype=torch.uint8)
        ordinary = torch.empty(2, 256, dtype=torch.bfloat16)
        block_elements, block_bytes = GGML_BLOCK[GGML_Q4_K]

        self.assertAlmostEqual(
            quantizer.param_element_size(None, "packed.weight", packed), block_bytes / block_elements
        )
        self.assertEqual(quantizer.param_element_size(None, "ordinary.weight", ordinary), ordinary.element_size())

    def test_persistent_quantizer_keeps_packed_path_without_kernel(self):
        quantizer = GgufHfQuantizer(GgufConfig())
        quantizer.supported = True
        quantizer.header = SimpleNamespace(has_quantized_weights=True)
        with patch("transformers.quantizers.quantizer_gguf.get_gguf_kernel", return_value=False):
            quantizer.validate_environment()
        self.assertFalse(quantizer.quantization_config.dequantize)
        self.assertFalse(quantizer.kernel)


if __name__ == "__main__":
    unittest.main()
