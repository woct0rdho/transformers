# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import unittest

from transformers.quantizers.quantizer_gguf import GgufHfQuantizer
from transformers.utils.quantization_config import GgufConfig


class GgufQuantizerTests(unittest.TestCase):
    def test_persistent_quantizer_allows_adapters_but_not_base_training(self):
        quantizer = GgufHfQuantizer(GgufConfig())
        quantizer.supported = True
        self.assertTrue(quantizer.is_trainable)
        self.assertFalse(quantizer.is_compileable)
        self.assertFalse(quantizer.is_serializable())

    def test_dense_compatibility_quantizer_keeps_main_lifecycle(self):
        quantizer = GgufHfQuantizer(GgufConfig(dequantize=True))
        self.assertFalse(quantizer.is_trainable)
        self.assertTrue(quantizer.is_compileable)
        self.assertFalse(quantizer.is_serializable())


if __name__ == "__main__":
    unittest.main()
