# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import unittest

import numpy as np
import torch

from transformers.integrations.gguf.dequant import GGML_Q8_0
from transformers.integrations.gguf.gguf_quantized_parameter import GgufQuantizedParameter
from transformers.integrations.gguf.reader import LazyGgufTensor


class GgufReaderTests(unittest.TestCase):
    def test_packed_tensor_exposes_physical_shape_and_logical_metadata(self):
        source = LazyGgufTensor(np.zeros(34, dtype=np.uint8), GGML_Q8_0, (1, 34), (1, 32))
        self.assertEqual(source.get_shape(), [1, 34])
        self.assertEqual(source.get_dtype(), "UINT8")
        packed = source[...]
        self.assertIsInstance(packed, GgufQuantizedParameter)
        self.assertEqual(packed.logical_shape, (1, 32))

    def test_scalar_tensor_reinterprets_dtype_and_allows_slicing(self):
        source = LazyGgufTensor(np.zeros(8, dtype=np.uint8), 26, (2,))
        self.assertEqual(source.get_shape(), [2])
        self.assertEqual(source.get_dtype(), "INT32")
        values = source[1:]
        self.assertEqual(values.dtype, torch.int32)
        self.assertEqual(tuple(values.shape), (1,))

    def test_packed_tensor_rejects_partial_slices(self):
        source = LazyGgufTensor(np.zeros(34, dtype=np.uint8), GGML_Q8_0, (1, 34), (1, 32))
        with self.assertRaisesRegex(ValueError, "complete tensor"):
            source[:]


if __name__ == "__main__":
    unittest.main()
