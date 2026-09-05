# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import unittest

import torch

from transformers.integrations.gguf.dequant import GGML_Q8_0
from transformers.integrations.gguf.gguf_conversion_mapping import (
    Concatenate,
    PermuteInputFeatures,
    PermuteRows,
)
from transformers.integrations.gguf.gguf_quantized_parameter import GgufQuantizedParameter


class GgufConversionOpsTests(unittest.TestCase):
    def test_concatenate_accepts_dense_values_and_rejects_packed_values(self):
        dense = Concatenate(dim=1).convert(
            {"gate": [torch.zeros(1, 2)], "up": [torch.ones(1, 2)]}, ["gate", "up"], ["gate_up"]
        )
        self.assertEqual(tuple(dense["gate_up"].shape), (1, 4))
        packed = GgufQuantizedParameter(torch.zeros(1, 34, dtype=torch.uint8), GGML_Q8_0, (1, 32))
        with self.assertRaisesRegex(ValueError, "cannot be concatenated"):
            Concatenate(dim=1).convert({"gate": [packed], "up": [packed]}, ["gate", "up"], ["gate_up"])

    def test_packed_row_permutation_preserves_metadata(self):
        packed = GgufQuantizedParameter(torch.zeros(2, 34, dtype=torch.uint8), GGML_Q8_0, (2, 32))
        result = PermuteRows(torch.tensor([1, 0])).convert({"weight": packed}, ["weight"], ["weight"])["weight"]
        self.assertIsInstance(result, GgufQuantizedParameter)
        self.assertEqual(result.quant_type, GGML_Q8_0)
        self.assertEqual(result.logical_shape, (2, 32))

    def test_packed_input_permutation_keeps_payload_unchanged(self):
        packed = GgufQuantizedParameter(torch.zeros(2, 34, dtype=torch.uint8), GGML_Q8_0, (2, 32))
        result = PermuteInputFeatures(torch.tensor([1, 0])).convert({"weight": packed}, ["weight"], ["weight"])[
            "weight"
        ]
        self.assertIs(result, packed)


if __name__ == "__main__":
    unittest.main()
