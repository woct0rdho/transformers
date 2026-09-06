# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import unittest
from types import SimpleNamespace

import torch

from transformers.core_model_loading import WeightConverter, WeightRenaming
from transformers.integrations.gguf.dequant import GGML_Q8_0
from transformers.integrations.gguf.gguf_conversion_mapping import (
    GGUF_ARCHS,
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

    def test_qwen35_moe_shared_expert_gate_mapping_restores_linear_weight_shape(self):
        text_config = SimpleNamespace(
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=4,
            linear_value_head_dim=4,
            num_hidden_layers=2,
        )
        config = SimpleNamespace(get_text_config=lambda: text_config)
        rules = GGUF_ARCHS["qwen35moe"](config)
        converters = [rule for rule in rules if isinstance(rule, WeightConverter)]
        converter = next(rule for rule in converters if "ffn_gate_inp_shexp" in rule.source_patterns[0])
        source = torch.zeros(32)
        result = converter.operations[0].convert({"gate": source}, ["gate"], ["shared_expert_gate.weight"])
        self.assertEqual(tuple(result["shared_expert_gate.weight"].shape), (1, 32))

    def test_qwen35_moe_mapping_converts_expert_names(self):
        text_config = SimpleNamespace(
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=4,
            linear_value_head_dim=4,
            num_hidden_layers=2,
        )
        config = SimpleNamespace(get_text_config=lambda: text_config)
        rules = GGUF_ARCHS["qwen35moe"](config)
        renamings = [rule for rule in rules if isinstance(rule, WeightRenaming)]
        for source, expected in (
            ("blk.0.ffn_gate_inp.weight", "model.layers.0.mlp.gate.weight"),
            ("blk.0.ffn_down_exps.weight", "model.layers.0.mlp.experts.down_proj"),
        ):
            name = source
            for rule in renamings:
                name, _ = rule.rename_source_key(name)
            self.assertEqual(name, expected)

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
