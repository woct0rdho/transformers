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
    Cast,
    Concatenate,
    PermuteInputFeatures,
    PermuteRows,
)
from transformers.integrations.gguf.gguf_quantized_parameter import GgufQuantizedParameter
from transformers.integrations.gguf.utils import add_gguf_load_ops
from transformers.modeling_utils import PreTrainedModel


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

    def test_qwen3_moe_shared_expert_gate_mapping_restores_linear_weight_shape(self):
        rules = GGUF_ARCHS["qwen3moe"](None)
        converters = [rule for rule in rules if isinstance(rule, WeightConverter)]
        converter = next(rule for rule in converters if "ffn_gate_inp_shexp" in rule.source_patterns[0])
        source = torch.zeros(32)
        result = converter.operations[0].convert({"gate": source}, ["gate"], ["shared_expert_gate.weight"])
        self.assertEqual(tuple(result["shared_expert_gate.weight"].shape), (1, 32))

    def test_standard_decoder_mapping_covers_qwen3_moe_experts(self):
        rules = GGUF_ARCHS["qwen3moe"](None)
        renamings = [rule for rule in rules if isinstance(rule, WeightRenaming)]
        for source, expected in (
            ("blk.0.attn_q.weight", "model.layers.0.self_attn.q_proj.weight"),
            ("blk.0.ffn_gate_inp.weight", "model.layers.0.mlp.gate.weight"),
            ("blk.0.ffn_down_exps.weight", "model.layers.0.mlp.experts.down_proj"),
        ):
            name = source
            for rule in renamings:
                name, _ = rule.rename_source_key(name)
            self.assertEqual(name, expected)

    def test_gguf_plan_uses_converter_target_names(self):
        from transformers.integrations.gguf.reader import GgufHeader, TensorInfo
        from transformers.integrations.gguf.utils import get_gguf_plan

        header = GgufHeader(
            "synthetic.gguf",
            "qwen3moe",
            (
                TensorInfo("blk.0.ffn_gate_exps.weight", (2, 32), GGML_Q8_0, 0, 68),
                TensorInfo("blk.0.ffn_up_exps.weight", (2, 32), GGML_Q8_0, 68, 68),
            ),
            0,
        )
        quantized, packable, _, names = get_gguf_plan(header, GGUF_ARCHS["qwen3moe"](None))
        expected = "model.layers.0.mlp.experts.gate_up_proj"
        self.assertEqual(set(quantized), {expected})
        self.assertEqual(packable, {})
        self.assertEqual(names, [expected, expected])

    def test_deepseek_auxiliary_names_map_to_native_state(self):
        from transformers.integrations.gguf.gguf_conversion_mapping import _deepseek_v4

        rules = _deepseek_v4(None)
        cases = {
            "blk.0.indexer.attn_q_b.weight": "model.layers.0.self_attn.compressor.indexer.q_b_proj.weight",
            "blk.3.exp_probs_b.bias": "model.layers.3.mlp.gate.e_score_correction_bias",
            "blk.4.indexer_compressor_ape.weight": "model.layers.4.self_attn.compressor.indexer.position_bias",
        }
        for source, expected in cases.items():
            name = source
            for rule in rules:
                name, _ = rule.rename_source_key(name)
            self.assertEqual(name, expected)

    def test_deepseek_final_norm_survives_native_mapping(self):
        from transformers.conversion_mapping import get_checkpoint_conversion_mapping
        from transformers.core_model_loading import rename_source_key
        from transformers.integrations.gguf.gguf_conversion_mapping import _deepseek_v4

        rules = _deepseek_v4(None) + get_checkpoint_conversion_mapping("deepseek_v4")
        renamings = [rule for rule in rules if isinstance(rule, WeightRenaming)]
        converters = [rule for rule in rules if isinstance(rule, WeightConverter)]
        name, _ = rename_source_key("output_norm.weight", renamings, converters)
        self.assertEqual(name, "model.norm.weight")

    def test_qwen4_exp_ple_source_maps_to_native_embedding(self):
        text_config = SimpleNamespace(
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=4,
            linear_value_head_dim=4,
            num_hidden_layers=2,
            ple_layer_ids=[2],
        )
        config = SimpleNamespace(get_text_config=lambda: text_config)
        rules = GGUF_ARCHS["qwen4exp"](config)
        name = "per_layer_token_embd.weight"
        for rule in rules:
            name, _ = rule.rename_source_key(name)
        self.assertEqual(name, "model.layers.1.ple.ple_embedding.ngram_embedding.weight")

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

    def test_cast_keeps_fp32_strict_values_and_casts_everything_else(self):
        # `8.9697` and `9.0261` are not representable in bf16: the checkpoint stores values that are
        # finer than one bf16 step, which is exactly what the FP32-strict declaration protects.
        source = torch.tensor([8.9697, 9.0261], dtype=torch.float32)
        cast = Cast(torch.bfloat16, keep_fp32=("e_score_correction_bias", "norm"))

        kept = cast.convert(
            {"bias": source},
            ["bias"],
            ["bias"],
            full_layer_name="model.layers.0.mlp.gate.e_score_correction_bias",
        )
        bias = kept["model.layers.0.mlp.gate.e_score_correction_bias"]
        self.assertEqual(bias.dtype, torch.float32)
        torch.testing.assert_close(bias, source, rtol=0, atol=0)

        # A half-precision source keeps all of its mantissa rather than losing bits to the load dtype.
        half = torch.tensor([1.0009765625], dtype=torch.float16)
        kept = cast.convert(
            {"weight": half}, ["weight"], ["weight"], full_layer_name="model.layers.0.input_layernorm.weight"
        )
        normalized = kept["model.layers.0.input_layernorm.weight"]
        self.assertEqual(normalized.dtype, torch.float32)
        self.assertEqual(float(normalized), float(half))

        # Unmatched floating tensors, packed payloads, and tensors that already carry the load dtype
        # (what `Dequantize` produces) are untouched by the policy.
        unchanged = cast.convert(
            {"weight": source}, ["weight"], ["weight"], full_layer_name="model.layers.0.self_attn.o_proj.weight"
        )
        self.assertEqual(unchanged["model.layers.0.self_attn.o_proj.weight"].dtype, torch.bfloat16)
        packed = GgufQuantizedParameter(torch.zeros(2, 34, dtype=torch.uint8), GGML_Q8_0, (2, 32))
        self.assertIs(
            cast.convert({"weight": packed}, ["weight"], ["weight"], full_layer_name="model.layers.0.norm.weight")[
                "model.layers.0.norm.weight"
            ],
            packed,
        )
        already_cast = source.to(torch.bfloat16)
        result = cast.convert(
            {"bias": already_cast},
            ["bias"],
            ["bias"],
            full_layer_name="model.layers.0.mlp.gate.e_score_correction_bias",
        )
        self.assertEqual(result["model.layers.0.mlp.gate.e_score_correction_bias"].dtype, torch.bfloat16)

    def test_cast_leaves_integer_state_alone(self):
        # `tid2eid` is integer routing state (the model declares a long buffer); an id table has no
        # floating load dtype and must not be rounded into one.
        table = torch.tensor([[0, 255], [17, 3]], dtype=torch.int32)
        result = Cast(torch.bfloat16, keep_fp32=("norm",)).convert(
            {"ids": table}, ["ids"], ["ids"], full_layer_name="model.layers.0.mlp.gate.tid2eid"
        )
        self.assertEqual(result["model.layers.0.mlp.gate.tid2eid"].dtype, torch.int32)
        torch.testing.assert_close(result["model.layers.0.mlp.gate.tid2eid"], table, rtol=0, atol=0)

    def test_deepseek_v4_router_bias_survives_the_load_chain(self):
        from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4PreTrainedModel

        # The plan the safetensors path would use for the same model and dtype.
        dtype_plan = PreTrainedModel._get_dtype_plan(DeepseekV4PreTrainedModel, torch.bfloat16)
        self.assertIn("e_score_correction_bias", dtype_plan)
        self.assertEqual(dtype_plan["e_score_correction_bias"], torch.float32)

        name = "model.layers.10.mlp.gate.e_score_correction_bias"
        source = torch.tensor([8.9697, 9.0261], dtype=torch.float32)

        def run(keep_fp32):
            mapping = add_gguf_load_ops(GGUF_ARCHS["deepseek4"](None), {}, [name], torch.bfloat16, keep_fp32=keep_fp32)
            converter = next(
                rule for rule in mapping if isinstance(rule, WeightConverter) and rule.rename_source_key(name)[1]
            )
            tensors = {"bias": source}
            for op in converter.operations:
                tensors = op.convert(tensors, ["bias"], ["bias"], full_layer_name=name)
            return tensors[name]

        loaded = run(tuple(dtype_plan))
        self.assertEqual(loaded.dtype, torch.float32)
        torch.testing.assert_close(loaded, source, rtol=0, atol=0)
        # Without the model's plan the same chain silently rounds the value into bf16.
        self.assertEqual(run(()).dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
