# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Fast (non-slow) tests that the static GGUF→HF rule table is well-formed
and covers every model_type previously supported by the legacy
``TENSOR_PROCESSORS`` map (and every model_type exercised by ``test_ggml.py``).

These tests do not download any GGUF file — they only inspect the registry,
so they can run on every PR without RUN_SLOW.
"""

from __future__ import annotations

import re
import unittest
from types import SimpleNamespace

from parameterized import parameterized

from transformers.core_model_loading import WeightConverter, WeightRenaming, rename_source_key
from transformers.gguf_conversion_ops import Qwen3_5ReorderValueHeads
from transformers.modeling_gguf_pytorch_utils import (
    _GGUF_ARCH_CONVERTERS,
    _postprocess_qwen35_config,
    get_gguf_converters,
)
from transformers.quantizers.quantizer_gguf import GGUFQuantizer


# Every HF model_type the public GGUF integration tests exercise. Keep this
# list in sync with ``tests/quantization/ggml/test_ggml.py``.
EXPECTED_MODEL_TYPES = sorted(
    {
        # Llama / RoPE family
        "llama",
        "mistral",
        "phi3",
        "cohere",
        "qwen2",
        "qwen3",
        "qwen3_5_text",
        "qwen3_5_moe_text",
        "deci",
        "stablelm",
        "starcoder2",
        # MoE families
        "qwen2_moe",
        "qwen3_moe",
        "minimax_m2",
        "gpt_oss",
        # Norm-subtract-one variants
        "nemotron",
        "gemma2",
        "gemma3",
        "gemma3_text",
        # Misc encoder/decoder & misc archs
        "bloom",
        "gpt2",
        "mamba",
        "lfm2",
        "falcon",
        "t5",
        "t5encoder",
        "umt5",
    }
)


class GgufArchCoverageTests(unittest.TestCase):
    def test_every_expected_model_type_is_registered(self):
        missing = sorted(set(EXPECTED_MODEL_TYPES) - set(_GGUF_ARCH_CONVERTERS))
        self.assertFalse(
            missing,
            f"Model types previously supported by the GGUF loader are no longer registered "
            f"in `_GGUF_ARCH_CONVERTERS`: {missing}. Add an entry (or alias to an existing "
            f"converter list) so the static rule table matches the legacy coverage.",
        )

    @parameterized.expand([(m,) for m in sorted(_GGUF_ARCH_CONVERTERS)])
    def test_arch_entry_is_well_formed(self, model_type: str):
        rules = get_gguf_converters(model_type)
        self.assertGreater(len(rules), 0, f"{model_type}: empty converter list")
        for rule in rules:
            self.assertIsInstance(
                rule,
                (WeightRenaming, WeightConverter),
                f"{model_type}: every entry must be a WeightRenaming or WeightConverter, got {type(rule).__name__}",
            )

    def test_qwen_neox_rope_weights_are_not_permuted(self):
        """Qwen GGUF files retain Hugging Face's split-half Q/K layout for NeoX RoPE."""
        from transformers.gguf_conversion_ops import ReversePermuteAttnK, ReversePermuteAttnQ

        for model_type in (
            "qwen2",
            "qwen3",
            "qwen3_5_text",
            "qwen2_moe",
            "qwen3_moe",
            "qwen3_5_moe_text",
        ):
            rules = get_gguf_converters(model_type)
            source_patterns = [source for rule in rules for source in rule.source_patterns]
            self.assertTrue(any(re.search(source, "model.layers.0.attn_q.weight") for source in source_patterns))
            self.assertTrue(any(re.search(source, "model.layers.0.attn_k.weight") for source in source_patterns))
            for rule in rules:
                if isinstance(rule, WeightConverter):
                    self.assertFalse(
                        any(isinstance(op, ReversePermuteAttnQ | ReversePermuteAttnK) for op in rule.operations),
                        f"{model_type}: Q/K weights must not receive Llama-style permutation.",
                    )

    def test_qwen3_moe_qk_norms_are_renamed(self):
        rules = get_gguf_converters("qwen3_moe")
        for projection in ("q", "k"):
            key = f"blk.0.attn_{projection}_norm.weight"
            for rule in rules:
                if isinstance(rule, WeightRenaming):
                    key, _ = rule.rename_source_key(key)
            self.assertEqual(key, f"model.layers.0.self_attn.{projection}_norm.weight")

    def test_qwen35_config_reconstruction(self):
        config = {
            "model_type": "qwen3_5_text",
            "max_position_embeddings": 262144,
            "num_hidden_layers": 8,
            "intermediate_size": 9216,
            "hidden_size": 2560,
            "head_dim": 256,
            "_gguf_attention_value_length": 256,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "rms_norm_eps": 1e-6,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 6,
            "_gguf_linear_inner_size": 768,
            "_gguf_rope_dimension_count": 64,
            "_gguf_rope_dimension_sections": [11, 11, 10, 0],
            "_gguf_rope_theta": 10_000_000.0,
            "_gguf_full_attention_interval": 4,
        }
        _postprocess_qwen35_config(config)
        self.assertEqual(config["linear_value_head_dim"], 128)
        self.assertEqual(
            config["layer_types"],
            [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ],
        )
        self.assertEqual(
            config["rope_parameters"],
            {
                "rope_type": "default",
                "rope_theta": 10_000_000.0,
                "partial_rotary_factor": 0.25,
                "mrope_section": [11, 11, 10],
                "mrope_interleaved": True,
            },
        )
        self.assertFalse(any(key.startswith("_gguf_") for key in config))

    def test_qwen35_moe_config_reconstruction(self):
        config = {
            "model_type": "qwen3_5_moe_text",
            "max_position_embeddings": 262144,
            "num_hidden_layers": 4,
            "hidden_size": 2048,
            "head_dim": 256,
            "_gguf_attention_value_length": 256,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-6,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "_gguf_linear_inner_size": 4096,
            "_gguf_rope_dimension_count": 64,
            "_gguf_rope_dimension_sections": [11, 11, 10, 0],
            "_gguf_rope_theta": 10_000_000.0,
            "_gguf_full_attention_interval": 4,
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 512,
            "shared_expert_intermediate_size": 512,
        }
        _postprocess_qwen35_config(config)
        self.assertEqual(config["model_type"], "qwen3_5_moe_text")
        self.assertEqual(config["linear_value_head_dim"], 128)
        self.assertEqual(config["num_experts"], 256)
        self.assertEqual(config["num_experts_per_tok"], 8)
        self.assertEqual(config["moe_intermediate_size"], 512)
        self.assertEqual(config["shared_expert_intermediate_size"], 512)
        self.assertEqual(
            config["layer_types"],
            ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        )

    def test_qwen35_explicit_recurrent_layers_override_interval(self):
        config = {
            "num_hidden_layers": 4,
            "head_dim": 8,
            "_gguf_attention_value_length": 8,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "_gguf_linear_inner_size": 16,
            "_gguf_rope_dimension_count": 4,
            "_gguf_rope_dimension_sections": [1, 1, 0, 0],
            "_gguf_rope_theta": 10000.0,
            "_gguf_recurrent_layers": [True, False, True, False],
            "_gguf_full_attention_interval": 3,
        }
        _postprocess_qwen35_config(config)
        self.assertEqual(
            config["layer_types"],
            ["linear_attention", "full_attention", "linear_attention", "full_attention"],
        )

    def test_qwen35_converter_names_and_value_head_reorder(self):
        import torch

        rules = get_gguf_converters("qwen3_5_text")
        renamings = [rule for rule in rules if isinstance(rule, WeightRenaming)]
        converters = [rule for rule in rules if isinstance(rule, WeightConverter)]
        expected_names = {
            "blk.0.attn_qkv.weight": "model.layers.0.linear_attn.in_proj_qkv.weight",
            "blk.0.attn_gate.weight": "model.layers.0.linear_attn.in_proj_z.weight",
            "blk.0.ssm_alpha.weight": "model.layers.0.linear_attn.in_proj_a.weight",
            "blk.0.ssm_beta.weight": "model.layers.0.linear_attn.in_proj_b.weight",
            "blk.0.ssm_a": "model.layers.0.linear_attn.A_log",
            "blk.0.ssm_dt.bias": "model.layers.0.linear_attn.dt_bias",
            "blk.0.ssm_conv1d.weight": "model.layers.0.linear_attn.conv1d.weight",
            "blk.0.ssm_norm.weight": "model.layers.0.linear_attn.norm.weight",
            "blk.0.ssm_out.weight": "model.layers.0.linear_attn.out_proj.weight",
            "blk.3.attn_q.weight": "model.layers.3.self_attn.q_proj.weight",
            "blk.3.attn_q_norm.weight": "model.layers.3.self_attn.q_norm.weight",
            "blk.3.post_attention_norm.weight": "model.layers.3.post_attention_layernorm.weight",
        }
        for source, expected in expected_names.items():
            actual, _ = rename_source_key(source, renamings, converters)
            self.assertEqual(actual, expected)

        moe_rules = get_gguf_converters("qwen3_5_moe_text")
        moe_renamings = [rule for rule in moe_rules if isinstance(rule, WeightRenaming)]
        moe_converters = [rule for rule in moe_rules if isinstance(rule, WeightConverter)]
        expected_moe_names = {
            "blk.0.ffn_gate_inp.weight": "model.layers.0.mlp.gate.weight",
            "blk.0.ffn_gate_exps.weight": "model.layers.0.mlp.experts.gate_up_proj",
            "blk.0.ffn_up_exps.weight": "model.layers.0.mlp.experts.gate_up_proj",
            "blk.0.ffn_down_exps.weight": "model.layers.0.mlp.experts.down_proj",
            "blk.0.ffn_gate_shexp.weight": "model.layers.0.mlp.shared_expert.gate_proj.weight",
            "blk.0.ffn_up_shexp.weight": "model.layers.0.mlp.shared_expert.up_proj.weight",
            "blk.0.ffn_down_shexp.weight": "model.layers.0.mlp.shared_expert.down_proj.weight",
            "blk.0.ffn_gate_inp_shexp.weight": "model.layers.0.mlp.shared_expert_gate.weight",
        }
        for source, expected in expected_moe_names.items():
            actual, _ = rename_source_key(source, moe_renamings, moe_converters)
            self.assertEqual(actual, expected)

        config = SimpleNamespace(
            linear_num_key_heads=2,
            linear_num_value_heads=6,
            linear_key_head_dim=3,
            linear_value_head_dim=2,
        )
        physical_order = torch.arange(12).reshape(2, 3, 2).transpose(0, 1).reshape(-1)
        canonical = torch.arange(12, dtype=torch.float32)
        physical = canonical.index_select(0, physical_order)
        op = Qwen3_5ReorderValueHeads(head_dim=2)
        actual = op.convert({"source": physical}, ["source"], ["target"], config=config)["target"]
        torch.testing.assert_close(actual, canonical)

        canonical_conv = torch.arange(24 * 4, dtype=torch.float32).reshape(24, 4)
        physical_conv = canonical_conv.clone()
        physical_conv[12:] = canonical_conv[12:].index_select(0, physical_order)
        conv_op = Qwen3_5ReorderValueHeads(head_dim=None, value_offset="qkv")
        actual_conv = conv_op.convert({"source": physical_conv}, ["source"], ["target"], config=config)["target"]
        torch.testing.assert_close(actual_conv, canonical_conv)

    def test_quantizer_prepends_gguf_dequantize_to_every_converter(self):
        """``GGUFQuantizer.update_weight_conversions`` injects ``GGUFDequantize`` at the head
        of every ``WeightConverter`` op chain — same pattern as ``Fp8Quantizer``.
        ``WeightRenaming`` entries pass through unmodified.
        """
        from transformers.gguf_conversion_ops import GGUFDequantize

        for model_type, rules in _GGUF_ARCH_CONVERTERS.items():
            quantizer = GGUFQuantizer(weight_mapping=rules)
            out = quantizer.update_weight_conversions([])
            for rule in out:
                if isinstance(rule, WeightConverter):
                    self.assertIsInstance(
                        rule.operations[0],
                        GGUFDequantize,
                        f"{model_type}: WeightConverter not prefixed with GGUFDequantize.",
                    )
