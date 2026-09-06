# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import unittest

from transformers.integrations.gguf.gguf_config_mapping import _qwen4_exp_config, get_gguf_config


class GgufConfigTests(unittest.TestCase):
    def test_qwen4_exp_metadata_requires_ple_and_tensor_inventory(self):
        metadata = {
            "general.architecture": "qwen4exp",
            "tokenizer.ggml.tokens": 32,
            "qwen4exp.context_length": 128,
            "qwen4exp.embedding_length": 32,
            "qwen4exp.feed_forward_length": 16,
            "qwen4exp.block_count": 2,
            "qwen4exp.attention.key_length": 8,
            "qwen4exp.attention.head_count": 4,
            "qwen4exp.attention.head_count_kv": 1,
            "qwen4exp.attention.layer_norm_rms_epsilon": 1e-6,
            "qwen4exp.rope.freq_base": 10000.0,
            "qwen4exp.rope.dimension_count": 4,
            "qwen4exp.rope.dimension_sections": [1, 1, 0, 0],
            "qwen4exp.ssm.conv_kernel": 4,
            "qwen4exp.ssm.state_size": 4,
            "qwen4exp.ssm.group_count": 2,
            "qwen4exp.ssm.time_step_rank": 4,
            "qwen4exp.ssm.inner_size": 16,
            "qwen4exp.hyper_connection.count": 4,
            "qwen4exp.hyper_connection.low_rank": 8,
            "qwen4exp.attention.indexer.head_count": 2,
            "qwen4exp.attention.indexer.key_length": 4,
            "qwen4exp.attention.indexer.top_k": 4,
            "qwen4exp.expert_count": 4,
            "qwen4exp.expert_used_count": 2,
            "qwen4exp.expert_feed_forward_length": 8,
            "qwen4exp.expert_shared_feed_forward_length": 8,
            "qwen4exp.attention.compress_ratios": [0, 2],
        }
        with self.assertRaisesRegex(ValueError, "PLE|missing required tensors"):
            get_gguf_config(metadata, ("token_embd.weight",))

        metadata.update(
            {
                "qwen4exp.ple.layers": [0],
                "qwen4exp.ple.ngram_size": 3,
                "qwen4exp.ple.heads_per_ngram": 2,
                "qwen4exp.ple.conv_kernel": 4,
                "qwen4exp.embedding_length_per_layer_input": 4,
                "qwen4exp.ple.layer_multipliers": [17, 19, 23],
                "qwen4exp.ple.head_offsets": [0, 2, 4, 6],
                "qwen4exp.ple.head_vocab_sizes": [2, 2, 2, 2],
                "qwen4exp.ple.eos_token_id": 1,
            }
        )
        with self.assertRaisesRegex(ValueError, "missing required tensors"):
            get_gguf_config(metadata, ("token_embd.weight",))

        config = _qwen4_exp_config(metadata, (), {"per_layer_token_embd.weight": (144, 4)})
        self.assertEqual(config["output_gate_type"], "sigmoid")
        self.assertEqual(config["ple_layer_multipliers"], [17, 19, 23])
        self.assertEqual(config["ple_head_offsets"], [0, 2, 4, 6])
        self.assertEqual(config["ple_head_vocab_sizes"], [2, 2, 2, 2])
        self.assertEqual(config["ple_vocab_size"], 144)

    def test_qwen35_moe_maps_expert_widths(self):
        metadata = {
            "general.architecture": "qwen35moe",
            "tokenizer.ggml.tokens": 32,
            "qwen35moe.context_length": 128,
            "qwen35moe.embedding_length": 32,
            "qwen35moe.block_count": 1,
            "qwen35moe.attention.key_length": 8,
            "qwen35moe.attention.head_count": 4,
            "qwen35moe.attention.head_count_kv": 2,
            "qwen35moe.attention.layer_norm_rms_epsilon": 1e-6,
            "qwen35moe.rope.freq_base": 10000.0,
            "qwen35moe.rope.dimension_count": 8,
            "qwen35moe.rope.dimension_sections": [2, 1, 1],
            "qwen35moe.ssm.conv_kernel": 4,
            "qwen35moe.ssm.state_size": 4,
            "qwen35moe.ssm.group_count": 2,
            "qwen35moe.ssm.time_step_rank": 4,
            "qwen35moe.ssm.inner_size": 16,
            "qwen35moe.full_attention_interval": 4,
            "qwen35moe.expert_count": 8,
            "qwen35moe.expert_used_count": 2,
            "qwen35moe.expert_feed_forward_length": 7,
            "qwen35moe.expert_shared_feed_forward_length": 5,
        }

        config = get_gguf_config(metadata, ())

        self.assertEqual(config["moe_intermediate_size"], 7)
        self.assertEqual(config["shared_expert_intermediate_size"], 5)


if __name__ == "__main__":
    unittest.main()
