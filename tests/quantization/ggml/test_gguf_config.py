# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import unittest

from transformers.integrations.gguf.gguf_config_mapping import get_gguf_config


class GgufConfigTests(unittest.TestCase):
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
