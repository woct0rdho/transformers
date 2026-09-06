# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import unittest

import torch

from transformers.integrations.gguf.dequant import GGML_Q8_0
from transformers.integrations.gguf.gguf_quantized_parameter import GgufQuantizedParameter
from transformers.integrations.gguf.modules import GgufLinear


class GgufModuleTests(unittest.TestCase):
    def test_linear_backpropagates_to_activations_but_not_packed_weight(self):
        in_features, out_features = 32, 3
        payload = torch.zeros(out_features, in_features // 32 * 34, dtype=torch.uint8)
        module = GgufLinear(in_features, out_features, bias=False, compute_dtype=torch.float32)
        module.weight = GgufQuantizedParameter(payload, GGML_Q8_0, (out_features, in_features))
        inputs = torch.randn(2, in_features, requires_grad=True)
        module(inputs).sum().backward()
        self.assertIsNotNone(inputs.grad)
        self.assertFalse(module.weight.requires_grad)
        self.assertIsNone(module.weight.grad)

    def test_tied_output_projection_uses_a_packed_aware_module(self):
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from transformers.integrations.gguf.modules import GgufEmbedding, replace_with_gguf_modules

        config = Qwen3Config(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=8,
            tie_word_embeddings=True,
        )
        with torch.device("meta"):
            model = Qwen3ForCausalLM(config)
        replacements = replace_with_gguf_modules(
            model,
            compute_dtype=torch.float32,
            packed_parameter_names={"model.embed_tokens.weight"},
        )
        self.assertIsInstance(model.model.embed_tokens, GgufEmbedding)
        self.assertIsInstance(model.lm_head, GgufLinear)
        self.assertIn("lm_head.weight", replacements)
        model.tie_weights()
        self.assertIs(model.lm_head.weight, model.model.embed_tokens.weight)


if __name__ == "__main__":
    unittest.main()
