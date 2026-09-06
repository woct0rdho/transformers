# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import unittest
from types import SimpleNamespace

import torch

from transformers.integrations.gguf.dequant import GGML_Q8_0
from transformers.integrations.gguf.gguf_quantized_parameter import GgufQuantizedParameter
from transformers.integrations.gguf.moe import GgufExperts
from transformers.integrations.moe import _ExpertRoutingPlan


class GgufMoeTests(unittest.TestCase):
    @staticmethod
    def _module():
        config = SimpleNamespace(
            num_experts=3,
            hidden_size=32,
            moe_intermediate_size=32,
            hidden_act="silu",
            _experts_implementation="eager",
        )
        module = GgufExperts(config, compute_dtype=torch.float32)

        def packed(logical_shape):
            payload_shape = (*logical_shape[:-1], logical_shape[-1] // 32 * 34)
            payload = torch.zeros(payload_shape, dtype=torch.uint8)
            payload[..., 1] = 0x3C
            payload[..., 2:] = 1
            return GgufQuantizedParameter(payload, GGML_Q8_0, logical_shape)

        module.gate_proj = packed((3, 32, 32))
        module.up_proj = packed((3, 32, 32))
        module.down_proj = packed((3, 32, 32))
        return module

    def test_batched_experts_mask_expert_parallel_sentinels(self):
        config = SimpleNamespace(num_experts=2, hidden_size=32, moe_intermediate_size=32, hidden_act="silu")
        module = GgufExperts(config, compute_dtype=torch.float32)
        for name in ("gate_proj", "up_proj", "down_proj"):
            setattr(
                module,
                name,
                GgufQuantizedParameter(torch.zeros(2, 32, 34, dtype=torch.uint8), GGML_Q8_0, (2, 32, 32)),
            )
        routing = _ExpertRoutingPlan(torch.zeros(2, 32), torch.tensor([0, 2]), torch.ones(2), None, 2, 1, 32)
        plan = module._prepare_expert_execution(routing, "batched_mm")
        self.assertTrue(torch.equal(plan.input_mask, torch.tensor([[False], [True]])))
        self.assertTrue(torch.equal(plan.output_mask, plan.input_mask))

    def test_packed_experts_forward_modes_and_gradients(self):
        module = self._module()
        hidden_states = torch.randn(4, 32)
        top_k_index = torch.tensor([[0, 1], [2, 3], [1, 0], [2, 1]])
        top_k_weights = torch.tensor([[0.5, 0.5], [0.4, 0.0], [0.3, 0.7], [0.2, 0.8]])
        grad_output = torch.randn_like(hidden_states)
        results = {}
        for implementation in ("eager", "grouped_mm", "batched_mm"):
            module.config._experts_implementation = implementation
            inputs = hidden_states.detach().clone().requires_grad_(True)
            routing_weights = top_k_weights.detach().clone().requires_grad_(True)
            output = module(inputs, top_k_index, routing_weights)
            output.backward(grad_output)
            self.assertIsNotNone(inputs.grad)
            self.assertIsNotNone(routing_weights.grad)
            results[implementation] = (output.detach(), inputs.grad.detach(), routing_weights.grad.detach())

        for implementation in ("grouped_mm", "batched_mm"):
            with self.subTest(implementation=implementation):
                for actual, expected in zip(results[implementation], results["eager"]):
                    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
        self.assertFalse(module.gate_proj.requires_grad)
        self.assertFalse(module.up_proj.requires_grad)
        self.assertFalse(module.down_proj.requires_grad)


if __name__ == "__main__":
    unittest.main()
