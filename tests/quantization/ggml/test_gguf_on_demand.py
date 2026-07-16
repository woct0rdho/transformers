# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import tempfile
import unittest
from types import SimpleNamespace
from typing import Any

from transformers import GGUFConfig
from transformers.testing_utils import require_gguf, require_torch, require_torch_bf16, require_torch_gpu, torch_device
from transformers.utils import is_gguf_available, is_torch_available


if is_torch_available():
    import torch
    from torch.nn import functional as F

    from transformers.integrations.gguf import (
        ALL_GGUF_EXPERTS_FUNCTIONS,
        GGUFEmbedding,
        GGUFExperts,
        GGUFLinear,
        replace_with_gguf_modules,
    )
    from transformers.integrations.gguf_dequant import GGUFQuantizedTensor, dequantize_gguf_tensor
    from transformers.integrations.moe import use_experts_implementation
    from transformers.models.qwen3 import Qwen3Config, Qwen3ForCausalLM
    from transformers.models.qwen3_moe import Qwen3MoeConfig, Qwen3MoeForCausalLM
    from transformers.quantizers.quantizer_gguf import GGUFQuantizer

if is_gguf_available():
    import gguf
    import numpy as np


def _float_bytes(tensor):
    return tensor.detach().cpu().numpy().view(np.uint8).reshape(*tensor.shape[:-1], -1)


def _make_test_gguf_experts(*, packed=True, full_weights=None, hidden_size=8, intermediate_size=12):
    config = Qwen3MoeConfig(
        hidden_size=hidden_size,
        moe_intermediate_size=intermediate_size,
        num_experts=4,
        num_experts_per_tok=2,
        hidden_act="silu",
    )
    module = GGUFExperts(config, compute_dtype=torch.float32)
    if full_weights is None:
        torch.manual_seed(0)
        full_weights = {
            "gate_proj": torch.randn(4, intermediate_size, hidden_size),
            "up_proj": torch.randn(4, intermediate_size, hidden_size),
            "down_proj": torch.randn(4, hidden_size, intermediate_size),
        }
    for name, full_weight in full_weights.items():
        parameter = (
            GGUFQuantizedTensor(
                torch.from_numpy(_float_bytes(full_weight)),
                quant_type=gguf.GGMLQuantizationType.F32,
                logical_shape=full_weight.shape,
            )
            if packed
            else torch.nn.Parameter(full_weight.clone(), requires_grad=False)
        )
        setattr(module, name, parameter)
    return module, full_weights


@require_torch
@require_gguf
class GGUFOnDemandTests(unittest.TestCase):
    def test_quantized_tensor_has_frozen_parameter_semantics(self):
        tensor = GGUFQuantizedTensor(
            torch.arange(16, dtype=torch.uint8).reshape(2, 8),
            quant_type=gguf.GGMLQuantizationType.Q5_K,
            logical_shape=(2, 4),
        )
        self.assertIsInstance(tensor, torch.nn.Parameter)
        self.assertFalse(tensor.requires_grad)
        self.assertEqual(tensor.dtype, torch.uint8)
        self.assertEqual(tensor.logical_shape, (2, 4))
        self.assertEqual(tensor.logical_numel, 8)
        self.assertEqual(tensor.storage_nbytes, 16)

        with self.assertRaisesRegex(ValueError, "cannot require gradients"):
            GGUFQuantizedTensor(torch.empty(4, dtype=torch.uint8), requires_grad=True)
        with self.assertRaisesRegex(TypeError, "must use torch.uint8 storage"):
            GGUFQuantizedTensor(torch.empty(4, dtype=torch.float32))

    def test_quantized_tensor_movement_preserves_storage_and_metadata(self):
        tensor = GGUFQuantizedTensor(
            torch.arange(16, dtype=torch.uint8).reshape(2, 8),
            quant_type=gguf.GGMLQuantizationType.Q5_K,
            logical_shape=(2, 4),
        )
        other = torch.empty(0, dtype=torch.bfloat16)
        for moved in (
            tensor.to(dtype=torch.bfloat16),
            tensor.to("cpu", torch.float16),
            tensor.to(other),
            tensor.to(copy=True),
        ):
            self.assertIsInstance(moved, GGUFQuantizedTensor)
            self.assertIsInstance(moved, torch.nn.Parameter)
            self.assertEqual(moved.dtype, torch.uint8)
            self.assertEqual(moved.logical_shape, tensor.logical_shape)
            self.assertEqual(moved.quant_type, tensor.quant_type)

        copied = tensor.to(copy=True)
        self.assertNotEqual(copied.untyped_storage().data_ptr(), tensor.untyped_storage().data_ptr())

        if torch.cuda.is_available():
            on_device = tensor.cuda()
            self.assertEqual(on_device.device.type, "cuda")
            self.assertEqual(on_device.quant_type, tensor.quant_type)
            on_cpu = on_device.cpu()
            self.assertEqual(on_cpu.device.type, "cpu")
            self.assertEqual(on_cpu.logical_shape, tensor.logical_shape)
            self.assertEqual(on_cpu.quant_type, tensor.quant_type)

    def test_quantized_tensor_copy_and_plain_tensor_operations(self):
        tensor = GGUFQuantizedTensor(
            torch.arange(16, dtype=torch.uint8).reshape(2, 8),
            quant_type=gguf.GGMLQuantizationType.Q5_K,
            logical_shape=(2, 4),
        )
        shallow = copy.copy(tensor)
        deep = copy.deepcopy(tensor)
        self.assertIsInstance(shallow, GGUFQuantizedTensor)
        self.assertIsInstance(deep, GGUFQuantizedTensor)
        self.assertEqual(shallow.gguf_metadata, tensor.gguf_metadata)
        self.assertEqual(deep.gguf_metadata, tensor.gguf_metadata)
        self.assertEqual(shallow.untyped_storage().data_ptr(), tensor.untyped_storage().data_ptr())
        self.assertNotEqual(deep.untyped_storage().data_ptr(), tensor.untyped_storage().data_ptr())

        self.assertIs(tensor[...], tensor)
        for plain in (tensor.clone(), tensor.detach(), tensor[0]):
            self.assertIs(type(plain), torch.Tensor)

    def test_linear_preserves_input_gradients_and_compressed_parameter(self):
        full_weight = torch.arange(32, dtype=torch.float32).reshape(8, 4) / 32
        compressed = GGUFQuantizedTensor(
            torch.from_numpy(_float_bytes(full_weight)),
            quant_type=gguf.GGMLQuantizationType.F32,
            logical_shape=full_weight.shape,
        )
        module = GGUFLinear(4, 8, bias=False)
        module.weight = compressed
        inputs = torch.randn(2, 3, 4, requires_grad=True)
        torch.testing.assert_close(module(inputs), F.linear(inputs, full_weight))
        torch.testing.assert_close(module.materialize_logical_weight(), full_weight)
        logical_bf16 = module.materialize_logical_weight(dtype=torch.bfloat16, device="cpu")
        self.assertEqual(logical_bf16.dtype, torch.bfloat16)
        torch.testing.assert_close(logical_bf16, full_weight.to(torch.bfloat16))
        with self.assertRaisesRegex(TypeError, "floating-point dtype"):
            module.materialize_logical_weight(dtype=torch.int32)
        module(inputs).sum().backward()
        self.assertIsNotNone(inputs.grad)
        self.assertIn("weight", dict(module.named_parameters()))
        self.assertNotIn("weight", dict(module.named_buffers()))
        self.assertFalse(module.weight.requires_grad)

    def test_linear_backward_redequantizes_without_saving_dense_weight(self):
        from unittest.mock import patch

        full_weight = torch.linspace(-1, 1, 128, dtype=torch.float32).reshape(4, 32)
        packed = gguf.quantize(full_weight.numpy(), gguf.GGMLQuantizationType.Q4_0)
        reference_weight = torch.from_numpy(gguf.dequantize(packed, gguf.GGMLQuantizationType.Q4_0).copy())
        compressed = GGUFQuantizedTensor(
            torch.from_numpy(packed),
            quant_type=gguf.GGMLQuantizationType.Q4_0,
            logical_shape=full_weight.shape,
        )
        bias_value = torch.arange(4, dtype=torch.float32) / 4
        linear = GGUFLinear(32, 4, bias=True, compute_dtype=torch.float32)
        linear.weight = compressed
        linear.bias = torch.nn.Parameter(bias_value.clone())
        inputs = torch.randn(2, 3, 32, requires_grad=True)
        reference_inputs = inputs.detach().clone().requires_grad_(True)
        reference_bias = bias_value.clone().requires_grad_(True)
        expected = F.linear(reference_inputs, reference_weight, reference_bias)
        grad_output = torch.randn_like(expected)
        saved_tensors = []

        def pack_hook(tensor):
            saved_tensors.append(tensor)
            return tensor

        with patch(
            "transformers.integrations.gguf.dequantize_gguf_tensor", wraps=dequantize_gguf_tensor
        ) as dequantize:
            with torch.autograd.graph.saved_tensors_hooks(pack_hook, lambda tensor: tensor):
                actual = linear(inputs)
            self.assertEqual(dequantize.call_count, 1)
            self.assertEqual(len(saved_tensors), 1)
            torch.testing.assert_close(actual, expected)
            actual.backward(grad_output)
            self.assertEqual(dequantize.call_count, 2)

        expected.backward(grad_output)
        saved_weight = saved_tensors[0]
        self.assertEqual(saved_weight.dtype, torch.uint8)
        self.assertEqual(saved_weight.shape, compressed.shape)
        self.assertEqual(saved_weight.numel(), compressed.numel())
        assert inputs.grad is not None and reference_inputs.grad is not None
        torch.testing.assert_close(inputs.grad, reference_inputs.grad)
        assert linear.bias is not None and linear.bias.grad is not None and reference_bias.grad is not None
        torch.testing.assert_close(linear.bias.grad, reference_bias.grad)
        self.assertIsNone(linear.weight.grad)

    def test_unloaded_gguf_modules_fail_before_checkpoint_assignment(self):
        linear = GGUFLinear(4, 3, bias=False)
        with self.assertRaisesRegex(RuntimeError, "GGUFLinear weight has not been loaded"):
            linear(torch.randn(2, 4))
        with self.assertRaisesRegex(RuntimeError, "GGUFLinear weight has not been loaded"):
            linear.materialize_logical_weight()

        embedding = GGUFEmbedding(3, 4)
        with self.assertRaisesRegex(RuntimeError, "GGUFEmbedding weight has not been loaded"):
            embedding(torch.tensor([0, 2]))

        config = Qwen3MoeConfig(
            hidden_size=4,
            moe_intermediate_size=4,
            num_experts=2,
            num_experts_per_tok=1,
            hidden_act="silu",
        )
        experts = GGUFExperts(config)
        experts.config._experts_implementation_internal = "eager"
        self.assertEqual(experts.weight_state, "placeholder")
        with self.assertRaisesRegex(RuntimeError, "GGUFExperts weights have not been loaded"):
            experts(torch.randn(2, 4), torch.tensor([[0], [1]]), torch.ones(2, 1))

    def test_gguf_expert_weight_states_are_explicit(self):
        config = Qwen3MoeConfig(
            hidden_size=6,
            moe_intermediate_size=4,
            num_experts=3,
            num_experts_per_tok=1,
            hidden_act="silu",
        )
        experts = GGUFExperts(config, compute_dtype=torch.float32)
        experts.config._experts_implementation_internal = "eager"
        floating_weights = {
            "gate_proj": torch.randn(3, 4, 6),
            "up_proj": torch.randn(3, 4, 6),
            "down_proj": torch.randn(3, 6, 4),
        }
        for name, weight in floating_weights.items():
            setattr(experts, name, torch.nn.Parameter(weight.clone()))

        self.assertEqual(experts.weight_state, "floating")
        output = experts(torch.randn(2, 6), torch.tensor([[0], [2]]), torch.ones(2, 1))
        self.assertTrue(torch.isfinite(output).all())

        gate_weight = floating_weights["gate_proj"]
        experts.gate_proj = GGUFQuantizedTensor(
            torch.from_numpy(_float_bytes(gate_weight)),
            quant_type=gguf.GGMLQuantizationType.F32,
            logical_shape=gate_weight.shape,
        )
        self.assertEqual(experts.weight_state, "mixed")
        with self.assertRaisesRegex(RuntimeError, "must be all packed or all floating-point"):
            experts(torch.randn(2, 6), torch.tensor([[0], [2]]), torch.ones(2, 1))

    def test_gguf_expert_factory_uses_generic_source_contract_and_dtype(self):
        from transformers.integrations.gguf import ALL_GGUF_EXPERTS_FUNCTIONS
        from transformers.integrations.moe import batched_mm_experts_forward, grouped_mm_experts_forward

        config = SimpleNamespace(_experts_implementation="eager")

        @use_experts_implementation
        class CompatibleExperts(torch.nn.Module):
            def __init__(self, config):
                super().__init__()
                self.num_experts = 3
                self.hidden_dim = 6
                self.intermediate_dim = 4
                self.gate_up_proj = torch.nn.Parameter(torch.empty(3, 8, 6, dtype=torch.float64, device="meta"))
                self.down_proj = torch.nn.Parameter(torch.empty(3, 6, 4, dtype=torch.float64, device="meta"))
                self.act_fn = F.silu

            def forward(self, hidden_states, top_k_index, top_k_weights):
                raise NotImplementedError

        source = CompatibleExperts(config)
        self.assertEqual(source.projection_layout, "concatenated_gate_up")
        self.assertEqual(source.gate_implementation, "default")
        self.assertEqual(set(source._get_expert_projection_tensors()), {"gate_up", "down"})
        self.assertTrue(source.experts_implementation_switchable)
        self.assertIs(ALL_GGUF_EXPERTS_FUNCTIONS["grouped_mm"], grouped_mm_experts_forward)
        self.assertIs(ALL_GGUF_EXPERTS_FUNCTIONS["batched_mm"], batched_mm_experts_forward)

        container = torch.nn.Module()
        container.experts = source
        replace_with_gguf_modules(container)
        replacement: Any = container.experts
        self.assertIsInstance(replacement, GGUFExperts)
        self.assertEqual(replacement.compute_dtype, torch.float64)
        self.assertEqual((replacement.num_experts, replacement.hidden_dim, replacement.intermediate_dim), (3, 6, 4))
        self.assertEqual(replacement.projection_layout, "split_gate_up")
        self.assertIsNone(replacement.is_concatenated)
        self.assertEqual(
            replacement.supported_experts_implementations,
            ("eager", "grouped_mm", "batched_mm"),
        )
        self.assertTrue(replacement.experts_implementation_switchable)

        @use_experts_implementation
        class ProviderBackedExperts(torch.nn.Module):
            def __init__(self, config):
                super().__init__()
                self.num_experts = 3
                self.hidden_dim = 6
                self.intermediate_dim = 4
                self.fused_input = torch.nn.Parameter(torch.empty(3, 8, 6, dtype=torch.float64, device="meta"))
                self.output = torch.nn.Parameter(torch.empty(3, 6, 4, dtype=torch.float64, device="meta"))
                self.act_fn = F.silu

            def _get_expert_projection_tensors(self):
                return {"gate_up": self.fused_input, "down": self.output}

            def forward(self, hidden_states, top_k_index, top_k_weights):
                raise NotImplementedError

        provider_container = torch.nn.Module()
        provider_container.experts = ProviderBackedExperts(config)
        replace_with_gguf_modules(provider_container)
        provider_replacement: Any = provider_container.experts
        self.assertIsInstance(provider_replacement, GGUFExperts)
        self.assertEqual(provider_replacement.compute_dtype, torch.float64)
        self.assertEqual(
            (provider_replacement.num_experts, provider_replacement.hidden_dim, provider_replacement.intermediate_dim),
            (3, 6, 4),
        )

        invalid_provider_container = torch.nn.Module()
        invalid_provider_container.experts = ProviderBackedExperts(config)
        object.__setattr__(
            invalid_provider_container.experts,
            "_get_expert_projection_tensors",
            lambda: {"gate_up": object()},
        )
        with self.assertRaisesRegex(ValueError, "string keys and tensor values"):
            replace_with_gguf_modules(invalid_provider_container)

        for attribute, value, error in (
            ("has_bias", True, "projection bias"),
            ("is_transposed", True, "transposed expert projections"),
            ("projection_layout", "interleaved_gate_up", "source projection layout"),
            ("gate_implementation", "custom", "custom gate behavior"),
        ):
            with self.subTest(attribute=attribute):
                incompatible = CompatibleExperts(config)
                setattr(incompatible, attribute, value)
                with self.assertRaisesRegex(ValueError, error):
                    GGUFExperts.from_module(incompatible)

        incompatible_container = torch.nn.Module()
        incompatible_container.linear = torch.nn.Linear(2, 2)
        incompatible_experts: Any = CompatibleExperts(config)
        incompatible_experts.has_bias = True
        incompatible_container.experts = incompatible_experts
        with self.assertRaisesRegex(ValueError, "projection bias"):
            replace_with_gguf_modules(incompatible_container)
        self.assertIs(type(incompatible_container.linear), torch.nn.Linear)

    def test_gguf_module_factories_preserve_structure_and_meta_device(self):
        source_linear = torch.nn.Linear(4, 3, bias=True, device="meta", dtype=torch.bfloat16)
        linear = GGUFLinear.from_linear(source_linear, compute_dtype=torch.float32)
        self.assertEqual((linear.in_features, linear.out_features), (4, 3))
        self.assertIsNotNone(linear.bias)
        self.assertEqual(linear.weight.device.type, "meta")
        self.assertEqual(linear.weight.dtype, torch.uint8)
        self.assertEqual(linear.compute_dtype, torch.float32)

        source_embedding = torch.nn.Embedding(7, 4, padding_idx=0, device="meta", dtype=torch.bfloat16)
        embedding = GGUFEmbedding.from_embedding(source_embedding, compute_dtype=torch.float32)
        self.assertEqual((embedding.num_embeddings, embedding.embedding_dim), (7, 4))
        self.assertEqual(embedding.padding_idx, 0)
        self.assertEqual(embedding.weight.device.type, "meta")
        self.assertEqual(embedding.weight.dtype, torch.uint8)
        self.assertEqual(embedding.compute_dtype, torch.float32)

    def test_representative_q4_0_weight_runs_through_linear_and_embedding(self):
        full_weight = torch.linspace(-1, 1, 128, dtype=torch.float32).reshape(4, 32)
        packed = gguf.quantize(full_weight.numpy(), gguf.GGMLQuantizationType.Q4_0)
        reference_weight = torch.from_numpy(gguf.dequantize(packed, gguf.GGMLQuantizationType.Q4_0).copy())
        compressed = GGUFQuantizedTensor(
            torch.from_numpy(packed),
            quant_type=gguf.GGMLQuantizationType.Q4_0,
            logical_shape=full_weight.shape,
        )

        linear = GGUFLinear(32, 4, bias=False)
        linear.weight = compressed
        inputs = torch.randn(2, 32)
        torch.testing.assert_close(linear(inputs), F.linear(inputs, reference_weight))

        embedding = GGUFEmbedding(4, 32)
        embedding.weight = compressed
        input_ids = torch.tensor([[3, 0, 3]])
        torch.testing.assert_close(embedding(input_ids), F.embedding(input_ids, reference_weight))

    def test_tied_parameter_survives_device_round_trip(self):
        full_weight = torch.arange(40, dtype=torch.float32).reshape(10, 4) / 40
        compressed = GGUFQuantizedTensor(
            torch.from_numpy(_float_bytes(full_weight)),
            quant_type=gguf.GGMLQuantizationType.F32,
            logical_shape=full_weight.shape,
        )
        module = torch.nn.Module()
        module.embedding = GGUFEmbedding(10, 4)
        module.lm_head = GGUFLinear(4, 10, bias=False)
        module.embedding.weight = compressed
        module.lm_head.weight = module.embedding.weight

        self.assertIs(module.embedding.weight, module.lm_head.weight)
        if torch.cuda.is_available():
            module.cuda()
            self.assertIs(module.embedding.weight, module.lm_head.weight)
            self.assertEqual(module.embedding.weight.device.type, "cuda")
            input_ids = torch.tensor([[9, 2, 0]], device="cuda")
            torch.testing.assert_close(module.embedding(input_ids), F.embedding(input_ids, full_weight.cuda()))
            module.cpu()
            self.assertIs(module.embedding.weight, module.lm_head.weight)
            self.assertEqual(module.embedding.weight.quant_type, gguf.GGMLQuantizationType.F32)

    def test_gguf_modules_use_explicit_compute_dtype(self):
        from unittest.mock import patch

        from transformers.integrations.gguf import _dequantize_experts

        full_weight = torch.arange(32, dtype=torch.float32).reshape(8, 4) / 32
        compressed = GGUFQuantizedTensor(
            torch.from_numpy(_float_bytes(full_weight)),
            quant_type=gguf.GGMLQuantizationType.F32,
            logical_shape=full_weight.shape,
        )
        linear = GGUFLinear(4, 8, bias=True, compute_dtype=torch.float32)
        linear.weight = compressed
        linear.bias = torch.nn.Parameter(torch.arange(8, dtype=torch.float32) / 8)

        embedding = GGUFEmbedding(8, 4, compute_dtype=torch.float32)
        embedding.weight = compressed

        config = Qwen3MoeConfig(
            hidden_size=4,
            moe_intermediate_size=4,
            num_experts=2,
            num_experts_per_tok=1,
            hidden_act="silu",
        )
        experts = GGUFExperts(config, compute_dtype=torch.float32)
        expert_weight = torch.arange(32, dtype=torch.float32).reshape(2, 4, 4) / 32
        for name in ("gate_proj", "up_proj", "down_proj"):
            setattr(
                experts,
                name,
                GGUFQuantizedTensor(
                    torch.from_numpy(_float_bytes(expert_weight)),
                    quant_type=gguf.GGMLQuantizationType.F32,
                    logical_shape=expert_weight.shape,
                ),
            )
        experts.config._experts_implementation_internal = "eager"

        container = torch.nn.Module()
        container.linear = linear
        container.embedding = embedding
        container.experts = experts
        container.to(dtype=torch.float64)

        for module in (linear, embedding, experts):
            self.assertEqual(module.compute_dtype, torch.float64)
        self.assertEqual(linear.weight.dtype, torch.uint8)
        self.assertEqual(embedding.weight.dtype, torch.uint8)
        self.assertEqual(experts.gate_proj.dtype, torch.uint8)
        self.assertEqual(linear.bias.dtype, torch.float64)

        inputs = torch.randn(2, 4, dtype=torch.float32)
        expected = F.linear(inputs.to(torch.float64), full_weight.to(torch.float64), linear.bias).to(inputs.dtype)
        actual = linear(inputs)
        self.assertEqual(actual.dtype, inputs.dtype)
        torch.testing.assert_close(actual, expected)

        input_ids = torch.tensor([[7, 1, 7]])
        embedded = embedding(input_ids)
        self.assertEqual(embedded.dtype, torch.float64)
        torch.testing.assert_close(embedded, F.embedding(input_ids, full_weight.to(torch.float64)))

        hidden_states = torch.randn(3, 4, dtype=torch.float32)
        top_k_index = torch.tensor([[0], [1], [0]])
        top_k_weights = torch.ones(3, 1)
        with patch(
            "transformers.integrations.gguf._dequantize_experts", wraps=_dequantize_experts
        ) as dequantize_experts:
            expert_output = experts(hidden_states, top_k_index, top_k_weights)
        self.assertEqual(expert_output.dtype, hidden_states.dtype)
        self.assertTrue(all(call.args[2] == torch.float64 for call in dequantize_experts.call_args_list))

        with self.assertRaisesRegex(TypeError, "compute dtype must be a floating-point"):
            linear.set_compute_dtype(torch.int64)

    def test_linear_compute_dtype_matrix_and_bias_gradients(self):
        full_weight = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 12
        compressed = GGUFQuantizedTensor(
            torch.from_numpy(_float_bytes(full_weight)),
            quant_type=gguf.GGMLQuantizationType.F32,
            logical_shape=full_weight.shape,
        )
        bias_value = torch.arange(3, dtype=torch.float32) / 3
        dtypes = (torch.float16, torch.bfloat16, torch.float32)
        for input_dtype in dtypes:
            for compute_dtype in dtypes:
                with self.subTest(input_dtype=input_dtype, compute_dtype=compute_dtype):
                    linear = GGUFLinear(4, 3, bias=True, compute_dtype=compute_dtype)
                    linear.weight = compressed
                    linear.bias = torch.nn.Parameter(bias_value.clone())
                    inputs = torch.randn(2, 4, dtype=input_dtype, requires_grad=True)
                    reference_inputs = inputs.detach().clone().requires_grad_(True)
                    reference_bias = bias_value.clone().requires_grad_(True)
                    actual = linear(inputs)
                    expected = F.linear(
                        reference_inputs.to(compute_dtype),
                        full_weight.to(compute_dtype),
                        reference_bias.to(compute_dtype),
                    ).to(reference_inputs.dtype)
                    self.assertEqual(actual.dtype, inputs.dtype)
                    torch.testing.assert_close(actual, expected)

                    grad_output = torch.randn_like(actual)
                    actual.backward(grad_output)
                    expected.backward(grad_output)
                    assert inputs.grad is not None and reference_inputs.grad is not None
                    self.assertEqual(inputs.grad.dtype, inputs.dtype)
                    torch.testing.assert_close(inputs.grad, reference_inputs.grad)
                    assert linear.bias is not None and linear.bias.grad is not None and reference_bias.grad is not None
                    self.assertEqual(linear.bias.grad.dtype, linear.bias.dtype)
                    torch.testing.assert_close(linear.bias.grad, reference_bias.grad)
                    self.assertIsNone(linear.weight.grad)

    def test_linear_autocast_preserves_input_output_dtype_contract(self):
        full_weight = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 12
        compressed = GGUFQuantizedTensor(
            torch.from_numpy(_float_bytes(full_weight)),
            quant_type=gguf.GGMLQuantizationType.F32,
            logical_shape=full_weight.shape,
        )
        bias_value = torch.arange(3, dtype=torch.float32) / 3
        linear = GGUFLinear(4, 3, bias=True, compute_dtype=torch.float32)
        linear.weight = compressed
        linear.bias = torch.nn.Parameter(bias_value.clone())
        inputs = torch.randn(2, 4, dtype=torch.float32, requires_grad=True)
        reference_inputs = inputs.detach().clone().requires_grad_(True)
        reference_bias = bias_value.clone().requires_grad_(True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = linear(inputs)
            expected = F.linear(reference_inputs, full_weight, reference_bias).to(reference_inputs.dtype)
        self.assertEqual(actual.dtype, inputs.dtype)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

        grad_output = torch.randn_like(actual)
        actual.backward(grad_output)
        expected.backward(grad_output)
        assert inputs.grad is not None and reference_inputs.grad is not None
        torch.testing.assert_close(inputs.grad, reference_inputs.grad, rtol=0, atol=0)
        assert linear.bias is not None and linear.bias.grad is not None and reference_bias.grad is not None
        torch.testing.assert_close(linear.bias.grad, reference_bias.grad, rtol=0, atol=0)

    def test_floating_weight_fallback_uses_torch_dtype_semantics(self):
        full_weight = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 12
        bias = torch.arange(3, dtype=torch.float32) / 3
        inputs = torch.randn(2, 4, dtype=torch.float32)
        linear = GGUFLinear(4, 3, bias=True, compute_dtype=torch.float64)
        linear.weight = torch.nn.Parameter(full_weight.clone())
        linear.bias = torch.nn.Parameter(bias.clone())
        actual = linear(inputs)
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, F.linear(inputs, full_weight, bias))
        torch.testing.assert_close(
            linear.materialize_logical_weight(dtype=torch.float64), full_weight.to(torch.float64)
        )

        embedding = GGUFEmbedding(3, 4, compute_dtype=torch.float64)
        embedding.weight = torch.nn.Parameter(full_weight.clone())
        input_ids = torch.tensor([[2, 0]])
        embedded = embedding(input_ids)
        self.assertEqual(embedded.dtype, full_weight.dtype)
        torch.testing.assert_close(embedded, F.embedding(input_ids, full_weight))

    def test_embedding_rejects_unsupported_mutation_and_gradient_options(self):
        with self.assertRaisesRegex(ValueError, "does not support max_norm"):
            GGUFEmbedding(10, 4, max_norm=1.0)
        with self.assertRaisesRegex(ValueError, "only supports the default norm_type"):
            GGUFEmbedding(10, 4, norm_type=1.0)
        with self.assertRaisesRegex(ValueError, "does not support scale_grad_by_freq"):
            GGUFEmbedding(10, 4, scale_grad_by_freq=True)
        with self.assertRaisesRegex(ValueError, "does not support sparse gradients"):
            GGUFEmbedding(10, 4, sparse=True)

    def test_embedding_rows_and_lm_head_match_full_weight(self):
        from unittest.mock import patch

        from transformers.integrations.gguf import _dequantize_rows

        full_weight = torch.arange(40, dtype=torch.float32).reshape(10, 4) / 40
        compressed = GGUFQuantizedTensor(
            torch.from_numpy(_float_bytes(full_weight)),
            quant_type=gguf.GGMLQuantizationType.F32,
            logical_shape=full_weight.shape,
        )
        embedding = GGUFEmbedding(10, 4, padding_idx=0)
        embedding.weight = compressed
        self.assertEqual(embedding.padding_idx, 0)
        for compute_dtype in (torch.float16, torch.bfloat16, torch.float32):
            embedding.set_compute_dtype(compute_dtype)
            for input_ids in (
                torch.tensor([9, 0]),
                torch.tensor([[9, 2, 9, 0]]),
                torch.tensor([[[9, 2], [0, 9]]]),
                torch.empty((0, 2), dtype=torch.long),
            ):
                with (
                    self.subTest(compute_dtype=compute_dtype, shape=tuple(input_ids.shape)),
                    patch(
                        "transformers.integrations.gguf._dequantize_rows", wraps=_dequantize_rows
                    ) as dequantize_rows,
                ):
                    actual = embedding(input_ids)
                    expected = F.embedding(input_ids, full_weight.to(compute_dtype))
                    self.assertEqual(actual.dtype, compute_dtype)
                    torch.testing.assert_close(actual, expected)
                    self.assertEqual(dequantize_rows.call_count, 1)
                    selected_rows = dequantize_rows.call_args.args[1]
                    torch.testing.assert_close(selected_rows.cpu(), torch.unique(input_ids.reshape(-1), sorted=True))
                    self.assertEqual(selected_rows.numel(), torch.unique(input_ids).numel())

        self.assertIn("weight", dict(embedding.named_parameters()))
        self.assertNotIn("weight", dict(embedding.named_buffers()))

        hidden_states = torch.randn(2, 4)
        lm_head = GGUFLinear(4, 10, bias=False)
        lm_head.weight = compressed
        torch.testing.assert_close(lm_head(hidden_states), F.linear(hidden_states, full_weight))

    def test_large_embedding_selects_payload_rows_before_dequantization(self):
        from unittest.mock import patch

        from transformers.integrations.gguf import dequantize_gguf_tensor

        full_weight = torch.arange(4096 * 8, dtype=torch.float32).reshape(4096, 8) / (4096 * 8)
        compressed = GGUFQuantizedTensor(
            torch.from_numpy(_float_bytes(full_weight)),
            quant_type=gguf.GGMLQuantizationType.F32,
            logical_shape=full_weight.shape,
        )
        embedding = GGUFEmbedding(4096, 8)
        embedding.weight = compressed
        input_ids = torch.tensor([[4095, 7, 4095], [0, 7, 0]])
        with patch(
            "transformers.integrations.gguf.dequantize_gguf_tensor", wraps=dequantize_gguf_tensor
        ) as dequantize:
            actual = embedding(input_ids)

        torch.testing.assert_close(actual, F.embedding(input_ids, full_weight))
        self.assertEqual(dequantize.call_count, 1)
        selected_payload = dequantize.call_args.args[0]
        self.assertEqual(selected_payload.shape[0], 3)
        self.assertEqual(selected_payload.numel(), 3 * compressed.shape[1])
        self.assertLess(selected_payload.numel(), compressed.numel())

    def test_adapter_training_keeps_compressed_parameter_frozen(self):
        full_weight = torch.arange(16, dtype=torch.float32).reshape(4, 4) / 16
        base = GGUFLinear(4, 4, bias=False)
        base.weight = GGUFQuantizedTensor(
            torch.from_numpy(_float_bytes(full_weight)),
            quant_type=gguf.GGMLQuantizationType.F32,
            logical_shape=full_weight.shape,
        )
        adapter = torch.nn.Linear(4, 4, bias=False)
        before = adapter.weight.detach().clone()
        optimizer = torch.optim.SGD(adapter.parameters(), lr=0.1)
        inputs = torch.randn(2, 4)
        (base(inputs) + adapter(inputs)).sum().backward()
        optimizer.step()

        self.assertIsNone(base.weight.grad)
        self.assertIsNotNone(adapter.weight.grad)
        self.assertFalse(torch.equal(before, adapter.weight))

    def test_qwen3_replacement_compute_dtype_and_tied_parameter(self):
        config = Qwen3Config(
            vocab_size=8,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            tie_word_embeddings=True,
        )
        with torch.device("meta"):
            model = Qwen3ForCausalLM(config)
        logical_num_parameters = model.num_parameters()
        quantizer = GGUFQuantizer(GGUFConfig(architecture="qwen3"))
        quantizer.update_dtype(torch.float32)
        quantizer.preprocess_model(model, dtype=torch.float32, device_map={"": "cpu"})
        self.assertIsInstance(model.model.embed_tokens, GGUFEmbedding)
        self.assertIsInstance(getattr(model.model.layers[0].self_attn, "q_proj"), GGUFLinear)
        self.assertIsInstance(model.lm_head, GGUFLinear)
        self.assertEqual(model.model.embed_tokens.compute_dtype, torch.float32)
        self.assertEqual(model.lm_head.compute_dtype, torch.float32)

        full_weight = torch.arange(64, dtype=torch.float32).reshape(8, 8) / 64
        compressed = GGUFQuantizedTensor(
            torch.from_numpy(_float_bytes(full_weight)),
            quant_type=gguf.GGMLQuantizationType.F32,
            logical_shape=full_weight.shape,
        )
        model.model.embed_tokens.weight = compressed
        missing_keys = {"lm_head.weight"}
        model.tie_weights(missing_keys=missing_keys, recompute_mapping=False)
        self.assertIs(model.lm_head.weight, model.model.embed_tokens.weight)
        self.assertNotIn("lm_head.weight", missing_keys)
        self.assertEqual(model.num_parameters(), logical_num_parameters)
        self.assertLess(model.get_memory_footprint(), logical_num_parameters * torch.finfo(torch.float32).bits // 8)
        with self.assertRaisesRegex(ValueError, "Casting a persistent GGUF model"):
            getattr(model, "to")(dtype=torch.bfloat16)

    def test_tiny_qwen3_forward_and_generation_after_persistent_replacement(self):
        torch.manual_seed(0)
        config = Qwen3Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            tie_word_embeddings=True,
            bos_token_id=0,
            eos_token_id=1,
            pad_token_id=0,
        )
        model = Qwen3ForCausalLM(config).eval()
        input_ids = torch.tensor([[2, 5, 7, 3]])
        with torch.no_grad():
            expected_logits = model(input_ids).logits

        source_parameters = {}
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
                source_parameters[f"{name}.weight"] = module.weight.detach().clone()
                if getattr(module, "bias", None) is not None:
                    source_parameters[f"{name}.bias"] = module.bias.detach().clone()

        quantizer = GGUFQuantizer(GGUFConfig(architecture="qwen3"))
        quantizer.update_dtype(torch.float32)
        quantizer.preprocess_model(model, dtype=torch.float32, device_map={"": "cpu"})
        for name, module in model.named_modules():
            if isinstance(module, (GGUFLinear, GGUFEmbedding)):
                full_weight = source_parameters[f"{name}.weight"]
                module.weight = GGUFQuantizedTensor(
                    torch.from_numpy(_float_bytes(full_weight)),
                    quant_type=gguf.GGMLQuantizationType.F32,
                    logical_shape=full_weight.shape,
                )
                bias_name = f"{name}.bias"
                if bias_name in source_parameters:
                    module.bias = torch.nn.Parameter(source_parameters[bias_name])
        model.tie_weights()

        with torch.no_grad():
            actual_logits = model(input_ids).logits
        torch.testing.assert_close(actual_logits, expected_logits, rtol=1e-5, atol=1e-6)
        generated = model.generate(input_ids, max_new_tokens=2, do_sample=False)
        self.assertEqual(generated.shape, (1, input_ids.shape[1] + 2))

    def test_qwen3_module_sizing_uses_physical_checkpoint_storage(self):
        from transformers.integrations.accelerate import compute_module_sizes
        from transformers.modeling_gguf_pytorch_utils import get_gguf_converters
        from transformers.modeling_utils import expand_device_map, get_total_byte_count

        config = Qwen3Config(
            vocab_size=8,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            tie_word_embeddings=True,
        )
        with torch.device("meta"):
            model = Qwen3ForCausalLM(config)
        weight_mapping = get_gguf_converters("qwen3")
        quantizer = GGUFQuantizer(GGUFConfig(architecture="qwen3"), weight_mapping=weight_mapping)
        checkpoint_tensor = GGUFQuantizedTensor(
            torch.empty((8, 2), dtype=torch.uint8),
            quant_type=gguf.GGMLQuantizationType.Q5_K,
            logical_shape=(8, 8),
        )
        quantizer.set_weight_mapping(weight_mapping, {"token_embd.weight": checkpoint_tensor})
        quantizer.preprocess_model(model, dtype=torch.float32, device_map={"": "cpu"})
        quantizer.update_weight_conversions([])

        module_sizes, _ = compute_module_sizes(model, quantizer, only_modules=False)
        self.assertEqual(module_sizes["model.embed_tokens.weight"], checkpoint_tensor.storage_nbytes)
        expected_keys = [name for name, _ in model.named_parameters()] + [name for name, _ in model.named_buffers()]
        device_map = expand_device_map({"": "cpu"}, expected_keys)
        total_byte_count = get_total_byte_count(model, device_map, quantizer)
        self.assertEqual(list(total_byte_count.values()), [module_sizes[""]])

    def test_persistent_qwen3_save_pretrained_is_rejected(self):
        config = Qwen3Config(
            vocab_size=8,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
        )
        model = Qwen3ForCausalLM(config)
        quantizer = GGUFQuantizer(GGUFConfig(architecture="qwen3"))
        quantizer.preprocess_model(model, dtype=torch.float32, device_map={"": "cpu"})
        object.__setattr__(model, "hf_quantizer", quantizer)
        with tempfile.TemporaryDirectory() as tmpdir, self.assertRaisesRegex(ValueError, "not serializable"):
            model.save_pretrained(tmpdir)

    def test_qwen3_moe_expert_backends_match_non_square_reference_and_gradients(self):
        from unittest.mock import patch

        from transformers.integrations.gguf import _dequantize_experts

        config = Qwen3MoeConfig(
            hidden_size=8,
            moe_intermediate_size=12,
            num_experts=5,
            num_experts_per_tok=2,
            hidden_act="silu",
        )
        module = GGUFExperts(config, compute_dtype=torch.float32)
        torch.manual_seed(0)
        full_weights = {
            "gate_proj": torch.randn(5, 12, 8),
            "up_proj": torch.randn(5, 12, 8),
            "down_proj": torch.randn(5, 8, 12),
        }
        for name, full_weight in full_weights.items():
            setattr(
                module,
                name,
                GGUFQuantizedTensor(
                    torch.from_numpy(_float_bytes(full_weight)),
                    quant_type=gguf.GGMLQuantizationType.F32,
                    logical_shape=full_weight.shape,
                ),
            )

        hidden_states = torch.randn(4, 8)
        top_k_index = torch.tensor([[0, 1], [2, 1], [0, 2], [1, 1]])
        top_k_weights = torch.tensor([[0.7, 0.3], [0.4, 0.6], [0.2, 0.8], [0.55, 0.45]])
        grad_output = torch.randn_like(hidden_states)

        def reference_forward(inputs, routing_weights):
            outputs = []
            for token_idx in range(inputs.shape[0]):
                token_output = torch.zeros(8, dtype=module.compute_dtype)
                for top_k_pos in range(top_k_index.shape[1]):
                    expert_idx = top_k_index[token_idx, top_k_pos]
                    current_state = inputs[token_idx].to(module.compute_dtype)
                    gate = F.linear(current_state, full_weights["gate_proj"][expert_idx])
                    up = F.linear(current_state, full_weights["up_proj"][expert_idx])
                    output = F.linear(F.silu(gate) * up, full_weights["down_proj"][expert_idx])
                    token_output = token_output + output * routing_weights[token_idx, top_k_pos]
                outputs.append(token_output)
            return torch.stack(outputs).to(inputs.dtype)

        self.assertEqual(module.weight_state, "packed")
        for implementation in ("eager", "grouped_mm", "batched_mm"):
            module.config._experts_implementation_internal = implementation
            inputs = hidden_states.detach().clone().requires_grad_(True)
            routing_weights = top_k_weights.detach().clone().requires_grad_(True)
            reference_inputs = hidden_states.detach().clone().requires_grad_(True)
            reference_routing_weights = top_k_weights.detach().clone().requires_grad_(True)
            expected = reference_forward(reference_inputs, reference_routing_weights)
            with patch(
                "transformers.integrations.gguf._dequantize_experts", wraps=_dequantize_experts
            ) as dequantize_experts:
                actual = module(inputs, top_k_index, routing_weights)
            self.assertEqual(actual.dtype, inputs.dtype)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
            dequantized_expert_ids = {
                int(expert_id)
                for call in dequantize_experts.call_args_list
                for expert_id in call.args[1].detach().cpu().tolist()
            }
            self.assertEqual(dequantized_expert_ids, {0, 1, 2})
            self.assertTrue(all(call.args[2] == module.compute_dtype for call in dequantize_experts.call_args_list))

            actual.backward(grad_output)
            expected.backward(grad_output)
            assert inputs.grad is not None and reference_inputs.grad is not None
            torch.testing.assert_close(inputs.grad, reference_inputs.grad, rtol=1e-5, atol=1e-6)
            assert routing_weights.grad is not None and reference_routing_weights.grad is not None
            torch.testing.assert_close(
                routing_weights.grad,
                reference_routing_weights.grad,
                rtol=1e-5,
                atol=1e-6,
            )

        self.assertEqual(dict(module.named_buffers()), {})
        self.assertEqual(set(dict(module.named_parameters())), {"gate_proj", "up_proj", "down_proj"})
        self.assertTrue(all(not param.requires_grad for param in module.parameters()))

    def test_gguf_expert_backward_redequantizes_without_saving_dense_weights(self):
        from unittest.mock import patch

        module, _ = _make_test_gguf_experts()
        hidden_states = torch.randn(3, 8)
        top_k_index = torch.tensor([[0, 1], [2, 1], [0, 2]])
        top_k_weights = torch.tensor([[0.7, 0.3], [0.4, 0.6], [0.2, 0.8]])
        logical_weight_shapes = {
            (12, 8),
            (8, 12),
            (3, 12, 8),
            (3, 8, 12),
            (6, 12, 8),
            (6, 8, 12),
        }

        for implementation in ("eager", "grouped_mm", "batched_mm"):
            with self.subTest(implementation=implementation):
                module.config._experts_implementation_internal = implementation
                inputs = hidden_states.detach().clone().requires_grad_(True)
                routing_weights = top_k_weights.detach().clone().requires_grad_(True)
                saved_tensors = []

                def pack_hook(tensor):
                    saved_tensors.append(tensor)
                    return tensor

                with patch(
                    "transformers.integrations.gguf.dequantize_gguf_tensor", wraps=dequantize_gguf_tensor
                ) as dequantize:
                    with torch.autograd.graph.saved_tensors_hooks(pack_hook, lambda tensor: tensor):
                        output = module(inputs, top_k_index, routing_weights)
                    expected_forward_calls = 9 if implementation == "eager" else 3
                    self.assertEqual(dequantize.call_count, expected_forward_calls)
                    output.sum().backward()
                    self.assertEqual(dequantize.call_count, 2 * expected_forward_calls)

                packed_saved = [tensor for tensor in saved_tensors if tensor.dtype == torch.uint8]
                self.assertEqual(len(packed_saved), expected_forward_calls)
                self.assertTrue(all(tensor.numel() == module.gate_proj.numel() for tensor in packed_saved))
                self.assertFalse(
                    any(
                        tensor.is_floating_point() and tuple(tensor.shape) in logical_weight_shapes
                        for tensor in saved_tensors
                    )
                )
                self.assertIsNotNone(inputs.grad)
                self.assertIsNotNone(routing_weights.grad)
                self.assertTrue(all(parameter.grad is None for parameter in module.parameters()))

    def test_gguf_expert_dtype_matrix_matches_floating_reference(self):
        packed_module, full_weights = _make_test_gguf_experts(intermediate_size=16)
        floating_module, _ = _make_test_gguf_experts(
            packed=False,
            full_weights=full_weights,
            intermediate_size=16,
        )
        hidden_states = torch.randn(3, 8)
        top_k_index = torch.tensor([[0, 1], [2, 1], [0, 2]])
        top_k_weights = torch.tensor([[0.7, 0.3], [0.4, 0.6], [0.2, 0.8]])
        grad_output = torch.randn_like(hidden_states)

        for implementation in ("eager", "grouped_mm", "batched_mm"):
            packed_module.config._experts_implementation_internal = implementation
            floating_module.config._experts_implementation_internal = implementation
            for input_dtype in (torch.float16, torch.bfloat16, torch.float32):
                for compute_dtype in (torch.float16, torch.bfloat16, torch.float32):
                    with self.subTest(
                        implementation=implementation,
                        input_dtype=input_dtype,
                        compute_dtype=compute_dtype,
                    ):
                        packed_module.set_compute_dtype(compute_dtype)
                        floating_module.set_compute_dtype(compute_dtype)
                        inputs = hidden_states.to(input_dtype).detach().clone().requires_grad_(True)
                        routing_weights = top_k_weights.detach().clone().requires_grad_(True)
                        reference_inputs = hidden_states.to(input_dtype).detach().clone().requires_grad_(True)
                        reference_routing_weights = top_k_weights.detach().clone().requires_grad_(True)

                        actual = packed_module(inputs, top_k_index, routing_weights)
                        expected = floating_module(reference_inputs, top_k_index, reference_routing_weights)
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

                        actual.backward(grad_output.to(input_dtype))
                        expected.backward(grad_output.to(input_dtype))
                        assert inputs.grad is not None and reference_inputs.grad is not None
                        torch.testing.assert_close(inputs.grad, reference_inputs.grad, rtol=0, atol=0)
                        assert routing_weights.grad is not None and reference_routing_weights.grad is not None
                        torch.testing.assert_close(
                            routing_weights.grad, reference_routing_weights.grad, rtol=0, atol=0
                        )

    def test_gguf_expert_autocast_matches_floating_reference(self):
        packed_module, full_weights = _make_test_gguf_experts()
        floating_module, _ = _make_test_gguf_experts(packed=False, full_weights=full_weights)
        hidden_states = torch.randn(3, 8)
        top_k_index = torch.tensor([[0, 1], [2, 1], [0, 2]])
        top_k_weights = torch.tensor([[0.7, 0.3], [0.4, 0.6], [0.2, 0.8]])
        grad_output = torch.randn_like(hidden_states)

        for implementation in ("eager", "grouped_mm", "batched_mm"):
            with self.subTest(implementation=implementation):
                packed_module.config._experts_implementation_internal = implementation
                floating_module.config._experts_implementation_internal = implementation
                inputs = hidden_states.detach().clone().requires_grad_(True)
                routing_weights = top_k_weights.detach().clone().requires_grad_(True)
                reference_inputs = hidden_states.detach().clone().requires_grad_(True)
                reference_routing_weights = top_k_weights.detach().clone().requires_grad_(True)

                with torch.autocast("cpu", dtype=torch.bfloat16):
                    actual = packed_module(inputs, top_k_index, routing_weights)
                    expected = floating_module(reference_inputs, top_k_index, reference_routing_weights)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

                actual.backward(grad_output)
                expected.backward(grad_output)
                assert inputs.grad is not None and reference_inputs.grad is not None
                torch.testing.assert_close(inputs.grad, reference_inputs.grad, rtol=0, atol=0)
                assert routing_weights.grad is not None and reference_routing_weights.grad is not None
                torch.testing.assert_close(routing_weights.grad, reference_routing_weights.grad, rtol=0, atol=0)

    @require_torch_gpu
    @require_torch_bf16
    def test_gguf_expert_accelerator_autocast_matches_floating_reference(self):
        packed_module, full_weights = _make_test_gguf_experts(intermediate_size=16)
        floating_module, _ = _make_test_gguf_experts(
            packed=False,
            full_weights=full_weights,
            intermediate_size=16,
        )
        packed_module.to(torch_device)
        floating_module.to(torch_device)
        hidden_states = torch.randn(3, 8, device=torch_device)
        top_k_index = torch.tensor([[0, 1], [2, 1], [0, 2]], device=torch_device)
        top_k_weights = torch.tensor(
            [[0.7, 0.3], [0.4, 0.6], [0.2, 0.8]],
            device=torch_device,
        )
        grad_output = torch.randn_like(hidden_states)

        for implementation in ("eager", "grouped_mm", "batched_mm"):
            with self.subTest(implementation=implementation):
                packed_module.config._experts_implementation_internal = implementation
                floating_module.config._experts_implementation_internal = implementation
                inputs = hidden_states.detach().clone().requires_grad_(True)
                routing_weights = top_k_weights.detach().clone().requires_grad_(True)
                reference_inputs = hidden_states.detach().clone().requires_grad_(True)
                reference_routing_weights = top_k_weights.detach().clone().requires_grad_(True)

                with torch.autocast(torch_device, dtype=torch.bfloat16):
                    actual = packed_module(inputs, top_k_index, routing_weights)
                    expected = floating_module(reference_inputs, top_k_index, reference_routing_weights)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

                actual.backward(grad_output)
                expected.backward(grad_output)
                assert inputs.grad is not None and reference_inputs.grad is not None
                torch.testing.assert_close(inputs.grad, reference_inputs.grad, rtol=0, atol=0)
                assert routing_weights.grad is not None and reference_routing_weights.grad is not None
                torch.testing.assert_close(routing_weights.grad, reference_routing_weights.grad, rtol=0, atol=0)

    def test_gguf_expert_reentrant_checkpointing_preserves_gradients_and_counts(self):
        from unittest.mock import patch

        from torch.utils.checkpoint import checkpoint

        module, _ = _make_test_gguf_experts()
        hidden_states = torch.randn(3, 8)
        top_k_index = torch.tensor([[0, 1], [2, 1], [0, 2]])
        top_k_weights = torch.tensor([[0.7, 0.3], [0.4, 0.6], [0.2, 0.8]])
        grad_output = torch.randn_like(hidden_states)

        for implementation in ("eager", "grouped_mm", "batched_mm"):
            with self.subTest(implementation=implementation):
                module.config._experts_implementation_internal = implementation
                reference_inputs = hidden_states.detach().clone().requires_grad_(True)
                reference_routing_weights = top_k_weights.detach().clone().requires_grad_(True)
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    expected = module(reference_inputs, top_k_index, reference_routing_weights)
                expected.backward(grad_output)

                inputs = hidden_states.detach().clone().requires_grad_(True)
                routing_weights = top_k_weights.detach().clone().requires_grad_(True)
                with patch(
                    "transformers.integrations.gguf.dequantize_gguf_tensor", wraps=dequantize_gguf_tensor
                ) as dequantize:
                    with torch.autocast("cpu", dtype=torch.bfloat16):
                        actual = checkpoint(
                            lambda states, weights: module(states, top_k_index, weights),
                            inputs,
                            routing_weights,
                            use_reentrant=True,
                        )
                    expected_forward_calls = 9 if implementation == "eager" else 3
                    self.assertEqual(dequantize.call_count, expected_forward_calls)
                    actual.backward(grad_output)
                    self.assertEqual(dequantize.call_count, 3 * expected_forward_calls)

                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                assert inputs.grad is not None and reference_inputs.grad is not None
                torch.testing.assert_close(inputs.grad, reference_inputs.grad, rtol=0, atol=0)
                assert routing_weights.grad is not None and reference_routing_weights.grad is not None
                torch.testing.assert_close(routing_weights.grad, reference_routing_weights.grad, rtol=0, atol=0)

    def test_gguf_expert_custom_backward_requires_input_gradients(self):
        from unittest.mock import patch

        from transformers.integrations.gguf import _GGUFExpertProjectionFunction

        module, _ = _make_test_gguf_experts()
        hidden_states = torch.randn(3, 8)
        top_k_index = torch.tensor([[0, 1], [2, 1], [0, 2]])
        top_k_weights = torch.tensor([[0.7, 0.3], [0.4, 0.6], [0.2, 0.8]])

        for implementation in ("eager", "grouped_mm", "batched_mm"):
            with self.subTest(implementation=implementation):
                module.config._experts_implementation_internal = implementation
                routing_weights = top_k_weights.detach().clone().requires_grad_(True)
                with patch.object(
                    _GGUFExpertProjectionFunction,
                    "apply",
                    wraps=_GGUFExpertProjectionFunction.apply,
                ) as projection:
                    output = module(hidden_states, top_k_index, routing_weights)
                    self.assertEqual(projection.call_count, 0)
                    output.sum().backward()
                    self.assertIsNotNone(routing_weights.grad)

                with (
                    patch.object(
                        _GGUFExpertProjectionFunction,
                        "apply",
                        wraps=_GGUFExpertProjectionFunction.apply,
                    ) as projection,
                    torch.no_grad(),
                ):
                    module(hidden_states, top_k_index, top_k_weights)
                    self.assertEqual(projection.call_count, 0)

    def test_qwen3_moe_replacement_and_converter_rewrite(self):
        from transformers.core_model_loading import WeightRenaming
        from transformers.modeling_gguf_pytorch_utils import get_gguf_converters

        config = Qwen3MoeConfig(
            vocab_size=8,
            hidden_size=8,
            intermediate_size=16,
            moe_intermediate_size=4,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            num_experts=4,
            num_experts_per_tok=2,
        )
        with torch.device("meta"):
            model = Qwen3MoeForCausalLM(config)
        weight_mapping = get_gguf_converters("qwen3_moe")
        quantizer = GGUFQuantizer(GGUFConfig(architecture="qwen3_moe"), weight_mapping=weight_mapping)
        quantizer.update_dtype(torch.bfloat16)
        checkpoint_tensors = {
            "blk.0.ffn_gate_exps.weight": GGUFQuantizedTensor(
                torch.empty((4, 4, 2), dtype=torch.uint8),
                quant_type=gguf.GGMLQuantizationType.Q5_K,
                logical_shape=(4, 4, 8),
            )
        }
        quantizer.set_weight_mapping(weight_mapping, checkpoint_tensors)
        quantizer.preprocess_model(model, dtype=torch.float32, device_map={"": "cpu"})
        experts: Any = model.get_submodule("model.layers.0.mlp.experts")
        self.assertIsInstance(experts, GGUFExperts)
        self.assertFalse(hasattr(experts, "gate_up_proj"))
        self.assertEqual(experts.gate_proj.device.type, "meta")
        self.assertEqual(experts.compute_dtype, torch.bfloat16)

        conversions = quantizer.update_weight_conversions([])
        renamings = {
            source: rule.target_patterns[0]
            for rule in conversions
            if isinstance(rule, WeightRenaming)
            for source in rule.source_patterns
        }
        self.assertEqual(renamings[r"\.ffn_gate_exps\.weight"], ".mlp.experts.gate_proj")
        self.assertEqual(renamings[r"\.ffn_up_exps\.weight"], ".mlp.experts.up_proj")

        gate_name = "model.layers.0.mlp.experts.gate_proj"
        self.assertEqual(quantizer.param_storage_bytes[gate_name], 32)
        self.assertEqual(quantizer.param_element_size(model, gate_name, experts.gate_proj), 0.25)

        previous_implementation = model.config._experts_implementation
        with self.assertRaisesRegex(ValueError, "'deepgemm' is not supported by GGUFExpertsInterface"):
            model.set_experts_implementation("deepgemm")
        self.assertEqual(model.config._experts_implementation, previous_implementation)
        with self.assertRaisesRegex(ValueError, "'deepgemm' is not supported by GGUFExpertsInterface"):
            model.set_experts_implementation({"": "deepgemm"})
        self.assertEqual(model.config._experts_implementation, previous_implementation)

        model.set_experts_implementation({"": "batched_mm"})
        self.assertEqual(model.config._experts_implementation, "batched_mm")
        model.set_experts_implementation({"": previous_implementation})

        custom_implementation = "test_gguf_custom_experts"
        ALL_GGUF_EXPERTS_FUNCTIONS[custom_implementation] = lambda *args, **kwargs: None
        try:
            model.set_experts_implementation(custom_implementation)
            self.assertEqual(model.config._experts_implementation, custom_implementation)
            model.set_experts_implementation(previous_implementation)
        finally:
            del ALL_GGUF_EXPERTS_FUNCTIONS[custom_implementation]

        invalid_config = copy.deepcopy(config)
        with torch.device("meta"):
            invalid_model = Qwen3MoeForCausalLM(invalid_config)
        invalid_model.config._experts_implementation_internal = "deepgemm"
        with self.assertRaisesRegex(ValueError, "'deepgemm' is not supported by GGUFExpertsInterface"):
            quantizer.preprocess_model(invalid_model, dtype=torch.float32, device_map={"": "cpu"})
        self.assertNotIsInstance(invalid_model.get_submodule("model.layers.0.mlp.experts"), GGUFExperts)


if __name__ == "__main__":
    unittest.main()
