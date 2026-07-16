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
import unittest

from transformers.testing_utils import require_gguf, require_torch
from transformers.utils import is_gguf_available, is_torch_available


if is_torch_available():
    import torch
    from torch.nn import functional as F

    from transformers.integrations.gguf import GGUFEmbedding, GGUFLinear
    from transformers.integrations.gguf_dequant import GGUFQuantizedTensor, dequantize_gguf_tensor

if is_gguf_available():
    import gguf
    import numpy as np


def _float_bytes(tensor):
    return tensor.detach().cpu().numpy().view(np.uint8).reshape(*tensor.shape[:-1], -1)


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

        embedding = GGUFEmbedding(3, 4)
        with self.assertRaisesRegex(RuntimeError, "GGUFEmbedding weight has not been loaded"):
            embedding(torch.tensor([0, 2]))

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

        container = torch.nn.Module()
        container.linear = linear
        container.embedding = embedding
        container.to(dtype=torch.float64)

        for module in (linear, embedding):
            self.assertEqual(module.compute_dtype, torch.float64)
        self.assertEqual(linear.weight.dtype, torch.uint8)
        self.assertEqual(embedding.weight.dtype, torch.uint8)
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


if __name__ == "__main__":
    unittest.main()
