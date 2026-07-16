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

    from transformers.integrations.gguf_dequant import GGUFQuantizedTensor

if is_gguf_available():
    import gguf


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


if __name__ == "__main__":
    unittest.main()
