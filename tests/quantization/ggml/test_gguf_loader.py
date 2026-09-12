# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from transformers.core_model_loading import WeightConverter, WeightRenaming
from transformers.integrations.gguf.dequant import GGML_BLOCK, GGML_Q4_K
from transformers.integrations.gguf.gguf_conversion_mapping import Concatenate
from transformers.modeling_utils import PreTrainedModel
from transformers.quantizers.quantizer_gguf import GgufHfQuantizer, _persistent_conversion_mapping
from transformers.utils.quantization_config import GgufConfig


class GgufLoaderTests(unittest.TestCase):
    def test_persistent_quantizer_reports_packed_element_size(self):
        quantizer = GgufHfQuantizer(GgufConfig())
        quantizer.supported = True
        quantizer.quantized = {"packed.weight": GGML_Q4_K}
        quantizer.packed_modules = {"packed.weight": object()}

        packed = torch.empty(2, 256, dtype=torch.uint8)
        ordinary = torch.empty(2, 256, dtype=torch.bfloat16)
        block_elements, block_bytes = GGML_BLOCK[GGML_Q4_K]

        self.assertAlmostEqual(
            quantizer.param_element_size(None, "packed.weight", packed), block_bytes / block_elements
        )
        self.assertEqual(quantizer.param_element_size(None, "ordinary.weight", ordinary), ordinary.element_size())

    def test_persistent_quantizer_keeps_packed_path_without_kernel(self):
        quantizer = GgufHfQuantizer(GgufConfig())
        quantizer.supported = True
        quantizer.header = SimpleNamespace(has_quantized_weights=True)
        with patch("transformers.quantizers.quantizer_gguf.get_gguf_kernel", return_value=False):
            quantizer.validate_environment()
        self.assertFalse(quantizer.quantization_config.dequantize)
        self.assertFalse(quantizer.kernel)

    def test_persistent_quantizer_allows_multi_device_placement(self):
        metadata = {"general.architecture": "synthetic"}
        for device_map in ({"model": 0, "lm_head": 1}, "auto"):
            with self.subTest(device_map=device_map):
                quantizer = GgufHfQuantizer(GgufConfig())
                self.assertEqual(quantizer.update_device_map(device_map), device_map)
                header = SimpleNamespace(has_quantized_weights=True)
                with (
                    patch("transformers.quantizers.quantizer_gguf.read_gguf_metadata", return_value=(metadata, ())),
                    patch("transformers.quantizers.quantizer_gguf.is_gguf_arch_supported", return_value=True),
                    patch("transformers.quantizers.quantizer_gguf.GgufHeader.from_file", return_value=header),
                    patch.object(quantizer, "validate_environment"),
                ):
                    quantizer.read_header("unused.gguf")
                self.assertIs(quantizer.header, header)

    def test_quantizer_rejects_disk_after_automatic_device_map_resolution(self):
        quantizer = GgufHfQuantizer(GgufConfig())
        with self.assertRaisesRegex(RuntimeError, "Disk offload"):
            quantizer.validate_environment(device_map={"model": 0, "lm_head": "disk"})

    def test_persistent_quantizer_rejects_native_parameter_sharding(self):
        modes = (
            ("tensor parallelism", "update_tp_plan", SimpleNamespace(tp_size=2)),
            ("FSDP2", "update_tp_plan", SimpleNamespace(fsdp_size=2)),
            ("expert parallelism", "update_ep_plan", SimpleNamespace(enable_expert_parallel=True)),
        )
        metadata = {"general.architecture": "synthetic"}
        for mode, hook, distributed_config in modes:
            with self.subTest(mode=mode):
                quantizer = GgufHfQuantizer(GgufConfig())
                getattr(quantizer, hook)(SimpleNamespace(distributed_config=distributed_config))
                with (
                    patch("transformers.quantizers.quantizer_gguf.read_gguf_metadata", return_value=(metadata, ())),
                    patch("transformers.quantizers.quantizer_gguf.is_gguf_arch_supported", return_value=True),
                ):
                    with self.assertRaisesRegex(RuntimeError, mode):
                        quantizer.read_header("unused.gguf")

    def test_persistent_quantizer_allows_non_sharding_distributed_configs(self):
        configs = (
            SimpleNamespace(tp_size=1, fsdp_size=1, pp_size=1, enable_expert_parallel=False),
            SimpleNamespace(tp_size=1, fsdp_size=1, pp_size=2, enable_expert_parallel=False),
        )
        metadata = {"general.architecture": "synthetic"}
        for distributed_config in configs:
            with self.subTest(distributed_config=distributed_config):
                quantizer = GgufHfQuantizer(GgufConfig())
                config = SimpleNamespace(distributed_config=distributed_config)
                quantizer.update_tp_plan(config)
                quantizer.update_ep_plan(config)
                header = SimpleNamespace(has_quantized_weights=True)
                with (
                    patch("transformers.quantizers.quantizer_gguf.read_gguf_metadata", return_value=(metadata, ())),
                    patch("transformers.quantizers.quantizer_gguf.is_gguf_arch_supported", return_value=True),
                    patch("transformers.quantizers.quantizer_gguf.GgufHeader.from_file", return_value=header),
                    patch.object(quantizer, "validate_environment"),
                ):
                    quantizer.read_header("unused.gguf")

    def test_persistent_mapping_splits_fused_expert_targets(self):
        mapping = _persistent_conversion_mapping(
            [
                WeightConverter(
                    source_patterns=[".ffn_gate_exps.weight", ".ffn_up_exps.weight"],
                    target_patterns=".mlp.experts.gate_up_proj",
                    operations=[Concatenate(dim=1)],
                )
            ]
        )
        targets = {
            rule.target_patterns[0]
            for rule in mapping
            if isinstance(rule, WeightRenaming) and "experts." in rule.target_patterns[0]
        }
        self.assertIn(".mlp.experts.gate_proj", targets)
        self.assertIn(".mlp.experts.up_proj", targets)
        self.assertNotIn(".mlp.experts.gate_up_proj", targets)

    def test_quantizer_resolves_the_model_dtype_plan(self):
        class _Model:
            config = SimpleNamespace()
            _keep_in_fp32_modules = None
            _keep_in_fp32_modules_strict = ["e_score_correction_bias"]

            def _get_dtype_plan(self, dtype):
                return PreTrainedModel._get_dtype_plan(self, dtype)

        with (
            patch("transformers.quantizers.quantizer_gguf.get_gguf_conversion_mapping", return_value=[]),
            patch("transformers.quantizers.quantizer_gguf.get_gguf_plan", return_value=({}, {}, {}, [])),
        ):
            quantizer = GgufHfQuantizer(GgufConfig(dequantize=True))
            quantizer.supported = True
            quantizer.header = SimpleNamespace(architecture="deepseek4")
            quantizer.dtype = torch.bfloat16
            quantizer._process_model_before_weight_loading(_Model())
        self.assertEqual(quantizer.keep_fp32, ("e_score_correction_bias",))

    def test_quantizer_forwards_the_dtype_plan_to_the_load_ops(self):
        name = "model.layers.0.mlp.gate.e_score_correction_bias"
        quantizer = GgufHfQuantizer(GgufConfig())
        quantizer.supported = True
        quantizer.dtype = torch.bfloat16
        quantizer.mapping = []
        quantizer.names = [name]
        quantizer.quantized = {}
        quantizer.packed_modules = {}
        quantizer.keep_fp32 = ("e_score_correction_bias",)

        mapping = quantizer.update_weight_conversions([])
        converter = next(rule for rule in mapping if isinstance(rule, WeightConverter))
        tensors = {"bias": torch.tensor([8.9697, 9.0261], dtype=torch.float32)}
        for op in converter.operations:
            tensors = op.convert(tensors, ["bias"], ["bias"], full_layer_name=name)
        loaded = tensors[name]
        self.assertEqual(loaded.dtype, torch.float32)
        torch.testing.assert_close(loaded, torch.tensor([8.9697, 9.0261]), rtol=0, atol=0)

    def test_quantizer_without_a_dtype_plan_keeps_the_load_dtype(self):
        name = "model.layers.0.self_attn.o_proj.weight"
        quantizer = GgufHfQuantizer(GgufConfig())
        quantizer.supported = True
        quantizer.dtype = torch.bfloat16
        quantizer.mapping = []
        quantizer.names = [name]
        quantizer.quantized = {}
        quantizer.packed_modules = {}

        mapping = quantizer.update_weight_conversions([])
        converter = next(rule for rule in mapping if isinstance(rule, WeightConverter))
        tensors = {"weight": torch.zeros(2, dtype=torch.float32)}
        for op in converter.operations:
            tensors = op.convert(tensors, ["weight"], ["weight"], full_layer_name=name)
        self.assertEqual(tensors[name].dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
