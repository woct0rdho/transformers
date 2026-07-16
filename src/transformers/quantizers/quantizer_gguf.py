# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""GGUF quantizer lifecycle integration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..utils import logging
from .base import HfQuantizer


if TYPE_CHECKING:
    from ..utils.quantization_config import GGUFConfig


logger = logging.get_logger(__name__)


class GGUFQuantizer(HfQuantizer):
    """Keep Qwen3 GGUF checkpoints compressed or dequantize through the compatibility fallback."""

    requires_calibration = False
    quantization_config: GGUFConfig

    def __init__(self, quantization_config: GGUFConfig | None = None, weight_mapping=None, **kwargs):
        if quantization_config is None:
            from ..utils.quantization_config import GGUFConfig

            quantization_config = GGUFConfig()
        kwargs.setdefault("pre_quantized", True)
        super().__init__(quantization_config=quantization_config, **kwargs)
        self.persistent = quantization_config.architecture in {"qwen3", "qwen3_moe"}
        self.compute_dtype = None
        self.weight_mapping = list(weight_mapping or [])
        self.checkpoint_storage_bytes = {}
        self.param_storage_bytes = {}

    def update_dtype(self, dtype):
        self.compute_dtype = dtype
        return dtype

    def validate_environment(self, *args, **kwargs):
        if self.quantization_config.architecture and not self.persistent:
            logger.warning_once(
                f"Persistent GGUF weights currently support Qwen3 and Qwen3-MoE only; "
                f"{self.quantization_config.architecture!r} will use load-time dequantization."
            )

    def set_weight_mapping(self, weight_mapping, checkpoint_tensors=None):
        self.weight_mapping = list(weight_mapping or [])
        self.checkpoint_storage_bytes = {
            name: tensor.numel() * tensor.element_size() for name, tensor in (checkpoint_tensors or {}).items()
        }

    @property
    def renaming_quantization_op(self):
        if self.persistent:
            from ..gguf_conversion_ops import GGUFSetMetadata

            return GGUFSetMetadata()

        from ..gguf_conversion_ops import GGUFDequantize

        return GGUFDequantize()

    def update_weight_conversions(self, weight_conversions):
        from copy import deepcopy

        from ..core_model_loading import WeightConverter, WeightRenaming, rename_source_key
        from ..gguf_conversion_ops import GGUFDequantize, GGUFSetMetadata

        injected = []
        for conversion in self.weight_mapping:
            if not isinstance(conversion, WeightConverter):
                injected.append(conversion)
                continue

            sources = conversion._original_source_patterns
            if self.quantization_config.architecture == "qwen3_moe" and sources == [
                r"\.ffn_gate_exps\.weight",
                r"\.ffn_up_exps\.weight",
            ]:
                injected.extend(
                    [
                        WeightRenaming(sources[0], ".mlp.experts.gate_proj"),
                        WeightRenaming(sources[1], ".mlp.experts.up_proj"),
                    ]
                )
                continue

            if self.persistent:
                operations = [GGUFSetMetadata(), *conversion.operations]
            else:
                operations = [GGUFDequantize(), *conversion.operations]

            injected.append(
                WeightConverter(
                    source_patterns=sources,
                    target_patterns=conversion._original_target_patterns,
                    operations=operations,
                )
            )

        updated_conversions = injected + list(weight_conversions)
        if self.persistent and self.checkpoint_storage_bytes:
            sizing_conversions = deepcopy(updated_conversions)
            renamings = [entry for entry in sizing_conversions if isinstance(entry, WeightRenaming)]
            converters = [entry for entry in sizing_conversions if isinstance(entry, WeightConverter)]
            self.param_storage_bytes = {}
            for source_name, storage_bytes in sorted(self.checkpoint_storage_bytes.items()):
                target_name, _ = rename_source_key(source_name, renamings, converters)
                self.param_storage_bytes[target_name] = self.param_storage_bytes.get(target_name, 0) + storage_bytes

        return updated_conversions

    def preserve_checkpoint_dtype(self, tensor, **kwargs):
        from ..integrations.gguf_dequant import GGUFQuantizedTensor

        return isinstance(tensor, GGUFQuantizedTensor)

    def param_element_size(self, model, param_name, param):
        if self.persistent and param_name in self.param_storage_bytes and param.numel() > 0:
            return self.param_storage_bytes[param_name] / param.numel()
        return super().param_element_size(model, param_name, param)

    def param_needs_quantization(self, model, param_name, **kwargs):
        return False

    def _process_model_before_weight_loading(self, model, **kwargs):
        if self.persistent:
            if self.quantization_config.architecture == "qwen3_moe" and model.config._experts_implementation not in {
                "eager",
                "grouped_mm",
                "batched_mm",
            }:
                raise ValueError(
                    f"GGUF experts do not support {model.config._experts_implementation!r}; "
                    "use 'eager', 'grouped_mm', or 'batched_mm'."
                )

            from ..integrations.gguf import replace_with_gguf_modules
            from ..utils import is_torch_available

            if is_torch_available():
                import torch

                device_map = kwargs.get("device_map") or {}
                target_devices = device_map.values() if isinstance(device_map, dict) else [device_map]
                uses_rocm = torch.version.hip is not None and any(
                    str(device).startswith("cuda") for device in target_devices
                )
                if uses_rocm and model.config._attn_implementation == "sdpa":
                    logger.warning_once("GGUF on ROCm uses eager attention because SDPA is unstable for this path.")
                    model.config._attn_implementation = "eager"
            replace_with_gguf_modules(model, compute_dtype=self.compute_dtype)
        return model

    def _process_model_after_weight_loading(self, model, **kwargs):
        if not self.persistent:
            self.remove_quantization_config(model)
        return model

    @property
    def is_trainable(self):
        return True

    def is_serializable(self):
        return False
