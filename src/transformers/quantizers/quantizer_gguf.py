# Copyright 2026 The HuggingFace Team. All rights reserved.
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
"""Keeping GGUF weights in their blocks instead of unpacking them at load time."""

from ..utils import is_torch_available, is_torch_mps_available, logging
from ..utils.quantization_config import GgufConfig
from .base import HfQuantizer


if is_torch_available():
    import torch

    from ..core_model_loading import WeightConverter, WeightRenaming
    from ..integrations.gguf.dequant import GGML_BLOCK
    from ..integrations.gguf.kernels import get_gguf_kernel
    from ..integrations.gguf.reader import GgufHeader, load_gguf_state_dict, read_gguf_metadata
    from ..integrations.gguf.utils import (
        add_gguf_load_ops,
        get_gguf_conversion_mapping,
        get_gguf_plan,
        is_gguf_arch_supported,
    )


logger = logging.get_logger(__name__)


def _persistent_conversion_mapping(mapping):
    """Split fused expert conversions into targets that can retain independent packed tensors."""
    persistent = []
    for transform in mapping:
        if not isinstance(transform, WeightConverter):
            persistent.append(transform)
            continue
        target = transform.target_patterns[0]
        sources = transform.source_patterns
        if "experts.gate_up_proj" in target and len(sources) == 2:
            for source, projection in zip(sources, ("gate_proj", "up_proj")):
                persistent.append(
                    WeightRenaming(source, target.replace("experts.gate_up_proj", f"experts.{projection}"))
                )
        else:
            persistent.append(transform)
    return persistent


class GgufHfQuantizer(HfQuantizer):
    """Loads a quantized GGUF checkpoint with its weights left in GGUF blocks."""

    quantization_config: "GgufConfig"
    header: "GgufHeader"  # set by `read_header`, before any of the loading hooks run
    requires_calibration = False
    requires_parameters_quantization = False

    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)
        self.pre_quantized = True
        self.packed_modules = {}
        self.input_permutations = {}
        self.quantized = {}
        self.names = []
        self.mapping = []
        self.header = None
        self.kernel = None
        self.dtype = None
        # Name patterns the model declares FP32-strict, resolved once the model exists.
        self.keep_fp32 = ()
        # TODO: only for the legacy loader — drop this, and every hook that guards on it, once all
        # architectures go through this path and there is no fallback left
        self.supported = False
        self._unsupported_distributed_modes = set()

    @staticmethod
    def _validate_device_map(device_map):
        # Disk offload is not supported for GGUF files. Loading may apply GGUF-specific conversion operations,
        # while Accelerate's disk hooks require a checkpoint representation that can be reopened independently.
        if "disk" in {str(place) for place in getattr(device_map, "values", lambda: [device_map])()}:
            raise RuntimeError(
                "One or more modules is configured to be mapped to disk. Disk offload is not supported "
                "for models loaded from GGUF files."
            )

    def validate_environment(self, *args, **kwargs):
        # Called both before resolving an automatic device map and after inference has produced its concrete map.
        self._validate_device_map(kwargs.get("device_map"))
        if not self.supported:
            return
        if self.quantization_config.dequantize:
            return
        if not self.header.has_quantized_weights:
            # Nothing is in blocks, so there is nothing to keep packed and nothing to unpack: this is
            # the dequantized path already, on any device. Setting the flag is what leaves it an
            # ordinary dense model, and no fallback happened, so there is nothing to warn about.
            self.quantization_config.dequantize = True
            return
        # The persistent modules have a torch dequantization path, so a fused kernel is optional.
        self.kernel = get_gguf_kernel()

    def update_device_map(self, device_map):
        """Default to the backend the blocks are computed on, rather than the host."""
        self._validate_device_map(device_map)
        if self.quantization_config.dequantize:
            return device_map
        if device_map is None and is_torch_mps_available():
            device_map = {"": torch.device("mps")}
            logger.info(f"No `device_map` was passed; loading the GGUF weights on {device_map['']}.")
        return device_map

    def read_header(self, gguf_file: str):
        """Parse the file's metadata, once `from_pretrained` knows where the file is."""
        metadata, _ = read_gguf_metadata(gguf_file)
        self.supported = is_gguf_arch_supported(metadata["general.architecture"])
        if self.supported:
            if not self.quantization_config.dequantize and self._unsupported_distributed_modes:
                modes = ", ".join(sorted(self._unsupported_distributed_modes))
                raise RuntimeError(
                    f"Native {modes} is not supported for persistent GGUF weights because packed parameters "
                    "cannot be sharded while loading"
                )
            self.header = GgufHeader.from_file(gguf_file)
            self.validate_environment()

    def update_dtype(self, dtype):
        """Settle the dtype the model is loaded in, and keep it."""
        if dtype is None:
            if not self.supported:
                self.dtype = torch.get_default_dtype()
                return self.dtype
            dtype = self.header.dtype if self.header.dtype is not None else torch.float32
        self.dtype = dtype
        return dtype

    def get_state_dict(self, checkpoint_file: str, model):
        """The file's tensors, quantized ones kept as raw blocks."""
        if self.supported:
            return load_gguf_state_dict(self.header, mmap_policy=self.quantization_config.mmap_policy)

        from ..modeling_gguf_pytorch_utils import load_gguf_checkpoint

        legacy = load_gguf_checkpoint(
            checkpoint_file, return_tensors=True, model_to_load=model, torch_dtype=self.dtype
        )
        return legacy["tensors"]

    def _process_model_before_weight_loading(self, model, **kwargs):
        """Swap in `GgufLinear` wherever the weight can stay packed."""
        if not self.supported:
            return model
        mapping = get_gguf_conversion_mapping(self.header.architecture, model.config)
        if not self.quantization_config.dequantize:
            mapping = _persistent_conversion_mapping(mapping)
        self.mapping = mapping
        self.quantized, packable, self.input_permutations, self.names = get_gguf_plan(self.header, self.mapping)
        # Which resident dtypes the model requires is the model's decision, not the adapter's. The
        # safetensors path applies the same plan through `PreTrainedModel._get_dtype_plan`; asking for it
        # here keeps the two loaders of one model consistent. It matters because the final cast of the
        # conversion chain is the last chance to keep a value: a tensor rounded to the load dtype cannot
        # be recovered by the operation that consumes it.
        dtype_plan = getattr(model, "_get_dtype_plan", None)
        if callable(dtype_plan) and self.dtype is not None:
            self.keep_fp32 = tuple(dtype_plan(self.dtype))
        if self.quantization_config.dequantize:
            return model
        floating = set(self.names) - set(self.quantized)
        from ..integrations.gguf.modules import replace_with_gguf_modules

        self.packed_modules = replace_with_gguf_modules(
            model,
            compute_dtype=self.dtype,
            packed_parameter_names=set(packable),
            floating_checkpoint_params=floating,
        )
        return model

    def _process_model_after_weight_loading(self, model, **kwargs):
        """Fill in what needs the weights already in place: the input permutations, then the layer kernels."""
        if not self.supported:
            return model
        from ..integrations.gguf.kernels import kernelize_ggml_layers

        for param_name, module in self.packed_modules.items():
            permutation = self.input_permutations.get(param_name)
            if permutation is not None:
                module.input_permutation = permutation.to(module.weight.device)
        kernelize_ggml_layers(model)
        # `save_pretrained` writes a checkpoint back by reversing whatever loaded the model, and this
        # mapping does not reverse: `Dequantize` has no inverse worth defining, and nobody wants a GGUF
        # written back. Dropped once the weights are in place, so saving falls back on the model's own
        # mapping -- which is what it needs, a dequantized model being an ordinary dense one. A model
        # that kept its blocks never reaches that code: `is_serializable` refuses it first.
        model._weight_conversions = None
        return model

    def _record_unsupported_distributed_modes(self, config):
        distributed_config = getattr(config, "distributed_config", None)
        if distributed_config is None:
            return

        def get_option(name, default=None):
            if isinstance(distributed_config, dict):
                return distributed_config.get(name, default)
            return getattr(distributed_config, name, default)

        if (get_option("tp_size") or 1) > 1:
            self._unsupported_distributed_modes.add("tensor parallelism")
        if (get_option("fsdp_size") or 1) > 1:
            self._unsupported_distributed_modes.add("FSDP2")
        if get_option("enable_expert_parallel", False):
            self._unsupported_distributed_modes.add("expert parallelism")

    def update_tp_plan(self, config):
        self._record_unsupported_distributed_modes(config)
        return config

    def update_ep_plan(self, config):
        self._record_unsupported_distributed_modes(config)
        return config

    def update_weight_conversions(self, weight_conversions):
        """Prepend this file's conversions: the GGUF -> transformers mapping, and the unpacking."""
        if not self.supported:
            return weight_conversions
        packed_params = set(self.packed_modules)
        to_unpack = {name: t for name, t in self.quantized.items() if name not in packed_params}
        return add_gguf_load_ops(
            self.mapping + weight_conversions,
            to_unpack,
            self.names,
            self.dtype,
            keep_fp32=self.keep_fp32,
        )

    def param_element_size(self, model, param_name: str, param: "torch.Tensor") -> float:
        """Report packed bytes per logical element for parameters retained in GGUF blocks."""
        if not self.supported or self.quantization_config.dequantize or param_name not in self.packed_modules:
            return super().param_element_size(model, param_name, param)
        quant_type = self.quantized.get(param_name)
        if quant_type is None:
            return super().param_element_size(model, param_name, param)
        block_elements, block_bytes = GGML_BLOCK[quant_type]
        return block_bytes / block_elements

    def param_needs_quantization(self, model, param_name: str, **kwargs) -> bool:
        return False

    @property
    def is_trainable(self) -> bool:
        return self.supported and not self.quantization_config.dequantize

    @property
    def is_compileable(self) -> bool:
        return not self.supported or self.quantization_config.dequantize

    def is_serializable(self, safe_serialization=None) -> bool:
        return False
