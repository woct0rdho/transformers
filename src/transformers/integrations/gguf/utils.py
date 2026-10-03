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

import re
from copy import deepcopy

import torch

from ...core_model_loading import WeightConverter, WeightRenaming, WeightTransform, rename_source_key
from .dequant import GGML_BLOCK
from .gguf_conversion_mapping import GGUF_ARCHS, Cast, Dequantize
from .reader import GgufHeader


def is_gguf_arch_supported(gguf_arch: str) -> bool:
    """Whether this path handles the architecture, or the legacy loader has to."""
    return gguf_arch in GGUF_ARCHS


def get_gguf_conversion_mapping(gguf_arch: str, config) -> list[WeightTransform]:
    """Weight transforms turning a GGUF checkpoint of `gguf_arch` into transformers weights."""
    if gguf_arch not in GGUF_ARCHS:
        raise ValueError(f"GGUF architecture {gguf_arch!r} is not supported yet. Supported: {sorted(GGUF_ARCHS)}.")
    return GGUF_ARCHS[gguf_arch](config)


def get_gguf_plan(
    header: GgufHeader, mapping: list[WeightTransform]
) -> tuple[dict[str, int], dict[str, int], dict[str, torch.Tensor], list[str]]:
    """Work out, for every tensor the file stores as blocks, whether it can stay that way.

    It can if none of the conversions on its way to the model touch the bytes. `mapping` is the whole
    of what will run, so this is decided, not guessed.

    Returns:
        `quantized`: every block tensor, keyed by the model's name for it -> ggml type.
            `{"lm_head.weight": 14, "model.embed_tokens.weight": 8}`
        `packable`: the subset whose conversions are all safe on packed bytes, so the layer can be
            swapped for a `GgufLinear` / `GgufExperts` / `GgufEmbedding` and keep its blocks.
            `{"model.layers.0.mlp.experts.gate_up_proj": 12}`
        `permutations`: model name -> index tensor, for a packed weight whose *input* is gathered
            instead, since permuting its columns would mean requantizing.
            `{"model.layers.0.linear_attn.out_proj.weight": tensor([0, 1, 2, ...])}`
        `names`: every tensor's name after conversion, in the order the file stores them.
    """
    # On a copy: asking a transform whether it matches a name marks it as used and arms the stateful
    # renamings, and the mapping handed back to the loader has to be untouched by that.
    mapping = deepcopy(mapping)
    renamings = [entry for entry in mapping if isinstance(entry, WeightRenaming)]
    converters = [entry for entry in mapping if isinstance(entry, WeightConverter)]

    pattern_to_converter = {pattern: converter for converter in converters for pattern in converter.source_patterns}

    quantized, packable, permutations, names = {}, {}, {}, []
    for gguf_name, ggml_type in header.ggml_types.items():
        param_name, source_pattern = rename_source_key(gguf_name, renamings, converters)
        names.append(param_name)
        if ggml_type not in GGML_BLOCK:
            continue
        quantized[param_name] = ggml_type
        converter = pattern_to_converter.get(source_pattern)
        operations = getattr(converter, "operations", ())
        # every conversion applied to this tensor must be safe on packed bytes
        if converter is None or all(getattr(op, "supports_packed", False) for op in operations):
            packable[param_name] = ggml_type
            # an op that cannot reorder packed columns asks for its input to be reordered instead
            for operation in operations:
                if (permutation := getattr(operation, "input_permutation", None)) is not None:
                    permutations[param_name] = permutation
    return quantized, packable, permutations, names


def get_unconverted_keys(mapping: list[WeightTransform], header: GgufHeader) -> list[str]:
    """The keys of the tensors no converter claims, in the form the loader will match them."""
    mapping = deepcopy(mapping)
    renamings = [entry for entry in mapping if isinstance(entry, WeightRenaming)]
    converters = [entry for entry in mapping if isinstance(entry, WeightConverter)]
    renamed = [rename_source_key(name, renamings, converters) for name in header.ggml_types]
    return [name for name, converter_pattern in renamed if converter_pattern is None]


def add_gguf_load_ops(
    mapping: list[WeightTransform], needs_unpacking: dict[str, int], header: GgufHeader, dtype
) -> list:
    """Give every tensor the two ops it needs: unpack its blocks first, cast to `dtype` last.

    A tensor that already has a converter gets them added at either end of it. One that has none gets
    a converter created for it, which only runs those two ops and leaves the name alone.
    """
    unconverted = get_unconverted_keys(mapping, header)
    converters = [entry for entry in mapping if isinstance(entry, WeightConverter)]
    dequantize_op = Dequantize(needs_unpacking, dtype) if needs_unpacking else None
    cast_op = Cast(dtype)
    for converter in converters:
        if dequantize_op is not None:
            converter.operations.insert(0, dequantize_op)
        converter.operations.append(cast_op)
    # `Dequantize` passes through a name it was not given, so one converter serves both kinds here
    operations = [dequantize_op, cast_op] if dequantize_op is not None else [cast_op]
    if unconverted:
        return mapping + [
            WeightConverter(
                source_patterns=[f"({re.escape(name)})" for name in unconverted],
                target_patterns=[r"\1"],
                operations=operations,
            )
        ]
    return mapping
