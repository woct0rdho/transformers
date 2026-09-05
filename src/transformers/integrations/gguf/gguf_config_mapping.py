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
"""Rebuilding a model config from a GGUF file's metadata.

A GGUF repo ships no `config.json`: the metadata is the config, under llama.cpp's key names. One
function per architecture turns those into a transformers config dict, so a file loads on its own.

Everything the file states about the model's shape is set even where it matches the config class's
default, since those defaults are one checkpoint's values and would silently apply to another.
"""


def _required(metadata, *keys):
    for key in keys:
        if key in metadata:
            return metadata[key]
    raise ValueError(f"GGUF metadata is missing required field {keys[0]!r}")


def _qwen35_moe_config(metadata, tensor_names):
    prefix = "qwen35moe" if "qwen35moe.expert_count" in metadata else "qwen35"
    config = _qwen35_config(metadata, tensor_names, architecture=prefix)
    config["model_type"] = "qwen3_5_moe_text"
    config["architectures"] = ["Qwen3_5MoeForCausalLM"]
    config["num_experts"] = _required(metadata, f"{prefix}.expert_count")
    config["num_experts_per_tok"] = _required(metadata, f"{prefix}.expert_used_count")
    config["moe_intermediate_size"] = _required(
        metadata, f"{prefix}.expert_feed_forward_length", f"{prefix}.moe_intermediate_size"
    )
    config["shared_expert_intermediate_size"] = _required(
        metadata, f"{prefix}.expert_shared_feed_forward_length", f"{prefix}.shared_expert_intermediate_size"
    )
    return config


def _qwen35_config(metadata, tensor_names, architecture="qwen35", require_interval=True):
    """Qwen3.5: hybrid GatedDeltaNet + full attention, with an mrope and an MTP block."""
    key = lambda name: metadata[f"{architecture}.{name}"]  # noqa: E731
    head_dim = key("attention.key_length")
    num_attention_heads = int(key("attention.head_count"))
    num_key_value_heads = int(key("attention.head_count_kv"))
    if head_dim <= 0 or num_attention_heads <= 0 or num_key_value_heads <= 0:
        raise ValueError("Qwen3.5 GGUF attention dimensions must be positive")
    if num_attention_heads % num_key_value_heads:
        raise ValueError("Qwen3.5 GGUF attention heads must be divisible by key/value heads")
    value_heads = key("ssm.time_step_rank")
    key_heads = key("ssm.group_count")
    inner_size = key("ssm.inner_size")
    if value_heads <= 0 or key_heads <= 0 or value_heads % key_heads or inner_size % value_heads:
        raise ValueError("Qwen3.5 GGUF recurrent dimensions are incompatible")
    rope_dimension = key("rope.dimension_count")
    sections = key("rope.dimension_sections")
    if (
        rope_dimension <= 0
        or rope_dimension > head_dim
        or not isinstance(sections, list)
        or any(section < 0 for section in sections)
    ):
        raise ValueError("Qwen3.5 GGUF RoPE dimensions are invalid")
    if len(sections) == 4:
        if sections[-1] != 0:
            raise ValueError("Qwen3.5 GGUF RoPE sections must end with zero")
        sections = sections[:3]
    if len(sections) != 3 or 2 * sum(sections) != rope_dimension:
        raise ValueError("Qwen3.5 GGUF RoPE sections do not match the rotary dimension")
    explicit_recurrent = metadata.get(f"{architecture}.attention.recurrent_layers")
    if explicit_recurrent is not None:
        if not isinstance(explicit_recurrent, list):
            explicit_recurrent = [explicit_recurrent] * int(key("block_count"))
        if len(explicit_recurrent) < int(key("block_count")):
            raise ValueError("Qwen3.5 GGUF recurrent-layer metadata is incomplete")
        layer_types = ["linear_attention" if bool(value) else "full_attention" for value in explicit_recurrent]
    else:
        interval = int(key("full_attention_interval")) if require_interval else 4
        if interval <= 0:
            raise ValueError("Qwen3.5 GGUF full-attention interval must be positive")
        layer_types = [
            "linear_attention" if (index + 1) % interval else "full_attention"
            for index in range(int(key("block_count")))
        ]
    intermediate_size = metadata.get(f"{architecture}.feed_forward_length")
    if intermediate_size is None:
        intermediate_size = _required(
            metadata,
            f"{architecture}.expert_shared_feed_forward_length",
            f"{architecture}.expert_feed_forward_length",
        )
    mtp_layers = metadata.get(f"{architecture}.nextn_predict_layers", 0)
    config = {
        "model_type": "qwen3_5_text",
        # Nothing in the file states this, but callers that pick a class from `config.architectures`
        # would otherwise need a GGUF special case.
        "architectures": ["Qwen3_5ForCausalLM"],
        "max_position_embeddings": key("context_length"),
        "hidden_size": key("embedding_length"),
        "intermediate_size": intermediate_size,
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_key_value_heads,
        "head_dim": head_dim,
        "rms_norm_eps": key("attention.layer_norm_rms_epsilon"),
        "linear_conv_kernel_dim": key("ssm.conv_kernel"),
        "linear_key_head_dim": key("ssm.state_size"),
        "linear_num_key_heads": key_heads,
        "linear_num_value_heads": value_heads,
        # the file counts the multi-token-prediction block as a layer; the decoder stack does not
        "num_hidden_layers": key("block_count") - mtp_layers,
        "layer_types": layer_types[: int(key("block_count")) - mtp_layers],
        # the value dimension is stated whole, where transformers wants it per head
        "linear_value_head_dim": inner_size // value_heads,
        "rope_parameters": {
            "rope_theta": key("rope.freq_base"),
            # one section per axis, padded to four; transformers keeps the ones it uses
            "mrope_section": [section for section in sections if section],
            "mrope_interleaved": True,
            # a rotary width in dimensions, where transformers takes a fraction of the head
            "partial_rotary_factor": rope_dimension / head_dim,
        },
        # `read_gguf_metadata` leaves the vocabulary as its length
        "vocab_size": metadata["tokenizer.ggml.tokens"],
        # llama.cpp writes the output projection only when it is not the embedding matrix
        "tie_word_embeddings": "output.weight" not in tensor_names,
        # absent ids stay `None`, as they are on a config by default
        "eos_token_id": metadata.get("tokenizer.ggml.eos_token_id"),
        "bos_token_id": metadata.get("tokenizer.ggml.bos_token_id"),
        "pad_token_id": metadata.get("tokenizer.ggml.padding_token_id"),
    }
    if require_interval and explicit_recurrent is None:
        config["full_attention_interval"] = key("full_attention_interval")
    return config


GGUF_CONFIG_ARCHS = {
    "qwen35": _qwen35_config,
    "qwen35moe": _qwen35_moe_config,
}


def get_gguf_config(metadata: dict, tensor_names: tuple[str, ...]) -> dict:
    """The transformers config dict for a file with this metadata and these tensors.

    Raises for an architecture with no entry above; callers with a fallback check `GGUF_CONFIG_ARCHS`.
    """
    architecture = metadata["general.architecture"]
    if architecture not in GGUF_CONFIG_ARCHS:
        raise ValueError(
            f"Cannot rebuild a config from a GGUF file of architecture {architecture!r}. "
            f"Supported: {sorted(GGUF_CONFIG_ARCHS)}."
        )
    return GGUF_CONFIG_ARCHS[architecture](metadata, tensor_names)
