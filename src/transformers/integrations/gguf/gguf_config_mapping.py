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


def _standard_config(metadata, tensor_names, architecture, model_type, moe=False):
    key = lambda name: metadata[f"{architecture}.{name}"]  # noqa: E731
    vocab_size = metadata.get("tokenizer.ggml.tokens")
    if vocab_size is None:
        vocab_size = key("vocab_size")
    num_attention_heads = int(key("attention.head_count"))
    head_dim = metadata.get(f"{architecture}.attention.key_length")
    if head_dim is None:
        head_dim = key("embedding_length") // num_attention_heads
    config = {
        "model_type": model_type,
        "architectures": ["Qwen3MoeForCausalLM" if moe else "Qwen3ForCausalLM"],
        "max_position_embeddings": key("context_length"),
        "hidden_size": key("embedding_length"),
        "intermediate_size": key("feed_forward_length"),
        "num_hidden_layers": key("block_count"),
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": key("attention.head_count_kv"),
        "rms_norm_eps": key("attention.layer_norm_rms_epsilon"),
        "head_dim": head_dim,
        "rope_parameters": {"rope_type": "default", "rope_theta": key("rope.freq_base")},
        "vocab_size": vocab_size,
        "tie_word_embeddings": "output.weight" not in tensor_names,
        "eos_token_id": metadata.get("tokenizer.ggml.eos_token_id"),
        "bos_token_id": metadata.get("tokenizer.ggml.bos_token_id"),
        "pad_token_id": metadata.get("tokenizer.ggml.padding_token_id"),
    }
    if moe:
        config.update({"num_experts": key("expert_count"), "num_experts_per_tok": key("expert_used_count")})
        config["norm_topk_prob"] = True
    return config


def _qwen3_config(metadata, tensor_names):
    return _standard_config(metadata, tensor_names, "qwen3", "qwen3")


def _qwen3_moe_config(metadata, tensor_names):
    return _standard_config(metadata, tensor_names, "qwen3moe", "qwen3_moe", moe=True)


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


def _deepseek_v4_config(metadata, tensor_names):
    prefix = "deepseek4"
    key = lambda name: _required(metadata, f"{prefix}.{name}")  # noqa: E731
    vocab_size = metadata.get("tokenizer.ggml.tokens")
    if vocab_size is None:
        vocab_size = key("vocab_size")
    num_layers = int(key("block_count"))
    head_dim = int(key("attention.key_length"))
    value_length = metadata.get(f"{prefix}.attention.value_length", head_dim)
    if int(value_length) != head_dim:
        raise ValueError("DeepSeek V4 GGUF attention key and value dimensions must match")
    ratios = key("attention.compress_ratios")
    if not isinstance(ratios, list) or len(ratios) < num_layers:
        raise ValueError("DeepSeek V4 GGUF compression metadata must contain at least one entry per layer")
    ratios = ratios[:num_layers]
    names = {0: "sliding_attention", 4: "compressed_sparse_attention", 128: "heavily_compressed_attention"}
    unsupported = sorted(set(map(int, ratios)).difference(names))
    if unsupported:
        raise ValueError(f"DeepSeek V4 GGUF has unsupported compression ratios {unsupported}")

    rope_dimension = int(key("rope.dimension_count"))
    if not 0 < rope_dimension <= head_dim:
        raise ValueError("DeepSeek V4 GGUF rotary dimension is outside the attention head")
    hash_layers = int(_required(metadata, f"{prefix}.hash_layer_count", f"{prefix}.mlp.hash_layer_count"))
    if not 0 <= hash_layers <= num_layers:
        raise ValueError("DeepSeek V4 GGUF hash layer count is outside the layer schedule")

    clamp_values = _required(
        metadata, f"{prefix}.swiglu_clamp_exp", f"{prefix}.expert.swiglu_clamp_exp", f"{prefix}.expert.silu_limit"
    )
    if isinstance(clamp_values, list):
        if len(clamp_values) < num_layers:
            raise ValueError("DeepSeek V4 GGUF routed SwiGLU clamp metadata is incomplete")
        clamp_values = [float(value) for value in clamp_values[:num_layers]]
        if not clamp_values or clamp_values[0] <= 0 or any(value != clamp_values[0] for value in clamp_values[1:]):
            raise ValueError("DeepSeek V4 GGUF routed SwiGLU clamps must be one positive value")
        clamp_values = clamp_values[0]
    else:
        clamp_values = float(clamp_values)
    if clamp_values <= 0:
        raise ValueError("DeepSeek V4 GGUF routed SwiGLU clamp must be positive")
    shared_clamps = metadata.get(f"{prefix}.swiglu_clamp_shexp")
    if shared_clamps is not None:
        if not isinstance(shared_clamps, list):
            shared_clamps = [shared_clamps]
        if len(shared_clamps) < num_layers or any(
            float(value) != clamp_values for value in shared_clamps[:num_layers]
        ):
            raise ValueError("DeepSeek V4 GGUF shared-expert SwiGLU clamps must match routed experts")

    gating_func = _required(metadata, f"{prefix}.expert_gating_func", f"{prefix}.expert.gating_func")
    if int(gating_func) != 4:
        raise ValueError(f"DeepSeek V4 GGUF requires sqrtsoftplus expert gating enum 4, got {gating_func}")

    rope_parameters = {
        "main": {
            "rope_type": "default",
            "rope_theta": float(key("rope.freq_base")),
            "partial_rotary_factor": rope_dimension / head_dim,
        },
        "compress": {
            "rope_type": "default",
            "rope_theta": float(
                _required(metadata, f"{prefix}.attention.compress_rope_freq_base", f"{prefix}.compress_rope.freq_base")
            ),
            "partial_rotary_factor": rope_dimension / head_dim,
        },
    }
    scaling_type = str(metadata.get(f"{prefix}.rope.scaling.type", "none"))
    if scaling_type == "yarn":
        rope_parameters["compress"].update(
            {
                "rope_type": "yarn",
                "factor": float(_required(metadata, f"{prefix}.rope.scaling.factor")),
                "original_max_position_embeddings": int(
                    _required(
                        metadata,
                        f"{prefix}.rope.scaling.original_context_length",
                        f"{prefix}.rope.scaling.original_max_position_embeddings",
                    )
                ),
                "beta_fast": float(
                    _required(metadata, f"{prefix}.rope.scaling.yarn_beta_fast", f"{prefix}.rope.scaling.beta_fast")
                ),
                "beta_slow": float(
                    _required(metadata, f"{prefix}.rope.scaling.yarn_beta_slow", f"{prefix}.rope.scaling.beta_slow")
                ),
                "attention_factor": 1.0,
            }
        )
    elif scaling_type not in {"none", "default"}:
        raise ValueError(f"DeepSeek V4 GGUF has unsupported RoPE scaling type {scaling_type!r}")

    return {
        "model_type": "deepseek_v4",
        "architectures": ["DeepseekV4ForCausalLM"],
        "vocab_size": vocab_size,
        "hidden_size": key("embedding_length"),
        "moe_intermediate_size": _required(
            metadata, f"{prefix}.expert_feed_forward_length", f"{prefix}.moe_intermediate_size"
        ),
        "num_hidden_layers": num_layers,
        "num_attention_heads": key("attention.head_count"),
        "num_key_value_heads": key("attention.head_count_kv"),
        "head_dim": head_dim,
        "q_lora_rank": key("attention.q_lora_rank"),
        "o_lora_rank": _required(metadata, f"{prefix}.attention.output_lora_rank", f"{prefix}.attention.o_lora_rank"),
        "o_groups": _required(metadata, f"{prefix}.attention.output_group_count", f"{prefix}.attention.o_groups"),
        "n_routed_experts": key("expert_count"),
        "num_experts_per_tok": key("expert_used_count"),
        "n_shared_experts": key("expert_shared_count"),
        "norm_topk_prob": bool(metadata.get(f"{prefix}.expert_weights_norm", True)),
        "routed_scaling_factor": float(metadata.get(f"{prefix}.expert_weights_scale", 1.5)),
        "num_nextn_predict_layers": int(metadata.get(f"{prefix}.nextn_predict_layers", 1)),
        "max_position_embeddings": key("context_length"),
        "rope_theta": key("rope.freq_base"),
        "compress_rope_theta": _required(
            metadata, f"{prefix}.attention.compress_rope_freq_base", f"{prefix}.compress_rope.freq_base"
        ),
        "rms_norm_eps": key("attention.layer_norm_rms_epsilon"),
        "sliding_window": key("attention.sliding_window"),
        "index_n_heads": _required(metadata, f"{prefix}.attention.indexer.head_count", f"{prefix}.indexer.head_count"),
        "index_head_dim": _required(metadata, f"{prefix}.attention.indexer.key_length", f"{prefix}.indexer.head_dim"),
        "index_topk": _required(metadata, f"{prefix}.attention.indexer.top_k", f"{prefix}.indexer.top_k"),
        "hc_mult": _required(metadata, f"{prefix}.hyper_connection.count", f"{prefix}.hyper_connection.expansion"),
        "hc_sinkhorn_iters": _required(
            metadata, f"{prefix}.hyper_connection.sinkhorn_iterations", f"{prefix}.hyper_connection.sinkhorn_iters"
        ),
        "hc_eps": _required(metadata, f"{prefix}.hyper_connection.epsilon", f"{prefix}.hc_eps"),
        "swiglu_limit": clamp_values,
        "scoring_func": "sqrtsoftplus",
        "layer_types": [names[int(ratio)] for ratio in ratios],
        "compress_rates": {name: ratio for ratio, name in names.items() if ratio},
        "mlp_layer_types": ["hash_moe"] * hash_layers + ["moe"] * (num_layers - hash_layers),
        "partial_rotary_factor": rope_dimension / head_dim,
        "rope_parameters": rope_parameters,
        "tie_word_embeddings": "output.weight" not in tensor_names,
        "eos_token_id": metadata.get("tokenizer.ggml.eos_token_id"),
        "bos_token_id": metadata.get("tokenizer.ggml.bos_token_id"),
        "pad_token_id": metadata.get("tokenizer.ggml.padding_token_id"),
    }


GGUF_CONFIG_ARCHS = {
    "qwen3": _qwen3_config,
    "qwen3moe": _qwen3_moe_config,
    "qwen35": _qwen35_config,
    "qwen35moe": _qwen35_moe_config,
    "deepseek4": _deepseek_v4_config,
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
