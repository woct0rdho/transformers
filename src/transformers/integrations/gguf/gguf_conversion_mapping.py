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
"""GGUF -> transformers weight conversion mappings, one entry per architecture.

`WeightRenaming`s map GGUF names into the transformers namespace and chain; `WeightConverter`s then
undo llama.cpp's value/layout transforms, matching on the renamed key, at most one per key.
"""

import torch

from ...core_model_loading import (
    ConversionOps,
    WeightConverter,
    WeightRenaming,
    WeightTransform,
    build_glob_alternation,
)
from .dequant import dequantize
from .gguf_quantized_parameter import GgufQuantizedParameter


# Shared skeleton for decoder-only models. Norms are absent: whether llama.cpp offsets them by one is
# per-architecture, so each arch declares its own.
DENSE_DECODER_RENAMINGS = [
    WeightRenaming(r"^token_embd\.", "model.embed_tokens."),
    WeightRenaming(r"^output\.", "lm_head."),
    WeightRenaming(r"^blk\.", "model.layers."),
    WeightRenaming(r"\.attn_q\.", ".self_attn.q_proj."),
    WeightRenaming(r"\.attn_k\.", ".self_attn.k_proj."),
    WeightRenaming(r"\.attn_v\.", ".self_attn.v_proj."),
    WeightRenaming(r"\.attn_output\.", ".self_attn.o_proj."),
    WeightRenaming(r"\.ffn_gate\.", ".mlp.gate_proj."),
    WeightRenaming(r"\.ffn_up\.", ".mlp.up_proj."),
    WeightRenaming(r"\.ffn_down\.", ".mlp.down_proj."),
]


def _qwen35(config) -> list[WeightTransform]:
    """Qwen3.5: hybrid GatedDeltaNet linear attention + full attention every fourth layer.

    llama.cpp's converter differs in five ways, each undone below:

    1. Names: `blk.N.*` with ggml leaf names.
    2. Zero-centred norms stored as `w + 1`, except `ssm_norm` (`SubtractOne`, in fp32).
    3. `ssm_a` holds `-exp(A_log)` (`LogNegate`).
    4. `conv1d` is squeezed to 2D (`Unsqueeze`).
    5. Value heads are tiled rather than grouped, on every v-indexed tensor: `PermuteRows` where the
       tensor produces the value dimension, `PermuteInputFeatures` for the one that consumes it.

    Only (5)'s column permute crosses quantization blocks, so `out_proj` is the one tensor that has
    to be dequantized; everything else stays packed or is F32 in the file.
    """
    text_config = config.get_text_config()
    num_key_heads = text_config.linear_num_key_heads
    value_heads_per_key_head = text_config.linear_num_value_heads // num_key_heads
    value_head_dim = text_config.linear_value_head_dim

    # `in_proj_qkv` and `conv1d` are fused q/k/v; only the value block is reordered.
    query_key_rows = 2 * text_config.linear_key_head_dim * num_key_heads

    # The same reorder over the flattened value dimension, and over head indices alone.
    heads = (num_key_heads, value_heads_per_key_head)
    per_value = TiledToGroupedRows(*heads, value_head_dim)
    per_head = TiledToGroupedRows(*heads)

    # llama.cpp writes the multi-token-prediction block as `blk.{num_hidden_layers}.*`. Transformers
    # has no MTP module and drops `^mtp.*` by name, so give it that prefix before the blanket `blk.`
    # rule turns it into a layer the model does not have.
    mtp_block = [WeightRenaming(rf"^blk\.{text_config.num_hidden_layers}\.", "mtp.")]

    renamings = (
        mtp_block
        + DENSE_DECODER_RENAMINGS
        + [
            WeightRenaming(r"\.attn_qkv\.", ".linear_attn.in_proj_qkv."),
            WeightRenaming(r"\.attn_gate\.", ".linear_attn.in_proj_z."),
            WeightRenaming(r"\.ssm_alpha\.", ".linear_attn.in_proj_a."),
            WeightRenaming(r"\.ssm_beta\.", ".linear_attn.in_proj_b."),
            WeightRenaming(r"\.ssm_conv1d\.", ".linear_attn.conv1d."),
            WeightRenaming(r"\.ssm_norm\.", ".linear_attn.norm."),  # the one norm with no offset
            WeightRenaming(r"\.ssm_out\.", ".linear_attn.out_proj."),
            WeightRenaming(r"\.ssm_a$", ".linear_attn.A_log"),
            WeightRenaming(r"\.ssm_dt\.bias$", ".linear_attn.dt_bias"),
        ]
    )

    # (2) norms stored as `w + 1`: rename the leaf and un-offset in one place.
    offset_norms = [
        WeightConverter(
            source_patterns=r"^output_norm\.weight",
            target_patterns="model.norm.weight",
            operations=[SubtractOne()],
        ),
        WeightConverter(
            source_patterns=r"\.attn_norm\.weight",
            target_patterns=".input_layernorm.weight",
            operations=[SubtractOne()],
        ),
        WeightConverter(
            source_patterns=r"\.post_attention_norm\.weight",
            target_patterns=".post_attention_layernorm.weight",
            operations=[SubtractOne()],
        ),
        WeightConverter(
            source_patterns=r"\.attn_q_norm\.weight",
            target_patterns=".self_attn.q_norm.weight",
            operations=[SubtractOne()],
        ),
        WeightConverter(
            source_patterns=r"\.attn_k_norm\.weight",
            target_patterns=".self_attn.k_norm.weight",
            operations=[SubtractOne()],
        ),
    ]

    # (3) and (4): value-scoped scalars, and the conv1d reshape.
    per_value_head = [
        WeightConverter(
            source_patterns="linear_attn.A_log",
            target_patterns="linear_attn.A_log",
            operations=[LogNegate(), per_head],
        ),
        WeightConverter(
            source_patterns="linear_attn.dt_bias",
            target_patterns="linear_attn.dt_bias",
            operations=[per_head],
        ),
        WeightConverter(
            source_patterns="linear_attn.in_proj_a.weight",
            target_patterns="linear_attn.in_proj_a.weight",
            operations=[per_head],
        ),
        WeightConverter(
            source_patterns="linear_attn.in_proj_b.weight",
            target_patterns="linear_attn.in_proj_b.weight",
            operations=[per_head],
        ),
    ]

    # (5) the value-head reorder on tensors with a row or column per value element.
    value_reorder = [
        WeightConverter(
            source_patterns="linear_attn.in_proj_z.weight",
            target_patterns="linear_attn.in_proj_z.weight",
            operations=[per_value],
        ),
        WeightConverter(
            source_patterns="linear_attn.in_proj_qkv.weight",
            target_patterns="linear_attn.in_proj_qkv.weight",
            operations=[TiledToGroupedRows(*heads, value_head_dim, offset=query_key_rows)],
        ),
        WeightConverter(
            source_patterns="linear_attn.conv1d.weight",
            target_patterns="linear_attn.conv1d.weight",
            operations=[TiledToGroupedRows(*heads, value_head_dim, offset=query_key_rows), Unsqueeze(1)],
        ),
        # The only tensor that *consumes* the value dimension, so the reorder is on its columns. Columns
        # cross quantization blocks, so when it stays packed the reorder is applied to its input instead.
        WeightConverter(
            source_patterns="linear_attn.out_proj.weight",
            target_patterns="linear_attn.out_proj.weight",
            operations=[TiledToGroupedInputs(*heads, value_head_dim)],
        ),
    ]

    return renamings + offset_norms + per_value_head + value_reorder


def _standard_decoder(config, moe=False):
    """Mapping shared by Qwen3 and Qwen3-MoE."""
    renamings = list(DENSE_DECODER_RENAMINGS)
    renamings.extend(
        [
            WeightRenaming(r"^output_norm\.weight", "model.norm.weight"),
            WeightRenaming(r"\.attn_norm\.", ".input_layernorm."),
            WeightRenaming(r"\.attn_(q|k)_norm\.", r".self_attn.\1_norm."),
            WeightRenaming(r"\.ffn_norm\.", ".post_attention_layernorm."),
            WeightRenaming(r"\.attn_(q|k|v)\.bias", r".self_attn.\1_proj.bias"),
        ]
    )
    if moe:
        renamings.extend(
            [
                WeightRenaming(r"\.ffn_gate_inp\.", ".mlp.gate."),
                WeightRenaming(r"\.ffn_down_exps\.weight", ".mlp.experts.down_proj"),
                WeightConverter(
                    source_patterns=r"\.ffn_gate_inp_shexp\.weight",
                    target_patterns=".mlp.shared_expert_gate.weight",
                    operations=[Unsqueeze(0)],
                ),
                WeightRenaming(r"\.ffn_(gate|up|down)_shexp\.weight", r".mlp.shared_expert.\1_proj.weight"),
            ]
        )
        renamings.append(
            WeightConverter(
                source_patterns=[r"\.ffn_gate_exps\.weight", r"\.ffn_up_exps\.weight"],
                target_patterns=".mlp.experts.gate_up_proj",
                operations=[Concatenate(dim=1)],
            )
        )
    return renamings


class Concatenate(ConversionOps):
    """Concatenate the tensors collected from a one-to-many source conversion."""

    def __init__(self, dim: int):
        self.dim = dim

    @torch.no_grad
    def convert(self, input_dict, source_patterns, target_patterns, **kwargs):
        tensors = []
        for source_pattern in source_patterns:
            values = input_dict.get(source_pattern, ())
            tensors.extend(values if isinstance(values, list) else (values,))
        if any(isinstance(tensor, GgufQuantizedParameter) for tensor in tensors):
            raise ValueError("Packed GGUF tensors cannot be concatenated")
        return {target_patterns[0]: torch.cat(tensors, dim=self.dim)}


def _qwen35_moe(config):
    return _qwen35(config) + [
        WeightRenaming(r"\.ffn_gate_inp\.", ".mlp.gate."),
        WeightRenaming(r"\.ffn_down_exps\.weight", ".mlp.experts.down_proj"),
        WeightConverter(
            source_patterns=r"\.ffn_gate_inp_shexp\.weight",
            target_patterns=".mlp.shared_expert_gate.weight",
            operations=[Unsqueeze(0)],
        ),
        WeightRenaming(r"\.ffn_(gate|up|down)_shexp\.weight", r".mlp.shared_expert.\1_proj.weight"),
        WeightConverter(
            source_patterns=[r"\.ffn_gate_exps\.weight", r"\.ffn_up_exps\.weight"],
            target_patterns=".mlp.experts.gate_up_proj",
            operations=[Concatenate(dim=1)],
        ),
    ]


def _deepseek_v4(config):
    return [
        WeightRenaming(r"^blk\.", "model.layers."),
        WeightRenaming(r"^token_embd\.weight", "model.embed_tokens.weight"),
        WeightRenaming(r"^output_norm\.weight", "model.norm.weight"),
        WeightRenaming(r"^output\.weight", "lm_head.weight"),
        WeightRenaming(r"^output_hc_fn\.weight", "model.hc_head.hc_fn"),
        WeightRenaming(r"^output_hc_base\.weight", "model.hc_head.hc_base"),
        WeightRenaming(r"^output_hc_scale\.weight", "model.hc_head.hc_scale"),
        WeightRenaming(r"\.attn_norm\.weight", ".input_layernorm.weight"),
        WeightRenaming(r"\.ffn_norm\.weight", ".post_attention_layernorm.weight"),
        WeightRenaming(r"\.hc_attn_(fn|base|scale)\.weight", r".attn_hc.\1"),
        WeightRenaming(r"\.hc_ffn_(fn|base|scale)\.weight", r".ffn_hc.\1"),
        WeightRenaming(r"\.attn_q_a\.weight", ".self_attn.q_a_proj.weight"),
        WeightRenaming(r"\.attn_q_a_norm\.weight", ".self_attn.q_a_norm.weight"),
        WeightRenaming(r"\.indexer\.attn_q_b\.weight", ".self_attn.compressor.indexer.q_b_proj.weight"),
        WeightRenaming(r"(?<!indexer)\.attn_q_b\.weight", ".self_attn.q_b_proj.weight"),
        WeightRenaming(r"\.attn_kv\.weight", ".self_attn.kv_proj.weight"),
        WeightRenaming(r"\.attn_kv_a_norm\.weight", ".self_attn.kv_norm.weight"),
        WeightRenaming(r"\.attn_output_a\.weight", ".self_attn.o_a_proj.weight"),
        WeightRenaming(r"\.attn_output_b\.weight", ".self_attn.o_b_proj.weight"),
        WeightRenaming(r"\.attn_sinks\.weight", ".self_attn.sinks"),
        WeightRenaming(r"\.ffn_gate_inp\.weight", ".mlp.gate.weight"),
        WeightRenaming(r"\.ffn_gate_tid2eid\.weight", ".mlp.gate.tid2eid"),
        WeightRenaming(r"\.exp_probs_b\.bias", ".mlp.gate.e_score_correction_bias"),
        WeightRenaming(r"\.ffn_(gate|up|down)_shexp\.weight", r".mlp.shared_experts.\1_proj.weight"),
        WeightRenaming(r"\.ffn_down_exps\.weight", ".mlp.experts.down_proj"),
        WeightConverter(
            source_patterns=[r"\.ffn_gate_exps\.weight", r"\.ffn_up_exps\.weight"],
            target_patterns=".mlp.experts.gate_up_proj",
            operations=[Concatenate(dim=1)],
        ),
        WeightRenaming(r"\.attn_compressor_ape\.weight", ".self_attn.compressor.position_bias"),
        WeightRenaming(r"\.attn_compressor_(kv|gate)\.weight", r".self_attn.compressor.\1_proj.weight"),
        WeightRenaming(r"\.attn_compressor_norm\.weight", ".self_attn.compressor.kv_norm.weight"),
        WeightRenaming(r"\.indexer\.proj\.weight", ".self_attn.compressor.indexer.scorer.weights_proj.weight"),
        WeightRenaming(r"\.indexer_compressor_ape\.weight", ".self_attn.compressor.indexer.position_bias"),
        WeightRenaming(r"\.indexer_compressor_(kv|gate)\.weight", r".self_attn.compressor.indexer.\1_proj.weight"),
        WeightRenaming(r"\.indexer_compressor_norm\.weight", ".self_attn.compressor.indexer.kv_norm.weight"),
    ]


def _qwen4_exp(config):
    text_config = config.get_text_config()
    ple_layer_ids = getattr(text_config, "ple_layer_ids", []) or []
    if len(ple_layer_ids) > 1:
        raise ValueError("Qwen4-Exp GGUF loading supports only one PLE layer")
    ple_mapping = []
    if ple_layer_ids:
        ple_mapping.append(
            WeightRenaming(
                r"^per_layer_token_embd\.weight$",
                f"model.layers.{int(ple_layer_ids[0]) - 1}.ple.ple_embedding.ngram_embedding.weight",
            )
        )
    return (
        _qwen35_moe(config)
        + ple_mapping
        + [
            WeightConverter(
                source_patterns=r"^output_hc_norm\.weight",
                target_patterns="model.hyper_connection_mixer.hc_norm.weight",
                operations=[SubtractOne()],
            ),
            WeightRenaming(r"^output_hc_down\.weight", "model.hyper_connection_mixer.input_mix_weight_down.weight"),
            WeightRenaming(r"^output_hc_up\.weight", "model.hyper_connection_mixer.input_mix_weight_up.weight"),
            WeightConverter(
                source_patterns=r"\.hc_attn_norm\.weight",
                target_patterns=".attn_hyper_connection.hc_norm.weight",
                operations=[SubtractOne()],
            ),
            WeightRenaming(r"\.hc_attn_down\.weight", ".attn_hyper_connection.input_mix_weight_down.weight"),
            WeightRenaming(r"\.hc_attn_up\.weight", ".attn_hyper_connection.input_mix_weight_up.weight"),
            WeightRenaming(r"\.hc_attn_inject\.weight", ".attn_hyper_connection.block_inject_weight.weight"),
            WeightConverter(
                source_patterns=r"\.hc_ffn_norm\.weight",
                target_patterns=".mlp_hyper_connection.hc_norm.weight",
                operations=[SubtractOne()],
            ),
            WeightRenaming(r"\.hc_ffn_down\.weight", ".mlp_hyper_connection.input_mix_weight_down.weight"),
            WeightRenaming(r"\.hc_ffn_up\.weight", ".mlp_hyper_connection.input_mix_weight_up.weight"),
            WeightRenaming(r"\.hc_ffn_inject\.weight", ".mlp_hyper_connection.block_inject_weight.weight"),
            WeightConverter(
                source_patterns=[r"\.indexer\.q_proj\.weight", r"\.indexer\.k_proj\.weight"],
                target_patterns=".self_attn.indexer.index_qk_proj.weight",
                operations=[Concatenate(dim=0)],
            ),
            WeightConverter(
                source_patterns=r"\.indexer\.q_norm\.weight",
                target_patterns=".self_attn.indexer.q_layernorm.weight",
                operations=[SubtractOne()],
            ),
            WeightConverter(
                source_patterns=r"\.indexer\.k_norm\.weight",
                target_patterns=".self_attn.indexer.k_layernorm.weight",
                operations=[SubtractOne()],
            ),
            WeightRenaming(r"\.ple_key\.weight", ".ple.key_proj.weight"),
            WeightRenaming(r"\.ple_value\.weight", ".ple.value_proj.weight"),
            WeightConverter(r"\.ple_norm_key\.weight", ".ple.norm_key.weight", [SubtractOne()]),
            WeightConverter(r"\.ple_norm_query\.weight", ".ple.norm_query.weight", [SubtractOne()]),
            WeightConverter(r"\.ple_norm_conv\.weight", ".ple.norm_conv.weight", [SubtractOne()]),
            WeightConverter(r"\.ple_conv1d\.weight", ".ple.conv1d.weight", [Unsqueeze(1)]),
        ]
    )


# gguf `general.architecture` -> builder taking the model config
GGUF_ARCHS = {
    "qwen3": lambda config: _standard_decoder(config),
    "qwen3moe": lambda config: _standard_decoder(config, moe=True),
    "qwen35": _qwen35,
    "qwen35moe": _qwen35_moe,
    "qwen4exp": _qwen4_exp,
    "deepseek4": _deepseek_v4,
}


class SubtractOne(ConversionOps):
    """Undo llama.cpp storing zero-centred RMSNorm weights as `w + 1`."""

    def __init__(self, offset: float = 1.0):
        self.offset = offset

    @torch.no_grad
    def convert(
        self, input_dict: dict[str, torch.Tensor], source_patterns: list[str], target_patterns: list[str], **kwargs
    ) -> dict[str, torch.Tensor]:
        tensor = _single_tensor(input_dict)
        return {target_patterns[0]: (tensor.float() - self.offset).to(tensor.dtype)}


class LogNegate(ConversionOps):
    """`A_log = log(-a)`, undoing llama.cpp storing `ssm_a = -exp(A_log)`."""

    @torch.no_grad
    def convert(
        self, input_dict: dict[str, torch.Tensor], source_patterns: list[str], target_patterns: list[str], **kwargs
    ) -> dict[str, torch.Tensor]:
        tensor = _single_tensor(input_dict)
        return {target_patterns[0]: torch.log(-tensor.float()).to(tensor.dtype)}


class Unsqueeze(ConversionOps):
    """Add a size-1 dim, undoing llama.cpp squeezing dimensions of size one."""

    def __init__(self, dim: int):
        self.dim = dim

    @torch.no_grad
    def convert(
        self, input_dict: dict[str, torch.Tensor], source_patterns: list[str], target_patterns: list[str], **kwargs
    ) -> dict[str, torch.Tensor]:
        tensor = _single_tensor(input_dict)
        return {target_patterns[0]: tensor.unsqueeze(self.dim)}


class PermuteRows(ConversionOps):
    """Reorder rows (dim 0), optionally only those from `offset` onwards.

    `offset` covers tensors whose leading rows must stay put, e.g. Qwen3.5's fused `in_proj_qkv`.
    Safe on packed blocks: a block never spans two rows.
    """

    supports_packed = True

    def __init__(self, permutation: torch.Tensor, offset: int = 0):
        self.permutation = permutation
        self.offset = offset

    @torch.no_grad
    def convert(
        self, input_dict: dict[str, torch.Tensor], source_patterns: list[str], target_patterns: list[str], **kwargs
    ) -> dict[str, torch.Tensor]:
        tensor = _single_tensor(input_dict)
        packed = isinstance(tensor, GgufQuantizedParameter)
        payload = tensor.as_subclass(torch.Tensor) if packed else tensor
        perm = self.permutation.to(payload.device)
        if self.offset:
            head, tail = payload[: self.offset], payload[self.offset :]
            payload = torch.cat([head, tail[perm]], dim=0)
        else:
            payload = payload[perm]
        if packed:
            tensor = GgufQuantizedParameter(payload.contiguous(), tensor.quant_type, tensor.logical_shape)
        else:
            tensor = payload.contiguous()
        return {target_patterns[0]: tensor}


class PermuteInputFeatures(ConversionOps):
    """Reorder columns (dim 1), for a tensor that *consumes* an axis llama.cpp reordered.

    Columns cross quantization blocks, so permuting packed bytes would mean requantizing. It is not
    needed on the weight at all, since `x @ W[:, p].T == x[:, argsort(p)] @ W.T`: a packed weight is
    left as stored and `GgufLinear` gathers its input through `input_permutation` instead. A dense
    tensor still has its columns permuted here.
    """

    supports_packed = True

    def __init__(self, permutation: torch.Tensor):
        self.permutation = permutation

    @property
    def input_permutation(self) -> torch.Tensor:
        """Reordering for the input of the module holding this weight, when it stays packed.

        The inverse of the column permutation, not the permutation itself.
        """
        return torch.argsort(self.permutation)

    @torch.no_grad
    def convert(
        self, input_dict: dict[str, torch.Tensor], source_patterns: list[str], target_patterns: list[str], **kwargs
    ) -> dict[str, torch.Tensor]:
        tensor = _single_tensor(input_dict)
        if isinstance(tensor, GgufQuantizedParameter) or tensor.dtype == torch.uint8:
            # Packed: the reorder rides on the input instead, so the blocks pass through untouched.
            return {target_patterns[0]: tensor}
        perm = self.permutation.to(tensor.device)
        return {target_patterns[0]: tensor[:, perm].contiguous()}


class TiledToGroupedRows(PermuteRows):
    """`PermuteRows` undoing llama.cpp's head reorder, for a tensor that *produces* the head axis.

    llama.cpp stores head-indexed axes tiled (`v0k0 v0k1 ... v1k0 ...`), transformers groups them by
    key head (`k0v0 k0v1 k1v0 ...`). `head_dim` defaults to 1, for per-head vectors like `A_log`.
    """

    def __init__(self, num_k_heads: int, heads_per_k: int, head_dim: int = 1, offset: int = 0):
        total = num_k_heads * heads_per_k * head_dim
        # On the CPU explicitly: a mapping may be built under a meta default device, and a meta index
        # tensor silently permutes nothing.
        indices = torch.arange(total, device="cpu")
        tiled_from_grouped = indices.reshape(num_k_heads, heads_per_k, head_dim).transpose(0, 1).reshape(-1)
        permutation = torch.argsort(tiled_from_grouped)
        super().__init__(permutation, offset=offset)


class TiledToGroupedInputs(PermuteInputFeatures):
    """`PermuteInputFeatures` undoing the same reorder, for a tensor that *consumes* the head axis.

    The permutation is the one `TiledToGroupedRows` builds, applied to columns instead of rows.
    """

    def __init__(self, num_k_heads: int, heads_per_k: int, head_dim: int = 1):
        total = num_k_heads * heads_per_k * head_dim
        # On the CPU explicitly: a mapping may be built under a meta default device, and a meta index
        # tensor silently permutes nothing.
        indices = torch.arange(total, device="cpu")
        tiled_from_grouped = indices.reshape(num_k_heads, heads_per_k, head_dim).transpose(0, 1).reshape(-1)
        permutation = torch.argsort(tiled_from_grouped)
        super().__init__(permutation)


class Cast(ConversionOps):
    """Cast to the model's dtype, after every other transform has run.

    Last, not first: llama.cpp stores values this path does arithmetic on (`w + 1` norms, `-exp(A_log)`),
    and rounding those to bf16 before the arithmetic would spend the precision near 1.0. Blocks pass
    through untouched. The loader skips this itself for a pre-quantized checkpoint under renamed keys.

    `keep_fp32` carries the name patterns a model declares FP32-strict through
    `PreTrainedModel._keep_in_fp32_modules[ _strict]`. Those tensors are cast to FP32 instead of the load
    dtype, because a value rounded to the load dtype here cannot be recovered by whatever consumes it
    later; the safetensors path already keeps them in FP32 through `PreTrainedModel._get_dtype_plan`, and
    the two loaders have to agree on resident dtypes for the same model. Patterns are matched like the
    dtype plan is (regex search, `*` acting as a wildcard).

    A tensor that already carries the load dtype passes through before the pattern test: it either is a
    floating tensor the file stores at that width, or one `Dequantize` produced, which has already spent
    the extra precision. Forcing FP32 storage on it would change dtypes without restoring anything.

    Integer and boolean tensors are not a floating computation and have no load dtype, so they keep the
    dtype the file stores (`tid2eid` routing tables stay integer state).
    """

    def __init__(self, dtype: "torch.dtype", keep_fp32: tuple[str, ...] = ()):
        self.dtype = dtype
        self.keep_fp32 = tuple(keep_fp32)
        self.keep_fp32_alternation = build_glob_alternation(list(self.keep_fp32))[0] if self.keep_fp32 else None

    @torch.no_grad
    def convert(
        self,
        input_dict: dict[str, torch.Tensor],
        source_patterns: list[str],
        target_patterns: list[str],
        full_layer_name: str | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        tensor = _single_tensor(input_dict)
        name = full_layer_name if full_layer_name is not None else target_patterns[0]
        if tensor.dtype in (torch.uint8, self.dtype):
            return {name: tensor}
        if not tensor.is_floating_point():
            return {name: tensor}
        if self.keep_fp32_alternation is not None and self.keep_fp32_alternation.search(name):
            return {name: tensor.to(torch.float32)}
        return {name: tensor.to(self.dtype)}


class Dequantize(ConversionOps):
    """Unpack GGUF blocks into values, on the parameter's own device.

    First in its chain, since every later transform is defined on dense values. One instance serves the
    whole file: llama.cpp mixes quantization types, so the type is looked up per parameter, and a
    parameter that is not listed keeps its blocks.
    """

    def __init__(self, ggml_types: dict[str, int], dtype: "torch.dtype"):
        self.ggml_types = ggml_types
        self.dtype = dtype

    @torch.no_grad
    def convert(
        self,
        input_dict: dict[str, torch.Tensor],
        source_patterns: list[str],
        target_patterns: list[str],
        full_layer_name: str | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        if len(input_dict) > 1:
            converted = {}
            for source_pattern, values in input_dict.items():
                blocks = _single_tensor({source_pattern: values})
                ggml_type = getattr(blocks, "quant_type", self.ggml_types.get(full_layer_name))
                if ggml_type is None:
                    converted[source_pattern] = blocks
                elif isinstance(blocks, GgufQuantizedParameter):
                    converted[source_pattern] = blocks.dequantize(dtype=self.dtype)
                else:
                    converted[source_pattern] = dequantize(blocks, ggml_type, dtype=self.dtype)
            return converted
        ggml_type = self.ggml_types.get(full_layer_name)
        if ggml_type is None:
            return input_dict
        blocks = _single_tensor(input_dict)
        values = (
            blocks.dequantize(dtype=self.dtype)
            if isinstance(blocks, GgufQuantizedParameter)
            else dequantize(blocks, ggml_type, dtype=self.dtype)
        )
        return {full_layer_name: values}


def _single_tensor(input_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    """These ops are all one-to-one; unwrap the single (possibly listed) tensor."""
    if len(input_dict) != 1:
        raise ValueError(f"expected a single source tensor, got {list(input_dict)}")
    tensors = next(iter(input_dict.values()))
    return tensors[0] if isinstance(tensors, list) else tensors
