<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# GGUF

[GGUF](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) is a file format used to store models for inference with [GGML](https://github.com/ggerganov/ggml), a fast and lightweight inference framework written in C and C++. GGUF is a single-file format containing the model metadata and tensors.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/gguf-spec.png"/>
</div>

The GGUF format also supports many quantized data types (refer to [quantization type table](https://hf.co/docs/hub/en/gguf#quantization-types) for a complete list of supported quantization types) which saves a significant amount of memory, making inference with large models like Whisper and Llama feasible on local and edge devices.

Transformers supports two GGUF loading modes. Qwen3, Qwen3-MoE, text-only Qwen3.5, Qwen4-Exp, and DeepSeek V4 checkpoints keep quantized payloads as frozen model parameters and dequantize weights during `forward`. Other registered architectures use the compatibility path, which dequantizes weights while loading. Both modes use the normal Transformers quantizer lifecycle and weight-conversion pipeline.

> [!TIP]
> Architectures wired up for GGUF loading include Llama, Mistral, Phi3, Cohere, Qwen2, Qwen3, Deci, StableLM, Starcoder2, Nemotron, Gemma2, Gemma3 (text + multimodal), Gemma4, Bloom, GPT2, Mamba, LFM2, Falcon, Qwen2-MoE, Qwen3-MoE, Qwen3.5 text, Qwen3.5-MoE text, Qwen4-Exp, DeepSeek V4, MiniMax-M2, GPT-OSS, T5, UMT5. Qwen3, Qwen3-MoE, the Qwen3.5 and Qwen4-Exp text architectures, and DeepSeek V4 currently have persistent quantized weights. The authoritative registry is `_GGUF_ARCH_CONVERTERS` in [`modeling_gguf_pytorch_utils.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_gguf_pytorch_utils.py).

Add the `gguf_file` parameter to [`~PreTrainedModel.from_pretrained`] to specify the GGUF file to load.

```py
# pip install gguf accelerate
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen3-0.6B-GGUF"
filename = "Qwen3-0.6B-Q4_K_M.gguf"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    gguf_file=filename,
    dtype=torch.bfloat16,
    device_map="auto",
)
# Loading tokenizer files from the corresponding Transformers repository is
# a reliable fallback when a particular GGUF tokenizer cannot be converted.
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
```

For large checkpoints on systems where file-backed pages and accelerator allocations compete for the same physical memory, set `gguf_mmap_policy="release"`. GGUF tensors remain lazy, and each tensor's complete mmap pages are released with operating-system advice after the loader has copied its final consumer to independent storage. Accelerator loads first stage release-enabled tensors through anonymous CPU storage, preventing accelerator drivers from pinning the source mapping. Unaligned boundary pages are retained because adjacent tensors can share them. The default policy is `"keep"`, which preserves normal mmap caching. Release mode requires `mmap.madvise(MADV_DONTNEED)` support and can reread evicted pages if another process or a later load needs them.

Persistent Qwen models dequantize linear weights only for the active operation. Embeddings dequantize only requested token rows. A tied language-model head shares the compressed embedding payload rather than duplicating it, and dequantizes its complete logical weight for projection like other GGUF linear modules. When autograd needs a packed linear's input gradient, backward re-dequantizes that weight instead of retaining the forward's dense weight. Qwen3.5 hybrid models preserve their physical GGUF value-head layout inside packed GatedDeltaNet projections while exposing the canonical Transformers layout at projection and cache boundaries.

Persistent Qwen MoE models keep each layer's pre-stacked gate, up, and down expert tensors compressed separately. Their GGUF-aware `eager`, `grouped_mm`, and `batched_mm` implementations select routed expert IDs before dequantization. `grouped_mm` and `batched_mm` reuse the standard Transformers expert route planning and reduction while a GGUF storage provider selects and dequantizes only the required packed projections. `eager` processes one active expert at a time, `grouped_mm` dequantizes each active expert once per projection, and `batched_mm` expands those weights to routed token-expert pairs. When an expert projection needs an input gradient, autograd saves the packed payload and backend-specific route metadata, then reselects and re-dequantizes only those active experts during backward. On accelerator devices, generation may automatically switch configured `grouped_mm` experts to the GGUF-aware `batched_mm` implementation for decoding. Dense-only expert implementations and Hub MoE kernels cannot consume GGUF payload bytes and are rejected; compatible kernels for surrounding modules can still be used.

Persistent GGUF linear, embedding, and expert modules use the load-time `dtype` as their shared compute policy while keeping packed storage in `uint8`. Linear and expert outputs preserve the activation input dtype; embedding outputs use the configured compute dtype. Ordinary floating-point fallback weights retain native PyTorch layer dtype behavior. GGUF linear, embedding, and expert modules are initialized by the checkpoint loader and reject forward calls while they still contain raw placeholder weights. Packed embeddings support `padding_idx`, but reject `max_norm`, non-default `norm_type`, `scale_grad_by_freq=True`, and `sparse=True` because packed base weights are immutable and frozen.

Disk offload, base-weight training, TP/DTensor sharding, expert parallelism, and saving persistent GGUF parameters with `save_pretrained` are not supported yet. Load persistent GGUF models with the desired `dtype`; casting the complete model to another dtype after loading is not supported. Device-only moves remain supported, and tied compressed parameters keep shared storage across those moves. PEFT-style adapters can still train because activation and adapter gradients are preserved. Packed dense linears and expert projections save compressed payload references rather than logical floating-point weights for input-gradient computation and re-dequantize during backward. On ROCm, persistent GGUF currently selects eager attention because SDPA is unstable for this execution path. Qwen3.5 and Qwen4-Exp support text-only models; vision projectors, NextN/MTP blocks, and GGUF tokenizer conversion are not supported.

Architectures on the compatibility path are ordinary dequantized models after loading and can be trained or saved normally. Convert a saved Transformers model back to GGUF with [convert-hf-to-gguf.py](https://github.com/ggerganov/llama.cpp/blob/master/convert_hf_to_gguf.py).

## Adding GGUF support for a new architecture

GGUF loading plugs into the same weight-conversion framework that the rest of `transformers` uses for safetensors / Fp8 / etc. To support a new architecture you only need to:

1. **Pick the HF `model_type`.** The loader looks up the rename table by the HF `model_type` (e.g. `"llama"`, `"qwen2_moe"`, `"gemma3_text"`). This is the same string `config.model_type` produces.
2. **Write a list of `WeightRenaming` / `WeightConverter` rules** in [`src/transformers/modeling_gguf_pytorch_utils.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_gguf_pytorch_utils.py) and register it in `_GGUF_ARCH_CONVERTERS`.
3. **Add the `model_type` to `EXPECTED_MODEL_TYPES`** in [`tests/quantization/ggml/test_gguf_arch_coverage.py`](https://github.com/huggingface/transformers/blob/main/tests/quantization/ggml/test_gguf_arch_coverage.py) so the fast regression test enforces it.

### Rule types

The conventions are the same as in `convert_mistral4_weight_to_hf.py` and `Fp8Quantizer`:

```python
from transformers.core_model_loading import WeightConverter, WeightRenaming

# Pure key rename (cheap — no op chain). Substring `re.sub` style: escape dots,
# use alternation when the same target template covers several source names.
WeightRenaming(r"^blk\.", "model.layers.")
WeightRenaming(r"\.ffn_(gate|up|down)\.weight", r".mlp.\1_proj.weight")

# With an actual tensor transform (transpose, permute, concat, etc.).
WeightConverter(
    source_patterns=r"\.attn_q\.weight",
    target_patterns=".self_attn.q_proj.weight",
    operations=[ReversePermuteAttnQ()],
)
```

`WeightRenaming`s are applied **sequentially** (each one operates on the previous rule's output), so structural prefix renames (`^blk\.` → `model.layers.`, etc.) should come first and per-tensor renames after. `WeightConverter`s are evaluated after all renames; the first one whose source pattern matches is selected.

The GGUF quantizer adapts these rules to the selected runtime. The compatibility path prepends `GGUFDequantize` to converter chains. Persistent Qwen paths instead keep compressed tensors and attach only the metadata required by each architecture rule, so architecture rules should never add `GGUFDequantize` directly. Persistent MoE paths rewrite their gate/up many-to-one converter into two direct renames because compressed tensors with independent quantization metadata cannot be concatenated. Qwen2 and Qwen3 use NeoX-style split-half RoPE and retain the original Hugging Face Q/K layout in GGUF; unlike Llama, their Q/K weights must not be permuted while loading.

### Existing transform ops

These live in [`src/transformers/gguf_conversion_ops.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/gguf_conversion_ops.py) and are reusable:

| Op | Used for |
|----|----------|
| `ReversePermuteAttnQ`, `ReversePermuteAttnK` | Architectures such as Llama whose GGUF conversion permutes Q/K weights |
| `SubtractOne` | Gemma / Nemotron norm weights (stored as `weight + 1` in GGUF) |
| `Unsqueeze(dim)` | Mamba conv1d, LFM2 shortconv — add a singleton dim |
| `LogNegate` | Mamba SSM-A: `log(-x)` on load |
| `BloomReshapeQKVWeight`, `BloomReshapeQKVBias` | Bloom interleaved QKV |
| `Concatenate(dim)` (from `core_model_loading`) | MoE gate+up merge into `gate_up_proj` |
| `Transpose` (from `core_model_loading`) | GPT-2 transposed `c_attn` / `c_proj` / `c_fc` |

If your architecture needs a new tensor transform, add it to `gguf_conversion_ops.py` as a `ConversionOps` subclass with `convert(input_dict, source_patterns, target_patterns, **kwargs)` and a `reverse_op` property.

### Worked example: adding a new Llama-like architecture

For a model whose HF `model_type` is `"my_llama"` and which uses the standard rope-permuted Q/K, plain Llama norms, and `mlp.gate_proj` / `up_proj` / `down_proj`, the rule list is already provided as `_LLAMA_CONVERTERS`. You only need:

```python
# src/transformers/modeling_gguf_pytorch_utils.py
_GGUF_ARCH_CONVERTERS = {
    ...
    "my_llama": _LLAMA_CONVERTERS,
}
```

For a variant with `q_norm` / `k_norm` and attn biases (StableLM style) you compose:

```python
_MY_LLAMA_CONVERTERS = _LLAMA_CONVERTERS + [
    WeightRenaming(r"\.attn_(q|k)_norm\.weight", r".self_attn.\1_norm.weight"),
    WeightRenaming(r"\.attn_(q|k|v)\.bias", r".self_attn.\1_proj.bias"),
]
```

For a wholly new layout, start from one of the existing entries (`_BLOOM_CONVERTERS`, `_GPT2_CONVERTERS`, `_T5_CONVERTERS`, …) and adapt the renames + add any required `WeightConverter` ops.

### Dequantization performance

Byte-level dequantization is exposed through [`integrations/gguf_dequant.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/gguf_dequant.py), with kernels adapted from llama.cpp in `integrations/gguf_dequant_kernels.py`. The kernels run on CPU, CUDA/ROCm, and MPS through PyTorch operations. Supported formats include F32, F16, BF16, Q4_0/Q4_1, Q5_0/Q5_1, Q8_0, Q2_K through Q6_K, IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS, TQ1_0, TQ2_0, MXFP4, and NVFP4.

Persistent Qwen memory is the compressed payload plus ordinary unquantized tensors. Dense temporary workspace includes one complete active linear projection, including the full dequantized language-model head, or the selected embedding rows. A grad-enabled packed linear releases its dense weight after forward and materializes it again only while computing its input gradient; autograd retains the packed payload reference rather than the logical floating-point matrix. Qwen3-MoE workspace scales with routed experts: one expert at a time for eager, the unique active set for grouped, and routed token-expert pairs for batched. Grad-enabled expert projections also release those selected floating weights after forward and reconstruct the same active selection transiently for input-gradient computation. Allocator warmup and automatic device mapping use each checkpoint tensor's physical byte count rather than its logical placeholder size. The PyTorch allocator may retain reserved memory after a forward, so compare `memory_allocated` after loading and peak allocated deltas rather than reserved-memory totals.

If you add a quantization type, add its block kernel and dispatch entry in `gguf_dequant_kernels.py`, then compare representative blocks against gguf-py in the download-free GGUF tests.
