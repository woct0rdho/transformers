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
"""Dequantizing GGUF blocks with torch ops."""

from typing import NamedTuple

import numpy as np
import torch
from gguf.quants import (
    IQ1_M,
    IQ1_S,
    IQ2_S,
    IQ2_XS,
    IQ2_XXS,
    IQ3_S,
    IQ3_XXS,
    IQ4_NL,
    MXFP4,
    NVFP4,
)


QK_K = 256

# ggml type ids, as numbered by `enum ggml_type` in ggml.h
GGML_Q4_0, GGML_Q4_1, GGML_Q5_0, GGML_Q5_1 = 2, 3, 6, 7
GGML_Q8_0 = 8
GGML_Q2_K, GGML_Q3_K, GGML_Q4_K, GGML_Q5_K, GGML_Q6_K = 10, 11, 12, 13, 14
GGML_IQ2_XXS, GGML_IQ2_XS, GGML_IQ3_XXS, GGML_IQ1_S, GGML_IQ4_NL = 16, 17, 18, 19, 20
GGML_IQ3_S, GGML_IQ2_S, GGML_IQ4_XS = 21, 22, 23
GGML_IQ1_M = 29
GGML_TQ1_0, GGML_TQ2_0 = 34, 35
GGML_MXFP4, GGML_NVFP4 = 39, 40

# ggml type id -> (elements per block, bytes per block)
GGML_BLOCK = {
    GGML_Q4_0: (32, 18),
    GGML_Q4_1: (32, 20),
    GGML_Q5_0: (32, 22),
    GGML_Q5_1: (32, 24),
    GGML_Q8_0: (32, 34),
    GGML_Q2_K: (256, 84),
    GGML_Q3_K: (256, 110),
    GGML_Q4_K: (256, 144),
    GGML_Q5_K: (256, 176),
    GGML_Q6_K: (256, 210),
    GGML_IQ2_XXS: (256, 66),
    GGML_IQ2_XS: (256, 74),
    GGML_IQ3_XXS: (256, 98),
    GGML_IQ1_S: (256, 50),
    GGML_IQ4_NL: (32, 18),
    GGML_IQ3_S: (256, 110),
    GGML_IQ2_S: (256, 82),
    GGML_IQ4_XS: (256, 136),
    GGML_IQ1_M: (256, 56),
    GGML_TQ1_0: (256, 54),
    GGML_TQ2_0: (256, 66),
    GGML_MXFP4: (32, 17),
    GGML_NVFP4: (64, 36),
}

# ggml type id -> its name, for messages
GGML_NAME = {
    GGML_Q4_0: "Q4_0",
    GGML_Q4_1: "Q4_1",
    GGML_Q5_0: "Q5_0",
    GGML_Q5_1: "Q5_1",
    GGML_Q8_0: "Q8_0",
    GGML_Q2_K: "Q2_K",
    GGML_Q3_K: "Q3_K",
    GGML_Q4_K: "Q4_K",
    GGML_Q5_K: "Q5_K",
    GGML_Q6_K: "Q6_K",
    GGML_IQ2_XXS: "IQ2_XXS",
    GGML_IQ2_XS: "IQ2_XS",
    GGML_IQ3_XXS: "IQ3_XXS",
    GGML_IQ1_S: "IQ1_S",
    GGML_IQ4_NL: "IQ4_NL",
    GGML_IQ3_S: "IQ3_S",
    GGML_IQ2_S: "IQ2_S",
    GGML_IQ4_XS: "IQ4_XS",
    GGML_IQ1_M: "IQ1_M",
    GGML_TQ1_0: "TQ1_0",
    GGML_TQ2_0: "TQ2_0",
    GGML_MXFP4: "MXFP4",
    GGML_NVFP4: "NVFP4",
}


# Workaround for a bug with Tensor.view(dtype) in torch < 2.12, see https://github.com/pytorch/pytorch/issues/172747
def _uint16_from_bytes(x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.int32)
    return x[..., 0] | (x[..., 1] << 8)


def _uint32_from_bytes(x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.int32)
    return x[..., 0] | (x[..., 1] << 8) | (x[..., 2] << 16) | (x[..., 3] << 24)


def _fp16_from_bytes(x: torch.Tensor) -> torch.Tensor:
    return _uint16_from_bytes(x).to(torch.int16).view(torch.float16).unsqueeze(-1).to(torch.float32)


def _split_block_dims(blocks: torch.Tensor, *dims: int) -> tuple[torch.Tensor, ...]:
    return torch.split(blocks, [*dims, blocks.shape[1] - sum(dims)], dim=1)


def _bits_to_signs(bits: torch.Tensor, device: torch.device) -> torch.Tensor:
    return torch.where(
        bits == 0,
        torch.ones((), dtype=torch.float32, device=device),
        -torch.ones((), dtype=torch.float32, device=device),
    )


def _grid_tensor(cls) -> torch.Tensor:
    cls.init_grid()
    return torch.from_numpy(cls.grid.squeeze().copy())


class _GridConstants(NamedTuple):
    grid_iq1_s: torch.Tensor
    grid_iq2_s: torch.Tensor
    grid_iq2_xxs: torch.Tensor
    grid_iq2_xs: torch.Tensor
    grid_iq3_s: torch.Tensor
    grid_iq3_xxs: torch.Tensor
    kvalues_iq4_nl: torch.Tensor
    kvalues_mxfp4: torch.Tensor
    kvalues_nvfp4: torch.Tensor
    ksigns_iq2_xxs: torch.Tensor
    iq1_s_delta: float
    iq1_m_delta: float


_GRID_CONSTANTS = _GridConstants(
    grid_iq1_s=_grid_tensor(IQ1_S),
    grid_iq2_s=_grid_tensor(IQ2_S),
    grid_iq2_xxs=_grid_tensor(IQ2_XXS),
    grid_iq2_xs=_grid_tensor(IQ2_XS),
    grid_iq3_s=_grid_tensor(IQ3_S),
    grid_iq3_xxs=_grid_tensor(IQ3_XXS),
    kvalues_iq4_nl=torch.tensor(IQ4_NL.kvalues, dtype=torch.float32),
    kvalues_mxfp4=torch.tensor(MXFP4.kvalues, dtype=torch.float32),
    kvalues_nvfp4=torch.tensor(NVFP4.kvalues, dtype=torch.float32),
    ksigns_iq2_xxs=torch.from_numpy(np.frombuffer(IQ2_XXS.ksigns, dtype=np.uint8).copy()),
    iq1_s_delta=float(IQ1_S.delta),
    iq1_m_delta=float(IQ1_M.delta),
)


def dequantize(data: torch.Tensor, ggml_type: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Flat `uint8` GGUF bytes -> flat values of `dtype`."""
    if ggml_type not in GGML_BLOCK:
        supported = ", ".join(f"{name} ({type_id})" for type_id, name in sorted(GGML_NAME.items()))
        raise ValueError(f"ggml type {ggml_type} is not supported yet. Supported quantized types: {supported}.")
    block_elems, block_bytes = GGML_BLOCK[ggml_type]
    blocks = data.reshape(-1, block_bytes)
    values = _DEQUANT[ggml_type](blocks, dtype)
    return values.reshape(-1)[: blocks.shape[0] * block_elems]


def _half(blocks: torch.Tensor, start: int) -> torch.Tensor:
    """Read one fp16 scalar per block, as (nb, 1) float32."""
    return blocks[:, start : start + 2].contiguous().view(torch.float16).float()


def _k_scales(scales: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Unpack the 12 bytes of 6-bit scales/mins shared by Q4_K and Q5_K (ggml's get_scale_min_k4)."""
    q = scales.int()
    scale = torch.cat([q[:, :4] & 63, (q[:, 8:12] & 0xF) | ((q[:, 0:4] >> 6) << 4)], dim=1)
    minimum = torch.cat([q[:, 4:8] & 63, (q[:, 8:12] >> 4) | ((q[:, 4:8] >> 6) << 4)], dim=1)
    return scale.float(), minimum.float()


def _interleave_nibbles(qs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(nb, 128) nibble bytes -> low/high nibbles as (nb, 4, 32) each, still `uint8`."""
    q = qs.reshape(-1, 4, 32)
    return q & 0xF, q >> 4


def _dequant_q8_0(blocks: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    d = _half(blocks, 0)
    qs = blocks[:, 2:34].contiguous().view(torch.int8).to(torch.float32)
    return (d * qs).to(dtype)


def _dequant_q4_k(blocks: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    d, dmin = _half(blocks, 0), _half(blocks, 2)
    scale, minimum = _k_scales(blocks[:, 4:16])
    low, high = _interleave_nibbles(blocks[:, 16:144])
    q = torch.stack([low, high], dim=2).reshape(-1, 8, 32).to(torch.float32)
    return ((d * scale)[..., None] * q - (dmin * minimum)[..., None]).to(dtype)


def _dequant_q5_k(blocks: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    d, dmin = _half(blocks, 0), _half(blocks, 2)
    scale, minimum = _k_scales(blocks[:, 4:16])
    qh = blocks[:, 16:48].unsqueeze(1)  # (nb, 1, 32), one extra bit per value
    low, high = _interleave_nibbles(blocks[:, 48:176])
    shift = torch.arange(4, device=blocks.device, dtype=torch.uint8).reshape(1, 4, 1) * 2
    low = low + ((qh >> shift) & 1) * 16
    high = high + ((qh >> (shift + 1)) & 1) * 16
    q = torch.stack([low, high], dim=2).reshape(-1, 8, 32).to(torch.float32)
    return ((d * scale)[..., None] * q - (dmin * minimum)[..., None]).to(dtype)


def _dequant_q6_k(blocks: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    d = _half(blocks, 208)
    ql, qh = blocks[:, 0:128], blocks[:, 128:192]
    scales = blocks[:, 192:208].contiguous().view(torch.int8).float()
    # 16 values share a scale; the four quarters of each 128-element half use scales is+0/2/4/6
    which = torch.arange(32, device=blocks.device) // 16
    out = []
    for half in range(2):
        lo, hi = ql[:, half * 64 : half * 64 + 32], ql[:, half * 64 + 32 : (half + 1) * 64]
        h, sc = qh[:, half * 32 : (half + 1) * 32], scales[:, half * 8 : (half + 1) * 8]
        quants = [
            (lo & 0xF) | ((h & 3) << 4),
            (hi & 0xF) | (((h >> 2) & 3) << 4),
            (lo >> 4) | (((h >> 4) & 3) << 4),
            (hi >> 4) | (((h >> 6) & 3) << 4),
        ]
        for quarter, q in enumerate(quants):
            scale = d * sc[:, which + 2 * quarter]
            out.append(scale * (q.to(torch.float32) - 32))
    return torch.cat(out, dim=1).to(dtype)


def _dequant_q4_0(blocks: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    n_blocks = blocks.shape[0]

    d, qs = _split_block_dims(blocks, 2)
    d = _fp16_from_bytes(d)

    shifts = torch.tensor([0, 4], device=blocks.device, dtype=torch.int32).view(1, 1, 2, 1)
    qs = qs.to(torch.int32).view(n_blocks, -1, 1, 16) >> shifts
    qs = (qs & 0x0F).view(n_blocks, -1).to(torch.int8) - 8

    return (d * qs.to(torch.float32)).to(dtype)


def _dequant_q4_1(blocks: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    n_blocks = blocks.shape[0]

    d, m, qs = _split_block_dims(blocks, 2, 2)
    d = _fp16_from_bytes(d)
    m = _fp16_from_bytes(m)

    shifts = torch.tensor([0, 4], device=blocks.device, dtype=torch.int32).view(1, 1, 2, 1)
    qs = qs.to(torch.int32).view(n_blocks, -1, 1, 16) >> shifts
    qs = (qs & 0x0F).view(n_blocks, -1).to(torch.float32)

    return ((d * qs) + m).to(dtype)


def _dequant_q5_0(blocks: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    n_blocks = blocks.shape[0]

    d, qh, qs = _split_block_dims(blocks, 2, 4)
    d = _fp16_from_bytes(d)
    qh = _uint32_from_bytes(qh).unsqueeze(-1)

    qh = qh >> torch.arange(32, device=blocks.device, dtype=torch.int32).view(1, 32)
    shifts = torch.tensor([0, 4], device=blocks.device, dtype=torch.int32).view(1, 1, 2, 1)
    ql = qs.to(torch.int32).view(n_blocks, -1, 1, 16) >> shifts
    qh = (qh & 1).to(torch.int32)
    ql = (ql & 0x0F).view(n_blocks, -1)

    qs = (ql | (qh << 4)).to(torch.int8) - 16
    return (d * qs.to(torch.float32)).to(dtype)


def _dequant_q5_1(blocks: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    n_blocks = blocks.shape[0]

    d, m, qh, qs = _split_block_dims(blocks, 2, 2, 4)
    d = _fp16_from_bytes(d)
    m = _fp16_from_bytes(m)
    qh = _uint32_from_bytes(qh).unsqueeze(-1)

    qh = qh >> torch.arange(32, device=blocks.device, dtype=torch.int32).view(1, 32)
    shifts = torch.tensor([0, 4], device=blocks.device, dtype=torch.int32).view(1, 1, 2, 1)
    ql = qs.to(torch.int32).view(n_blocks, -1, 1, 16) >> shifts
    qh = (qh & 1).to(torch.int32)
    ql = (ql & 0x0F).view(n_blocks, -1)

    qs = (ql | (qh << 4)).to(torch.float32)
    return ((d * qs) + m).to(dtype)


def _dequant_q2_k(blocks: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    n_blocks = blocks.shape[0]

    scales, qs, d, dmin = _split_block_dims(blocks, QK_K // 16, QK_K // 4, 2)
    d = _fp16_from_bytes(d)
    dmin = _fp16_from_bytes(dmin)

    scales = scales.to(torch.int32)
    dl = (d * (scales & 0x0F).to(torch.float32)).view(n_blocks, QK_K // 16, 1)
    ml = (dmin * (scales >> 4).to(torch.float32)).view(n_blocks, QK_K // 16, 1)

    shift = torch.tensor([0, 2, 4, 6], device=blocks.device, dtype=torch.int32).view(1, 1, 4, 1)
    qs = (qs.to(torch.int32).view(n_blocks, -1, 1, 32) >> shift) & 3
    qs = qs.view(n_blocks, QK_K // 16, 16).to(torch.float32)

    return (dl * qs - ml).view(n_blocks, -1).to(dtype)


def _dequant_q3_k(blocks: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    n_blocks = blocks.shape[0]

    hmask, qs, scales, d = _split_block_dims(blocks, QK_K // 8, QK_K // 4, 12)
    d = _fp16_from_bytes(d)

    scales = scales.to(torch.int32)
    lscales, hscales = scales[:, :8], scales[:, 8:]
    lshifts = torch.tensor([0, 4], device=blocks.device, dtype=torch.int32).view(1, 2, 1)
    hshifts = torch.tensor([0, 2, 4, 6], device=blocks.device, dtype=torch.int32).view(1, 4, 1)
    lscales = (lscales.view(n_blocks, 1, 8) >> lshifts).view(n_blocks, 16)
    hscales = (hscales.view(n_blocks, 1, 4) >> hshifts).view(n_blocks, 16)
    scales = (lscales & 0x0F) | ((hscales & 0x03) << 4)
    scales = scales.to(torch.int8).to(torch.float32) - 32

    dl = (d * scales).view(n_blocks, 16, 1)

    ql_shifts = torch.tensor([0, 2, 4, 6], device=blocks.device, dtype=torch.int32).view(1, 1, 4, 1)
    qh_shifts = torch.arange(8, device=blocks.device, dtype=torch.int32).view(1, 1, 8, 1)
    ql = qs.to(torch.int32).view(n_blocks, -1, 1, 32) >> ql_shifts
    qh = hmask.to(torch.int32).view(n_blocks, -1, 1, 32) >> qh_shifts
    ql = ql.view(n_blocks, 16, QK_K // 16) & 3
    qh = (qh.view(n_blocks, 16, QK_K // 16) & 1) ^ 1
    q = ql.to(torch.int8) - (qh << 2).to(torch.int8)

    return (dl * q.to(torch.float32)).view(n_blocks, QK_K).to(dtype)


def _dequant_tq1_0(blocks: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    n_blocks = blocks.shape[0]

    qs, qh, d = _split_block_dims(blocks, (QK_K - 4 * QK_K // 64) // 5, QK_K // 64)
    d = _fp16_from_bytes(d)

    qs0, qs1 = qs[:, :32], qs[:, 32:]
    weights5 = torch.tensor([1, 3, 9, 27, 81], device=blocks.device, dtype=torch.int32).view(1, 1, 5, 1)
    weights4 = torch.tensor([1, 3, 9, 27], device=blocks.device, dtype=torch.int32).view(1, 1, 4, 1)
    qs0 = qs0.to(torch.int32).view(n_blocks, -1, 1, 32) * weights5
    qs0 = (qs0 & 0xFF).view(n_blocks, -1)
    qs1 = qs1.to(torch.int32).view(n_blocks, -1, 1, 16) * weights5
    qs1 = (qs1 & 0xFF).view(n_blocks, -1)
    qh = qh.to(torch.int32).view(n_blocks, -1, 1, 4) * weights4
    qh = (qh & 0xFF).view(n_blocks, -1)
    qs = torch.cat([qs0, qs1, qh], dim=-1)
    qs = ((qs * 3) >> 8).to(torch.int8) - 1

    return (d * qs.to(torch.float32)).to(dtype)


def _dequant_tq2_0(blocks: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    n_blocks = blocks.shape[0]

    qs, d = _split_block_dims(blocks, QK_K // 4)
    d = _fp16_from_bytes(d)

    shifts = torch.tensor([0, 2, 4, 6], device=blocks.device, dtype=torch.int32).view(1, 1, 4, 1)
    qs = qs.to(torch.int32).view(n_blocks, -1, 1, 32) >> shifts
    qs = (qs & 0x03).view(n_blocks, -1).to(torch.int8) - 1

    return (d * qs.to(torch.float32)).to(dtype)


def _e8m0_to_fp32_half(x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.int32)
    bits = torch.where(x < 2, torch.tensor(0x00200000, device=x.device, dtype=torch.int32) << x, (x - 1) << 23)
    return bits.view(torch.float32)


def _dequant_mxfp4(blocks: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    n_blocks = blocks.shape[0]

    e, qs = _split_block_dims(blocks, 1)
    d = _e8m0_to_fp32_half(e)

    shifts = torch.tensor([0, 4], device=blocks.device, dtype=torch.int32).view(1, 2, 1)
    qs = qs.to(torch.int32).view(n_blocks, 1, 16) >> shifts
    qs = (qs & 0x0F).view(n_blocks, -1).to(torch.long)

    kvalues = _GRID_CONSTANTS.kvalues_mxfp4.to(device=blocks.device)
    qs = kvalues[qs].view(n_blocks, 32)

    return (d * qs).to(dtype)


def _ue4m3_to_fp32(x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.int32)
    exp = (x >> 3) & 0x0F
    man = (x & 0x07).to(torch.float32)
    raw = torch.where(
        exp == 0,
        man * 2.0**-9,
        (1.0 + man / 8.0) * torch.pow(torch.tensor(2.0, device=x.device), exp.to(torch.float32) - 7.0),
    )
    return torch.where((x == 0) | (x == 0x7F), torch.zeros((), device=x.device, dtype=torch.float32), raw * 0.5)


def _dequant_nvfp4(blocks: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    n_blocks = blocks.shape[0]

    d_bytes, qs = _split_block_dims(blocks, 4)
    d = _ue4m3_to_fp32(d_bytes).view(n_blocks, 4, 1)

    qs = qs.to(torch.int32).view(n_blocks, 4, 8)
    lo = qs & 0x0F
    hi = qs >> 4
    vals = torch.cat([lo, hi], dim=-1).to(torch.long)

    kvalues = _GRID_CONSTANTS.kvalues_nvfp4.to(device=blocks.device)
    vals = kvalues[vals]

    return (d * vals).view(n_blocks, 64).to(dtype)


def _dequant_iq2_xxs(blocks: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    n_blocks = blocks.shape[0]

    d, qs = _split_block_dims(blocks, 2)
    d = _fp16_from_bytes(d)

    qs_bytes = qs.view(n_blocks, 8, 2, 4)
    qs_u32 = _uint32_from_bytes(qs_bytes)
    q0 = qs_bytes[:, :, 0, :]
    q1 = qs_u32[:, :, 1]

    db = d * (0.5 + ((q1 >> 28) & 0x0F).to(torch.float32)) * 0.25
    db = db.view(n_blocks, 8, 1, 1)

    shifts = torch.tensor([0, 7, 14, 21], device=blocks.device, dtype=torch.int32).view(1, 1, 4)
    sign_indices = (q1.unsqueeze(-1) >> shifts) & 0x7F
    sign_bytes = _GRID_CONSTANTS.ksigns_iq2_xxs.to(device=blocks.device)[sign_indices.to(torch.long)]

    shifts_bits = torch.arange(8, device=blocks.device, dtype=torch.int32).view(1, 1, 1, 8)
    signs = (sign_bytes.to(torch.int32).unsqueeze(-1) >> shifts_bits) & 1
    signs = _bits_to_signs(signs, blocks.device)
    signs = signs.view(n_blocks, 8, 4, 8)

    grid = _GRID_CONSTANTS.grid_iq2_xxs.to(device=blocks.device)[q0.to(torch.long)]
    grid = grid.view(n_blocks, 8, 4, 8)

    return (db * grid * signs).view(n_blocks, -1).to(dtype)


def _dequant_iq2_xs(blocks: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    n_blocks = blocks.shape[0]

    d, qs, scales = _split_block_dims(blocks, 2, 2 * QK_K // 8)
    d = _fp16_from_bytes(d)
    qs = qs.view(n_blocks, -1, 2)
    qs = _uint16_from_bytes(qs)

    scale_shifts = torch.tensor([0, 4], device=blocks.device, dtype=torch.int32).view(1, 1, 2)
    scales = scales.to(torch.int32).view(n_blocks, -1, 1) >> scale_shifts
    scales = (scales & 0x0F).view(n_blocks, -1)
    db = d * (0.5 + scales.to(torch.float32)) * 0.25
    db = db.view(n_blocks, -1, 1, 1)

    signs = _GRID_CONSTANTS.ksigns_iq2_xxs.to(device=blocks.device)[(qs >> 9).to(torch.long)]
    shifts_bits = torch.arange(8, device=blocks.device, dtype=torch.int32).view(1, 1, 8)
    signs = (signs.to(torch.int32).view(n_blocks, -1, 1) >> shifts_bits) & 1
    signs = _bits_to_signs(signs, blocks.device)
    signs = signs.view(n_blocks, -1, 2, 8)

    grid = _GRID_CONSTANTS.grid_iq2_xs.to(device=blocks.device)[(qs & 511).to(torch.long)]
    grid = grid.view(n_blocks, -1, 2, 8)

    return (db * grid * signs).view(n_blocks, -1).to(dtype)


def _dequant_iq2_s(blocks: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    n_blocks = blocks.shape[0]

    d, qs, signs, qh, scales = _split_block_dims(blocks, 2, QK_K // 8, QK_K // 8, QK_K // 32)
    d = _fp16_from_bytes(d)

    scale_shifts = torch.tensor([0, 4], device=blocks.device, dtype=torch.int32).view(1, 1, 2)
    scales = scales.to(torch.int32).view(n_blocks, -1, 1) >> scale_shifts
    scales = (scales & 0x0F).view(n_blocks, -1)
    db = d * (0.5 + scales.to(torch.float32)) * 0.25
    db = db.view(n_blocks, -1, 1, 1)

    shifts_bits = torch.arange(8, device=blocks.device, dtype=torch.int32).view(1, 1, 8)
    signs = (signs.to(torch.int32).view(n_blocks, -1, 1) >> shifts_bits) & 1
    signs = _bits_to_signs(signs, blocks.device)
    signs = signs.view(n_blocks, -1, 2, 8)

    qh_shifts = torch.tensor([0, 2, 4, 6], device=blocks.device, dtype=torch.int32).view(1, 1, 4)
    qh = qh.to(torch.int32).view(n_blocks, -1, 1) >> qh_shifts
    qs = qs.to(torch.int32) | ((qh & 0x03) << 8).view(n_blocks, -1)

    grid = _GRID_CONSTANTS.grid_iq2_s.to(device=blocks.device)[qs.to(torch.long)]
    grid = grid.view(n_blocks, -1, 2, 8)

    return (db * grid * signs).view(n_blocks, -1).to(dtype)


def _dequant_iq3_xxs(blocks: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    n_blocks = blocks.shape[0]

    d, qs, scales = _split_block_dims(blocks, 2, QK_K // 4)
    d = _fp16_from_bytes(d)
    scales = scales.view(n_blocks, -1, 4)
    scales = _uint32_from_bytes(scales)

    db = d * (0.5 + ((scales >> 28) & 0x0F).to(torch.float32)) * 0.5
    db = db.view(n_blocks, -1, 1, 1)

    shifts = torch.tensor([0, 7, 14, 21], device=blocks.device, dtype=torch.int32).view(1, 1, 4)
    signs = (scales.view(n_blocks, -1, 1) >> shifts) & 0x7F
    signs = _GRID_CONSTANTS.ksigns_iq2_xxs.to(device=blocks.device)[signs.to(torch.long)]
    shifts_bits = torch.arange(8, device=blocks.device, dtype=torch.int32).view(1, 1, 1, 8)
    signs = (signs.to(torch.int32).view(n_blocks, -1, 4, 1) >> shifts_bits) & 1
    signs = _bits_to_signs(signs, blocks.device)
    signs = signs.view(n_blocks, -1, 4, 8)

    grid = _GRID_CONSTANTS.grid_iq3_xxs.to(device=blocks.device)[qs.to(torch.long)]
    grid = grid.view(n_blocks, -1, 4, 8)

    return (db * grid * signs).view(n_blocks, -1).to(dtype)


def _dequant_iq3_s(blocks: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    n_blocks = blocks.shape[0]

    d, qs, qh, signs, scales = _split_block_dims(blocks, 2, QK_K // 4, QK_K // 32, QK_K // 8)
    d = _fp16_from_bytes(d)

    scale_shifts = torch.tensor([0, 4], device=blocks.device, dtype=torch.int32).view(1, 1, 2)
    scales = scales.to(torch.int32).view(n_blocks, -1, 1) >> scale_shifts
    scales = (scales & 0x0F).view(n_blocks, -1)
    db = d * (1 + 2 * scales.to(torch.float32))
    db = db.view(n_blocks, -1, 1, 1)

    shifts_bits = torch.arange(8, device=blocks.device, dtype=torch.int32).view(1, 1, 8)
    signs = (signs.to(torch.int32).view(n_blocks, -1, 1) >> shifts_bits) & 1
    signs = _bits_to_signs(signs, blocks.device)
    signs = signs.view(n_blocks, -1, 4, 8)

    qh = qh.to(torch.int32).view(n_blocks, -1, 1) >> torch.arange(8, device=blocks.device, dtype=torch.int32)
    qh = (qh & 0x01).view(n_blocks, -1)
    qs = qs.to(torch.int32) | (qh << 8)

    grid = _GRID_CONSTANTS.grid_iq3_s.to(device=blocks.device)[qs.to(torch.long)]
    grid = grid.view(n_blocks, -1, 4, 8)

    return (db * grid * signs).view(n_blocks, -1).to(dtype)


def _dequant_iq1_s(blocks: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    n_blocks = blocks.shape[0]

    d, qs, qh = _split_block_dims(blocks, 2, QK_K // 8)
    d = _fp16_from_bytes(d)
    qh = qh.view(n_blocks, -1, 2)
    qh = _uint16_from_bytes(qh)

    dl = d * (2 * ((qh >> 12) & 7).to(torch.float32) + 1)
    dl = dl.view(n_blocks, -1, 1, 1)
    delta = torch.where(
        (qh & 0x8000) == 0,
        torch.full((), _GRID_CONSTANTS.iq1_s_delta, dtype=torch.float32, device=blocks.device),
        torch.full((), -_GRID_CONSTANTS.iq1_s_delta, dtype=torch.float32, device=blocks.device),
    )
    delta = delta.view(n_blocks, -1, 1, 1)

    qh_shifts = torch.tensor([0, 3, 6, 9], device=blocks.device, dtype=torch.int32).view(1, 1, 4)
    qh = qh.view(n_blocks, -1, 1) >> qh_shifts
    qs = qs.to(torch.int32) | ((qh & 7) << 8).view(n_blocks, -1)

    grid = _GRID_CONSTANTS.grid_iq1_s.to(device=blocks.device)[qs.to(torch.long)]
    grid = grid.view(n_blocks, -1, 4, 8)

    return (dl * (grid + delta)).view(n_blocks, -1).to(dtype)


def _dequant_iq1_m(blocks: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    n_blocks = blocks.shape[0]

    qs, qh, scales = _split_block_dims(blocks, QK_K // 8, QK_K // 16)
    scales = scales.view(n_blocks, -1, 2)
    scales = _uint16_from_bytes(scales)

    d_shifts = torch.tensor([12, 8, 4, 0], device=blocks.device, dtype=torch.int32).view(1, 4)
    d = (scales.view(n_blocks, 4) & 0xF000) >> d_shifts
    d = d[:, 0] | d[:, 1] | d[:, 2] | d[:, 3]
    d = d.to(torch.int16).view(torch.float16).to(torch.float32).view(n_blocks, 1)

    scale_shifts = torch.tensor([0, 3, 6, 9], device=blocks.device, dtype=torch.int32).view(1, 1, 4)
    scales = scales.view(n_blocks, -1, 1) >> scale_shifts
    scales = (scales & 0x07).view(n_blocks, -1)
    dl = d * (2 * scales.to(torch.float32) + 1)
    dl = dl.view(n_blocks, -1, 2, 1, 1)

    qh_shifts = torch.tensor([0, 4], device=blocks.device, dtype=torch.int32).view(1, 1, 2)
    qh = qh.to(torch.int32).view(n_blocks, -1, 1) >> qh_shifts
    qs = qs.to(torch.int32) | ((qh & 0x07) << 8).view(n_blocks, -1)

    delta = torch.where(
        qh & 0x08 == 0,
        torch.full((), _GRID_CONSTANTS.iq1_m_delta, dtype=torch.float32, device=blocks.device),
        torch.full((), -_GRID_CONSTANTS.iq1_m_delta, dtype=torch.float32, device=blocks.device),
    )
    delta = delta.view(n_blocks, -1, 2, 2, 1)

    grid = _GRID_CONSTANTS.grid_iq1_s.to(device=blocks.device)[qs.to(torch.long)]
    grid = grid.view(n_blocks, -1, 2, 2, 8)

    return (dl * (grid + delta)).view(n_blocks, -1).to(dtype)


def _dequant_iq4_nl(blocks: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    n_blocks = blocks.shape[0]

    d, qs = _split_block_dims(blocks, 2)
    d = _fp16_from_bytes(d)

    shifts = torch.tensor([0, 4], device=blocks.device, dtype=torch.int32).view(1, 1, 2, 1)
    qs = qs.to(torch.int32).view(n_blocks, -1, 1, 16) >> shifts
    qs = (qs & 0x0F).view(n_blocks, -1).to(torch.long)

    kvalues = _GRID_CONSTANTS.kvalues_iq4_nl.to(device=blocks.device)
    qs = kvalues[qs].view(n_blocks, -1)

    return (d * qs).to(dtype)


def _dequant_iq4_xs(blocks: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    n_blocks = blocks.shape[0]

    d, scales_h, scales_l, qs = _split_block_dims(blocks, 2, 2, QK_K // 64)
    d = _fp16_from_bytes(d)
    scales_h = _uint16_from_bytes(scales_h).unsqueeze(-1)

    scales_l_shifts = torch.tensor([0, 4], device=blocks.device, dtype=torch.int32).view(1, 1, 2)
    scales_l = scales_l.to(torch.int32).view(n_blocks, -1, 1) >> scales_l_shifts
    scales_h_shifts = torch.arange(0, 2 * (QK_K // 32), 2, device=blocks.device, dtype=torch.int32)
    scales_h = scales_h.view(n_blocks, 1, -1) >> scales_h_shifts.view(1, -1, 1)
    scales_l = scales_l.view(n_blocks, -1) & 0x0F
    scales_h = scales_h.view(n_blocks, -1) & 0x03

    scales = (scales_l | (scales_h << 4)).to(torch.int8).to(torch.float32) - 32
    dl = (d * scales).view(n_blocks, -1, 1)

    qs_shifts = torch.tensor([0, 4], device=blocks.device, dtype=torch.int32).view(1, 1, 2, 1)
    qs = qs.to(torch.int32).view(n_blocks, -1, 1, 16) >> qs_shifts
    qs = (qs.view(n_blocks, -1, 32) & 0x0F).to(torch.long)

    kvalues = _GRID_CONSTANTS.kvalues_iq4_nl.to(device=blocks.device)
    qs = kvalues[qs].view(n_blocks, -1, 32)

    return (dl * qs).view(n_blocks, -1).to(dtype)


_DEQUANT = {
    GGML_Q8_0: _dequant_q8_0,
    GGML_Q4_K: _dequant_q4_k,
    GGML_Q5_K: _dequant_q5_k,
    GGML_Q6_K: _dequant_q6_k,
    GGML_Q4_0: _dequant_q4_0,
    GGML_Q4_1: _dequant_q4_1,
    GGML_Q5_0: _dequant_q5_0,
    GGML_Q5_1: _dequant_q5_1,
    GGML_Q2_K: _dequant_q2_k,
    GGML_Q3_K: _dequant_q3_k,
    GGML_TQ1_0: _dequant_tq1_0,
    GGML_TQ2_0: _dequant_tq2_0,
    GGML_MXFP4: _dequant_mxfp4,
    GGML_NVFP4: _dequant_nvfp4,
    GGML_IQ2_XXS: _dequant_iq2_xxs,
    GGML_IQ2_XS: _dequant_iq2_xs,
    GGML_IQ2_S: _dequant_iq2_s,
    GGML_IQ3_XXS: _dequant_iq3_xxs,
    GGML_IQ3_S: _dequant_iq3_s,
    GGML_IQ1_S: _dequant_iq1_s,
    GGML_IQ1_M: _dequant_iq1_m,
    GGML_IQ4_NL: _dequant_iq4_nl,
    GGML_IQ4_XS: _dequant_iq4_xs,
}
