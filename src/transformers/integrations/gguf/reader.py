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
"""Reading a GGUF file: architecture, tensor types, and tensors keyed by their GGUF names."""

import mmap
import os
import struct
import threading
import warnings
from collections.abc import Container
from math import prod
from typing import Literal, NamedTuple, overload

import numpy as np
import torch

from .dequant import GGML_BLOCK, GGML_NAME
from .gguf_quantized_parameter import GgufQuantizedParameter


# The ggml type IDs for tensors stored as individual scalar values rather than quantized blocks,
# together with their corresponding torch dtypes.
# Integer tensors are semantic model state (for example DeepSeek V4's tid2eid routing table), not weights.
_GGML_F32, _GGML_F16 = 0, 1
_GGML_I8, _GGML_I16, _GGML_I32, _GGML_I64, _GGML_F64, _GGML_BF16 = 24, 25, 26, 27, 28, 30
_TORCH_DTYPE = {
    _GGML_F32: torch.float32,
    _GGML_F16: torch.float16,
    _GGML_I8: torch.int8,
    _GGML_I16: torch.int16,
    _GGML_I32: torch.int32,
    _GGML_I64: torch.int64,
    _GGML_F64: torch.float64,
    _GGML_BF16: torch.bfloat16,
}

_GGUF_VERSIONS = (2, 3)  # v1 counted tensors in 32 bits; no file in the wild still uses it

# the fixed-size metadata value types: u8/i8/bool, u16/i16, u32/i32/f32, u64/i64/f64. The two that are
# not fixed-size are 8 (string) and 9 (array), handled in `_read_value`.
_KV_FORMAT = {0: "<B", 1: "<b", 7: "<?", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 10: "<Q", 11: "<q", 12: "<d"}
_KV_WIDTH = {0: 1, 1: 1, 7: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 10: 8, 11: 8, 12: 8}


class TensorInfo(NamedTuple):
    """Where one tensor lives in the file, and how it is stored."""

    name: str
    shape: tuple[int, ...]  # in torch order
    ggml_type: int
    offset: int  # from the start of the data section
    nbytes: int


class GgufHeader(NamedTuple):
    """A GGUF file's metadata: everything the loading path needs except the tensor data."""

    path: str
    architecture: str
    tensors: tuple[TensorInfo, ...]
    data_start: int  # where the tensor data begins, after the aligned metadata

    @property
    def ggml_types(self) -> dict[str, int]:
        """`{gguf_name: ggml_type}`."""
        return {info.name: info.ggml_type for info in self.tensors}

    @property
    def has_quantized_weights(self) -> bool:
        """Whether any tensor is stored as GGUF blocks rather than as plain floats."""
        return any(ggml_type in GGML_BLOCK for ggml_type in self.ggml_types.values())

    @property
    def dtype(self) -> "torch.dtype | None":
        """The float type this file was written in, or `None` if it holds quantized blocks."""
        types = set(self.ggml_types.values())
        if not types <= set(_TORCH_DTYPE):  # blocks, not values, somewhere in the file
            return None
        # a file's F32 tensors sit alongside its half ones — the norms — so the half type is the model's
        if _GGML_BF16 in types:
            return torch.bfloat16
        if _GGML_F16 in types:
            return torch.float16
        return torch.float32

    @classmethod
    def from_file(cls, gguf_path: str) -> "GgufHeader":
        """Parse a file's metadata and tensor table, without touching its tensor data."""
        blob = _mapped(gguf_path)
        metadata, tensor_count, pos = _read_metadata(blob, gguf_path)
        architecture = metadata["general.architecture"]
        alignment = metadata.get("general.alignment", 32)  # llama.cpp's default

        entries, pos = _read_tensor_table(blob, tensor_count, pos)
        names = [name for name, *_ in entries]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"GGUF tensor table contains duplicate names: {duplicates[:8]}")
        infos = tuple(
            TensorInfo(name, shape, ggml_type, offset, _byte_count(ggml_type, prod(shape), gguf_path))
            for name, shape, ggml_type, offset in entries
        )
        # the data section starts at the next alignment boundary after the tensor table
        data_start = (pos + alignment - 1) // alignment * alignment
        return cls(gguf_path, architecture, infos, data_start)


@overload
def read_gguf_metadata(
    gguf_path: str, string_arrays: "Container[str]" = (), return_tensor_shapes: Literal[False] = False
) -> tuple[dict, tuple[str, ...]]: ...


@overload
def read_gguf_metadata(
    gguf_path: str, string_arrays: "Container[str]" = (), return_tensor_shapes: Literal[True] = True
) -> tuple[dict, tuple[str, ...], dict[str, tuple[int, ...]]]: ...


def read_gguf_metadata(
    gguf_path: str, string_arrays: "Container[str]" = (), return_tensor_shapes: bool = False
) -> tuple[dict, tuple[str, ...]] | tuple[dict, tuple[str, ...], dict[str, tuple[int, ...]]]:
    """A file's metadata keys and tensor names, without reading any tensor data.

    When ``return_tensor_shapes`` is true, also return the tensor table's logical shapes keyed by GGUF name.
    """
    blob = _mapped(gguf_path)
    metadata, tensor_count, pos = _read_metadata(blob, gguf_path, string_arrays)
    entries, _ = _read_tensor_table(blob, tensor_count, pos)
    tensor_names = tuple(name for name, *_ in entries)
    if return_tensor_shapes:
        return metadata, tensor_names, {name: shape for name, shape, *_ in entries}
    return metadata, tensor_names


def _read_tensor_table(blob: np.ndarray, tensor_count: int, pos: int) -> tuple[list[tuple], int]:
    """`([(name, shape, ggml_type, offset), ...], offset just past the table)`."""
    entries = []
    for _ in range(tensor_count):
        name, pos = _read_string(blob, pos)
        (dim_count,) = struct.unpack_from("<I", blob, pos)
        pos += 4
        dims = struct.unpack_from(f"<{dim_count}Q", blob, pos)
        pos += 8 * dim_count
        ggml_type, offset = struct.unpack_from("<IQ", blob, pos)
        pos += 12
        # ggml stores dimensions fastest-moving first, torch the other way round
        entries.append((name, tuple(reversed(dims)), ggml_type, offset))
    return entries, pos


def _read_metadata(blob: np.ndarray, gguf_path: str, string_arrays: "Container[str]" = ()) -> tuple[dict, int, int]:
    """`(metadata, tensor_count, offset of the tensor table)`."""
    if bytes(blob[:4]) != b"GGUF":
        raise ValueError(f"{gguf_path} does not start with the GGUF magic bytes, so it is not a GGUF file.")
    version, tensor_count, metadata_count = struct.unpack_from("<IQQ", blob, 4)
    if version not in _GGUF_VERSIONS:
        raise ValueError(
            f"{gguf_path} is GGUF v{version}; this reader handles v{' and v'.join(map(str, _GGUF_VERSIONS))}."
        )

    pos = 24
    metadata = {}
    for _ in range(metadata_count):
        key, pos = _read_string(blob, pos)
        (value_type,) = struct.unpack_from("<I", blob, pos)
        metadata[key], pos = _read_value(blob, pos + 4, value_type, key in string_arrays)
    if "general.architecture" not in metadata:
        raise ValueError(f"{gguf_path} has no `general.architecture` in its metadata.")
    return metadata, tensor_count, pos


def _read_value(blob: np.ndarray, pos: int, value_type: int, keep_strings: bool = False):
    """One metadata value, and the offset just past it."""
    if value_type in _KV_WIDTH:
        (value,) = struct.unpack_from(_KV_FORMAT[value_type], blob, pos)
        return value, pos + _KV_WIDTH[value_type]
    if value_type == 8:  # string
        return _read_string(blob, pos)
    if value_type == 9:  # array
        element_type, count = struct.unpack_from("<IQ", blob, pos)
        pos += 12
        if element_type in _KV_WIDTH:  # small, and a config can want them: the mrope sections
            values = struct.unpack_from(f"<{count}{_KV_FORMAT[element_type][1]}", blob, pos)
            return list(values), pos + count * _KV_WIDTH[element_type]
        if element_type != 8:
            raise ValueError(f"GGUF metadata holds an array of type {element_type}, which this reader cannot read.")
        if keep_strings:  # asked for: a vocabulary or a merge table
            values = []
            for _ in range(count):
                value, pos = _read_string(blob, pos)
                values.append(value)
            return values, pos
        for _ in range(count):  # variable-length, so there is nothing to do but walk it
            (length,) = struct.unpack_from("<Q", blob, pos)
            pos += 8 + length
        return count, pos
    raise ValueError(f"GGUF metadata holds a value of type {value_type}, which this reader cannot read.")


def _mapped(gguf_path: str) -> np.ndarray:
    """The file as a `uint8` memory map. Cheap: pages are only read when a tensor is materialized."""
    return np.memmap(gguf_path, mode="r", dtype=np.uint8)


def _read_string(blob: np.ndarray, pos: int) -> tuple[str, int]:
    (length,) = struct.unpack_from("<Q", blob, pos)
    pos += 8
    return bytes(blob[pos : pos + length]).decode("utf-8"), pos + length


def _byte_count(ggml_type: int, elements: int, gguf_path: str) -> int:
    """How many bytes `elements` of this type occupy in the file."""
    if ggml_type in _TORCH_DTYPE:
        return elements * _TORCH_DTYPE[ggml_type].itemsize
    if ggml_type not in GGML_BLOCK:
        supported = ", ".join(f"{name} ({type_id})" for type_id, name in sorted(GGML_NAME.items()))
        raise ValueError(
            f"{gguf_path} holds tensors of ggml type {ggml_type}, which is not supported yet. "
            f"Supported quantized types: {supported}."
        )
    block_elements, block_bytes = GGML_BLOCK[ggml_type]
    if elements % block_elements:
        raise ValueError(
            f"{gguf_path} stores {elements} elements of ggml type {ggml_type}, which is not a whole number of blocks."
        )
    return elements // block_elements * block_bytes


class LazyGgufTensor:
    """One tensor of the file, read only when the loading pipeline asks for it."""

    def __init__(
        self,
        data: np.ndarray,
        ggml_type: int,
        shape: tuple[int, ...],
        logical_shape: tuple[int, ...] | None = None,
        releaser=None,
        offset: int = 0,
    ):
        self.data = data  # a read-only mmap view, untouched until materialized
        self.ggml_type = ggml_type
        self.shape = shape
        self.logical_shape = logical_shape or shape
        self._releaser = releaser
        self._offset = offset
        self._remaining_materializations = None
        self._release_lock = threading.Lock()

    def get_shape(self) -> list[int]:
        """Return the physical shape expected by the generic loading and sharding APIs."""
        return list(self.shape)

    def get_dtype(self) -> str:
        if self.ggml_type in _TORCH_DTYPE:
            return str(_TORCH_DTYPE[self.ggml_type]).removeprefix("torch.").upper()
        return "UINT8"

    def __getitem__(self, key) -> torch.Tensor:
        if self.ggml_type not in _TORCH_DTYPE and key is not Ellipsis:
            raise ValueError("Slicing a packed GGUF tensor is unsupported unless the complete tensor is selected")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
            raw = torch.from_numpy(np.ascontiguousarray(self.data))
        if self.ggml_type not in _TORCH_DTYPE:
            return GgufQuantizedParameter(raw.reshape(self.shape), self.ggml_type, self.logical_shape)
        # The file is mapped as bytes, since numpy has no bfloat16, so the values are reinterpreted here.
        # Left in the type the file wrote: the transforms need it -- a norm is stored as `w + 1`, and
        # rounding before the subtraction spends the precision available near 1.0 on a much smaller
        # weight. `Cast` is the last op of every chain, so this lands in the model's dtype anyway.
        values = raw.view(_TORCH_DTYPE[self.ggml_type]).reshape(self.shape)
        return values[key]

    @property
    def release_after_materialization(self):
        return self._release_after_materialization if self._releaser is not None else None

    def prepare_materializations(self, count: int):
        if count <= 0:
            raise ValueError(f"GGUF tensor materialization count must be positive, got {count}")
        with self._release_lock:
            if self._remaining_materializations is not None:
                raise RuntimeError("GGUF tensor source was prepared more than once")
            self._remaining_materializations = count

    def is_materialized_view(self, tensor):
        if tensor.device.type != "cpu" or tensor.numel() == 0:
            return False
        try:
            pointer = tensor.data_ptr()
        except RuntimeError:
            return False
        source = int(self.data.__array_interface__["data"][0])
        return source <= pointer < source + self.data.nbytes

    def _release_after_materialization(self):
        with self._release_lock:
            remaining = 1 if self._remaining_materializations is None else self._remaining_materializations
            if remaining <= 0:
                raise RuntimeError("GGUF tensor source was released more than expected")
            self._remaining_materializations = remaining - 1
            should_release = remaining == 1
        if should_release:
            self._releaser.release(self._offset, self.data.nbytes)


def _page_aligned_interior(offset: int, length: int, page_size: int):
    start = ((offset + page_size - 1) // page_size) * page_size
    end = ((offset + length) // page_size) * page_size
    return (start, end - start) if end > start else None


class _GgufFileRangeReleaser:
    def __init__(self, path, mapped_array):
        mapped_file = getattr(mapped_array, "_mmap", None)
        if mapped_file is None or not hasattr(mapped_file, "madvise") or not hasattr(mmap, "MADV_DONTNEED"):
            raise RuntimeError("GGUF mmap page release requires mmap.madvise(MADV_DONTNEED) support")
        self._mapped_file = mapped_file
        self._page_size = mmap.PAGESIZE
        self._lock = threading.Lock()
        self._fd = (
            os.open(os.fspath(path), os.O_RDONLY)
            if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED")
            else None
        )

    def release(self, offset: int, length: int):
        aligned = _page_aligned_interior(offset, length, self._page_size)
        if aligned is None:
            return
        start, size = aligned
        with self._lock:
            self._mapped_file.madvise(mmap.MADV_DONTNEED, start, size)
            if self._fd is not None:
                os.posix_fadvise(self._fd, start, size, os.POSIX_FADV_DONTNEED)

    def close(self):
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def __del__(self):
        try:
            self.close()
        except (AttributeError, OSError):
            pass


def load_gguf_state_dict(header: GgufHeader, mmap_policy: str = "keep") -> dict[str, LazyGgufTensor]:
    """`{gguf_name: LazyGgufTensor}` — the file's tensors, none of them read yet."""
    if mmap_policy not in {"keep", "release"}:
        raise ValueError(f"GGUF mmap policy must be 'keep' or 'release', got {mmap_policy!r}")
    blob = _mapped(header.path)
    releaser = _GgufFileRangeReleaser(header.path, blob) if mmap_policy == "release" else None

    state_dict = {}
    for info in header.tensors:
        logical_shape = info.shape
        shape = logical_shape
        if info.ggml_type not in _TORCH_DTYPE:  # blocks, not values: as many bytes per row as it takes
            block_elements, block_bytes = GGML_BLOCK[info.ggml_type]
            if not shape or shape[-1] % block_elements:
                raise ValueError(f"GGUF tensor {info.name!r} does not have a whole number of quantization blocks")
            shape = (*shape[:-1], shape[-1] // block_elements * block_bytes)
        start = header.data_start + info.offset
        state_dict[info.name] = LazyGgufTensor(
            blob[start : start + info.nbytes], info.ggml_type, shape, logical_shape, releaser, start
        )

    return state_dict
