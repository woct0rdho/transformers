# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import os
import tempfile
import unittest

import numpy as np
import torch

from transformers.core_model_loading import _materialize_copy, _stage_releasable_source_for_accelerator
from transformers.integrations.gguf.dequant import GGML_Q8_0
from transformers.integrations.gguf.reader import (
    GgufHeader,
    LazyGgufTensor,
    TensorInfo,
    _GgufFileReader,
    _page_aligned_interior,
    load_gguf_state_dict,
)


class GgufMmapTests(unittest.TestCase):
    def test_source_release_waits_for_all_consumers(self):
        class Releaser:
            def __init__(self):
                self.calls = 0

            def release(self, offset, length):
                self.calls += 1

        releaser = Releaser()
        source = LazyGgufTensor(np.zeros(34, dtype=np.uint8), GGML_Q8_0, (1, 34), (1, 32), releaser=releaser)
        source.prepare_materializations(2)
        source.release_after_materialization()
        self.assertEqual(releaser.calls, 0)
        source.release_after_materialization()
        self.assertEqual(releaser.calls, 1)

    def test_page_release_excludes_unowned_boundaries(self):
        self.assertIsNone(_page_aligned_interior(0, 100, 4096))
        self.assertEqual(_page_aligned_interior(100, 9000, 4096), (4096, 4096))

    def test_failed_materialization_does_not_release_source(self):
        class FailingSource:
            def __init__(self):
                self.releases = 0

            def __getitem__(self, _):
                raise RuntimeError("synthetic materialization failure")

            def release_after_materialization(self):
                self.releases += 1

        source = FailingSource()
        with self.assertRaisesRegex(RuntimeError, "synthetic materialization failure"):
            _materialize_copy(source)
        self.assertEqual(source.releases, 0)

    def test_accelerator_staging_detaches_a_releasable_view(self):
        class Source:
            def release_after_materialization(self):
                pass

        source = Source()
        tensor = torch.ones(3)
        staged = _stage_releasable_source_for_accelerator(source, tensor, torch.device("cuda"))
        self.assertEqual(staged.device.type, "cpu")
        self.assertIsNot(staged, tensor)
        self.assertTrue(torch.equal(staged, tensor))

    def test_accelerator_staging_leaves_a_source_that_owns_its_bytes_alone(self):
        class Source:
            def release_after_materialization(self):
                pass

            def is_materialized_view(self, tensor):
                return False

        tensor = torch.ones(3)
        staged = _stage_releasable_source_for_accelerator(Source(), tensor, torch.device("cuda"))
        self.assertIs(staged, tensor)

    def test_pread_source_reads_its_range_and_shares_no_mapped_page(self):
        values = np.arange(8, dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "tiny.gguf")
            with open(path, "wb") as handle:
                handle.write(values.tobytes())
            mapped = LazyGgufTensor(np.memmap(path, mode="r", dtype=np.uint8), 0, (8,))
            source = load_gguf_state_dict(
                GgufHeader(path, "synthetic", (TensorInfo("blk.0.weight", (8,), 0, 0, values.nbytes),), 0),
                mmap_policy="pread",
            )["blk.0.weight"]

            self.assertTrue(torch.equal(source[...], mapped[...]))
            self.assertTrue(torch.equal(source[...], torch.from_numpy(values)))
            self.assertFalse(source.is_materialized_view(source[...]))
            with self.assertRaisesRegex(ValueError, "mmap policy"):
                load_gguf_state_dict(GgufHeader(path, "synthetic", (), 0), mmap_policy="invalid")

    def test_pread_reader_rejects_a_file_that_ends_early(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "short.bin")
            with open(path, "wb") as handle:
                handle.write(b"\x00" * 4)
            reader = _GgufFileReader(path)
            with self.assertRaisesRegex(OSError, "ended after"):
                reader.read(0, 16)


if __name__ == "__main__":
    unittest.main()
