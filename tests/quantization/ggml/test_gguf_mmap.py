# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import unittest

import numpy as np
import torch

from transformers.core_model_loading import _materialize_copy, _stage_releasable_source_for_accelerator
from transformers.integrations.gguf.dequant import GGML_Q8_0
from transformers.integrations.gguf.reader import LazyGgufTensor, _page_aligned_interior


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


if __name__ == "__main__":
    unittest.main()
