# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import unittest

from transformers.utils.quantization_config import GgufConfig


class GgufMmapConfigTests(unittest.TestCase):
    def test_config_validates_and_serializes_mmap_policy(self):
        config = GgufConfig(mmap_policy="release")
        self.assertEqual(config.get_loading_attributes(), {"mmap_policy": "release"})
        with self.assertRaisesRegex(ValueError, "mmap policy"):
            GgufConfig(mmap_policy="invalid")


if __name__ == "__main__":
    unittest.main()
