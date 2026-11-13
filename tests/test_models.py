# SPDX-License-Identifier: Apache-2.0
import tempfile
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

from kaillm.models import resolve_model


class ModelResolutionTests(unittest.TestCase):
    def test_explicit_hub_reference_downloads_only_requested_file(self):
        module = types.ModuleType("huggingface_hub")
        module.hf_hub_download = Mock(return_value="/cache/checkpoint.gguf")
        module.list_repo_files = Mock(side_effect=AssertionError("No repository scan needed"))
        with patch.dict(sys.modules, {"huggingface_hub": module}):
            result = resolve_model("example/models:folder/checkpoint.gguf", revision="pinned")
        self.assertEqual(result, "/cache/checkpoint.gguf")
        module.hf_hub_download.assert_called_once_with(
            repo_id="example/models", filename="folder/checkpoint.gguf", revision="pinned")

    def test_dry_run_does_not_import_or_download_hub(self):
        with patch.dict(sys.modules, {"huggingface_hub": None}):
            self.assertEqual(resolve_model("example/models:model.gguf", dry_run=True),
                             "example/models:model.gguf")

    def test_ambiguous_repository_requires_filename(self):
        module = types.ModuleType("huggingface_hub")
        module.hf_hub_download = Mock(side_effect=AssertionError("Must not choose a model"))
        module.list_repo_files = Mock(return_value=["small.gguf", "large.gguf"])
        with patch.dict(sys.modules, {"huggingface_hub": module}):
            with self.assertRaisesRegex(ValueError, "explicit GGUF"):
                resolve_model("example/models")

    def test_local_file_and_invalid_parent_path(self):
        with tempfile.TemporaryDirectory() as directory:
            local = Path(directory) / "model.gguf"
            local.write_bytes(b"fixture")
            self.assertEqual(resolve_model(str(local)), str(local))
        for reference in ("example/models:../model.gguf", "example/models:/model.gguf"):
            with self.assertRaises(ValueError):
                resolve_model(reference, dry_run=True)


if __name__ == "__main__":
    unittest.main()
