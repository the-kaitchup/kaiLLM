"""CPU checks for the actual relocation failure, without importing Torch."""

import ast
from functools import lru_cache
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from kaillm import native_assets


ROOT = Path(__file__).resolve().parents[1]
PAYLOAD = ROOT / "kaillm/_runtime"


class NativeAssetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.library = self.root / "asset/libexample.so"
        self.library.parent.mkdir()
        self.library.write_bytes(b"immutable qualified native bytes")
        self.source = self.root / "kernel.cu"
        self.source.write_text("qualified CUDA source\n")
        self.architecture = {"compute_capability": [12, 0], "target": "sm_120"}
        identity = {"source_sha256": native_assets.digest(self.source),
                    "contract": "sm120-example", "architecture": self.architecture,
                    "library": "/historical/build/libexample.so",
                    "library_sha256": native_assets.digest(self.library)}
        self.metadata = self.library.with_name("identity.json")
        self.metadata.write_text(json.dumps(identity, indent=2) + "\n")
        self.entry = {**identity, "stem": "example", "library": "asset/libexample.so",
                      "metadata": "asset/identity.json",
                      "metadata_sha256": native_assets.digest(self.metadata)}
        self.manifest = self.root / "native-assets.json"
        self.write_manifest([self.entry])

    def tearDown(self):
        native_assets.catalog.cache_clear()
        self.temporary.cleanup()

    def write_manifest(self, entries):
        self.manifest.write_text(json.dumps({"schema": "kaillm-frozen-native-assets-v1", "assets": entries}))
        native_assets.catalog.cache_clear()

    def resolve(self, **options):
        return native_assets.resolve(self.manifest, self.source, options.get("contract", "sm120-example"),
                                     "example", options.get("architecture", self.architecture))

    def test_relocation_preserves_raw_identity(self):
        before = self.metadata.read_bytes()
        result = self.resolve()
        self.assertEqual(result["library"], str(self.library))
        self.assertEqual(self.metadata.read_bytes(), before)
        self.assertEqual(json.loads(before)["library"], "/historical/build/libexample.so")

    def test_wrong_architecture_or_contract_uses_original_builder(self):
        self.assertIsNone(self.resolve(architecture={"compute_capability": [8, 6]}))
        self.assertIsNone(self.resolve(contract="other-contract"))

    def test_changed_native_bytes_rejected(self):
        self.library.write_bytes(b"different binary")
        with self.assertRaisesRegex(ValueError, "library bytes differ"):
            self.resolve()

    def test_changed_raw_metadata_rejected(self):
        self.metadata.write_text(self.metadata.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "metadata bytes differ"):
            self.resolve()

    def test_ambiguous_asset_rejected(self):
        self.write_manifest([self.entry, self.entry])
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            self.resolve()

    def test_path_escape_rejected(self):
        self.write_manifest([{**self.entry, "library": "../outside.so"}])
        with self.assertRaisesRegex(ValueError, "leaves its bundle"):
            self.resolve()

    @unittest.skipUnless((PAYLOAD / "assembly.json").exists(), "Assemble the frozen payload first")
    def test_actual_legacy_guards_pass_from_relocated_cache_without_nvcc(self):
        manifest = PAYLOAD / "runtime-native-cache/native-assets.json"
        catalog = json.loads(manifest.read_text())["assets"]
        guards = []
        for path in (PAYLOAD / "vllm_gguf_plugin/quantization").glob("*resource.py"):
            if "def legacy_metadata(" not in path.read_text():
                continue
            spec = importlib.util.spec_from_file_location("native_guard_under_test", path)
            guard = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(guard)
            guards.append(guard)
        self.assertGreaterEqual(len(guards), 2)
        for guard in guards:
            with self.subTest(guard=guard.__file__):
                entries = [entry for entry in catalog
                           if entry["library_sha256"] == guard.LEGACY_LIBRARY_SHA]
                self.assertEqual(len(entries), 1, "Every strict legacy guard needs its frozen asset")
                self.check_actual_guard(manifest, entries[0], guard)

    def check_actual_guard(self, manifest, entry, guard):
        sources = list((PAYLOAD / "vllm_gguf_plugin/native/kernels").glob("*.cu"))
        source = next(p for p in sources if native_assets.digest(p) == entry["source_sha256"])
        loader = PAYLOAD / "vllm_gguf_plugin/quantization/yaqa_cuda.py"
        tree = ast.parse(loader.read_text())
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == "build_cuda_library")
        namespace = {"lru_cache": lru_cache, "architecture_identity": lambda _: entry["architecture"],
                     "os": os, "shutil": shutil, "Path": Path}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(loader), "exec"), namespace)
        raw = manifest.parent / entry["metadata"]
        before = raw.read_bytes()
        with patch.dict(os.environ, {"KAILLM_NATIVE_ASSETS": str(manifest)}), \
                patch.object(shutil, "which", side_effect=AssertionError("Prebuilt cache must bypass nvcc")):
            identity = namespace["build_cuda_library"](
                str(source), entry["contract"], entry["stem"], (12, 0))
        self.assertEqual(guard.legacy_metadata(identity["library"])[str(raw)], entry["metadata_sha256"])
        self.assertEqual(raw.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
