import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest


class GraphCoreContractTest(unittest.TestCase):
    def test_runtime_overrides_match_both_installed_serving_catalogs(self):
        source = Path(__file__).resolve().parents[1] / "runtime-source"
        lock = json.loads((source / "source-lock.json").read_bytes())["source_files"]
        prefix = "vllm-overrides/vllm/"
        overrides = {name[len(prefix):]: digest for name, digest in lock.items() if name.startswith(prefix)}
        self.assertIn("v1/worker/gpu/model_runner.py", overrides)
        for name in ("deployment/serving-catalog.json", "vllm_gguf_plugin/native/serving-catalog.json"):
            content = (source / name).read_bytes()
            self.assertEqual(hashlib.sha256(content).hexdigest(), lock[name])
            for cc, target in json.loads(content)["targets"].items():
                with self.subTest(catalog=name, capability=cc):
                    actual = target["core_contract"]["files"]
                    self.assertEqual({file: actual.get(file) for file in overrides}, overrides)


    def test_assembly_keeps_existing_runtime_source_stamps_independent(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks/long_context"))
        try:
            from runtime import assemble
        finally:
            sys.path.pop(0)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "release"
            (base / "compatible-vllm/core").mkdir(parents=True)
            stamp_name = "benchmark-source.sha256"
            (base / stamp_name).write_text("existing release stamp\n")
            source = root / "source"
            source.mkdir()
            lock = source / "source-lock.json"

            first = root / "first"
            lock.write_text('{"source_files": {}}\n')
            first_stamp = hashlib.sha256(lock.read_bytes()).hexdigest() + "\n"
            assemble(first, release=base, source=source)
            self.assertEqual((base / stamp_name).read_text(), "existing release stamp\n")

            second = root / "second"
            lock.write_text('{"source_files": {}, "revision": "second"}\n')
            second_stamp = hashlib.sha256(lock.read_bytes()).hexdigest() + "\n"
            assemble(second, release=first, source=source)
            self.assertEqual((first / stamp_name).read_text(), first_stamp)
            self.assertEqual((second / stamp_name).read_text(), second_stamp)
