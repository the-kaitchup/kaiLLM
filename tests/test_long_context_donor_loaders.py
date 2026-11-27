"""A new, verified loader reopens the correct engine's cells and old allocation limits."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks/long_context"))
from benchmark import MANIFEST, full_graph_runtime, kai_loader_runtime
from coverage_matrix import loader_exclusion
from report import confirmed_limit, current_limit_build


class DonorLoaderTest(unittest.TestCase):
    def test_hy4_plain_capacity_uses_the_registered_graph_source(self):
        for name in ("hy4-iq2m", "hy4-dondon"):
            graph_pin = full_graph_runtime(name, "plain")
            self.assertIsNotNone(graph_pin)
            self.assertIsNone(full_graph_runtime(name, "dfly5"))
            historical = kai_loader_runtime(MANIFEST["models"][name], 2)
            for count in (1, 2):
                stamp = dict(state="failed", gpu_names=["GPU"] * count, checkpoint=name,
                             engine="kaiLLM", mode="plain", batch=1, reason="CUDA out of memory")
                self.assertIsNone(confirmed_limit({**stamp, "kai_source": historical[0]}))
                limit = confirmed_limit({**stamp, "kai_source": graph_pin[0]})
                self.assertEqual(limit["kai_source"], graph_pin[0])

    def test_fused_daisy_donor_reopens_for_full_graph_runtime(self):
        self.assertIsNone(loader_exclusion("daisy-iq2m", "kaiLLM"))
        pin = full_graph_runtime("daisy-iq2m", "plain")
        stamp = dict(state="failed", gpu_names=["GPU"], checkpoint="daisy-iq2m",
                     engine="kaiLLM", mode="plain", batch=1, reason="CUDA out of memory")
        self.assertIsNone(confirmed_limit({**stamp, "kai_source": "obsolete"}))
        self.assertEqual(confirmed_limit({**stamp, "kai_source": pin[0]})["kai_source"], pin[0])

    def test_old_load_failure_does_not_block_qualified_runtime(self):
        model = MANIFEST["models"]["inkling-udq2"]
        kai = MANIFEST["kaillm"]
        variant = "test_loader"
        current = {"checkpoint": "inkling-udq2", "engine": "kaiLLM", "kai_source": "new", "model_revision": model["revision"]}
        with patch.dict(model, kai_loader_runtime=variant), patch.dict(kai, test_loader_revision="new"), \
                patch.dict(kai["runtime_locks"], {variant: {"source": "digest"}}):
            self.assertEqual(kai_loader_runtime(model), ("new", "digest"))
            self.assertIsNone(loader_exclusion("inkling-udq2", "kaiLLM"))
            self.assertIsNone(loader_exclusion("inkling-udq2", "llama.cpp"))
            self.assertTrue(current_limit_build(current))
            self.assertFalse(current_limit_build({**current, "kai_source": "old"}))

    def test_pipeline_pin_only_changes_multi_gpu_hy4(self):
        model = MANIFEST["models"]["hy4-iq2m"]
        kai = MANIFEST["kaillm"]
        with patch.dict(model, kai_pipeline_runtime="hy4_pipeline"), \
             patch.dict(kai, hy4_pipeline_revision="new"), \
             patch.dict(kai["runtime_locks"], hy4_pipeline={"source": "digest"}):
            self.assertIsNone(kai_loader_runtime(model, 1))
            for count in (2, 4):
                self.assertEqual(kai_loader_runtime(model, count), ("new", "digest"))
                row = {"checkpoint": "hy4-iq2m", "engine": "kaiLLM", "gpu": ["GPU"] * count}
                self.assertFalse(current_limit_build({**row, "kai_source": "old"}))
                self.assertTrue(current_limit_build({**row, "kai_source": "new"}))
