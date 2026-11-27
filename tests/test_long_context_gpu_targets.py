"""The replacement GPU starts empty, and historical receipts keep their physical identities."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks/long_context"))
from benchmark import BATCHES, DRAFT_TOKENS, MANIFEST, timing_domains
from coverage_matrix import active_rows, counts, device_group
from matrix import DGX_SPARK, modes


class GpuCoverageTest(unittest.TestCase):
    def test_timing_domains_follow_speculation(self):
        self.assertEqual(timing_domains("plain"), ("code",))
        for mode, limit in DRAFT_TOKENS.items():
            if limit:
                self.assertEqual(timing_domains(mode), ("prose", "code"))

    def test_replacement_and_archive(self):
        old = {"gpu": ["NVIDIA GeForce RTX 3060"]}
        new = {"gpu": ["NVIDIA GeForce RTX 5070 Ti"], "checkpoint": "lfm-qad"}
        self.assertEqual(active_rows([old, new]), [new])
        self.assertEqual(device_group(old["gpu"]), ("3060", 1))
        hardware = {(g["gpu"], g["count"]): g for g in counts([old], [])["hardware"]}
        self.assertNotIn(("3060", 1), hardware)
        self.assertEqual(hardware[("5070 Ti", 1)]["filled"], 0)
        self.assertGreater(hardware[("5070 Ti", 1)]["remaining"], 0)

    def test_5070_is_not_a_5070_ti(self):
        with self.assertRaises(ValueError):
            device_group(["NVIDIA GeForce RTX 5070"])

    def test_dondon_xs_adds_only_singleton_16_gib_cells(self):
        key = "inkling-dondon-xs"
        for engine in ("kaiLLM", "llama.cpp"):
            for gpu in ("5070 Ti", "5080"):
                self.assertEqual(modes(key, engine, 1, gpu=gpu), ["plain", "dspark5"])
                for batch in BATCHES[1:]:
                    self.assertEqual(modes(key, engine, batch, gpu=gpu), ())
            self.assertEqual(modes(key, engine, 2, gpu="3090"), ["plain", "dspark5"])
            self.assertEqual(modes(key, engine, 2, gpu="5090"), ["plain", "dspark5"])
            self.assertEqual(modes("inkling-dondon-s", engine, 1, gpu="5070 Ti"), ())
        with patch.dict(MANIFEST["models"][key], {"benchmark_additional_gpu_batches": {DGX_SPARK: [1]}}):
            previous = counts([], [])["planned"]
        self.assertEqual(counts([], [])["planned"], previous + 12)

    def test_dondon_s_dspark3_is_scoped_to_5090_batch_one_for_both_engines(self):
        key = "inkling-dondon-s"
        for engine in ("kaiLLM", "llama.cpp"):
            self.assertEqual(modes(key, engine, 1, gpu="5090"), ["plain", "dspark5", "dspark3"])
            self.assertEqual(modes(key, engine, 2, gpu="5090"), ["plain", "dspark5"])
            self.assertEqual(modes(key, engine, 1, gpu="3090"), ["plain", "dspark5"])

    def test_dgx_spark_has_a_dedicated_inkling_matrix(self):
        self.assertEqual(device_group([DGX_SPARK]), (DGX_SPARK, 1))
        with self.assertRaises(ValueError):
            device_group(["NVIDIA GB10"])
        for engine in ("kaiLLM", "llama.cpp"):
            self.assertEqual(modes("inkling-dondon-xs", engine, 1, gpu=DGX_SPARK), ["plain", "dspark5"])
            self.assertEqual(modes("inkling-dondon-xs", engine, 2, gpu=DGX_SPARK), ())
            self.assertEqual(modes("lfm-qad", engine, 1, gpu=DGX_SPARK), ())
        hardware = {(g["gpu"], g["count"]): g for g in counts([], [])["hardware"]}
        self.assertEqual(modes("inkling-chima-xs", "kaiLLM", 1, gpu=DGX_SPARK), ["plain", "dspark5"])
        self.assertEqual(modes("inkling-chima-xs", "kaiLLM", 2, gpu=DGX_SPARK), ())
        self.assertEqual(modes("inkling-chima-xs", "llama.cpp", 1, gpu=DGX_SPARK), ())
        self.assertEqual(modes("inkling-chima-s", "kaiLLM", 1, gpu=DGX_SPARK), ["plain", "dspark5", "dspark3"])
        self.assertEqual(modes("inkling-chima-s", "kaiLLM", 2, gpu=DGX_SPARK), ())
        self.assertEqual(modes("inkling-chima-s", "llama.cpp", 1, gpu=DGX_SPARK), ())
        self.assertEqual(modes("inkling-dondon-s", "llama.cpp", 1, gpu=DGX_SPARK), ["plain", "dspark5"])
        self.assertEqual(hardware[(DGX_SPARK, 1)]["remaining"], 20)

    def test_chimachima_xs_and_s_enroll_verified_slots_on_5080(self):
        for key in ("inkling-chima-xs", "inkling-chima-s"):
            self.assertEqual(modes(key, "kaiLLM", 1, gpu="5080"), ["plain", "dspark5"])
            self.assertEqual(modes(key, "llama.cpp", 1, gpu="5080"), ())
            for batch in BATCHES[1:]:
                expected = ["plain"] if batch == 2 else ()
                self.assertEqual(modes(key, "kaiLLM", batch, gpu="5080"), expected)
                self.assertEqual(modes(key, "llama.cpp", batch, gpu="5080"), ())

    def test_pro_6000_includes_dondon_and_chimachima_m_at_batch_eight(self):
        for engine in ("kaiLLM", "llama.cpp"):
            for batch in BATCHES:
                expected = ["plain", "dspark5"] if batch in (1, 2, 8) else ()
                self.assertEqual(modes("inkling-dondon-m", engine, batch, gpu="PRO 6000"), expected)
        for batch in BATCHES:
            expected = ["plain", "dspark5"] if batch in (1, 8) else ()
            self.assertEqual(modes("inkling-chima-m", "kaiLLM", batch, gpu="PRO 6000"), expected)
            self.assertEqual(modes("inkling-chima-m", "llama.cpp", batch, gpu="PRO 6000"), ())
