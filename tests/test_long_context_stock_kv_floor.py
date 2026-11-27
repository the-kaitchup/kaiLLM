import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks/long_context"))
import stock_kv_floor as floor
from coverage_matrix import known_exclusion
from matrix import modes


class TestStockMemory(unittest.TestCase):
    def test_lfm_global_only(self):
        self.assertEqual(floor.minimum_bytes("lfm-suisui", 1), 864 * 2**20)
        self.assertEqual(floor.minimum_bytes("lfm-qad", 32), 27 * 2**30)
        self.assertIsNone(known_exclusion("5080", 1, "lfm-suisui", "llama.cpp", 16))
        self.assertTrue(known_exclusion("5080", 1, "lfm-suisui", "llama.cpp", 32))
        self.assertIsNone(known_exclusion("5080", 1, "lfm-suisui", "kaiLLM", 32))

    def test_distinct_inkling_donors(self):
        self.assertEqual(floor.minimum_bytes("inkling-udq2", 4), 7_888_306_176 + 9 * 2**30)
        self.assertTrue(known_exclusion("5080", 1, "inkling-udq2", "llama.cpp", 4))
        self.assertIsNone(known_exclusion("5090", 1, "inkling-udq2", "llama.cpp", 8))
        self.assertTrue(known_exclusion("5090", 1, "inkling-loop-iq4", "llama.cpp", 8))
        self.assertIsNone(known_exclusion("5090", 2, "inkling-loop-iq4", "llama.cpp", 8))

    def test_hybrid_qwen_target_omits_mtp_and_recurrent_state(self):
        for key in ("daisy-iq2m", "daisy-dondon", "qwen-udq2"):
            self.assertEqual(floor.minimum_bytes(key, 2), 9 * 2**30)
            self.assertTrue(known_exclusion("5090", 1, key, "llama.cpp", 8))
            self.assertIsNone(known_exclusion("PRO 6000", 1, key, "llama.cpp", 16))

    def test_published_inkling_stock_limit_and_native_schedule(self):
        for tier in ("xs", "s", "m"):
            dondon = "inkling-dondon-" + tier
            chima = "inkling-chima-" + tier
            self.assertIsNone(known_exclusion("3090", 1, dondon, "llama.cpp", 4))
            self.assertTrue(known_exclusion("3090", 1, dondon, "llama.cpp", 8))
            self.assertIsNone(known_exclusion("5090", 1, dondon, "llama.cpp", 8))
            self.assertTrue(known_exclusion("5090", 1, dondon, "llama.cpp", 16))
            self.assertIsNone(known_exclusion("3090", 1, chima, "kaiLLM", 8))
            self.assertEqual(modes(chima, "kaiLLM", 8, gpu="3090"), ["plain", "dspark5"])
            self.assertEqual(modes(chima, "llama.cpp", 1, gpu="3090"), ())
            expected_5080 = ["plain", "dspark5"] if tier in ("xs", "s") else ["plain"]
            self.assertEqual(modes(dondon, "llama.cpp", 1, gpu="5080"), expected_5080)
            self.assertEqual(modes(dondon, "llama.cpp", 2, gpu="5080"), ())

    def test_pin_drift_and_equality(self):
        with patch.object(floor, "llama_revision", return_value="different"):
            self.assertIsNone(floor.minimum_bytes("lfm-qad", 32))
        model = floor.MANIFEST["models"]["lfm-qad"]
        with patch.dict(model, {"files": {"replacement.gguf": {"sha256": "different"}}}):
            self.assertIsNone(floor.minimum_bytes("lfm-qad", 32))
        with patch.object(floor, "minimum_bytes", return_value=16 * 2**30):
            self.assertIsNone(known_exclusion("5080", 1, "lfm-qad", "llama.cpp", 32))
