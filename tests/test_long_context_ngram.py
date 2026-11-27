"""Keep an inactive stock n-gram configuration out of speculative comparisons."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks/long_context"))
from check_lfm3_ngram_graphs import compare_controls
from coverage_matrix import INACTIVE_NGRAM, MISSING_NGRAM_COUNTS, NO_NGRAM_PROPOSALS, graphs_verified, provisional_reason
from test_long_context_inkling_native import draft_graph_proof, graph_proof, head_proof


class LlamaNgramEvidenceTest(unittest.TestCase):
    def test_old_batch_eight_receipts_require_a_rerun(self):
        row = {"engine": "llama.cpp", "mode": "ngram1", "batch": 8}
        self.assertEqual(provisional_reason(row), INACTIVE_NGRAM)
        simple = {"method": "ngram-simple", "lookup_tokens": 2, "draft_tokens": 1, "proposed": 0}
        self.assertEqual(provisional_reason({**row, "llama_ngram": simple}), INACTIVE_NGRAM)

    def test_one_token_map_requires_actual_proposals(self):
        setup = {"method": "ngram-map-k", "lookup_tokens": 2, "draft_tokens": 1, "min_hits": 1}
        row = {"engine": "llama.cpp", "mode": "ngram1", "batch": 8}
        self.assertEqual(provisional_reason({**row, "llama_ngram": setup}), MISSING_NGRAM_COUNTS)
        self.assertEqual(provisional_reason({**row, "llama_ngram": {**setup, "proposed": 0}}),
                         NO_NGRAM_PROPOSALS)
        self.assertIsNone(provisional_reason({**row, "llama_ngram": {**setup, "proposed": 123}}))

    def test_other_measured_configurations_remain_eligible(self):
        self.assertIsNone(provisional_reason({"engine": "kaiLLM", "mode": "ngram1", "batch": 8}))
        setup = {"method": "ngram-simple", "lookup_tokens": 2, "draft_tokens": 3, "proposed": 42}
        self.assertIsNone(provisional_reason({"engine": "llama.cpp", "mode": "ngram3", "batch": 4,
                                              "llama_ngram": setup}))


    def test_gpu_lookup_needs_one_graph_per_call_and_measured_draft_tokens(self):
        row = {"engine": "kaiLLM", "mode": "ngram3", "gpu": ["RTX 3090"], "cuda_graphs": graph_proof(),
               "target_head_cuda_graphs": head_proof(), "drafter_cuda_graphs": draft_graph_proof(),
               "kai_ngram": {"scope": "full-wave", "lookup_min": 2, "lookup_max": 8, "draft_tokens": 3, "proposed": 201, "accepted": 43, "rejected": 158}}
        self.assertFalse(graphs_verified(row))
        row["drafter_cuda_graphs"]["graph_replays"] = 2
        row["drafter_cuda_graphs"]["per_wave_by_rank"]["full"][0]["graph_replays"] = 2
        self.assertTrue(graphs_verified(row))
        for bad in ({"proposed": 0}, {"accepted": 202}, {"rejected": 0}, {"scope": "warmup"}, {"accepted": True}):
            self.assertFalse(graphs_verified({**row, "kai_ngram": {**row["kai_ngram"], **bad}}))
        del row["drafter_cuda_graphs"]
        self.assertFalse(graphs_verified(row))
        self.assertTrue(graphs_verified({**row, "engine": "llama.cpp"}))


    def test_qualification_compares_every_generated_token_and_slot(self):
        sample = {"code": [[4] * 48, [6] * 48], "prose": [[8] * 48, [9] * 48]}
        self.assertEqual(compare_controls(sample, sample), {"code": 96, "prose": 96})
        with self.assertRaisesRegex(ValueError, "differ"):
            compare_controls(sample, {**sample, "prose": list(reversed(sample["prose"]))})
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            compare_controls({**sample, "code": [[]]}, sample)
