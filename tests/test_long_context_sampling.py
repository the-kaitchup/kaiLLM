import json
from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "long_context"))
import benchmark


class SamplingTests(unittest.TestCase):
    def test_llama_completion_verifies_applied_parameters(self):
        sampling = benchmark.sampling_for("lfm")
        response = MagicMock()
        result = {"tokens_evaluated": 1, "tokens_predicted": 1, "content": "sample", "stop_type": "limit",
                  "timings": {"prompt_n": 1}, "generation_settings": benchmark.sampling_payload("llama.cpp", sampling)}
        with patch.object(benchmark.urllib.request, "urlopen", return_value=response) as call:
            response.__enter__.return_value.read.return_value = json.dumps(result)
            benchmark.completion("llama.cpp", "http://127.0.0.1", [1], 1, 5, sampling=sampling)
            payload = json.loads(call.call_args.args[0].data)
            self.assertEqual(payload["temperature"], 0.1)
            self.assertEqual(payload["top_k"], 50)
            self.assertEqual(payload["repeat_penalty"], 1.1)
            self.assertEqual(payload["repeat_last_n"], 73728)
            self.assertFalse(payload["cache_prompt"])
            self.assertNotIn("source", payload)
            result["generation_settings"]["temperature"] = 0
            response.__enter__.return_value.read.return_value = json.dumps(result)
            with self.assertRaisesRegex(ValueError, "reported sampling temperature"):
                benchmark.completion("llama.cpp", "http://127.0.0.1", [1], 1, 5, sampling=sampling)

    def test_missing_greedy_peer_uses_the_historical_payload(self):
        self.assertIsNone(benchmark.sampling_for("inkling", "greedy"))
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps({
            "tokens_evaluated": 1, "tokens_predicted": 1, "content": "sample", "stop_type": "limit",
            "timings": {"prompt_n": 1},
        })
        with patch.object(benchmark.urllib.request, "urlopen", return_value=response) as call:
            benchmark.completion("llama.cpp", "http://127.0.0.1", [1], 1, 5,
                                 sampling=benchmark.sampling_for("inkling", "greedy"))
            payload = json.loads(call.call_args.args[0].data)
            self.assertEqual((payload["temperature"], payload["seed"]), (0, 42))
            self.assertFalse(payload["cache_prompt"])
            self.assertNotIn("top_k", payload)

    def test_kai_completion_passes_full_distribution_to_the_server(self):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps({
            "choices": [{"finish_reason": "length", "text": "sample"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        })
        with patch.object(benchmark.urllib.request, "urlopen", return_value=response) as call:
            benchmark.completion("kaiLLM", "http://127.0.0.1", [1], 1, 5,
                                 sampling=benchmark.sampling_for("inkling"))
            payload = json.loads(call.call_args.args[0].data)
            self.assertEqual((payload["temperature"], payload["top_p"], payload["top_k"], payload["min_p"]),
                             (1.0, 1.0, 0, 0.0))
            self.assertEqual(payload["seed"], 42)
            self.assertTrue(payload["ignore_eos"])
            self.assertNotIn("repeat_penalty", payload)
