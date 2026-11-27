"""Exercise the multi-stage Hy4 qualification's graph admission without a model or device."""
import json
from contextlib import ExitStack
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks/long_context"))
import check_hy4_pipeline as check
import cuda_graphs
import llama_graph
import verify_gguf_loader


class PipelineGraphTest(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.uuids = ["GPU-first", "GPU-last"]
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for owner, name in ((check, "wait_server"), (check, "stop_server"), (check.subprocess, "Popen")):
            self.stack.enter_context(patch.object(owner, name))

    def kai_counters(self, head_rank):
        evidence = self.directory / "evidence"
        evidence.mkdir()
        for index, uuid in enumerate(self.uuids):
            path = evidence / f"{uuid}.{index + 10}.json"
            path.write_text(json.dumps({
                "uuid":uuid, "pid":index + 10, "schema":1, "mode":"FULL", "enforce_eager":False,
                "captured_token_sizes":[1,512], "prefill_capture_sizes":[512], "max_prefill_tokens":512,
                "output_head_capture_sizes":[1] if index == head_rank else [],
            }))
            path.with_suffix(".counts").write_bytes(cuda_graphs.COUNTERS.pack(*[0]*8))
            if index == head_rank:
                path.with_suffix(".head-counts").write_bytes(cuda_graphs.HEAD_COUNTERS.pack(0,0,0,0))
        return evidence

    def run_with(self, engine, evidence):
        return check.run_capture(engine, ["server"], {}, self.directory, {}, 10234, evidence, self.uuids)

    def test_pipeline_gate_exercises_dense_and_compressed_indices(self):
        source = list(range(check.MAX_MODEL_LEN))
        bank = {"domains": {domain: {"rows": [source] * 3} for domain in ("prose", "code")}}
        with patch.object(verify_gguf_loader, "predict", return_value={}), \
                patch.object(verify_gguf_loader, "verify_prompt_bank"):
            result = check.capture("kaiLLM", "test", bank, "hy4", check.PREFIX_SIZES)
        self.assertEqual(len(result["rows"]), 24)
        for domain in bank["domains"]:
            sizes = [r["prompt_tokens"] for r in result["rows"] if r["domain"] == domain]
            self.assertEqual(sizes, list(check.PREFIX_SIZES) * 3)
            self.assertLess(min(sizes), 2048)
            self.assertGreater(max(sizes), 2048)
            self.assertLessEqual(max(sizes) + 8, check.MAX_MODEL_LEN)

    def test_pipeline_and_output_head_use_their_actual_devices(self):
        evidence = self.kai_counters(1)

        def execute(*_):
            for path in evidence.glob("*.counts"):
                path.write_bytes(cuda_graphs.COUNTERS.pack(24,24,0,0,168,168,0,0))
            for path in evidence.glob("*.head-counts"):
                path.write_bytes(cuda_graphs.HEAD_COUNTERS.pack(24,24,168,168))
            return {"engine":"kaiLLM", "rows":[]}

        with patch.object(check, "capture", side_effect=execute):
            actual = self.run_with("kaiLLM", evidence)
        self.assertEqual(actual["graph_launches"]["decode"], {uid:168 for uid in self.uuids})
        self.assertEqual(actual["head_graph_launches"]["prefill"], {self.uuids[-1]:24})

    def test_wrong_head_rank_rejects_before_sending_any_prompt(self):
        evidence = self.kai_counters(0)
        with patch.object(check, "capture") as requests:
            with self.assertRaisesRegex(ValueError, "final pipeline rank"):
                self.run_with("kaiLLM", evidence)
            requests.assert_not_called()

    def test_reference_rejects_a_stage_without_decode_launch(self):
        path = self.directory / "llama.bin"
        words = [0] * (llama_graph.LAYOUT.size // 8)
        words[:4] = [llama_graph.MAGIC, llama_graph.VERSION, 2, 0]
        path.write_bytes(llama_graph.LAYOUT.pack(*words))

        def execute(*_):
            for phase, devices in (("prefill",[0,1]), ("decode",[0])):
                for device in devices:
                    offset = 4+(llama_graph.PHASES.index(phase)*4+device)*len(llama_graph.EVENTS)
                    words[offset+llama_graph.EVENTS.index("batches")] = 1
                    words[offset+llama_graph.EVENTS.index("replays")] = 1
            path.write_bytes(llama_graph.LAYOUT.pack(*words))
            return {"engine":"llama.cpp", "rows":[]}

        with patch.object(check, "capture", side_effect=execute):
            with self.assertRaisesRegex(ValueError, "decode on GPU-last"):
                self.run_with("llama.cpp", path)


if __name__ == "__main__":
    unittest.main()
