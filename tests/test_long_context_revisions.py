"""Checkpoint provenance across Hub revisions with identical and changed GGUFs."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks/long_context"))
from benchmark import MANIFEST, OUTPUT, PROMPT, PROTOCOL, current_weights, full_graph_runtime, model_at_revision
from coverage_matrix import counts
from matrix import modes
from report import confirmed_limit, get_record, main as render
from test_long_context_reports import graph_proof, head_proof


def raw_plain(key, revision):
    model = model_at_revision(key, revision)
    kai = MANIFEST["kaillm"]
    variant = model.get("kai_loader_runtime", "default")
    sha256 = [f["sha256"] for f in model["files"].values()]
    def wave(length):
        return {"requests": [{"prompt_tokens": PROMPT, "output_tokens": length,
                              "output_sha256": "generated", "cache_hits": 0}], "seconds": 2.0 if length == 1 else 1000.0}
    return dict(checkpoint=key, model_revision=revision, sha256=sha256, prompt_tokens=PROMPT,
                generated_tokens=OUTPUT, protocol=PROTOCOL, batch=1, prompt_hashes=["tokens"], domain="prose",
                prefill=[wave(1) for _ in range(3)], full=wave(OUTPUT), mode="plain", engine="kaiLLM",
                timestamp_utc="2026-11-22T11:12:13Z", environment=dict(
                    devices=[{"uuid": "example", "driver_version": "570", "power.limit": 350}],
                    gpu_names=["RTX 3090"], llama_source=MANIFEST["llama_cpp"]["revision"],
                    kai_source=kai[variant + "_revision"], engine_revision=next(iter(kai["runtime_locks"][variant].values())),
                    kai_humming_deterministic=model.get("kai_humming_deterministic", False), max_concurrency=1.1,
                    server_argv=["kaiLLM", "--max-num-seqs", "1", "--max-num-batched-tokens", "2048",
                                 "--max-model-len", str(PROMPT + OUTPUT), "--kv-cache-memory-bytes", "100"]))


class CheckpointRevisionTests(unittest.TestCase):
    def test_older_raw_and_compact_keep_their_actual_revision_and_hashes(self):
        for key, eligible in (("inkling-udq2", False), ("lfm-chima", True), ("hy4-iq2m", True)):
            for previous in MANIFEST["model_history"][key]:
                with self.subTest(checkpoint=key, revision=previous):
                    raw = raw_plain(key, previous)
                    compact = get_record(raw)
                    self.assertEqual((compact["model_revision"], compact["sha256"]), (previous, raw["sha256"]))
                    compact.update(cuda_graphs=graph_proof(), target_head_cuda_graphs=head_proof())
                    self.assertEqual(current_weights(compact), eligible)
                    self.assertEqual(counts([compact], [])["filled"], int(eligible))
                    with self.assertRaisesRegex(ValueError, "Weight digests differ"):
                        get_record({**raw, "sha256": ["wrong"]})
                    with self.assertRaisesRegex(ValueError, "Unknown Hugging Face revision"):
                        get_record({**raw, "model_revision": "unknown"})
                    with tempfile.TemporaryDirectory() as tmp:
                        root = Path(tmp)
                        archive = root / "results.jsonl"
                        archived = dict(compact)
                        del archived["sha256"]  # The early compact schema omitted shard digests.
                        archive.write_text(json.dumps(archived) + "\n")
                        (root / "sample-prose.json").write_text(json.dumps(raw))
                        with patch.object(sys, "argv", ["report.py", str(archive), str(root / "speed.md"), str(root)]):
                            render()
                        self.assertEqual(json.loads(archive.read_text())["sha256"], compact["sha256"])
                        for name in ("speed.md", "speed-by-model.md"):
                            self.assertEqual("| 1 | plain | prose |" in (root / name).read_text(), eligible)
                        archive.write_text(json.dumps({**compact, "sha256": ["wrong"]}) + "\n")
                        with patch.object(sys, "argv", ["report.py", str(archive), str(root / "speed.md")]):
                            with self.assertRaisesRegex(ValueError, "Weight digests differ"):
                                render()

    def test_updated_donor_reopens_both_engines_and_requires_new_limits(self):
        key = "inkling-udq2"
        current = MANIFEST["models"][key]
        old, = MANIFEST["model_history"][key]
        self.assertEqual(current["gguf_architecture"], "inkling2_loop")
        self.assertIn("plain", modes(key, "kaiLLM", 1))
        self.assertIn("dspark5", modes(key, "llama.cpp", 1))
        for engine in ("kaiLLM", "llama.cpp"):
            stamp = dict(state="failed", stage="startup", gpu_names=["RTX 5080"], checkpoint=key,
                         engine=engine, mode="plain", batch=2, reason="CUDA out of memory",
                         llama_source=MANIFEST["llama_cpp"]["inkling_revision"])
            if engine == "kaiLLM":
                stamp["kai_source"] = full_graph_runtime(key, "plain")[0]
            self.assertIsNone(confirmed_limit(stamp))
            self.assertIsNone(confirmed_limit({**stamp, "model_revision": old}))
            limit = confirmed_limit({**stamp, "model_revision": current["revision"]})
            self.assertEqual(limit["model_revision"], current["revision"])
            self.assertEqual(counts([], [limit])["unavailable"] - counts([], [])["unavailable"], 1)

    def test_current_revision_and_same_bytes_produce_one_timing_cell(self):
        key = "lfm-chima"
        previous, = MANIFEST["model_history"][key]
        current = MANIFEST["models"][key]["revision"]
        rows = []
        for revision in (previous, current):
            compact = get_record(raw_plain(key, revision))
            compact.update(cuda_graphs=graph_proof(), target_head_cuda_graphs=head_proof(),
                           prefill_seconds=[1.0 if revision == previous else 0.5] * 3, full_seconds=5.0)
            rows.append(compact)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "results.jsonl"
            for order in (rows, list(reversed(rows))):
                archive.write_text("".join(json.dumps(r) + "\n" for r in order))
                with patch.object(sys, "argv", ["report.py", str(archive), str(root / "speed.md")]):
                    render()
                self.assertEqual(len(archive.read_text().splitlines()), 2)
                for name in ("speed.md", "speed-by-model.md"):
                    table = (root / name).read_text()
                    self.assertEqual(table.count("| 1 | plain | prose | 16384.0 |"), 1)
                    self.assertNotIn("| 1 | plain | prose | 8192.0 |", table)
