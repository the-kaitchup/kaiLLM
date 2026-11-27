import hashlib
import json
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks/long_context"))
import prefix_checked
import report
from benchmark import MANIFEST, dspark_asset
from coverage_matrix import graphs_verified, timing_accepted
from test_long_context_inkling_native import graph_proof, head_proof, draft_graph_proof


def sample():
    model = MANIFEST["models"]["inkling-udq2"]
    draft = dspark_asset(model)[1]
    target_sha, = [f["sha256"] for f in model["files"].values()]
    filename, = draft["files"]
    draft_sha = draft["files"][filename]["sha256"]
    scope = {"gpu_uuid": "test-board", "runtime": {"revision": "c" * 40, "source_lock": "d" * 64},
             "target_sha256": target_sha, "drafter_sha256": draft_sha}
    raw = {"domain": "prose", "checkpoint": "inkling-udq2", "model_revision": model["revision"], "batch": 1,
           "sha256": [target_sha], "engine": "kaiLLM", "mode": "dspark5", "timestamp_utc": "2099-01-01",
           "protocol": "8k64k-v1", "prompt_tokens": 8192, "generated_tokens": 65536, "prompt_hashes": ["a" * 64],
           "kai_dspark": {"scope": "full-wave", "accepted": 4, "proposed": 15, "rejected": 11},
           "environment": {"gpu_names": ["NVIDIA GeForce RTX 5070 Ti"], "devices": [{"uuid": "test-board",
               "name": "NVIDIA GeForce RTX 5070 Ti", "driver_version": "test-driver", "power.limit": "300.00"}],
               "numerical_scope": scope, "engine_revision": "d" * 64, "kai_source": "c" * 40,
               "kai_humming_deterministic": True, "measured_gguf_sha256": [target_sha],
               "measured_drafter_sha256": [draft_sha], "max_concurrency": 1.001,
               "server_argv": ["--max-num-seqs", "1", "--max-num-batched-tokens", "2048", "--max-model-len", "73728",
                               "--kv-cache-memory-bytes", str(3764 << 20), "--spec-tokens", "5", "--dspark",
                               f"{draft['repo']}/{draft['revision']}/{filename}"]}}
    tiny = {"requests": [{"prompt_tokens": 8192, "output_tokens": 1, "cache_hits": 0}], "seconds": 2.0}
    raw.update(prefill=[deepcopy(tiny) for _ in range(3)], full={"seconds": 1002.0,
               "requests": [{"prompt_tokens": 8192, "output_tokens": 65536, "cache_hits": 0, "output_sha256": "e" * 64}]},
               prefill_tps=4096, decode_tps=65.535)
    return raw, hashlib.sha256((json.dumps(scope, indent=2) + "\n").encode()).hexdigest()


class PromptLengthTimings(unittest.TestCase):
    def test_scope_keeps_the_row_outside_standard_coverage_with_or_without_graph_evidence(self):
        raw, scope_sha = sample()
        with patch.object(prefix_checked, "SCOPE_SHA256", scope_sha):
            row = report.get_record(raw)
            self.assertFalse(timing_accepted(row))
            self.assertFalse(graphs_verified(row))
            control = {**row, "mode": "plain", "kai_source": "ordinary", "cuda_graphs": graph_proof(),
                       "target_head_cuda_graphs": head_proof()}
            del control["numerical_scope_sha256"]
            self.assertTrue(timing_accepted(control))
            rows = [control]
            plain_views = report.tables(rows)
            views = report.tables(rows, prefix_timings=[row])
            for old, new in zip(plain_views, views):
                before, section = new.split("## " + prefix_checked.LABEL, 1)
                diagnostic, after = section.split("## Availability", 1)
                self.assertEqual(old, before + "## Availability" + after)
                self.assertIn("4/15 (26.7%) | Unverified | `ordinary` yes", diagnostic)
                self.assertIn("Three shorter prose cases diverged", diagnostic)
            row.update(cuda_graphs=graph_proof(), target_head_cuda_graphs=head_proof(),
                       drafter_cuda_graphs=draft_graph_proof())
            self.assertTrue(graphs_verified(row))
            self.assertFalse(timing_accepted(row))
            self.assertIn("| Verified |", "\n".join(report.prefix_checked_section([row], rows)))
            with self.assertRaisesRegex(ValueError, "ordinary full-length control"):
                report.prefix_checked_section([row], [{**control, "hardware_id": "different-board"}])

    def test_matches_only_identical_stock_workloads_and_marks_the_slower_kai_rate(self):
        raw, scope_sha = sample()
        with patch.object(prefix_checked, "SCOPE_SHA256", scope_sha):
            row = report.get_record(raw)
            ordinary = {**row, "mode": "plain", "cuda_graphs": graph_proof(), "target_head_cuda_graphs": head_proof()}
            del ordinary["numerical_scope_sha256"]
            reference = {**row, "engine": "llama.cpp", "llama_source": "e" * 40, "prefill_seconds": [1.] * 3,
                         "llama_default": {"policy": "default", "microbatch": 256, "speculative": row["kai_dspark"],
                                           "measured_drafter_sha256": row["drafter"]["sha256"]}}
            del reference["numerical_scope_sha256"]
            def section(stock):
                # The llama_default suite checks certified builds. Here isolate pairing and rendering.
                with patch.object(report.llama_default, "verified",
                                  side_effect=lambda value: value.get("llama_default", {}).get("policy") == "default"):
                    return "\n".join(report.prefix_checked_section([row], [ordinary, stock]))
            matched = section(reference)
            self.assertIn(r"$\color{red}{4096.0}$ | 65.5", matched)
            self.assertIn("llama.cpp default `eeeeeeee` (ub256) | 8192.0 |", matched)
            for changed in ({"hardware_id": "other"}, {"domain": "code"}, {"driver": ["other"]}, {"power_limit_w": ["250"]},
                            {"model_revision": "other"}, {"prompt_set_sha256": "other"}, {"sha256": ["other"]},
                            {"llama_default": {**reference["llama_default"], "policy": "strict"}},
                            {"llama_default": {**reference["llama_default"], "measured_drafter_sha256": "wrong"}}):
                no_match = section({**reference, **changed})
                self.assertNotIn("llama.cpp default `", no_match)
                self.assertNotIn(r"\color{red}", no_match)

    def test_multiple_qualified_builds_share_the_same_stock_control(self):
        raw, tracer = sample()
        with patch.object(prefix_checked, "SCOPE_SHA256", tracer), \
             patch.dict(prefix_checked.ADDITIONAL_SCOPES, {"new-scope": "Three shorter prose cases diverged."}):
            old = report.get_record(raw)
            new = {**old, "kai_source": "b" * 40, "timestamp_utc": "2099-02-01", "numerical_scope_sha256": "new-scope"}
            ordinary = {**old, "mode": "plain", "cuda_graphs": graph_proof(), "target_head_cuda_graphs": head_proof()}
            del ordinary["numerical_scope_sha256"]
            reference = {**old, "engine": "llama.cpp", "llama_source": "e" * 40,
                         "llama_default": {"policy": "default", "microbatch": 128, "speculative": old["kai_dspark"],
                                           "measured_drafter_sha256": old["drafter"]["sha256"]}}
            with patch.object(report.llama_default, "verified", side_effect=lambda row: row is reference):
                for view in report.tables([ordinary, reference], prefix_timings=[old, new]):
                    section = view.split("## " + prefix_checked.LABEL)[1].split("## Availability")[0]
                    self.assertEqual(section.count("| llama.cpp default `"), 1)
                    self.assertEqual(section.count("| kaiLLM `"), 2)
                    self.assertIn("Each kaiLLM build matched ordinary decoding on four 48-token outputs", section)
                    self.assertFalse(timing_accepted(new))

    def test_packed_gate_receipt_requires_both_switches_in_the_actual_cuda_worker(self):
        row = {"kai_source": "candidate", "engine_revision": "lock"}
        valid = dict(source="candidate", source_lock="lock", physical_gpu="test-board",
                     active_cuda_process=True, serving_task_descendant=True,
                     selector_environment=dict.fromkeys(prefix_checked.EXPERT_FLAGS, "1"))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tensors = root / "selector-witness"
            tensors.mkdir()
            f = root / "options.json"
            operator = "c8e509a6bde9c9035b65a3f56e4ef9ea9406b23837ccc90fef6d724bd4a81aae"
            selected = {str(k): prefix_checked.packed_tile_map(k) for k in (16, 17, 19, 22)}
            hashes = {}
            for k, maps in selected.items():
                path = tensors / f"stage-{k}.json"
                path.write_text(json.dumps(dict(kind=int(k), selected=maps, flags=valid["selector_environment"],
                                                source_file_sha256=operator, method="dflash", k=5,
                                                draft_architectures=["DFlashInklingDSparkDraftModel"])))
                hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            valid.update(selected_tiles=selected, operator_source_sha256=operator, stage_receipts_sha256=hashes)
            captures = {}
            for kind in (16, 17, 19, 22):
                for width in (6, 2048):
                    path = tensors / f"capture-{kind}-{width}.json"
                    path.write_text(json.dumps(dict(kind=kind, rows=width, capturing=True,
                                                    tile=selected[str(kind)][str(width)])))
                    captures[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            valid["capture_receipts_sha256"] = captures
            def verify(body):
                f.write_text(json.dumps(body))
                digest = hashlib.sha256(f.read_bytes()).hexdigest()
                self.assertEqual(prefix_checked.packed_expert_evidence(f, row, "test-board", digest), digest)
            verify(valid)
            with self.assertRaises(ValueError):
                prefix_checked.packed_expert_evidence(f, row, "test-board", "wrong-physical-server")
            for field in ({"selector_environment": {prefix_checked.EXPERT_FLAGS[0]: "1"}},
                          {"source_lock": "different"}, {"active_cuda_process": False}, {"physical_gpu": "other"},
                          {"serving_task_descendant": False}, {"selected_tiles": {**selected, "17": None}},
                          {"operator_source_sha256": "different"}, {"stage_receipts_sha256": {}}, {"capture_receipts_sha256": {}}):
                with self.assertRaises(ValueError):
                    verify({**valid, **field})
            (tensors / "stage-16.json").write_text("{}")
            with self.assertRaises(ValueError):
                verify(valid)

    def test_raw_validation_and_partial_archive(self):
        raw, scope_sha = sample()
        with patch.object(prefix_checked, "SCOPE_SHA256", scope_sha), tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for label in ("before", "after"):
                (root / f"metrics-{label}-prose.txt").write_text("vllm:num_preemptions_total{} 1.0\n")
            f = root / "prose.json"
            f.write_text(json.dumps(raw))
            records = prefix_checked.read_run(root, report.get_record, report.rates)
            self.assertEqual(len(records), 1)
            self.assertNotIn("test-board", json.dumps(records))
            self.assertEqual(records[0]["raw_sha256"], hashlib.sha256(f.read_bytes()).hexdigest())
            for key, value in (("kai_source", "other-runtime"), ("measured_drafter_sha256", ["wrong"])):
                changed = deepcopy(raw)
                changed["environment"][key] = value
                with self.assertRaises(ValueError):
                    report.get_record(changed)
            for key, value in (("scope", "observed-request-interval"), ("rejected", 0)):
                changed = deepcopy(raw)
                changed["kai_dspark"][key] = value
                with self.assertRaises(ValueError):
                    report.get_record(changed)
            (root / "metrics-after-prose.txt").write_text("vllm:num_preemptions_total{} 2.0\n")
            with self.assertRaisesRegex(ValueError, "preemption during"):
                prefix_checked.read_run(root, report.get_record, report.rates)
