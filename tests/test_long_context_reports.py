import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from html import escape
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks/long_context"))
from report import FAMILIES, MANIFEST, checkpoint_label, tables
import report
from test_long_context_inkling_native import draft_graph_proof


QAD_WEIGHTS = dict(model_revision=MANIFEST["models"]["lfm-qad"]["revision"],
                   sha256=[f["sha256"] for f in MANIFEST["models"]["lfm-qad"]["files"].values()])


def graph_proof():
    waves = {"prefill": ("prefill-1", "prefill-2", "prefill-3", "full"), "decode": ("full",)}
    return {phase: {"scope": "timed-wave", "graph_replays": 3 * len(labels), "eager_steps": 0, "unclassified_steps": 0,
                    "per_wave_by_rank": {label: [3] for label in labels}} for phase, labels in waves.items()}


def head_proof():
    return {**graph_proof(), "pipeline_stages": 1, "device_ranks": [0]}


def displayed_results(markdown, model_first):
    rows = Counter()
    gpu = model = section = None
    summaries = {checkpoint_summary(key): checkpoint_label(key) for key in MANIFEST["models"]}
    depth = 0
    for line in markdown.splitlines():
        if model_first and line in ("<details>", "<details open>"):
            depth += 1
        elif model_first and line == "</details>":
            depth -= 1
            if depth == 0:
                gpu = model = section = None
        elif model_first and line in summaries:
            gpu = section = None
            model = summaries[line]
        elif line.startswith("## "):
            gpu = None if model_first else line[3:]
            model = section = None
        elif line.startswith("### "):
            section = None
        elif line.startswith("#### "):
            if model_first:
                gpu = line[5:]
            else:
                model = line[5:]
            section = None
        elif line.startswith("| Batch | Mode | Domain |"):
            section = "timing"
        elif line.startswith("| Engine | Mode | Batch |"):
            section = "limit"
        elif section and line.startswith("| ") and gpu and model:
            rows[(gpu, model, section, line)] += 1
    return rows


def checkpoint_summary(key):
    model = MANIFEST["models"][key]
    return f"<summary><strong>{escape(model['repo'])} ({escape(model['gguf_type'])})</strong></summary>"


class LongContextReports(unittest.TestCase):
    def test_draft_lengths_match_without_a_plain_measurement(self):
        shared = dict(gpu=["NVIDIA GeForce RTX 5090"], hardware_id="board", driver=["1"],
                      power_limit_w=["575"], checkpoint="inkling-chima-m", engine="kaiLLM",
                      batch=1, domain="prose", sha256=["weights"], protocol=report.PROTOCOL,
                      input_tokens_per_request=report.PROMPT, output_tokens_per_request=report.OUTPUT,
                      prompt_set_sha256="prompts", sampling=report.benchmark.sampling_for("inkling"),
                      prefill_seconds=[1.] * 3)
        five = dict(shared, mode="dspark5", full_seconds=4.)
        three = dict(shared, mode="dspark3", full_seconds=3.)
        group = ("5090", 1, "inkling-chima-m")
        with patch.object(report, "timing_accepted", return_value=True), patch.object(report, "current_weights", return_value=True):
            matches = report.inkling_draft_comparisons([five, three])
            self.assertEqual(matches[group], [(five, three)])
            self.assertIn("| prose | dspark5 | dspark3 |", "\n".join(report.draft_comparison_block(matches[group])))
            for changes in ({"hardware_id": "other board"}, {"sha256": ["other weights"]},
                            {"sampling": {**shared["sampling"], "seed": 43}}):
                self.assertFalse(report.inkling_draft_comparisons([five, {**three, **changes}]))

    def test_closest_inkling_reference_matches_board_sampler_and_drafter(self):
        shared = dict(gpu=["NVIDIA GeForce RTX 5080"], hardware_id="board", driver=["1"],
                      power_limit_w=["360"], mode="dspark5", batch=1, domain="code",
                      protocol="8k64k-v1", input_tokens_per_request=8192, output_tokens_per_request=65536,
                      prompt_set_sha256="prompts", sampling=report.benchmark.sampling_for("inkling"),
                      timestamp_utc="2026-11-26T00:00:00Z")

        def row(model, engine, **settings):
            return dict(shared, checkpoint=model, engine=engine, model_revision=MANIFEST["models"][model]["revision"], **settings)

        target = row("inkling-chima-s", "kaiLLM", measured_drafter_sha256="draft",
                     prefill_seconds=[1.] * 3, full_seconds=3.)
        close = row("inkling-dondon-s", "llama.cpp", llama_default={"measured_drafter_sha256": "draft"},
                    prefill_seconds=[2.] * 3, full_seconds=4.)
        farther = {**close, "checkpoint": "inkling-dondon-xs"}
        group = ("5080", 1, "inkling-chima-s")
        with patch.object(report, "timing_accepted", return_value=True), patch.object(report.llama_default, "verified", return_value=True):
            expect = report.closest_inkling_references([farther, close, target])
            self.assertEqual(expect[group], [(target, close)])
            text = "\n".join(report.closest_reference_block(expect[group]))
            self.assertIn(checkpoint_label("inkling-dondon-s"), text)
            self.assertIn("200.0% | 100.0%", text)
            for mismatch in ({"hardware_id": "another board"}, {"power_limit_w": ["350"]},
                             {"prompt_set_sha256": "different inputs"},
                             {"sampling": {**close["sampling"], "seed": 43}},
                             {"llama_default": {"measured_drafter_sha256": "different weights"}}):
                self.assertFalse(report.closest_inkling_references([target, {**close, **mismatch}]))

    def test_full_output_agreement_requires_the_same_board_and_inputs(self):
        original = dict(engine="kaiLLM", kai_source="ordinary", timestamp_utc="01", gpu=["NVIDIA GeForce RTX 5090"],
                        hardware_id="original board", driver=["driver"], power_limit_w=[600], checkpoint="inkling-loop-iq4",
                        mode="plain", batch=2, domain="code", prompt_set_sha256="prompts", sha256=["weights"],
                        protocol="8k64k-v1", input_tokens_per_request=8192, output_tokens_per_request=65536,
                        generated_text_hashes=["a","b"])
        candidate = {**original, "kai_source":"candidate", "timestamp_utc":"02", "generated_text_hashes":["a","c"]}
        pins = [{"revision":"candidate", "output_control_revision":"ordinary"}]
        notes = report.output_agreement_notes([candidate,original],pins)
        self.assertIn("1 of 2", notes[("5090",1,"inkling-loop-iq4")][0])
        for changed in ({"hardware_id":"other board"}, {"prompt_set_sha256":"other"}, {"sha256":["different weights"]}):
            self.assertFalse(report.output_agreement_notes([original,{**candidate,**changed}],pins))

    def test_sampling_is_preserved_and_isolates_all_reference_paths(self):
        from test_long_context_hybrid_report import hybrid_raw
        from test_long_context_llama_default import raw_stock

        raw = hybrid_raw()
        family = MANIFEST["models"][raw["checkpoint"]]["family"]
        sampling = report.benchmark.sampling_for(family)
        old_kai = report.get_record(raw)
        old_stock = report.get_record(raw_stock())
        old_stock["hardware_id"] = old_kai["hardware_id"]
        self.assertNotIn("sampling", old_kai)
        new_kai = report.get_record({**raw, "sampling": sampling})
        self.assertEqual(new_kai["sampling"], sampling)
        for wrong in (None, {}, {**sampling, "temperature": 0}, {**sampling, "top_p": True}):
            with self.subTest(wrong=wrong), self.assertRaisesRegex(ValueError, "sampler"):
                report.get_record({**raw, "sampling": wrong})
        new_kai.update(timestamp_utc="2027-01-02T00:00:00Z", full_seconds=1.7*old_kai["full_seconds"])
        new_stock = {**old_stock, "sampling": sampling, "timestamp_utc":"2027-01-01T00:00:00Z"}
        self.assertFalse(report.same_default_workload(new_kai, old_stock))
        self.assertFalse(report.matched_performance([{"kaiLLM": new_kai, "llama.cpp": old_stock}]))
        # A mismatch in effective parameters is sufficient even with an identical profile name.
        self.assertFalse(report.same_default_workload(new_kai, {**new_stock, "sampling":{**sampling, "seed":43}}))
        for view in report.tables([new_kai, old_stock]):
            self.assertNotIn("## Matched engine performance", view)
            self.assertNotIn(r"\color{red}", view)
            self.assertIn("| Archived greedy |", view)
            self.assertIn("| Model-recommended |", view)
        all_runs = [old_stock, new_kai, new_stock, old_kai]
        self.assertEqual(report.counts(all_runs, [])["filled"], 2)
        by_gpu, by_model = report.tables(all_runs)
        self.assertNotIn("## Matched engine performance", by_gpu)
        self.assertEqual(by_model.count("1 matched pair shares"), 2)
        self.assertIn("### Model-recommended", by_model)
        self.assertIn("### Archived greedy", by_model)
        for view in (by_gpu, by_model):
            self.assertIn(f"{report.rates(old_kai)[1]:.1f}", view)
            self.assertIn(f"{report.rates(new_kai)[1]:.1f}", view)

    def test_plain_vs_draft_hash_is_scoped_to_sampler(self):
        shared = dict(cuda_graphs=graph_proof(), target_head_cuda_graphs=head_proof(),
                      gpu=["RTX 5080"], driver=["1"], power_limit_w=["360"], hardware_id="fixture",
                      checkpoint="lfm-qad", **QAD_WEIGHTS, batch=1, domain="prose", kai_source="kai",
                      llama_source=MANIFEST["llama_cpp"]["revision"], prompt_set_sha256="prompt", engine="kaiLLM",
                      prefill_seconds=[1.0]*3, full_seconds=3.0, timestamp_utc="2027-01-01T00:00:00Z")
        sampling = report.benchmark.sampling_for("lfm")
        old = dict(shared, mode="plain", generated_text_hashes=["a"])
        new = dict(shared, mode="plain", sampling=sampling, generated_text_hashes=["b"])
        graph = draft_graph_proof()
        graph["graph_replays"] = graph["proposals"]
        graph["per_wave_by_rank"]["full"][0]["graph_replays"] = graph["proposals"]
        draft = dict(shared, mode="ngram3", sampling=sampling, generated_text_hashes=["b"],
                     drafter_cuda_graphs=graph, kai_ngram=dict(scope="full-wave", lookup_min=2, lookup_max=8,
                                                              draft_tokens=3, proposed=12, accepted=6, rejected=6))
        for rows, expected in (([old, draft], False), ([new, old, draft], False),
                               ([old, {**new, "generated_text_hashes":["c"]}, draft], True)):
            for view in report.tables(rows):
                line, = [line for line in view.splitlines() if line.startswith("| 1 | ngram3 | prose |")]
                self.assertEqual("†" in line, expected)

    def test_both_layouts_preserve_paired_and_single_engine_observations(self):
        reference = MANIFEST["llama_cpp"]["revision"]
        gpu = ["NVIDIA GeForce RTX 5080"]
        shared = dict(cuda_graphs=graph_proof(), target_head_cuda_graphs=head_proof(), gpu=gpu, driver=["1"], power_limit_w=["360"], hardware_id="fixture",
                      checkpoint="lfm-qad", **QAD_WEIGHTS, batch=1, domain="prose", kai_source="kai",
                      llama_source=reference, prompt_set_sha256="prompt")
        ngram = draft_graph_proof()
        ngram["graph_replays"] = ngram["proposals"]
        ngram["per_wave_by_rank"]["full"][0]["graph_replays"] = ngram["proposals"]
        rows = [
            dict(shared, engine="kaiLLM", mode="plain", prefill_seconds=[1.0] * 3,
                 full_seconds=3.0, generated_text_hashes=["target"]),
            dict(shared, engine="llama.cpp", mode="plain", prefill_seconds=[0.5] * 3,
                 full_seconds=1.5, generated_text_hashes=["target"]),
            dict(shared, engine="kaiLLM", mode="ngram3", drafter_cuda_graphs=ngram,
                 kai_ngram=dict(scope="full-wave", lookup_min=2, lookup_max=8, draft_tokens=3, proposed=17, accepted=8, rejected=9), prefill_seconds=[1.0] * 3,
                 full_seconds=2.0, generated_text_hashes=["different"]),
        ]
        limits = [dict(gpu=gpu, checkpoint="lfm-qad", engine="kaiLLM", mode="plain", batch=32,
                       kai_source=report.full_graph_runtime("lfm-qad", "plain")[0],
                       reason="Memory allocation failed at the requested context")]
        by_gpu, by_model = tables(rows, limits)
        a, b = displayed_results(by_gpu, False), displayed_results(by_model, True)
        self.assertEqual(a, b)
        self.assertEqual(sum(a.values()), 3)
        self.assertEqual({r[1] for r in a}, {checkpoint_label("lfm-qad")})
        self.assertEqual(sum(line.count(r"\color{red}") for _, _, _, line in a), 2)
        self.assertTrue(any("†" in line and "—" in line for _, _, _, line in a))
        self.assertTrue(any("Memory allocation failed" in line for _, _, _, line in a))
        self.assertNotIn("speed-by-model.md", by_gpu)
        self.assertIn('<a id="gpu-5080-1-model-lfm"></a>\n\n### LFM3 1.7B', by_gpu)
        self.assertIn("[Browse by GPU](speed.md)", by_model)
        coverage = report.counts(rows, limits)
        for column in ("filled", "remaining", "unavailable"):
            self.assertEqual(sum(f[column] for f in coverage["families"].values()), coverage[column])
        self.assertEqual(coverage["families"]["lfm"]["filled"], 3)
        self.assertIn("[LFM3 1.7B](#model-lfm)", by_model)
        for view in (by_gpu, by_model):
            self.assertNotIn("runnable cells measured", view)
            self.assertNotIn("Timing coverage", view)
            self.assertNotIn("candidate cells", view)
        self.assertIn(f"<details>\n{checkpoint_summary('lfm-qad')}\n\n", by_model)
        self.assertIn("<details open>\n<summary>Configuration limits (1)</summary>", by_model)
        self.assertNotIn("<details open>", by_gpu)
        details = []
        for line in by_model.splitlines():
            if line in ("<details>", "<details open>"):
                details.append(line)
            elif line == "</details>":
                details.pop()
            elif line.startswith("|"):
                self.assertIn("<details>", details, line)
        self.assertFalse(details)

    def test_missing_eager_partial_and_unclassified_receipts_do_not_fill_cells(self):
        row = dict(engine="kaiLLM", target_head_cuda_graphs=head_proof(), gpu=["NVIDIA GeForce RTX 5080"], checkpoint="lfm-qad", mode="plain",
                   batch=1, domain="code", **QAD_WEIGHTS, hardware_id="fixture", timestamp_utc="2026-01-01T00:00:00Z")
        for phase in ("prefill", "decode"):
            for change in ({"scope": "startup"}, {"graph_replays": 0}, {"graph_replays": True},
                           {"unclassified_steps": 1}, {"eager_steps": 1}, {"eager_steps": False}):
                proof = graph_proof()
                proof[phase].update(change)
                with self.subTest(phase=phase, change=change):
                    self.assertFalse(report.graphs_verified(dict(row, cuda_graphs=proof)))
        self.assertFalse(report.graphs_verified(dict(row, cuda_graphs=graph_proof(), kai_enforce_eager=True)))
        partial = graph_proof()
        del partial["prefill"]
        rows = [row, dict(row, cuda_graphs=partial)]
        for view in report.tables(rows):
            self.assertNotIn("| Batch | Mode | Domain |", view)
        self.assertEqual(report.counts(rows, [])["filled"], 0)
        self.assertEqual(report.counts([dict(row, cuda_graphs=graph_proof())], [])["filled"], 1)

    def test_approved_historical_timing_retains_its_value_and_remains_separate_from_reruns(self):
        import json
        from historical_timings import previously_accepted
        archive = Path(__file__).resolve().parents[1] / "benchmarks/long_context/results.jsonl"
        rows = [json.loads(x) for x in archive.read_text().splitlines()]
        eager = next(r for r in rows if report.historical_graph_mode(r) == "eager" and previously_accepted(r))
        self.assertFalse(report.graphs_verified(eager))
        self.assertEqual(report.counts([eager], [])["filled"], 1)
        changed = {**eager, "full_seconds": eager["full_seconds"] + 1.}
        self.assertFalse(previously_accepted(changed))
        self.assertEqual(report.counts([changed], [])["filled"], 0)
        for view in report.tables([eager]):
            self.assertIn(f"{report.rates(eager)[1]:.1f}", view)
            self.assertIn("| Eager |", view)

    def test_rank_evidence_covers_every_probe_and_does_not_expose_device_ids(self):
        devices = [{"uuid": "private-device-a"}, {"uuid": "private-device-b"}]
        raw = {}
        for phase, labels in report.GRAPH_WAVES.items():
            raw[phase] = {"scope": "timed-wave", "eager_steps": 0, "unclassified_steps": 0,
                          "graph_replays": 5 * len(labels),
                          "per_wave_by_gpu": {label: {"private-device-b": 3, "private-device-a": 2} for label in labels}}
        safe = report.compact_graph_proof(raw, devices)
        row = {"cuda_graphs": safe, "gpu": ["RTX 3090", "RTX 3090"]}
        self.assertTrue(report.graphs_verified(row))
        self.assertNotIn("private-device", str(safe))
        self.assertEqual(safe["decode"]["per_wave_by_rank"]["full"], [2, 3])
        for phase in report.GRAPH_WAVES:
            current = safe[phase]["graph_replays"]
            safe[phase]["graph_replays"] += 1
            self.assertFalse(report.graphs_verified(row))
            safe[phase]["graph_replays"] = current
        del raw["prefill"]["per_wave_by_gpu"]["prefill-2"]["private-device-a"]
        self.assertFalse(report.graphs_verified({**row, "cuda_graphs": report.compact_graph_proof(raw, devices)}))

    def test_kai_logits_need_phase_specific_replay_on_all_terminal_stage_ranks(self):
        gpus = ["GPU-a", "GPU-b", "GPU-c", "GPU-d"]
        devices = [{"uuid": uuid} for uuid in gpus]
        def raw(ids):
            return {phase: {"scope": "timed-wave", "eager_steps": 0, "unclassified_steps": 0,
                            "graph_replays": len(labels) * len(ids),
                            "per_wave_by_gpu": {label: {uuid: 1 for uuid in ids} for label in labels}}
                    for phase, labels in report.GRAPH_WAVES.items()}
        row = {"gpu": ["5090"] * 4, "engine": "kaiLLM", "cuda_graphs": report.compact_graph_proof(raw(gpus), devices)}
        self.assertFalse(report.graphs_verified(row))
        for stages, ranks in ((1, [0, 1, 2, 3]), (2, [2, 3]), (4, [3])):
            head = [gpus[i] for i in ranks]
            proof = report.compact_head_graph_proof(raw(head), devices, head, stages)
            row["target_head_cuda_graphs"] = proof
            self.assertTrue(report.graphs_verified(row))
            self.assertNotIn("GPU-", str(proof))
            self.assertEqual(proof["device_ranks"], ranks)
            del proof["decode"]
            self.assertFalse(report.graphs_verified(row))
            self.assertTrue(report.graph_phase_verified(row, "prefill"))
        self.assertEqual(report.compact_head_graph_proof(raw(gpus[:1]), devices, gpus[:1], 4), {})

    def test_neural_proposals_require_replay_on_each_drafter_rank(self):
        devices = [{"uuid": "a"}, {"uuid": "b"}]
        zero = {"proposals": 0, "completed_proposals": 0, "graph_replays": 0}
        full = {"proposals": 10, "completed_proposals": 10, "graph_replays": 12}
        waves = {label: {"b": zero.copy()} for label in report.GRAPH_WAVES["prefill"]}
        waves["full"] = {"b": full.copy()}
        raw = dict(full, scope="timed-wave", eager_steps=0, unclassified_steps=0, per_wave_by_gpu=waves)
        row = {"cuda_graphs": graph_proof(), "gpu": ["RTX 3090"], "mode": "dspark5"}
        self.assertFalse(report.graphs_verified(row))
        proof = report.compact_draft_graph_proof(raw, devices, ["b"])
        self.assertEqual(proof["device_ranks"], [1])
        target = graph_proof()
        for part in target.values():
            for counts in part["per_wave_by_rank"].values():
                counts.append(3)
            part["graph_replays"] *= 2
        row.update(gpu=["RTX 3090", "RTX 3090"], cuda_graphs=target, drafter_cuda_graphs=proof)
        self.assertTrue(report.graphs_verified(row))
        proof["per_wave_by_rank"]["full"][0]["completed_proposals"] -= 1
        self.assertFalse(report.graphs_verified(row))
        self.assertEqual(report.compact_draft_graph_proof(raw, devices, ["unallocated-gpu"]), {})

    def test_family_navigation_and_pending_checkpoints(self):
        by_gpu, by_model = tables([], gpu_filename="custom.md")
        self.assertIn("[Browse by GPU](custom.md)", by_model)
        for family, title in FAMILIES.items():
            self.assertEqual(by_model.count(f"[{title}](#model-{family})"), 1)
            self.assertEqual(by_model.count(f'<a id="model-{family}"></a>\n\n## {title}'), 1)
        for key, model in MANIFEST["models"].items():
            self.assertIn(f"<details>\n{checkpoint_summary(key)}\n\n"
                          f"[{model['repo']} ({model['gguf_type']})]({report.checkpoint_url(model)})\n\n"
                          "Full-length timings pending.", by_model)
        self.assertFalse(displayed_results(by_model, True))
        self.assertFalse(displayed_results(by_gpu, False))

    def test_existing_cli_refreshes_both_files_on_every_run(self):
        speed = Path(report.__file__).resolve().parents[2] / "speed.md"
        _, heading, appendix = speed.read_text().partition(report.EARLIER_BENCHMARKS)
        self.assertTrue(appendix)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            primary = directory / "GPU results.md"
            secondary = directory / "speed-by-model.md"
            args = [sys.executable, report.__file__, str(directory / "results.jsonl"), str(primary)]
            for _ in range(2):
                primary.write_text("stale")
                secondary.write_text("stale")
                subprocess.run(args, check=True)
                gpu_view = primary.read_text()
                self.assertNotIn("speed-by-model.md", gpu_view)
                self.assertEqual(gpu_view.count(heading), 1)
                self.assertTrue(gpu_view.endswith(heading + appendix))
                self.assertIn("[Browse by GPU](GPU%20results.md)", secondary.read_text())


if __name__ == "__main__":
    unittest.main()
