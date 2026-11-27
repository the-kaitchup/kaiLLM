"""Keep qualified Inkling timings, their matched controls, and earlier builds distinct."""
from copy import deepcopy
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks/long_context"))
from benchmark import MANIFEST, OUTPUT, PROMPT, PROTOCOL, inkling_draft_runtime, model_at_revision, runtime_pin
from coverage_matrix import counts
from devices import physical_id
from matrix import modes
from report import current_limit_build, main as render_reports, tables, verify_kai_native_record


def graph_proof():
    waves = {"prefill": ("prefill-1", "prefill-2", "prefill-3", "full"), "decode": ("full",)}
    return {phase: {"scope": "timed-wave", "graph_replays": 3 * len(labels), "eager_steps": 0, "unclassified_steps": 0,
                    "per_wave_by_rank": {label: [3] for label in labels}} for phase, labels in waves.items()}


def head_proof():
    return {**graph_proof(), "pipeline_stages": 1, "device_ranks": [0]}


def draft_graph_proof():
    counts = {"proposals": 2, "completed_proposals": 2, "graph_replays": 3}
    waves = {label: [{key: 0 for key in counts}] for label in ("prefill-1", "prefill-2", "prefill-3")}
    waves["full"] = [counts]
    return dict(counts, scope="timed-wave", eager_steps=0, unclassified_steps=0, device_ranks=[0], per_wave_by_rank=waves)


class InklingNativeTest(unittest.TestCase):
    def setUp(self):
        pins = MANIFEST["kaillm"]
        ref = MANIFEST["dspark"]
        name, = ref["files"]
        model = MANIFEST["models"]["inkling-loop-iq4"]
        self.plain = dict(checkpoint="inkling-loop-iq4", model_revision=model["revision"],
                          sha256=[f["sha256"] for f in model["files"].values()], mode="plain", batch=1, gpu=["RTX 3090"],
                          kai_source=pins["inkling_dspark_revision"],
                          engine_revision=pins["runtime_locks"]["inkling_dspark"]["source"],
                          llama_source=MANIFEST["llama_cpp"]["inkling_revision"], kai_humming_deterministic=True)
        self.native = {**self.plain, "mode": "dspark5", "drafter_cuda_graphs": draft_graph_proof(),
                       "kai_dspark": dict(scope="full-wave", proposed=68, accepted=42, rejected=26),
                       "drafter": dict(repo=ref["repo"], revision=ref["revision"],
                                       filename=name, sha256=ref["files"][name]["sha256"])}

    def test_only_the_qualified_target_and_slot_count_enable_native_kai(self):
        self.assertIn("dspark5", modes("inkling-loop-iq4", "kaiLLM", 1))
        self.assertNotIn("dspark5", modes("inkling-loop-iq4", "kaiLLM", 2))
        self.assertNotIn("dspark5", modes("inkling-udq2", "kaiLLM", 1))
        self.assertIn("dspark5", modes("inkling-loop-iq4", "llama.cpp", 2))

    def test_graph_qualification_is_limited_to_its_card_source_and_slot(self):
        model = MANIFEST['models']['inkling-udq2']
        placement = dict(revision='candidate', source='test-lock', gpu='NVIDIA GeForce RTX 5080', matrix_gpu='5080',
                         batch=1, prefill_bias_mib=24, kv_mib=3764, hardware_id=physical_id(['example']),
                         compact_graphs=True, serial_verify_post=True, serial_verify_bias=True, serial_verify_route=True, capture_sizes=[1, 2, 4, 6, 128, 256, 512, 1024, 2048])
        with patch.dict(model, kai_dspark_placements=[]):
            base_cells = counts([], [])['planned']
        with patch.dict(model, kai_dspark_placements=[placement]):
            self.assertIn('dspark5', modes('inkling-udq2', 'kaiLLM', 1, gpu='5080'))
            self.assertNotIn('dspark5', modes('inkling-udq2', 'kaiLLM', 2, gpu='5080'))
            self.assertNotIn('dspark5', modes('inkling-udq2', 'kaiLLM', 1, gpu='5090'))
            self.assertEqual(counts([], [])['planned'], base_cells + 2)
            hashes = [f['sha256'] for f in model['files'].values()]
            row = dict(self.native, checkpoint='inkling-udq2', model_revision=model['revision'], sha256=hashes,
                       measured_gguf_sha256=hashes, measured_drafter_sha256=self.native['drafter']['sha256'],
                       engine='kaiLLM', gpu=[placement['gpu']], hardware_id=placement['hardware_id'],
                       kai_source='candidate', engine_revision='test-lock', kai_inkling_prefill_bias_mib=24,
                       kai_inkling_capture_sizes=placement['capture_sizes'], kai_inkling_serial_verify_post=True, kai_inkling_serial_verify_bias=True, kai_inkling_serial_verify_route=True,
                       kai_kv={'bytes_per_device': 3764 << 20}, cuda_graphs=graph_proof(),
                       target_head_cuda_graphs=head_proof())
            verify_kai_native_record(row)
            for change in ({'gpu': ['NVIDIA GeForce RTX 5090']}, {'hardware_id':'another-board'},
                           {'kai_kv': {'bytes_per_device': 3840 << 20}}, {'kai_inkling_prefill_bias_mib':64},
                           {'measured_drafter_sha256':None}, {'measured_gguf_sha256':None}, {'kai_inkling_serial_verify_post':False},
                           {'kai_inkling_serial_verify_post':1}, {'kai_inkling_serial_verify_bias':False}, {'kai_inkling_serial_verify_bias':1},
                           {'kai_inkling_serial_verify_route':False}, {'kai_inkling_serial_verify_route':1},
                           {'target_head_cuda_graphs':{}}, {'drafter_cuda_graphs':{}}, {'kai_inkling_capture_sizes':[1,2048]}):
                with self.subTest(change=change), self.assertRaises(ValueError):
                    verify_kai_native_record(dict(row, **change))

    def test_verify_source_humming_and_drafter_identity(self):
        verify_kai_native_record(self.plain)
        verify_kai_native_record(self.native)
        wrong_sha = {**self.native["drafter"], "sha256": "wrong"}
        for overrides in ({"kai_humming_deterministic": False}, {"batch": 2}, {"engine_revision": "other"},
                          {"drafter": wrong_sha}, {"checkpoint": "inkling-udq2"}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                verify_kai_native_record({**self.native, **overrides})

    def test_original_checkpoint_preserves_its_own_historical_runtime(self):
        key = "inkling-udq2"
        revision, = MANIFEST["model_history"][key]
        original = model_at_revision(key, revision)
        legacy_pin = runtime_pin("inkling_legacy")
        current = MANIFEST["models"][key]
        self.assertNotIn("kai_loader_runtime", current)
        self.assertIsNone(inkling_draft_runtime(current, 1))
        with patch.dict(original, kai_dspark_batches=[1]):
            native_pin = inkling_draft_runtime(original, 1)
            shared = dict(cuda_graphs=graph_proof(), target_head_cuda_graphs=head_proof(), driver=["570"],
                          power_limit_w=["350"], hardware_id="test", domain="prose", prompt_set_sha256="public-bank",
                          prefill_seconds=[10.0] * 3, full_seconds=1000.0, generated_text_hashes=["same"],
                          protocol=PROTOCOL, input_tokens_per_request=PROMPT, output_tokens_per_request=OUTPUT,
                          timestamp_utc="test", sha256=[file["sha256"] for file in original["files"].values()])
            old = {**self.native, **shared, "checkpoint": key, "model_revision": revision, "engine": "kaiLLM",
                   "kai_source": native_pin[0], "engine_revision": native_pin[1]}
            verify_kai_native_record(old)
            for other in (legacy_pin, inkling_draft_runtime(MANIFEST["models"]["inkling-loop-iq4"], 1)):
                with self.assertRaises(ValueError):
                    verify_kai_native_record({**old, "kai_source": other[0], "engine_revision": other[1]})
            self.assertFalse(current_limit_build(old))
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                archive = root / "results.jsonl"
                archive.write_text(json.dumps(old) + "\n")
                with patch.object(sys, "argv", ["report.py", str(archive), str(root / "speed.md")]):
                    render_reports()
                self.assertEqual(json.loads(archive.read_text())["model_revision"], revision)
                for name in ("speed.md", "speed-by-model.md"):
                    self.assertNotIn("| 1 | dspark5", (root / name).read_text())

    def test_counter_evidence_keeps_its_scope(self):
        observed = deepcopy(self.native)
        observed["kai_dspark"].update(scope="observed-request-interval", observed_generated_tokens=62311)
        verify_kai_native_record(observed)
        for change in ({"observed_generated_tokens": 65537}, {"rejected": 22}, {"scope": "full-wave"}):
            broken = deepcopy(observed)
            broken["kai_dspark"].update(change)
            with self.assertRaises(ValueError):
                verify_kai_native_record(broken)

    def test_historical_control_and_deterministic_pair_render_separately(self):
        pins = MANIFEST["kaillm"]
        workload = dict(cuda_graphs=graph_proof(), target_head_cuda_graphs=head_proof(), driver=["570"], power_limit_w=["350"], hardware_id="example", domain="prose",
                        prompt_set_sha256="same", prefill_seconds=[10.0] * 3, full_seconds=1000.0,
                        generated_text_hashes=["current"])
        native = {**workload, **self.native, "engine": "kaiLLM"}
        plain = {**workload, **self.plain, "engine": "kaiLLM"}
        old = {**plain, "kai_source": pins["default_revision"], "engine_revision": "earlier",
               "generated_text_hashes": ["earlier"]}
        del old["kai_humming_deterministic"]
        llama = {**workload, **self.plain, "engine": "llama.cpp", "kai_source": pins["default_revision"]}
        del llama["kai_humming_deterministic"]
        gpu, model = tables([native, old, plain, llama])
        for report in (gpu, model):
            self.assertIn("earlier kaiLLM · Humming unrecorded", report)
            self.assertIn("| 1 | plain (deterministic kaiLLM · Inkling PR) | prose | 819.2 | 819.2 |", report)
            self.assertIn("| 1 | dspark5 (deterministic kaiLLM · Inkling PR) | prose | 819.2 | — | 66.2 |", report)

        current = MANIFEST["models"]["inkling-udq2"]
        ordinary = {**plain, "checkpoint": "inkling-udq2", "batch": 2, "model_revision": current["revision"],
                    "sha256": [file["sha256"] for file in current["files"].values()]}
        for view in tables([ordinary]):
            self.assertIn("| 2 | plain (Inkling PR) | prose | 1638.4 | — |", view)

        # A registration on another board of the same GPU type does not relabel
        # an earlier ordinary row or claim its reference in either report view.
        same_sku = dict(gpu=ordinary['gpu'][0], matrix_gpu='5080', hardware_id='second-board', batch=2,
                        model_revision=current['revision'], revision='separately-qualified', source='distinct-lock')
        with patch.dict(current, kai_dspark_placements=[same_sku]):
            for view in tables([ordinary]):
                self.assertIn("| 2 | plain (Inkling PR) | prose | 1638.4 | — |", view)
                self.assertNotIn('earlier kaiLLM', view)
