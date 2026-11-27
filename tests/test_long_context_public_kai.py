import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks/long_context"))
from published_kai import CLIENT, RELEASE, SOURCE, full_samples, verified
from published_llama import timing_signature


def snapshot(time, generated):
    return {
        "reading": (
            f'{time}\n{{"stage":"code","state":"running"}}\ndevice\nNVIDIA GPU, GPU-one, 595, 300\n'
            "gpu_processes\nGPU-one, 201, VLLM::EngineCore\nowned_by_public_worker\n201 1280 200\nmetrics\n"
            'vllm:num_requests_running{model_name="bench"} 4\nvllm:num_preemptions_total{model_name="bench"} 0\n'
            f'vllm:generation_tokens_total{{model_name="bench"}} {generated}\n'
        )
    }


class PublicKaiTest(unittest.TestCase):
    def test_full_samples_verify_process_ownership_concurrency_and_progress(self):
        samples = [snapshot(10, 30), snapshot(20, 50)]
        self.assertEqual(full_samples(samples, "code", 5, 25, "GPU-one", 4), (2, ((201, 1280, 200),)))
        for wrong in (
            samples[1]["reading"].replace("GPU-one, 201", "GPU-other, 201"),
            samples[1]["reading"].replace("201 1280 200", "202 1280 200"),
            samples[1]["reading"].replace("201 1280 200", "201 1281 200"),
            samples[1]["reading"].replace(
                'preemptions_total{model_name="bench"} 0', 'preemptions_total{model_name="bench"} 1'
            ),
            samples[1]["reading"].replace(" 50\n", " 12\n"),
        ):
            with self.subTest(wrong=wrong), self.assertRaises(ValueError):
                full_samples([samples[0], {"reading": wrong}], "code", 5, 25, "GPU-one", 4)
        with self.assertRaises(ValueError):
            full_samples(samples, "code", 5, 25, "GPU-one", 8)

    def test_shareable_release_receipt_binds_the_full_timing(self):
        h = "a" * 64
        row = dict(
            gpu=["NVIDIA GPU"],
            hardware_id="board",
            engine="kaiLLM",
            mode="plain",
            checkpoint="lfm-suisui",
            kai_source=CLIENT,
            engine_revision=SOURCE,
            batch=4,
            domain="prose",
            full_seconds=575,
            sha256=[h],
        )
        row["published_kai"] = dict(
            source_lock=SOURCE,
            release_sha256=RELEASE,
            hardware_id="board",
            measured_gguf_sha256=[h],
            timing_signature=timing_signature(row),
            graph_capture_logged=True,
            phase_replay_counts=None,
            preemptions=0,
            full_wave_samples=3,
            max_active_requests=4,
            receipt_sha256=h,
            process_samples_sha256=h,
            launch_preview_sha256=h,
            server_log_sha256=h,
        )
        self.assertTrue(verified(row))
        for change in (
            {"kai_enforce_eager": True},
            {"mode": "ngram3"},
            {"batch": 2},
            {"domain": "code"},
            {"full_seconds": 560},
        ):
            self.assertFalse(verified({**row, **change}))
        self.assertFalse(verified({**row, "published_kai": {**row["published_kai"], "full_wave_samples": 0}}))
