import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks/long_context"))
from published_llama import (
    CLIENT,
    MANIFEST,
    libraries,
    native_draft,
    native_proof_matches,
    process,
    startup_attempts,
    timing_signature,
    verified,
)


class PublishedLlamaTest(unittest.TestCase):
    def setUp(self):
        self.sha = "3" * 64
        self.rev = "5" * 40
        self.binary = "/build/bin/llama-server"
        self.libs = ["libllama", "libggml", "libggml-base", "libggml-cpu", "libggml-cuda"]
        head = [
            self.rev,
            "CMAKE_CUDA_ARCHITECTURES:UNINITIALIZED=120",
            "GGML_CUDA:BOOL=ON",
            "GGML_CUDA_GRAPHS:BOOL=ON",
        ]
        maps = []
        for i, n in enumerate(["llama-server", *(x + ".so.1" for x in self.libs)]):
            path = "/build/bin/" + n
            head.extend([f"2b {100 + i} 30 {path}", f"{self.sha}  {path}"])
            if i:
                maps.append(f"00:2b {100 + i} {path}")
        self.build = "\n".join(head)
        self.command = [self.binary, "-m", "/model.gguf"]
        self.device = "GPU-unit"
        self.observe = "\n".join(
            [
                "1400",
                "pid=40",
                "start_ticks=31",
                *self.command,
                "policy_override_count",
                "0",
                "library_maps",
                *maps,
                "gpu_processes",
                "GPU-unit, 40",
                "status",
                json.dumps({"stage": "prose", "state": "running"}),
            ]
        )

    def test_original_binary_and_loaded_cuda_inodes_must_match(self):
        files = libraries(self.build, self.binary, self.rev)
        when, pid, ticks, observed, stage = process(self.observe, self.command, self.device, files)
        self.assertEqual((when, pid, ticks, stage["stage"]), (1400, 40, 31, "prose"))
        self.assertEqual(len(observed), 5)
        for changed in (
            self.observe.replace("GPU-unit, 40", "GPU-unit, 49"),
            self.observe.replace("00:2b 101", "00:2b 102"),
            self.observe.replace("override_count\n0", "override_count\n1"),
            self.observe.replace("/model.gguf", "/other.gguf"),
        ):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                process(changed, self.command, self.device, files)
        with self.assertRaises(ValueError):
            libraries(
                self.build.replace("GGML_CUDA_GRAPHS:BOOL=ON", "GGML_CUDA_GRAPHS:BOOL=OFF"),
                self.binary,
                self.rev,
            )

    def test_smaller_microbatch_requires_every_larger_compute_allocation_failure(self):
        failure = (
            "ggml_backend_cuda_buffer_type_alloc_buffer: cudaMalloc failed: out of memory\n"
            "ggml_gallocr_reserve_n_impl: failed to allocate CUDA0 buffer of size 7671716096\n"
            "graph_reserve: failed to allocate compute buffers\n"
        )
        with tempfile.TemporaryDirectory() as temp:
            area = Path(temp)
            (area / "selected-ub").write_text("128")
            (area / "startup-ub128.log").write_text("server ready")
            for ub in (512, 256):
                (area / f"startup-ub{ub}.log").write_text(failure)
            chosen, attempts = startup_attempts(area)
            self.assertEqual(chosen, 128)
            self.assertEqual([a["microbatch"] for a in attempts], [512, 256, 128])
            self.assertEqual(attempts[0]["requested_bytes"], 7671716096)
            (area / "startup-ub256.log").write_text("cudaMalloc failed: out of memory")
            with self.assertRaises(ValueError):
                startup_attempts(area)

    def test_native_checks_require_both_greedy_agreement_and_full_run_proposals(self):
        draft = MANIFEST["dspark"]
        ref = {key: draft[key] for key in ("repo", "revision", "files")}
        row = dict(
            checkpoint="inkling-udq2", model_revision=MANIFEST["models"]["inkling-udq2"]["revision"],
            mode="dspark5", drafter={k: ref[k] for k in ("repo", "revision")}
        )
        full = {"requests": [{"server_timings": {"draft_n": 125, "draft_n_accepted": 53}}]}
        with tempfile.TemporaryDirectory() as temp:
            area = Path(temp)
            (area / "drafter-download.json").write_text(json.dumps({"source": "huggingface.co", **ref}))
            slots = [[d, l, self.sha, 0, 0] for d in ("prose", "code") for l in (128, 511, 2048, 8192)]

            def write(mode, rows):
                (area / f"parity-{mode}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

            write("plain", slots)
            proposed = [[d, l, h, 7, 3] for d, l, h, _, _ in slots]
            write("dspark5", proposed)
            proof = native_draft(area, row, full)
            self.assertTrue(native_proof_matches(row, proof))
            self.assertFalse(native_proof_matches(row, {**proof, "measured_drafter_sha256": "0" * 64}))
            self.assertFalse(
                native_proof_matches(
                    row,
                    {
                        **proof,
                        "speculative": {"scope": "full-wave", "proposed": 0, "rejected": 0, "accepted": 0},
                    },
                )
            )
            proposed[2][2] = "0" * 64
            write("dspark5", proposed)
            divergent = native_draft(area, row, full)
            self.assertEqual(divergent["greedy_matching_outputs"], 7)
            self.assertTrue(native_proof_matches(row, divergent))
            proposed[2][1] = 4096
            write("dspark5", proposed)
            with self.assertRaises(ValueError):
                native_draft(area, row, full)

    def test_shareable_proof_binds_workload_and_real_process_sample(self):
        row = dict(
            engine="llama.cpp",
            mode="plain",
            gpu=["Card"],
            kai_source=CLIENT,
            llama_source=self.rev,
            hardware_id="board",
            sha256=[self.sha],
            timestamp_utc="2020-01-01T00:01:00Z",
            batch=2,
            domain="code",
            full_seconds=120,
            input_tokens_per_request=8192,
        )
        proof = dict(
            published_client=CLIENT,
            policy="default",
            upstream=self.rev,
            measured_gguf_sha256=row["sha256"],
            hardware_id="board",
            device_count=1,
            domain="code",
            batch=2,
            completed_utc=row["timestamp_utc"],
            microbatch=512,
            observer_configured=False,
            environment_overrides=[],
            full_wave_samples=2,
            executable_sha256=self.sha,
            timing_receipt_sha256=self.sha,
            process_samples_sha256=self.sha,
            client_receipt_sha256=self.sha,
            build_receipt_sha256=self.sha,
            timing_signature=timing_signature(row),
            mapped_dso_sha256={name + ".so.1": self.sha for name in self.libs},
        )
        self.assertTrue(verified(row, proof))
        for update in (
            {"mode": "ngram3"},
            {"domain": "prose"},
            {"hardware_id": "other"},
            {"full_seconds": 12},
        ):
            self.assertFalse(verified({**row, **update}, proof))
        for update in (
            {"full_wave_samples": 0},
            {"microbatch": 256},
            {"observer_configured": True},
            {"environment_overrides": ["GGML_CUDA_DISABLE_GRAPHS"]},
            {"mapped_dso_sha256": {}},
        ):
            self.assertFalse(verified(row, {**proof, **update}))
