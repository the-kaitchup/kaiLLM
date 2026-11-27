"""Check GPU capacity evidence without requiring a GPU or starting a model."""
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks/long_context"))
import report
import worker


SHORT = ("To serve at least one request with the model's max seq len (73728), "
         "(4.69 GiB KV cache is needed, which is larger than the available KV cache memory (2.95 GiB).")
CAPACITY = "Maximum concurrency for 73,728 tokens per request: {}x"


class LongContextCapacityTest(unittest.TestCase):
    def test_allocation_failure_belongs_to_the_measured_reference(self):
        stamp = dict(state="failed", stage="startup", gpu_names=["RTX 3090"],
                     checkpoint="inkling-loop-iq4", engine="llama.cpp", mode="plain",
                     batch=1, reason="CUDA out of memory")
        pins = report.MANIFEST["llama_cpp"]
        self.assertIsNone(report.confirmed_limit(stamp))
        self.assertIsNone(report.confirmed_limit({**stamp, "llama_source": pins["revision"]}))
        confirmed = report.confirmed_limit({**stamp, "llama_source": pins["inkling_revision"]})
        self.assertEqual(confirmed["llama_source"], pins["inkling_revision"])

    def test_insufficient_kv_increases_allocation_before_timing(self):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "server.log"
            cmd = ["server", "--kv-cache-memory-bytes", str(3 * worker.GIB)]
            sizes = []

            def readiness(proc, url, path):
                sizes.append(int(cmd[-1]) / worker.GIB)
                if len(sizes) == 1:
                    path.write_text(SHORT)
                    raise RuntimeError(worker.startup_failure(SHORT))
                path.write_text(CAPACITY.format("1.01"))

            with patch.object(worker.subprocess, "Popen") as processes, patch.object(worker, "stop_server") as stop, \
                    patch.object(worker, "wait_server", side_effect=readiness):
                server = worker.start_server(cmd, {}, "", log, 1, True, 16 * worker.GIB)
            self.assertIs(server, processes.return_value)
            self.assertEqual(sizes, [3.0, 4.75])
            stop.assert_called_once()
            self.assertEqual(log.with_suffix(".attempt1.log").read_text(), SHORT)

    def test_growth_shrinkage_and_fixed_context(self):
        gib = worker.GIB
        self.assertEqual(worker.next_kv_allocation(CAPACITY.format("2"), 8*gib, 4, 15*gib, False), 15*gib)
        self.assertEqual(worker.next_kv_allocation(CAPACITY.format("20"), 10*gib, 5, 16*gib, True), 2.75*gib)
        self.assertIsNone(worker.next_kv_allocation("CUDA out of memory", 8*gib, 1, 16*gib, True))
        self.assertIsNone(worker.next_kv_allocation(CAPACITY.format("6"), 8*gib, 4, 16*gib, False))
        measured = "GPU KV cache size: 74,480 tokens, " + CAPACITY.format("1.01")
        self.assertEqual(worker.next_kv_allocation(measured, 4.75*gib, 1, 16*gib, True), 4.7109375*gib)

    def test_rounded_slot_count_cannot_admit_an_underfilled_cache(self):
        under = "GPU KV cache size: 73,696 tokens, " + CAPACITY.format("1.00")
        full = "GPU KV cache size: 73,728 tokens, " + CAPACITY.format("1.00")
        other_group = "GPU KV cache size: 120,000 tokens, " + CAPACITY.format("1.00")
        self.assertLess(worker.reported_capacity(under), 1)
        self.assertEqual(worker.reported_capacity(full), 1)
        self.assertEqual(worker.reported_capacity(other_group), 1)
        self.assertEqual(worker.reported_capacity(CAPACITY.format("1.00")), 1)
        self.assertEqual(worker.reported_capacity(full + "\n" + under), 73696 / 73728)
        self.assertGreater(worker.next_kv_allocation(under, 4*worker.GIB, 1, 16*worker.GIB, False),
                           4*worker.GIB)

    def test_legacy_log_requires_matching_receipt(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "test.log"
            text = "failed to allocate CUDA0 buffer of size 100\n" + "failed to create context\n" * 6
            p.write_text(text)
            stamp = dict(state="failed", stage="startup", gpu_names=["RTX 5080"], checkpoint="lfm-qad",
                         engine="llama.cpp", mode="plain", batch=1, reason=worker.startup_tail(text), time=time.time())
            self.assertEqual(report.confirmed_limit(stamp, p)["reason"], "Memory allocation failed at the requested context")
            self.assertIsNone(report.confirmed_limit({**stamp, "time": 0}, p))
            self.assertIsNone(report.confirmed_limit({**stamp, "reason": "different failure"}, p))
            for reason in ("Insufficient VRAM after weights and working memory", "Full-context KV capacity 3.84 sequences < batch 4",
                           worker.startup_failure(SHORT)):
                self.assertIsNone(report.confirmed_limit({**stamp, "reason": reason}, p))
            self.assertIn("hy_v4_flash", worker.startup_failure("error loading model: unknown model architecture: 'hy_v4_flash'"))
