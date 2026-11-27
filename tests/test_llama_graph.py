"""Check the llama.cpp build admission and cross-DSO, per-device CUDA graph proof wire format."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parents[1] / "benchmarks/long_context"
sys.path.insert(0, str(HERE))
import llama_graph as proof


class GraphEvidenceTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.p = self.root / "counter"
        self.rev, self.sources = next(
            iter(json.loads(proof.SOURCE_LOCK.read_bytes()).items())
        )
        for name in proof.BINARY_NAMES:
            (self.root / name).write_bytes(name.encode())
        self.cert = {
            "protocol": proof.VERSION,
            "upstream": self.rev,
            "source_sha256": self.sources,
            "binary_sha256": {
                n: proof.digest(self.root / n) for n in proof.BINARY_NAMES
            },
        }
        (self.root / proof.CERTIFICATE).write_text(json.dumps(self.cert))

    def prepare(self):
        return proof.prepare(
            self.root / "llama-server",
            self.p,
            self.rev,
            2,
            base_env={"LD_LIBRARY_PATH": "/lib"},
        )

    def test_rejects_missing_library_or_source_before_launch(self):
        (self.root / "libggml-cuda.so").write_text("stale")
        with self.assertRaisesRegex(ValueError, "certificate"):
            self.prepare()
        self.assertFalse(self.p.exists())
        (self.root / "libggml-cuda.so").write_text("libggml-cuda.so")
        self.cert["source_sha256"] = {}
        (self.root / proof.CERTIFICATE).write_text(json.dumps(self.cert))
        with self.assertRaisesRegex(ValueError, "source"):
            self.prepare()
        self.assertFalse(self.p.exists())

    def test_file_is_private_unique_and_initialized(self):
        self.assertEqual(
            self.prepare(),
            {
                "KAILLM_LLAMA_GRAPH_FILE": str(self.p),
                "LD_LIBRARY_PATH": str(self.root) + ":/lib",
            },
        )
        self.assertEqual(self.p.stat().st_mode & 0o777, 0o600)
        with self.assertRaises(FileExistsError):
            self.prepare()
        state = proof.snapshot(self.p)
        self.assertEqual(state["device_count"], 2)
        self.assertTrue(
            all(
                all(not any(item.values()) for item in rows)
                for rows in state["phases"].values()
            )
        )
        with self.assertRaisesRegex(ValueError, "No successful"):
            proof.verify_interval(state, state, ["decode"], ["a", "b"])

    @unittest.skipUnless(
        shutil.which("c++"), "C++ compiler required for cross-DSO counters"
    )
    def test_cpp_library_and_server_share_the_phase_for_each_gpu(self):
        env = dict(os.environ, **self.prepare())
        before = proof.snapshot(self.p)
        backend = self.root / "backend.cpp"
        backend.write_text("""#include "llama_graph.hpp"
extern "C" void batch(int gpu, bool capture) {
    using namespace kaillm_graph;
    record(gpu, batches);
    if (capture) {
        record(gpu, captures);
        record(gpu, capture_launches);
    } else {
        record(gpu, replays);
    }
}
""")
        server = self.root / "server.cpp"
        server.write_text("""#include "llama_graph.hpp"
extern "C" void batch(int, bool);
int main() {
    using namespace kaillm_graph;
    { scope prompt(prefill); batch(0, true); batch(1, true); }
    { scope both(mixed);     batch(0, false); batch(1, false); }
    { scope token(decode);   batch(0, false); batch(1, false); }
    { scope draft(proposal); batch(0, true);  batch(1, false); }
    { scope state(draft_decode); batch(0, false); batch(1, true); }
}
""")
        common = ["c++", "-std=c++17", "-I", str(HERE)]
        subprocess.run(
            [
                *common,
                "-shared",
                "-fPIC",
                str(backend),
                "-o",
                str(self.root / "libbackend.so"),
            ],
            check=True,
        )
        exe = self.root / "sample"
        subprocess.run(
            [
                *common,
                str(server),
                "-o",
                str(exe),
                "-L",
                str(self.root),
                "-lbackend",
                "-Wl,-rpath,$ORIGIN",
            ],
            check=True,
        )
        subprocess.run([str(exe)], env=env, check=True)
        after = proof.snapshot(self.p)
        actual = proof.verify_interval(before, after, ["prefill", "decode"], ["A", "B"])
        self.assertEqual(
            actual["prefill"]["A"],
            dict(
                batches=2,
                captures=1,
                capture_launches=1,
                replays=1,
                rejected=0,
                mixed_replays=1,
                graph_launches=2,
                updates=0,
                proposals=0,
                completed_proposals=0,
            ),
        )
        self.assertEqual(actual["decode"]["B"]["replays"], 2)
        expected = {"proposals": 1, "completed_proposals": 1, "graph_replays": 1}
        self.assertEqual(
            proof.verify_draft_interval(before, after, ["A", "B"], True),
            {"A": expected, "B": expected},
        )
        self.assertEqual(after["phases"]["draft_decode"][1]["capture_launches"], 1)
        # One proposal with zero GPU forwards cannot be covered by a second successful call.
        after["phases"]["proposal"][0]["proposals"] += 1
        after["phases"]["proposal"][0]["replays"] += 10
        with self.assertRaisesRegex(ValueError, "proposal graph execution on A"):
            proof.verify_draft_interval(before, after, ["A", "B"], True)
        # A new capture followed by a real CUDA launch executes the model via a graph.
        after["phases"]["mixed"][0]["replays"] = 0
        after["phases"]["mixed"][0]["batches"] = 0
        observed = proof.verify_interval(before, after, ["prefill"], ["A", "B"])
        self.assertEqual(observed["prefill"]["A"]["graph_launches"], 1)
        self.assertEqual(observed["prefill"]["A"]["replays"], 0)
        after["phases"]["prefill"][0]["capture_launches"] = 0
        with self.assertRaisesRegex(ValueError, "Missing successful"):
            proof.verify_interval(before, after, ["prefill"], ["A", "B"])

    def test_rejects_unattributed_rejected_and_active_batches(self):
        self.prepare()
        before = proof.snapshot(self.p)
        for phase, key in (("unattributed", "batches"), ("mixed", "rejected")):
            after = proof.snapshot(self.p)
            after["phases"][phase][0][key] = 1
            with self.assertRaisesRegex(ValueError, "unattributable"):
                proof.verify_interval(before, after, [], ["a", "b"])
        values = list(proof.LAYOUT.unpack(self.p.read_bytes()))
        values[3] = 1
        self.p.write_bytes(proof.LAYOUT.pack(*values))
        with self.assertRaisesRegex(ValueError, "active"):
            proof.snapshot(self.p)


if __name__ == "__main__":
    unittest.main()
