"""Check admission order, exact token coverage and score matching for two concurrent requests."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks/long_context"))
from verify_batch_schedule import append_step, capture, compare, joint_decode_steps, prompt_hash, serve, signature, valid_scores
from unittest.mock import patch


def step(admissions, scheduled):
    return {"admissions": admissions, "scheduled": scheduled}


class BatchedNumerics(unittest.TestCase):
    def setUp(self):
        self.hashes = [prompt_hash([1, 2]), prompt_hash([3, 4])]
        a, b = self.hashes
        self.split = [step([("a", a)], [("a", 2)]), step([("b", b)], [("a", 1), ("b", 2)]),
                      step([], [("b", 1)])]
        self.join = [step([("left", a), ("right", b)], [("left", 2), ("right", 2)]),
                     step([], [("left", 1), ("right", 1)])]

    def test_admission_and_decode_order_are_preserved(self):
        split = signature(self.split, self.hashes, [2, 2], 2)
        joint = signature(self.join, self.hashes, [2, 2], 2)
        self.assertEqual(split, [[[0], [[0, 2]]], [[1], [[0, 1], [1, 2]]], [[], [[1, 1]]]])
        self.assertEqual(joint, [[[0, 1], [[0, 2], [1, 2]]], [[], [[0, 1], [1, 1]]]])
        legacy = [{"new": dict(s["admissions"]), "runs": s["scheduled"], "total": sum(n for _, n in s["scheduled"])}
                  for s in self.join]
        self.assertEqual(signature(legacy, self.hashes, [2, 2], 2), joint)
        legacy[0]["total"] -= 1
        with self.assertRaisesRegex(ValueError, "counts disagree"):
            signature(legacy, self.hashes, [2, 2], 2)

    def test_joint_decode_excludes_combined_prefills_and_mixed_steps(self):
        joined = signature(self.join, self.hashes, [2, 2], 2)
        split = signature(self.split, self.hashes, [2, 2], 2)
        self.assertEqual(joint_decode_steps(joined, 2, 2), 1)
        self.assertEqual(joint_decode_steps(split, 2, 2), 0)

    def test_nonfinite_scores_never_establish_parity(self):
        top = {str(i): -float(i) for i in range(8)}
        self.assertTrue(valid_scores(0, top))
        for bad in (float("nan"), float("inf"), None):
            self.assertFalse(valid_scores(0, {**top, "7": bad}))

    def test_four_rows_require_a_full_four_request_forward(self):
        hashes = [prompt_hash([i]) for i in range(4)]
        requests = [(f"r{i}", digest) for i, digest in enumerate(hashes)]
        counts = [(name, 1) for name, _ in requests]
        four = [step(requests, counts), step([], counts)]
        self.assertEqual(signature(four, hashes, [1] * 4, 2)[0][0], [0, 1, 2, 3])
        pairs = [step(requests[:2], counts[:2]), step([], counts[:2]),
                 step(requests[2:], counts[2:]), step([], counts[2:])]
        with self.assertRaisesRegex(ValueError, "never shared"):
            signature(pairs, hashes, [1] * 4, 2)

    def test_incomplete_or_foreign_work_fails(self):
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            signature(self.split[:-1], self.hashes, [2, 2], 2)
        with self.assertRaisesRegex(ValueError, "Unexpected scheduled request"):
            signature([step([], [("unknown", 1)])], self.hashes, [2, 2], 2)

    def test_observer_hashes_prompts_and_opaque_request_ids(self):
        with tempfile.TemporaryDirectory() as folder:
            trial = SimpleNamespace(num_scheduled_tokens={"request-123": 2}, total_num_scheduled_tokens=2,
                                    scheduled_new_reqs=[SimpleNamespace(req_id="request-123", prefill_token_ids=[1, 2])])
            out = Path(folder) / "steps.jsonl"
            append_step(trial, out)
            row = json.loads(out.read_bytes())
            self.assertEqual(row["admissions"][0][1], self.hashes[0])
            self.assertEqual(row["admissions"][0][0], row["scheduled"][0][0])
            self.assertNotIn("request-123", out.read_text())

    def test_server_preparation_keeps_the_real_deployment_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            hooks = directory / "deployment"
            hooks.mkdir()
            (hooks / "sitecustomize.py").write_text('"""Normal deployment startup."""')
            model = directory / "test.gguf"
            model.write_bytes(b"gguf")
            argv = ["/trusted/python", "-m", "vllm.entrypoints.cli.main", "serve", str(model),
                    "--port", "12345", "--max-num-seqs", "2"]
            response = {"command": argv, "environment": {"PYTHONPATH": str(hooks)}}
            output = directory / "qa"
            args = SimpleNamespace(directory=output, command=["--", "deployment", str(model)])
            with patch("subprocess.check_output", side_effect=[json.dumps(response), '["gpu-a"]']), \
                 patch("os.execve") as execute:
                serve(args)
            self.assertEqual(execute.call_args.args[:2], (argv[0], argv))
            identity = json.loads((output / "launch.json").read_bytes())
            self.assertEqual(identity["arguments"], ["--max-num-seqs", "2"])
            self.assertEqual(identity["cuda_uuid"], "gpu-a")
            self.assertEqual(execute.call_args.args[2]["PYTHONPATH"].split(":")[-1], str(hooks))

    def test_capture_writes_real_forward_evidence_and_cleans_recording_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "launch.json").write_text('{"cuda_uuid":"a"}')
            bank = folder / "bank.json"
            bank.write_text(json.dumps({"domains": {"code": {"rows": [[1, 2], [3, 4]]}}}))
            args = SimpleNamespace(directory=folder, bank=bank, domain="code", rows=(0, 1), length=2,
                                   steps=2, attempts=2, both_admissions=True, model="bench", url="local",
                                   timeout=3, output=folder / "scores.json")
            schedules = iter((self.split, self.join))
            def generate(*args):
                self.assertTrue((folder / "recording").is_file())
                with (folder / "steps.jsonl").open("a") as log:
                    for item in next(schedules):
                        print(json.dumps(item), file=log)
                return [{"tokens": [0, 0], "scores": [{str(i): -float(i) for i in range(8)}] * 2}] * 2
            with patch("verify_batch_schedule.request_scores", side_effect=generate):
                capture(args)
            self.assertFalse((folder / "recording").exists())
            receipt = json.loads(args.output.read_bytes())
            self.assertEqual(compare(receipt, receipt, True)["matching_schedules"], 2)

    def test_distinct_score_sets_can_both_be_exact_under_their_own_schedule(self):
        scores = {str(i): float(-i) for i in range(8)}
        def result(rows, token):
            answers = [{"tokens": [token, token], "scores": [dict(scores), dict(scores)]} for _ in range(2)]
            return {"schedule": signature(rows, self.hashes, [2, 2], 2), "answers": answers}
        first = {"prompt_sha256": self.hashes, "domain": "code", "length": 2, "generated": 2,
                 "cases": [result(self.split, 0), result(self.join, 1)]}
        second = copy.deepcopy(first)
        second["cases"].reverse()
        self.assertEqual(compare(first, second, True)["matching_schedules"], 2)
        second["cases"][0]["answers"][0]["scores"][1]["1"] -= .25
        with self.assertRaisesRegex(ValueError, "Numerical disagreement"):
            compare(first, second, True)
        second["cases"].pop(0)
        with self.assertRaisesRegex(ValueError, "Insufficient matching"):
            compare(first, second, True)
        second["cases"][0]["answers"][0]["scores"][0] = {}
        with self.assertRaisesRegex(ValueError, "Incomplete scores"):
            compare(first, second)


if __name__ == "__main__":
    unittest.main()
