"""Compare greedy outputs of isolated kaiLLM validation servers under identical request schedules.

Use serve with the normal deployment arguments, capture the same prompt rows on each
server, and compare their receipts. This instrumentation belongs to numerical checks.
Start the normal deployment directly for throughput measurements.
"""
from __future__ import annotations

import argparse
import ast
import functools
import hashlib
import importlib.machinery
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import uuid
from urllib.request import Request, urlopen


def prompt_hash(tokens):
    return hashlib.sha256(struct.pack(f"<{len(tokens)}I", *tokens)).hexdigest()


def append_step(output, path):
    if not output.num_scheduled_tokens:
        return
    def key(request_id):
        return hashlib.sha256(request_id.encode()).hexdigest()

    admissions = [[key(request.req_id), prompt_hash(request.prefill_token_ids)]
                  for request in output.scheduled_new_reqs if request.prefill_token_ids is not None]
    counts = [(key(request), n) for request, n in output.num_scheduled_tokens.items()]
    if sum(n for _, n in counts) != output.total_num_scheduled_tokens:
        raise ValueError("Scheduled token counts disagree")
    event = {"admissions": admissions, "scheduled": counts}
    proposals = getattr(output, "scheduled_spec_decode_tokens", {})
    if proposals:
        event["draft_widths"] = [[key(request), len(tokens)] for request, tokens in proposals.items() if tokens]
    with path.open("a") as stream:
        print(json.dumps(event), file=stream)


def install_observer(directory):
    """Record scheduler metadata without reading CUDA tensors or altering the model forward."""
    marker = directory / "recording"
    trace = directory / "steps.jsonl"

    class RunnerFinder:
        @staticmethod
        def find_spec(name, path, target=None):
            if name != "vllm.v1.worker.gpu.model_runner":
                return None
            spec = importlib.machinery.PathFinder.find_spec(name, path)
            execute = spec.loader.exec_module

            def load(module):
                execute(module)
                runner = module.GPUModelRunner
                original = runner.execute_model

                @functools.wraps(original)
                def observed(self, scheduler_output, *args, **kwargs):
                    if marker.is_file() and not kwargs.get("dummy_run"):
                        append_step(scheduler_output, trace)
                    return original(self, scheduler_output, *args, **kwargs)

                runner.execute_model = observed

            spec.loader.exec_module = load
            return spec

    sys.meta_path.insert(0, RunnerFinder())


def serve(args):
    directory = args.directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    # A preparation run returns the exact catalog-verified command and its environment changes.
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or "--dry-run" in command:
        raise ValueError("Pass the deployment command used for ordinary numerical validation")
    prepared = json.loads(subprocess.check_output([*command, "--dry-run"]))
    environment = {**os.environ, **prepared["environment"]}
    roots = environment["PYTHONPATH"].split(os.pathsep)
    startup = Path(roots[0]) / "sitecustomize.py"
    body = ast.parse(startup.read_text()).body
    if body and not (len(body) == 1 and ast.get_docstring(ast.Module(body=body, type_ignores=[])) is not None):
        raise ValueError(f"The deployment has an active Python startup hook at {startup}")
    argv = prepared["command"]
    model_index = argv.index("serve") + 1
    model = Path(argv[model_index])
    if model.suffix.lower() != ".gguf" or "-of-" in model.stem:
        raise ValueError("This validator supports single-file GGUF targets")
    with model.open("rb") as stream:
        weight_digest = hashlib.file_digest(stream, "sha256").hexdigest()
    devices = Path(__file__).with_name("devices.py").resolve()
    device_query = f"import runpy,json; print(json.dumps(runpy.run_path({str(devices)!r})['visible_cuda_uuids']()))"
    uuids = json.loads(subprocess.check_output([argv[0], "-c", device_query], env=environment))
    if len(uuids) != 1:
        raise ValueError("Expose one physical CUDA GPU for each paired validation server")
    numeric_arguments = []
    values = iter(argv[model_index + 1:])
    for value in values:
        if value in ("--port", "--host", "--served-model-name"):
            next(values)
        else:
            numeric_arguments.append(value)
    identity = {"checkpoint_sha256": weight_digest, "cuda_uuid": uuids[0], "arguments": numeric_arguments,
                "humming_deterministic": environment.get("KAILLM_HUMMING_DETERMINISTIC")}
    server = {"id": uuid.uuid4().hex}
    for arg in command:
        if Path(arg).name == 'serve.py':
            stamp = Path(arg).resolve().parent.parent / 'benchmark-source.sha256'
            if stamp.is_file():
                server['runtime_source_lock'] = stamp.read_text().strip()
            break
    with (directory / "launch.json").open("x") as stream:
        json.dump(identity, stream)
    with (directory / "server-start.json").open("x") as stream:
        json.dump(server, stream)
    hook = directory / "observer"
    hook.mkdir(exist_ok=True)
    # The engine starts additional Python processes. All of them need the same import observer.
    code = ("from pathlib import Path\nimport runpy\n"
            f"runpy.run_path({str(Path(__file__).resolve())!r})['install_observer'](Path({str(directory)!r}))\n")
    (hook / "sitecustomize.py").write_text(code)
    environment["PYTHONPATH"] = os.pathsep.join([str(hook), *roots])
    os.execve(prepared["command"][0], prepared["command"], environment)


def signature(rows, prompt_hashes, prompt_lengths, predictions):
    """Normalize server-generated request IDs while retaining admission and execution order."""
    batch = len(prompt_hashes)
    if not 2 <= batch <= 32 or len(set(prompt_hashes)) != batch or len(prompt_lengths) != batch:
        raise ValueError("Expected two to 32 distinct prompt hashes with matching lengths")
    indices = {}
    totals = [0] * batch
    schedule = []
    for row in rows:
        if "runs" in row:
            if sum(n for _, n in row["runs"]) != row["total"]:
                raise ValueError("Scheduled token counts disagree")
            row = {"admissions": list(row["new"].items()), "scheduled": row["runs"]}
        admitted = []
        for request_id, digest in row["admissions"]:
            if request_id in indices or digest not in prompt_hashes:
                raise ValueError("Unexpected admission during the probe")
            index = prompt_hashes.index(digest)
            if index in indices.values():
                raise ValueError("The same prompt was submitted more than once")
            indices[request_id] = index
            admitted.append(index)
        run = []
        for request_id, count in row["scheduled"]:
            if request_id not in indices or type(count) is not int or count < 1:
                raise ValueError("Unexpected scheduled request or length")
            index = indices[request_id]
            totals[index] += count
            run.append([index, count])
        if len(run) != len({index for index, _ in run}):
            raise ValueError("A request appeared twice in one scheduled step")
        schedule.append([admitted, run])
    if totals != [length + predictions - 1 for length in prompt_lengths]:
        raise ValueError(f"Incomplete model forward counts for these prompts: {totals}")
    if not any(len({index for index, _ in work}) == batch for _, work in schedule):
        raise ValueError("The requested batch never shared a model forward")
    return schedule


def valid_scores(token, scores):
    return (len(scores) >= 8 and str(token) in scores
            and all(type(value) in (float, int) and math.isfinite(value) for value in scores.values()))


def joint_decode_steps(schedule, batch, prompt_length):
    """Count pure decode forwards in which all request slots already have their prompt KV."""
    totals = [0] * batch
    joined = 0
    for _, work in schedule:
        if (min(totals) >= prompt_length and len(work) == batch
                and {slot for slot, _ in work} == set(range(batch)) and all(n == 1 for _, n in work)):
            joined += 1
        for slot, n in work:
            totals[slot] += n
    return joined


def request_scores(url, model, prompts, steps, timeout):
    data = dict(model=model, prompt=prompts, max_tokens=steps, temperature=0, ignore_eos=True, seed=42,
                logprobs=8, return_token_ids=True, return_tokens_as_token_ids=True)
    req = Request(url.rstrip("/") + "/v1/completions", data=json.dumps(data).encode(),
                  headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=timeout) as response:
        result = json.load(response)
    if result["usage"]["prompt_tokens"] != sum(map(len, prompts)) or result["usage"]["completion_tokens"] != len(prompts) * steps:
        raise ValueError("The server returned an incomplete batch")
    answers = []
    for slot, choice in enumerate(sorted(result["choices"], key=lambda row: row["index"])):
        tokens = choice["token_ids"]
        scores = [{str(int(key.removeprefix("token_id:"))): value for key, value in top.items()}
                  for top in choice["logprobs"]["top_logprobs"]]
        if (choice["index"] != slot or len(tokens) != steps or len(scores) != steps
                or any(not valid_scores(token, top) for token, top in zip(tokens, scores))):
            raise ValueError("Missing token IDs or top-eight scores")
        answers.append({"tokens": tokens, "scores": scores})
    if len(answers) != len(prompts):
        raise ValueError("Missing response for one or more requests")
    return answers


def capture(args):
    identity = json.loads((args.directory / "launch.json").read_bytes())
    rows = json.loads(args.bank.read_bytes())["domains"][args.domain]["rows"]
    prompts = [rows[i][:args.length] for i in args.rows]
    hashes = [prompt_hash(row) for row in prompts]
    batch = len(prompts)
    if not 2 <= batch <= 32 or len(set(hashes)) != batch or any(len(row) != args.length for row in prompts):
        raise ValueError("Expected two to 32 distinct prompts of the requested length")
    if min(args.steps, args.length, args.attempts) < 1 or args.output.exists():
        raise ValueError("Use positive lengths and a new receipt path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    marker = args.directory / "recording"
    trace = args.directory / "steps.jsonl"
    recorded = []
    admissions = set()
    for _ in range(args.attempts):
        begin = trace.stat().st_size if trace.exists() else 0
        marker.touch(exist_ok=False)
        try:
            answers = request_scores(args.url, args.model, prompts, args.steps, args.timeout)
        finally:
            marker.unlink(missing_ok=True)
        with trace.open("rb") as stream:
            stream.seek(begin)
            schedule = signature([json.loads(line) for line in stream], hashes, list(map(len, prompts)), args.steps)
        same = next((case for case in recorded if case["schedule"] == schedule), None)
        if same and same["answers"] != answers:
            raise ValueError("Same-source repeated outputs changed under the same schedule")
        if same is None:
            recorded.append({"schedule": schedule, "answers": answers})
        admissions.add(len(schedule[0][0]))
        receipt = dict(identity=identity, prompt_sha256=hashes, domain=args.domain, length=args.length, generated=args.steps, cases=recorded)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(receipt) + "\n")
        temporary.replace(args.output)
        if args.both_admissions and {1, batch} <= admissions:
            break
    if args.both_admissions and not {1, batch} <= admissions:
        raise ValueError("The server did not naturally produce both single-row and all-row initial admissions")


def compare(first, second, both=False):
    for key in ("prompt_sha256", "domain", "length", "generated"):
        if first[key] != second[key]:
            raise ValueError(f"Input mismatch for {key}")
    if first.get("identity") != second.get("identity"):
        raise ValueError("The checkpoint, CUDA device or validation settings differ")
    batch = len(first["prompt_sha256"])
    for receipt in (first, second):
        if (not receipt["cases"] or receipt["generated"] < 1 or not 2 <= batch <= 32
                or len(set(receipt["prompt_sha256"])) != batch):
            raise ValueError("Incomplete pair receipt")
        for case in receipt["cases"]:
            if not case["schedule"] or len(case["answers"]) != batch:
                raise ValueError("Missing schedule or numerical evidence")
            for answer in case["answers"]:
                n = receipt["generated"]
                if (len(answer["tokens"]) != n or len(answer["scores"]) != n
                        or any(not valid_scores(token, scores)
                               for token, scores in zip(answer["tokens"], answer["scores"]))):
                    raise ValueError("Incomplete scores for one of the requests")
    matches = 0
    joint_steps = []
    admission_sizes = set()
    for control in first["cases"]:
        for candidate in second["cases"]:
            if control["schedule"] != candidate["schedule"]:
                continue
            if control["answers"] != candidate["answers"]:
                raise ValueError("Numerical disagreement under the same complete request schedule")
            matches += 1
            joint_steps.append(joint_decode_steps(control["schedule"], batch, first["length"]))
            admission_sizes.add(len(control["schedule"][0][0]))
    if not matches or (both and not {1, batch} <= admission_sizes):
        raise ValueError("Insufficient matching schedules")
    return {"matching_schedules": matches, "initial_admission_sizes": sorted(admission_sizes),
            "joint_decode_steps_per_match": joint_steps,
            "scored_positions_per_schedule": len(first["prompt_sha256"]) * first["generated"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    launch = commands.add_parser("serve")
    launch.add_argument("--directory", type=Path, required=True)
    launch.add_argument("command", nargs=argparse.REMAINDER)
    probe = commands.add_parser("capture")
    probe.add_argument("--directory", type=Path, required=True)
    probe.add_argument("--url", required=True)
    probe.add_argument("--model", default="bench")
    probe.add_argument("--bank", type=Path, required=True)
    probe.add_argument("--domain", choices=("prose", "code"), required=True)
    probe.add_argument("--rows", type=int, nargs="+", default=[0, 1])
    probe.add_argument("--length", type=int, required=True)
    probe.add_argument("--steps", type=int, default=32)
    probe.add_argument("--attempts", type=int, default=12)
    probe.add_argument("--timeout", type=int, default=300)
    probe.add_argument("--both-admissions", action="store_true")
    probe.add_argument("--output", type=Path, required=True)
    compare_args = commands.add_parser("compare")
    compare_args.add_argument("control", type=Path)
    compare_args.add_argument("candidate", type=Path)
    compare_args.add_argument("--both-admissions", action="store_true")
    args = parser.parse_args()
    if args.action == "serve":
        serve(args)
    elif args.action == "capture":
        capture(args)
    else:
        first, second = [json.loads(p.read_bytes()) for p in (args.control, args.candidate)]
        if not first.get("identity") or not second.get("identity"):
            raise ValueError("Both captures must include the server identity")
        print(json.dumps(compare(first, second, args.both_admissions)))


if __name__ == "__main__":
    main()
