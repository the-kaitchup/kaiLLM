"""Compare plain and GPU n-gram LFM3 against each other on the same physical card and batch."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from benchmark import MANIFEST, full_graph_runtime, verify_prompt_bank, wave, write_json
from cuda_graphs import ENV, PHASES, snapshot, validate_kai_launch, verify_draft_interval, verify_head_interval, verify_interval
from devices import visible_cuda_uuids
from verify_inkling_drafter import counters
from worker import admission_fraction, checkpoint_files, gpu_inventory, selected_devices, stop_server, wait_server


def compare_controls(plain, speculative):
    matched = {}
    for domain in ("prose", "code"):
        rows = plain[domain]
        if not rows or any(len(row) != 48 for row in rows):
            raise ValueError(f"Incomplete LFM3 plain continuations for {domain}")
        if rows != speculative[domain]:
            raise ValueError(f"GPU n-gram greedy tokens differ from plain for {domain}")
        matched[domain] = sum(map(len, rows))
    return matched


def run(args, mode, target, model, bank, devices, lock):
    out = args.out / mode
    directory = out / "graphs"
    directory.mkdir(parents=True)
    uuid = devices[0]["uuid"]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, OMP_NUM_THREADS=str(min(8, max(1, len(os.sched_getaffinity(0)) // 2))),
               KAILLM_NATIVE_ASSETS=str(args.runtime / "runtime-native-cache/native-assets.json"))
    env[ENV] = str(directory)
    tokens = 0 if mode == "plain" else int(mode.removeprefix("ngram"))
    cmd = [sys.executable, str(args.runtime / "deployment/serve.py"), str(target), "--port", str(args.port),
           "--served-model-name", "bench", "--spec-tokens", str(tokens), "--lookup-min", "2", "--lookup-max", "8",
           "--max-num-seqs", str(args.batch), "--max-model-len", "8256", "--max-num-batched-tokens", "2048",
           "--kv-cache-memory-bytes", str(args.kv_mib << 20), "--gpu-memory-utilization", str(admission_fraction(devices))]
    dry = subprocess.run([*cmd, "--dry-run"], env=env, text=True, capture_output=True, check=True)
    argv = json.loads(dry.stdout)["command"]
    validate_kai_launch(argv)
    if tokens:
        spec = json.loads(argv[argv.index("--speculative-config") + 1])
        expected = ("ngram_gpu", tokens, 2, 8, tokens == 3)
        actual = (spec["method"], spec["num_speculative_tokens"], spec["prompt_lookup_min"], spec["prompt_lookup_max"],
                  spec.get("ngram_gpu_single_active", False))
        if actual != expected:
            raise ValueError("Serving policy differs from the benchmark n-gram configuration")
    write_json(out / "launch.json", {"argv": argv, "gpu": uuid, "runtime_lock": lock, "batch": args.batch,
                                    "model_revision": model["revision"], "files": model["files"]})
    url = f"http://127.0.0.1:{args.port}"
    log = out / "server.log"
    with log.open("w") as stream:
        proc = subprocess.Popen(cmd, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        wait_server(proc, url, log)
        outputs = {}
        for domain in ("prose", "code"):
            prompts = bank["domains"][domain]["rows"][:args.batch]
            before = snapshot(directory, [uuid])
            if not before[uuid]["head_sizes"] or bool(before[uuid]["draft_captures"]) != bool(tokens):
                raise ValueError("Expected captured vocabulary head and matching GPU draft inventory")
            n0 = counters(url) if tokens else None
            result = wave("kaiLLM", url, prompts, 48, args.timeout, return_token_ids=True)
            outputs[domain] = [r["token_ids"] for r in result["requests"]]
            write_json(out / f"{domain}.json", outputs[domain])
            after = snapshot(directory, [uuid])
            proof = {"target": verify_interval(before, after, PHASES),
                     "head": verify_head_interval(before, after, PHASES, [uuid])}
            if tokens:
                draft = verify_draft_interval(before, after, [uuid], require_proposal=True)
                proof["lookup"] = draft
                n1 = counters(url, required=True)
                proposed, accepted = (n1[k] - n0[k] for k in ("draft_tokens", "accepted_tokens"))
                if (draft[uuid]["proposals"] != draft[uuid]["graph_replays"] or not 0 <= accepted < proposed):
                    raise ValueError("Expected one graph per lookup and observed rejected n-gram tokens")
                proof["tokens"] = {"proposed": proposed, "accepted": accepted, "rejected": proposed-accepted}
            write_json(out / f"{domain}-proof.json", proof)
        return outputs
    finally:
        stop_server(proc)


def main(args):
    model = MANIFEST["models"][args.checkpoint]
    target, = checkpoint_files(args.weights, args.checkpoint, model)
    mode = "ngram1" if args.batch == 8 else "ngram3"
    lock = (args.runtime / "benchmark-source.sha256").read_text().strip()
    if full_graph_runtime(args.checkpoint, mode)[1] != lock:
        raise ValueError("LFM3 qualification requires the manifest's exact GPU n-gram runtime")
    bank = json.loads((args.banks / "lfm.json").read_bytes())
    for domain in ("prose", "code"):
        verify_prompt_bank(bank, "lfm", domain)
    devices = selected_devices(gpu_inventory(), [args.device], visible_cuda_uuids())
    args.out.mkdir(parents=True, exist_ok=False)
    plain, draft = [run(args, name, target, model, bank, devices, lock) for name in ("plain", mode)]
    write_json(args.out / "comparison.json", compare_controls(plain, draft))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime", "weights", "banks", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--checkpoint", choices=("lfm-qad", "lfm-chima", "lfm-suisui"), required=True)
    p.add_argument("--batch", type=int, choices=(1, 2, 4, 8), required=True)
    p.add_argument("--kv-mib", type=int, default=2048)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--port", type=int, default=18889)
    p.add_argument("--timeout", type=int, default=600)
    main(p.parse_args())
