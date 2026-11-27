"""Verify Inkling generation against an archived fixed-prefix capture with a measured CUDA dispatch policy."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import kai_default
from benchmark import MANIFEST, write_json
from cuda_graphs import ENV, snapshot, validate_kai_launch, verify_head_interval, verify_interval
from devices import visible_cuda_uuids
from verify_gguf_loader import capture, compare
from worker import admission_fraction, checkpoint_files, gpu_inventory, selected_devices, stop_server, wait_server


def qualify(args):
    spec = MANIFEST["models"][args.checkpoint]
    if spec["family"] != "inkling" or len(spec["files"]) != 1:
        raise ValueError("Expected a single-file Inkling checkpoint")
    model, = checkpoint_files(args.weights, args.checkpoint, spec)
    sha256 = spec["files"][model.name]["sha256"]
    if args.reference_model_sha256 != sha256:
        raise ValueError("Fixed-prefix reference and target must use identical checkpoint bytes")
    devices = selected_devices(gpu_inventory(), [args.device], visible_cuda_uuids())
    uuid = devices[0]["uuid"]
    bank = json.loads((args.banks / "inkling.json").read_bytes())
    ref = json.loads(args.reference.read_bytes())
    if len(ref["rows"]) != 24:
        raise ValueError("Expected the archived 24 fixed prefixes")
    args.out.mkdir(parents=True, exist_ok=False)
    graphs = args.out / "graphs"
    graphs.mkdir()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, KAILLM_HUMMING_DETERMINISTIC="1",
               KAILLM_NATIVE_ASSETS=str(args.runtime / "runtime-native-cache/native-assets.json"),
               OMP_NUM_THREADS=str(min(8, max(1, len(os.sched_getaffinity(0)) // 2))))
    normal = getattr(args, 'kai_policy', 'strict') == 'default'
    env.pop(ENV, None)
    env.pop(kai_default.ENV, None)
    if normal:
        for override in ('KAILLM_INKLING_FULL_GRAPHS','KAILLM_PREFILL_GRAPHS'):
            env.pop(override, None)
        lock = (args.runtime/'benchmark-source.sha256').read_text().strip()
        if lock not in {source for _, source in kai_default.pins(MANIFEST, args.checkpoint, 'plain')}:
            raise ValueError('The passive dispatch checker needs its pinned runtime')
        with model.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != sha256:
                raise ValueError('Checkpoint digest differs from its Hugging Face revision')
    env[kai_default.ENV if normal else ENV] = str(graphs)
    command = [sys.executable, str(args.runtime / "deployment/serve.py"), str(model),
               "--port", str(args.port), "--served-model-name", "bench", "--spec-tokens", "0",
               "--max-num-seqs", "1", "--max-model-len", "4096", "--max-num-batched-tokens", "2048",
               "--kv-cache-memory-bytes", str(1 << 30), "--gpu-memory-utilization", str(admission_fraction(devices)),
               "--extra-vllm-args", json.dumps(["--enable-prompt-tokens-details", "--no-enable-prefix-caching"])]
    dry = subprocess.run([*command, "--dry-run"], env=env, text=True, capture_output=True, check=True)
    if normal:
        kai_default.launch(json.loads(dry.stdout)['command'])
    else:
        validate_kai_launch(json.loads(dry.stdout)['command'])
    log = args.out / "server.log"
    with log.open("w") as stream:
        process = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        url = f"http://127.0.0.1:{args.port}"
        wait_server(process, url, log)
        observe = kai_default.read if normal else snapshot
        before = observe(graphs, [uuid])
        if not normal and not before[uuid]["head_sizes"]:
            raise RuntimeError("The target output projection has no captured graph")
        candidate = capture("kaiLLM", url, bank, "inkling")
        after = observe(graphs, [uuid])
        if normal:
            proof = {'kai_default_dispatch': kai_default.interval(before, after, ('prefill','decode'))}
        else:
            proof = {'graph_replays': verify_interval(before, after, ('prefill', 'decode')),
                     'output_head_replays': verify_head_interval(before, after, ('prefill','decode'), [uuid])}
        outcome = compare(ref, candidate)
        write_json(args.out / "candidate.json", candidate)
        write_json(args.out / "comparison.json", {**outcome, **proof,
                                                  "model_revision": spec["revision"], "model_sha256": sha256,
                                                  "reference_capture_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest()})
        if outcome["greedy_tokens_matching"] != 192 or outcome["first_differences"]:
            raise RuntimeError("Fixed-prefix token parity failed; inspect the captured distributions")
    finally:
        stop_server(process)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("runtime", "out", "reference", "weights", "banks"):
        p.add_argument("--" + arg, required=True, type=Path)
    p.add_argument('--kai-policy', choices=('strict','default'), default='strict')
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--reference-model-sha256", required=True)
    p.add_argument("--port", type=int, default=18885)
    p.add_argument("--device", type=int, default=0)
    qualify(p.parse_args())
