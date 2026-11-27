"""Qualify and screen Hy4 index pooling on a single GPU with the original GGUF."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from benchmark import MANIFEST, completion, hash_tokens, verify_prompt_bank, write_json
from verify_gguf_loader import predict
from worker import admission_fraction, gpu_inventory, reported_capacity, stop_server, wait_server


MODEL_FILE = Path("compatible-vllm/core/vllm/model_executor/models/hy_v4_flash.py")
SIZES = (2047, 2048, 2049, 2063, 8192, 65536)
CHECKPOINT = "hy4-iq2m"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prompt_for(rows, size):
    # The 64k diagnostic joins eight independently verified 8k windows in order.
    return [token for row in rows[:(size + 8191) // 8192] for token in row][:size]


def make_trace_runtime(runtime, destination):
    """Instrument a private copy. The clean launch remains available for independent timings."""
    original = runtime / MODEL_FILE
    marker = "        self.indexer(h, q_c, positions)\n"
    addition = (
        "        if isinstance(self.indexer, Hy4CompressedIndexer):\n"
        "            from hy4_index_observer import observe\n"
        "            observe(self.layer_idx, positions, self.indexer.topk_indices_buffer)\n"
    )
    code = original.read_text()
    if code.count(marker) != 1:
        raise ValueError("Expected one Hy4 attention indexer call")
    revised = code.replace(marker, marker + addition)
    if destination.exists():
        if ((destination / MODEL_FILE).read_text() != revised or
                digest(destination / "hy4_index_observer.py") != digest(Path(__file__).with_name("hy4_index_observer.py"))
                or (destination / "benchmark-source.sha256").read_bytes() != (runtime / "benchmark-source.sha256").read_bytes()):
            raise ValueError("The diagnostic runtime belongs to a different source or observer")
    else:
        shutil.copytree(runtime, destination, copy_function=os.link, symlinks=True)
        # The remaining inputs share read-only inodes with the clean runtime.
        local = destination / MODEL_FILE
        local.unlink()
        local.write_text(revised)
        shutil.copyfile(Path(__file__).with_name("hy4_index_observer.py"), destination / "hy4_index_observer.py")
    return digest(destination / MODEL_FILE)


def index_proof(path, tag, expected):
    records = [json.loads(line) for line in path.read_text().splitlines()]
    result = {}
    for record in records:
        if record["tag"] != tag:
            continue
        key = (record["layer"], record["position"])
        if key in result or record["width"] != 2048:
            raise ValueError("Duplicate or incomplete Hy4 index vector")
        result[key] = record["sha256"]
    layers = {layer for layer, _ in result}
    if len(layers) != 12 or set(result) != {(layer, pos) for pos in expected for layer in layers}:
        raise ValueError("Each sampled position must cover all twelve indexed layers")
    return {f"{layer}/{pos}": value for (layer, pos), value in sorted(result.items())}


def check(args):
    model = MANIFEST["models"][CHECKPOINT]
    folder = args.weights / model["repo"] / model["revision"]
    received = json.loads((folder / f"download-{CHECKPOINT}.json").read_bytes())
    provenance = {"repo": model["repo"], "revision": model["revision"], "source": "huggingface.co", "files": model["files"]}
    if received != provenance:
        raise ValueError("Unexpected checkpoint provenance")
    for name, data in model["files"].items():
        if (folder / name).stat().st_size != data["bytes"]:
            raise ValueError(f"Incomplete shard {name}")
    path = folder / next(iter(model["files"]))
    bank = json.loads((args.banks / "hy4.json").read_bytes())
    for domain in ("code", "prose"):
        verify_prompt_bank(bank, "hy4", domain)
    devices = [d for d in gpu_inventory() if d["uuid"] == args.gpu]
    if len(devices) != 1:
        raise ValueError("Select exactly one physical GPU")
    source_hash = (args.runtime / "benchmark-source.sha256").read_text().strip()
    if source_hash != args.lock:
        raise ValueError("Unexpected runtime source lock")
    args.out.mkdir(parents=True, exist_ok=True)
    clean = args.runtime.resolve()
    traced = args.action == "quality"
    rt = args.out / "trace-runtime" if traced else clean
    observed_hash = make_trace_runtime(clean, rt) if traced else None
    control = args.out / "observation.json"
    events = args.out / "indices.jsonl"
    control.unlink(missing_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, OMP_NUM_THREADS="4", HF_HUB_OFFLINE="1",
               KAILLM_NATIVE_ASSETS=str(rt / "runtime-native-cache/native-assets.json"))
    if traced:
        env.update(KAILLM_HY4_OBSERVATION=str(control), KAILLM_HY4_INDICES=str(events))
    metrics = ["--enable-prompt-tokens-details", "--enable-per-request-metrics"]
    command = [sys.executable, str(rt / "deployment/serve.py"), str(path), "--tokenizer", str(path),
               "--spec-tokens", "0", "--port", str(args.port), "--served-model-name", "bench",
               "--max-num-seqs", "1", "--max-num-batched-tokens", "2048", "--max-model-len", "73728",
               "--kv-cache-memory-bytes", str(args.kv_gib * 2**30), "--gpu-memory-utilization",
               str(admission_fraction(devices)), "--extra-vllm-args", json.dumps(metrics)]
    preview = subprocess.run(command + ["--dry-run", "--compute-capability", devices[0]["compute_cap"]],
                             env=env, check=True, text=True, capture_output=True)
    effective = json.loads(preview.stdout)["command"]
    if "--no-enable-prefix-caching" not in effective:
        raise ValueError("The measurement requires cold prompt processing")
    identity = {"checkpoint": provenance, "gpu": devices[0], "lock": args.lock, "clean_model": digest(clean / MODEL_FILE),
                "trace_model": observed_hash, "action": args.action, "argv": command, "effective_argv": effective}
    write_json(args.out / "setup.json", identity)
    url = f"http://127.0.0.1:{args.port}"
    log = args.out / "server.log"
    with log.open("w") as stream:
        proc = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            wait_server(proc, url, log)
            capacity = reported_capacity(log.read_text(errors="replace"))
            if capacity is None or capacity < 1:
                raise ValueError("The runtime could not admit one entire 73,728-token sequence")
            identity["capacity"] = capacity
            write_json(args.out / "setup.json", identity)
            for domain in ("prose", "code"):
                windows = bank["domains"][domain]["rows"]
                for size in (SIZES if traced else (8192, 65536)):
                    prompt = prompt_for(windows, size)
                    count = 129 if size >= 8192 else 8
                    for repeat in range(1 if traced or size == 65536 else 3):
                        tag = f"{domain}-{size}-{repeat}"
                        dst = args.out / (tag + ".json")
                        if dst.is_file():
                            row = json.loads(dst.read_bytes())
                            if (row["lock"], row["gpu"], row["prompt_sha256"], row["action"]) != (
                                    args.lock, args.gpu, hash_tokens(prompt), args.action):
                                raise ValueError("Saved observation differs from the requested source or inputs")
                            continue
                        row = dict(lock=args.lock, gpu=args.gpu, action=args.action, prompt_sha256=hash_tokens(prompt),
                                   domain=domain, input_tokens=size, output_tokens=count, repeat=repeat)
                        if traced:
                            selected = sorted({size - 1, size, size + count - 2})
                            # A unique attempt id keeps partial observations from an interrupted server separate.
                            attempt = tag + "-" + str(time.time_ns())
                            write_json(control, {"tag": attempt, "positions": selected})
                            row["prediction"] = predict("kaiLLM", url, prompt, output_tokens=count, timeout=3600)
                            control.unlink()
                            row["indices"] = index_proof(events, attempt, selected)
                        else:
                            start = time.monotonic()
                            result = completion("kaiLLM", url, prompt, count, 3600, return_token_ids=True)
                            row["wall_seconds"] = time.monotonic() - start
                            if result["cache_hits"] != 0:
                                raise ValueError("The timed diagnostic reused prompt tokens")
                            row.update(result)
                        write_json(dst, row)
                        print(tag, "complete", flush=True)
        finally:
            control.unlink(missing_ok=True)
            stop_server(proc)


def compare(left, right):
    before = json.loads((left / "setup.json").read_bytes())
    after = json.loads((right / "setup.json").read_bytes())
    same_device = ("uuid", "driver_version", "name", "power.limit")
    if (before["checkpoint"] != after["checkpoint"] or before["lock"] == after["lock"]
            or before["action"] != "quality" or after["action"] != "quality"
            or any(before["gpu"][k] != after["gpu"][k] for k in same_device)):
        raise ValueError("Quality comparisons require two distinct sources on identical hardware and weights")
    totals = dict(prefixes=0, output_positions=0, index_vectors=0, differing_indices=0, differing_outputs=0, differing_scores=0)
    for domain in ("prose", "code"):
        for size in SIZES:
            name = f"{domain}-{size}-0.json"
            x, y = [json.loads((root / name).read_bytes()) for root in (left, right)]
            if (x["prompt_sha256"] != y["prompt_sha256"] or x["indices"].keys() != y["indices"].keys()
                    or (x["lock"], y["lock"]) != (before["lock"], after["lock"])):
                raise ValueError("Quality observations differ in source, position or prompt")
            a, b = x["prediction"], y["prediction"]
            totals["prefixes"] += 1
            totals["output_positions"] += len(a["tokens"])
            totals["index_vectors"] += len(x["indices"])
            totals["differing_indices"] += sum(value != y["indices"][key] for key, value in x["indices"].items())
            totals["differing_outputs"] += sum(i != j for i, j in zip(a["tokens"], b["tokens"], strict=True))
            totals["differing_scores"] += sum(i != j for i, j in zip(a["top_logprobs"], b["top_logprobs"], strict=True))
    totals["passed"] = not any(totals[key] for key in ("differing_indices", "differing_outputs", "differing_scores"))
    return totals


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    actions = cli.add_subparsers(dest="action", required=True)
    for action in ("quality", "screen"):
        cmd = actions.add_parser(action)
        for name in ("weights", "banks", "runtime", "out"):
            cmd.add_argument("--" + name, type=Path, required=True)
        for name in ("gpu", "lock"):
            cmd.add_argument("--" + name, required=True)
        cmd.add_argument("--port", type=int, default=18879)
        cmd.add_argument("--kv-gib", type=int, default=8)
    cmd = actions.add_parser("compare")
    for name in ("left", "right", "output"):
        cmd.add_argument(name, type=Path)
    args = cli.parse_args()
    if args.action == "compare":
        result = compare(args.left, args.right)
        write_json(args.output, result)
        print(json.dumps(result))
        sys.exit(0 if result["passed"] else 2)
    check(args)
