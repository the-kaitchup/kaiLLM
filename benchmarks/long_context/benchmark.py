"""Matched prefill and decode throughput for an already running local server."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import statistics
import struct
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST = json.loads((ROOT / "manifest.json").read_text())
PROMPT = 8192
OUTPUT = 65536
BATCHES = (1, 2, 4, 8, 16, 32)
DRAFT_TOKENS = {"plain": 0, "mtp5": 5, "dspark5": 5, "dspine7": 7, "dspine15": 15,
                "ngram3": 3, "ngram1": 1}
PROTOCOL = "8k64k-v1"


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def hash_tokens(ids):
    return hashlib.sha256(struct.pack(f"<{len(ids)}I", *ids)).hexdigest()


def prompt_bank(family, dest):
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    ref = MANIFEST["tokenizers"][family]
    path = hf_hub_download(ref["repo"], "tokenizer.json", revision=ref["revision"])
    tokenizer = Tokenizer.from_file(path)
    bank = {"family": family, "tokenizer_revision": ref["revision"],
            "tokenizer_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(), "domains": {}}
    for domain, source in MANIFEST["sources"].items():
        with urllib.request.urlopen(source["url"], timeout=120) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != source["sha256"]:
            raise ValueError(f"{domain} source hash mismatch")
        passage = content.decode("utf-8-sig")
        if domain == "prose":
            chapter = "CHAPTER 1. Loomings."
            # The second occurrence starts the novel after the table of contents.
            at = passage.upper().rfind(chapter.upper())
            if at < 0:
                raise ValueError("First chapter missing")
            passage = passage[at:]
        ids = tokenizer.encode(passage, add_special_tokens=False).ids
        usable = len(ids) - PROMPT
        if usable < 1024:
            raise ValueError(f"Insufficient distinct windows in {domain}")
        # Index 32 is disjoint from the timed batch by starting position and prefix.
        rows = [ids[i * usable // 32:i * usable // 32 + PROMPT] for i in range(33)]
        if len({tuple(row[:64]) for row in rows}) != 33:
            raise ValueError(f"Duplicate prompt prefixes in {domain}")
        bank["domains"][domain] = {"hashes": [hash_tokens(row) for row in rows], "rows": rows}
    write_json(dest, bank)


def completion(engine, url, ids, count, timeout):
    if engine == "kaiLLM":
        endpoint = "/v1/completions"
        payload = {"model": "bench", "prompt": ids, "max_tokens": count, "temperature": 0,
                   "seed": 42, "ignore_eos": True}
    else:
        endpoint = "/completion"
        payload = {"prompt": ids, "n_predict": count, "temperature": 0, "seed": 42,
                   "ignore_eos": True, "cache_prompt": False}
    req = urllib.request.Request(url.rstrip("/") + endpoint, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as res:
        raw = json.load(res)
    if engine == "kaiLLM":
        choice, = raw["choices"]
        u = raw["usage"]
        cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
        received = (u["prompt_tokens"], u["completion_tokens"])
        if choice["finish_reason"] != "length":
            raise ValueError(f"Unexpected finish reason: {choice['finish_reason']}")
        output = choice["text"]
        timings = raw.get("metrics") or {}
    else:
        received = raw["tokens_evaluated"], raw["tokens_predicted"]
        timings = raw["timings"]
        cached = timings.get("cache_n", 0)
        # llama.cpp also sets truncated when the final requested token lands exactly on
        # the context boundary. Distinguish that stop from an early cut or dropped input.
        if timings["prompt_n"] != len(ids) or raw.get("stop_type") != "limit":
            raise ValueError("llama.cpp reused input or stopped before the requested length")
        if raw.get("truncated") and (count != OUTPUT or received != (len(ids), OUTPUT)):
            raise ValueError("llama.cpp reached the context boundary prematurely")
        output = raw["content"]
    if received != (len(ids), count) or cached not in (None, 0):
        raise ValueError(f"Expected {(len(ids), count)} with empty cache, got {received}, cached={cached}")
    return {"output_tokens": count, "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
            "prompt_tokens": len(ids), "cache_hits": cached, "server_timings": timings}


def wave(engine, url, prompts, count, timeout):
    gate = threading.Barrier(len(prompts) + 1)
    def one(ids):
        gate.wait(timeout=30)
        return completion(engine, url, ids, count, timeout)
    with concurrent.futures.ThreadPoolExecutor(len(prompts)) as pool:
        pending = [pool.submit(one, ids) for ids in prompts]
        start = time.monotonic()
        gate.wait(timeout=30)
        results = [f.result() for f in pending]
        elapsed = time.monotonic() - start
    return {"seconds": elapsed, "requests": results}


def run(args):
    bank = json.loads(args.prompts.read_text())
    env = json.loads(args.environment.read_text())
    model = MANIFEST["models"][args.checkpoint]
    if bank["family"] != model["family"]:
        raise ValueError("Wrong tokenizer family")
    if not env.get("gpu_names") or not env.get("engine_revision") or not env.get("server_argv"):
        raise ValueError("Missing hardware, engine revision or launch configuration")
    data = bank["domains"][args.domain]
    for row, expected in zip(data["rows"], data["hashes"], strict=True):
        if len(row) != PROMPT or hash_tokens(row) != expected:
            raise ValueError("Prompt bank integrity check failed")
    batch = args.batch
    rows = data["rows"][:batch]
    # A separate untimed request primes prefill and decode kernels without sharing a timed prefix.
    completion(args.engine, args.url, data["rows"][32], 32, args.timeout)
    prefill = [wave(args.engine, args.url, rows, 1, args.timeout) for _ in range(3)]
    full = wave(args.engine, args.url, rows, OUTPUT, args.timeout)
    pp_seconds = statistics.median(x["seconds"] for x in prefill)
    decode_seconds = full["seconds"] - pp_seconds
    if not math.isfinite(decode_seconds) or min(pp_seconds, decode_seconds) <= 0:
        raise ValueError("Invalid timing interval")
    record = {"protocol": PROTOCOL, "checkpoint": args.checkpoint, "model_revision": model["revision"],
              "sha256": [f["sha256"] for f in model["files"].values()], "engine": args.engine,
              "mode": args.mode, "domain": args.domain, "batch": batch, "prompt_tokens": PROMPT,
              "generated_tokens": OUTPUT, "prompt_hashes": data["hashes"][:batch], "environment": env,
              "prefill_tps": batch * PROMPT / pp_seconds,
              "decode_tps": batch * (OUTPUT - 1) / decode_seconds, "prefill": prefill, "full": full,
              "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    write_json(args.output, record)
    print(f"{args.engine} {args.checkpoint} {args.domain} B{batch}: "
          f"prefill {record['prefill_tps']:.1f}, decode {record['decode_tps']:.1f} tok/s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("prompts")
    p.add_argument("family", choices=MANIFEST["tokenizers"])
    p.add_argument("output", type=Path)
    b = sub.add_parser("run")
    b.add_argument("--engine", required=True, choices=("kaiLLM", "llama.cpp"))
    b.add_argument("--checkpoint", required=True, choices=MANIFEST["models"])
    b.add_argument("--url", required=True)
    b.add_argument("--prompts", type=Path, required=True)
    b.add_argument("--domain", required=True, choices=("prose", "code"))
    b.add_argument("--batch", required=True, choices=BATCHES, type=int)
    b.add_argument("--mode", choices=DRAFT_TOKENS, required=True)
    b.add_argument("--environment", type=Path, required=True)
    b.add_argument("--timeout", type=int, default=43200)
    b.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    if a.action == "prompts":
        prompt_bank(a.family, a.output)
    else:
        run(a)
