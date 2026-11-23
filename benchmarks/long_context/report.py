"""Rebuild speed.md from completed runs. Never match across GPUs or prompt banks."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

from benchmark import BATCHES, MANIFEST, OUTPUT, PROMPT, PROTOCOL


def digest(items):
    return hashlib.sha256(json.dumps(items, separators=(",", ":")).encode()).hexdigest()


def get_record(r):
    key = r["checkpoint"]
    m = MANIFEST["models"][key]
    if (r["protocol"], r["prompt_tokens"], r["generated_tokens"], r["model_revision"]) != (
            PROTOCOL, PROMPT, OUTPUT, m["revision"]):
        raise ValueError(f"Invalid protocol or revision for {key}")
    if r["sha256"] != [f["sha256"] for f in m["files"].values()]:
        raise ValueError("Weight digests differ from manifest")
    n = r["batch"]
    if n not in BATCHES or len(r["prompt_hashes"]) != n or len(set(r["prompt_hashes"])) != n:
        raise ValueError("Invalid prompt batch")
    if len(r["prefill"]) != 3:
        raise ValueError("Expected three prefill probes")
    for wave, length in [(w, 1) for w in r["prefill"]] + [(r["full"], OUTPUT)]:
        if len(wave["requests"]) != n or any((x["prompt_tokens"], x["output_tokens"]) != (PROMPT, length)
                or x["cache_hits"] not in (0, None) for x in wave["requests"]):
            raise ValueError("Incomplete request or cached prefix")
    env = r["environment"]
    dev = env["devices"]
    return {"protocol": PROTOCOL, "gpu": env["gpu_names"], "hardware_id": digest([d["uuid"] for d in dev])[:16],
            "driver": [d["driver_version"] for d in dev], "power_limit_w": [d["power.limit"] for d in dev],
            "checkpoint": key, "model_revision": m["revision"], "engine": r["engine"],
            "kai_source": env["kai_source"], "llama_source": env["llama_source"],
            "engine_revision": env["engine_revision"], "mode": r["mode"], "batch": n, "domain": r["domain"],
            "prompt_set_sha256": digest(r["prompt_hashes"]), "input_tokens_per_request": PROMPT,
            "output_tokens_per_request": OUTPUT, "prefill_seconds": [w["seconds"] for w in r["prefill"]],
            "full_seconds": r["full"]["seconds"], "generated_text_hashes": [x["output_sha256"] for x in r["full"]["requests"]],
            "timestamp_utc": r["timestamp_utc"]}


def rates(row):
    prefill = statistics.median(row["prefill_seconds"])
    return row["batch"] * PROMPT / prefill, row["batch"] * (OUTPUT - 1) / (row["full_seconds"] - prefill)


def confirmed_limit(stamp):
    if stamp.get("state") != "failed" or not stamp.get("gpu_names"):
        return None
    reason = stamp["reason"]
    match = re.search(r"Full-context KV capacity [\d.]+ sequences < batch \d+", reason)
    if match:
        detail = match.group()
    elif re.search(r"out of memory|insufficient vram|cannot allocate memory|failed to allocate.*buffer|no available memory for the cache blocks", reason, re.I):
        detail = "Memory allocation failed at the requested context"
    elif "No kaiLLM release native catalog for compute capability" in reason:
        detail = reason
    elif reason in ("GGUF has no native MTP tensor head", "kaiLLM Inkling has no DSpark verifier",
                    "The merged adapter has no native draft verifier for this family"):
        detail = reason
    else:
        return None
    return {"gpu": stamp["gpu_names"], **{k: stamp[k] for k in ("checkpoint", "engine", "mode", "batch")}, "reason": detail}


def table(records, limits=()):
    rows = defaultdict(dict)
    plain = {}
    def workload(r):
        return (r["hardware_id"], tuple(r["driver"]), tuple(r["power_limit_w"]), r["checkpoint"],
                r["batch"], r["domain"], r["prompt_set_sha256"], r["engine"])
    for r in records:
        key = (tuple(r["gpu"]), r["hardware_id"], tuple(r["driver"]), tuple(r["power_limit_w"]), r["checkpoint"],
               r["mode"], r["batch"], r["domain"], r["prompt_set_sha256"], r["kai_source"], r["llama_source"])
        if r["engine"] in rows[key]:
            raise ValueError(f"Duplicate result for {key}")
        rows[key][r["engine"]] = r
        if r["mode"] == "plain":
            plain[workload(r)] = r["generated_text_hashes"]
    lines = ["# kaiLLM prefill and decode speed", "",
             "Each request uses **8,192 input tokens and 65,536 output tokens**. Rates are aggregate tokens per second at the listed concurrency. A red kaiLLM value marks a workload as **not optimized** when llama.cpp was more than 10% faster on the same checkpoint, GPU, workload, batch, and decoding mode.",
             "", "[Protocol, commands and prompt sources](benchmarks/long_context/README.md) · [Pinned checkpoints](benchmarks/long_context/manifest.json) · [Timing data](benchmarks/long_context/results.jsonl)", "",
             "## Results", ""]
    if not rows:
        lines.append("The first full-length runs are in progress.")
    else:
        lines += ["| GPU | Checkpoint | Mode | Batch | Input | kaiLLM prefill | llama.cpp prefill | kaiLLM decode | llama.cpp decode |",
                  "|---|---|---|---:|---|---:|---:|---:|---:|"]
    for key, pair in sorted(rows.items()):
        gpus, _, _, _, model, mode, batch, domain, *_ = key
        x = rates(pair["kaiLLM"]) if "kaiLLM" in pair else (None, None)
        y = rates(pair["llama.cpp"]) if "llama.cpp" in pair else (None, None)
        nums = []
        for metric, (kai, llama) in enumerate(zip(x, y)):
            value = "—" if kai is None else f"{kai:.1f}"
            if kai is not None and llama is not None and llama > 1.10 * kai:
                value = "$\\color{red}{" + value + "}$"
            values = [value, "—" if llama is None else f"{llama:.1f}"]
            if metric == 1 and mode != "plain":
                for i, engine in enumerate(("kaiLLM", "llama.cpp")):
                    r = pair.get(engine)
                    if r and workload(r) in plain and r["generated_text_hashes"] != plain[workload(r)]:
                        values[i] += "†"
            nums.extend(values)
        gpu = " + ".join(g.replace("NVIDIA ", "") for g in gpus)
        lines.append("| " + " | ".join(map(str, [gpu, model, mode, batch, domain, *nums])) + " |")
    lines += ["", "† The generated text differed from that engine’s plain run on at least one request. All scored runs completed the specified output length. A dash means there is no matched measurement in that cell. kaiPTQ/chimachima requires kaiLLM. DSpine uses batch 1.", "", "## Hardware coverage", "", "The campaign covers the RTX 3060, 3090, 5080, 5090, RTX PRO 6000, and GB10.", ""]
    if limits:
        lines += ["## Confirmed configuration limits", "",
                  "| GPU | Checkpoint | Engine | Mode | Batch | Observation |",
                  "|---|---|---|---|---:|---|"]
        for r in limits:
            gpu = " + ".join(x.removeprefix("NVIDIA ") for x in r["gpu"])
            lines.append("| " + " | ".join(map(str, [gpu, r["checkpoint"], r["engine"], r["mode"], r["batch"], r["reason"]])) + " |")
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("directories", nargs="*", type=Path)
    a = parser.parse_args()
    records = {}
    def add(r):
        k = tuple(r[x] for x in ("hardware_id", "checkpoint", "engine", "mode", "batch", "domain", "kai_source", "llama_source"))
        if k in records and records[k] != r:
            raise ValueError("Multiple runs for the same cell; retain and aggregate repetitions explicitly")
        records[k] = r
    if a.archive.exists():
        for line in a.archive.read_text().splitlines():
            if line.strip():
                add(json.loads(line))
    for root in a.directories:
        for domain in ("prose", "code"):
            for f in root.rglob(f"*-{domain}.json"):
                add(get_record(json.loads(f.read_text())))
    ordered = [records[k] for k in sorted(records)]
    index = a.archive.with_name("limits.json")
    def limit_key(r):
        return (tuple(r["gpu"]), r["checkpoint"], r["engine"], r["mode"], r["batch"])
    limits = {limit_key(r): r for r in json.loads(index.read_text())} if index.exists() else {}
    for root in a.directories:
        for f in root.rglob("*.status.json"):
            r = confirmed_limit(json.loads(f.read_text()))
            if r:
                limits[limit_key(r)] = r
    for r in ordered:
        limits.pop(limit_key(r), None)
    limits = [limits[k] for k in sorted(limits)]
    index.write_text(json.dumps(limits, indent=2) + "\n")
    a.archive.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in ordered))
    a.report.write_text(table(ordered, limits))


if __name__ == "__main__":
    main()
