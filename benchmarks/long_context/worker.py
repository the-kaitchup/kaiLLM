"""Launch one engine configuration, check full-context capacity, and collect both domains."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace

from benchmark import BATCHES, DRAFT_TOKENS, MANIFEST, OUTPUT, PROMPT, run, write_json

CONTEXT = PROMPT + OUTPUT
GIB = 1024 ** 3


def gpu_inventory():
    fields = "name,uuid,driver_version,compute_cap,memory.total,memory.free,power.limit"
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=" + fields, "--format=csv,noheader,nounits"], text=True)
    return [dict(zip(fields.split(","), (v.strip() for v in row), strict=True))
            for row in csv.reader(raw.splitlines())]


def wait_server(proc, url, log):
    until = time.monotonic() + 1500
    while proc.poll() is None and time.monotonic() < until:
        try:
            with urllib.request.urlopen(url + "/health", timeout=3) as res:
                if res.status == 200:
                    return
        except OSError:
            pass
        time.sleep(3)
    tail = log.read_text(errors="replace")[-12000:]
    lines = [s for s in tail.splitlines() if re.search(r"error|failed|cache|context|memory", s, re.I)]
    raise RuntimeError("Server startup failed: " + " ".join(lines[-4:])[-1200:])


def keyname(model, engine, mode, batch):
    return f"{model}-{engine}-{mode}-b{batch}"


def inspect_family(path):
    source = Path(__file__).resolve().parents[2] / "runtime-source/vllm_gguf_plugin/serving_defaults.py"
    spec = importlib.util.spec_from_file_location("bench_gguf_defaults", source)
    defaults = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(defaults)
    return defaults.inspect(path)


def main(a):
    inventory = gpu_inventory()
    devices = [inventory[i] for i in a.devices]
    spec = MANIFEST["models"][a.checkpoint]
    base = a.weights / spec["repo"] / spec["revision"]
    files = [base / n for n in spec["files"]]
    # Either fetch entry point checks all SHA-256 before emitting a receipt.
    receipts = [base / ("download-" + a.checkpoint + ".json"), base / "download.json"]
    valid = False
    for p in receipts:
        if p.exists():
            obj = json.loads(p.read_text())
            valid |= (obj.get("source") == "huggingface.co" and obj.get("repo") == spec["repo"]
                      and obj.get("revision") == spec["revision"] and obj.get("files") == spec["files"])
    if not valid or any(p.stat().st_size != spec["files"][p.name]["bytes"] for p in files):
        raise ValueError("Verified HF download receipt required")
    a.out.mkdir(parents=True, exist_ok=True)
    key = keyname(a.checkpoint, a.engine, a.mode, a.batch)
    outputs = {d: a.out / f"{key}-{d}.json" for d in ("prose", "code")}
    if all(p.exists() for p in outputs.values()):
        return
    log = a.out / (key + ".log")
    stamp = a.out / (key + ".status.json")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, a.devices))
    env["OMP_NUM_THREADS"] = str(min(8, max(1, len(os.sched_getaffinity(0)) // 2)))
    stage = "preflight"
    proc = None
    try:
        tokens = DRAFT_TOKENS[a.mode]
        dspine = a.mode.startswith("dspine")
        ngram = a.mode.startswith("ngram")
        if ngram and (spec["family"] != "lfm" or tokens != max(0, min(3, 16 // a.batch - 1))):
            raise ValueError("LFM3 n-gram draft length must follow its batch-dependent serving policy")
        if a.mode == "mtp5" and not inspect_family(files[0])["mtp_layers"]:
            raise ValueError("GGUF has no native MTP tensor head")
        if dspine or a.mode == "dspark5":
            auxiliary = "dspine" if dspine else "dspark"
            ref = MANIFEST[auxiliary]
            directory = a.weights / ref["repo"] / ref["revision"]
            receipt = json.loads((directory / ("download-" + auxiliary + ".json")).read_text())
            if receipt != {"source": "huggingface.co", "repo": ref["repo"], "revision": ref["revision"], "files": ref["files"]}:
                raise ValueError("Drafter receipt differs from the pinned version")
            path = directory if dspine else directory / next(iter(ref["files"]))
            if a.drafter is not None and a.drafter.resolve() != path.resolve():
                raise ValueError("Drafter must match the pinned Hugging Face download")
            a.drafter = path
        if a.engine == "kaiLLM":
            if a.runtime is None or a.python is None:
                raise ValueError("Set --runtime and --python for the merged kaiLLM runtime")
            if any(d["compute_cap"] not in ("8.6", "12.0") for d in devices):
                raise ValueError("No kaiLLM release native catalog for compute capability " + devices[0]["compute_cap"])
            if ngram and inspect_family(files[0])["speculation"] != "ngram":
                raise ValueError("GGUF does not select n-gram drafting")
            if dspine and (a.batch != 1 or spec["family"] not in ("qwen", "daisy")):
                raise ValueError("DSpine requires one slot and a Qwen3.8-family target")
            if a.mode == "dspark5":
                raise ValueError("kaiLLM Inkling has no DSpark verifier")
            if a.mode != "plain" and spec["family"] in ("inkling", "hy4"):
                raise ValueError("The merged adapter has no native draft verifier for this family")
            env["KAILLM_NATIVE_ASSETS"] = str(a.runtime / "runtime-native-cache/native-assets.json")
            free = min(float(d["memory.free"]) * 2**20 for d in devices)
            weight_bytes = sum(p.stat().st_size for p in files)
            if spec["family"] == "hy4":
                # Compute the resident size from the tensor directory; the table stays in host RAM.
                sys.path.insert(0, str(a.runtime / "gguf-reader"))
                import gguf
                table_bytes = sum(t.n_bytes for p in files for t in gguf.GGUFReader(str(p)).tensors
                                  if t.name == "per_layer_token_embd.weight")
                if not table_bytes:
                    raise ValueError("Missing Hy4 host Engram tensor")
                weight_bytes -= table_bytes
            reserve = max(3 * GIB, free * .10) + (3 * GIB if dspine else 0)
            kv = a.kv_gib * GIB if a.kv_gib else int((free - weight_bytes / len(devices) - reserve) // GIB) * GIB
            if kv < GIB:
                raise ValueError("Insufficient VRAM after weights and working memory")
            extra = ["--enable-prompt-tokens-details", "--enable-per-request-metrics"]
            if len(devices) > 1:
                extra += ["--tensor-parallel-size", str(len(devices))]
            cmd = [str(a.python), str(a.runtime / "deployment/serve.py"), str(files[0]),
                   "--max-num-seqs", str(a.batch), "--max-model-len", str(CONTEXT),
                   "--kv-cache-memory-bytes", str(kv), "--gpu-memory-utilization", "0.94",
                   "--max-num-batched-tokens", "2048", "--port", str(a.port), "--served-model-name", "bench",
                   "--spec-tokens", str(tokens), "--extra-vllm-args", json.dumps(extra)]
            if dspine:
                cmd += ["--dspine", str(a.drafter)]
            if spec["family"] == "hy4":
                # Hy4's model adapter creates the correct GGUF tokenizer. Override the generic LFM fallback.
                cmd += ["--tokenizer", str(files[0])]
            revision = (a.runtime / "benchmark-source.sha256").read_text().strip()
        else:
            if spec["kind"] == "chimachima":
                raise ValueError("llama.cpp has no kaiPTQ ITQ tensor loader")
            if a.llama is None:
                raise ValueError("Set --llama to the pinned llama-server executable")
            if dspine:
                raise ValueError("The Tencent kaiLLM DSpine fixture is not a llama.cpp draft GGUF")
            cmd = [str(a.llama), "-m", str(files[0]), "--host", "127.0.0.1", "--port", str(a.port),
                   "--ctx-size", str(CONTEXT * a.batch), "--parallel", str(a.batch), "-ngl", "999",
                   "--fit", "off", "--flash-attn", "on", "--cache-type-k", "bf16", "--cache-type-v", "bf16",
                   "--cache-ram", "0", "--no-context-shift", "-b", "2048", "-ub", "512"]
            if spec["family"] == "hy4":
                cmd += ["--override-tensor", "per_layer_token_embd.weight=CPU"]
            if a.mode == "mtp5":
                cmd += ["--spec-type", "draft-mtp", "--spec-draft-n-max", "5"]
            elif ngram:
                cmd += ["--spec-type", "ngram-simple", "--spec-ngram-simple-size-n", "2", "--spec-ngram-simple-size-m", str(tokens),
                        "--spec-draft-n-max", str(tokens)]
            elif a.mode == "dspark5":
                if a.drafter is None:
                    raise ValueError("A pinned DSpark GGUF is required")
                cmd += ["--spec-type", "draft-dspark", "--model-draft", str(a.drafter), "--spec-draft-n-max", "5"]
            else:
                cmd += ["--spec-type", "none"]
            revision = subprocess.check_output([str(a.llama), "--version"], stderr=subprocess.STDOUT, text=True).strip()
            if MANIFEST["llama_cpp"]["revision"][:7] not in revision:
                raise ValueError("llama.cpp binary revision differs from manifest: " + revision[-200:])
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", a.port))
        stage = "startup"
        with log.open("w") as f:
            proc = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        url = f"http://127.0.0.1:{a.port}"
        wait_server(proc, url, log)
        stage = "capacity"
        logs = log.read_text(errors="replace")
        cap = None
        if a.engine == "kaiLLM":
            reported = re.findall(r"Maximum concurrency for [\d,]+ tokens per request: (\d+(?:\.\d+)?)x", logs)
            if reported:
                cap = float(reported[-1])
                if cap < a.batch:
                    raise ValueError(f"Full-context KV capacity {cap} sequences < batch {a.batch}")
            else:
                raise ValueError("kaiLLM did not confirm full-context KV capacity")
        envfile = a.out / (key + ".environment.json")
        write_json(envfile, {"gpu_names": [d["name"] for d in devices], "devices": devices,
                             "engine_revision": revision, "llama_source": MANIFEST["llama_cpp"]["revision"],
                             "kai_source": a.kai_commit, "server_argv": cmd, "max_concurrency": cap})
        for domain, output in outputs.items():
            if output.exists():
                continue
            stage = domain
            write_json(stamp, {"state": "running", "stage": stage, "time": time.time()})
            run(SimpleNamespace(prompts=a.banks / (spec["family"] + ".json"), environment=envfile,
                                checkpoint=a.checkpoint, domain=domain, batch=a.batch, engine=a.engine,
                                url=url, timeout=a.timeout, mode=a.mode, output=output))
        write_json(stamp, {"state": "complete", "time": time.time()})
    except Exception as exc:
        write_json(stamp, {"state": "failed", "stage": stage, "reason": str(exc)[-1400:], "time": time.time(),
                           "gpu_names": [d["name"] for d in devices], "checkpoint": a.checkpoint,
                           "engine": a.engine, "mode": a.mode, "batch": a.batch})
        raise
    finally:
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=50)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("weights", "banks", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("runtime", "python", "llama", "drafter"):
        p.add_argument("--" + name, type=Path)
    p.add_argument("--checkpoint", required=True, choices=MANIFEST["models"])
    p.add_argument("--engine", required=True, choices=("kaiLLM", "llama.cpp"))
    p.add_argument("--mode", required=True, choices=DRAFT_TOKENS)
    p.add_argument("--batch", required=True, type=int, choices=BATCHES)
    p.add_argument("--devices", type=int, nargs="+", default=[0])
    p.add_argument("--port", type=int, default=8876)
    p.add_argument("--kv-gib", type=int)
    p.add_argument("--kai-commit", required=True)
    p.add_argument("--timeout", type=int, default=43200)
    main(p.parse_args())
