# SPDX-License-Identifier: Apache-2.0
"""--cpu-fit CONTEXT: fit a requested context length by enabling memory measures one at a time (opt-in).

Without --cpu-fit the server keeps its usual plan (everything on the GPU, the KV reservation given on the command
line).  With it, the launcher tries the rungs below in order, each adding to the previous ones, and starts vLLM with
the largest KV reservation the GPU can hold at that rung.  vLLM itself checks whether CONTEXT fits ("estimated maximum
model length is N"); a rung that cannot hold CONTEXT moves on to the next one, an out-of-memory start retries the same
rung with a smaller reservation.  The plan that started is remembered (~/.cache/kaillm/cpu-fit.json) and reused.

Rungs (all keep the same outputs; the CPU ones keep tables in pinned host memory read over PCIe):
  1. gpu-lossless  one GDN state block per request for batch-1 verification (KAILLM_GDN_SINGLE_SLOT) and a KV-cache
                   group size without padding on full attention (KAILLM_KV_GROUPS=auto).  No CPU memory.
  2. host-embed    untied input embedding rows in host memory (KAILLM_HOST_EMBEDDING); ~20 us per step.
  3. host-draft    DSpine TSV / TG lookup tables as int8 rows in host memory (KAILLM_DSPINE_INT8_TABLES +
                   KAILLM_DSPINE_HOST_TABLES), only with a DSpine (dflash) drafter.
KV-cache quantization and KV offloading are not rungs: they change outputs or speed and need their own flags.

CONTEXT: a token count (32768, 128k) or "max" (the largest context the last rung can hold).
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

GIB = 2 ** 30
MIB = 2 ** 20
CACHE = Path(os.getenv("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "kaillm" / "cpu-fit.json"

RUNGS = [
    ("gpu-lossless", {"KAILLM_GDN_SINGLE_SLOT": "1", "KAILLM_KV_GROUPS": "auto"}),
    ("host-embed", {"KAILLM_HOST_EMBEDDING": "1"}),
    ("host-draft", {"KAILLM_DSPINE_INT8_TABLES": "1", "KAILLM_DSPINE_HOST_TABLES": "1"}),
]
# Calibrated on the RTX 3060 (Qwen3.8 7.0 GB ITQ + DSpine 7, 256-token prefill chunks, 7 capture sizes): CUDA
# context, kernels, graph pools and cached activation blocks besides the weights and the KV cache.
RUNTIME_OVERHEAD = float(os.getenv("KAILLM_CPU_FIT_OVERHEAD_GIB", "1.7")) * GIB
HEADROOM = float(os.getenv("KAILLM_CPU_FIT_HEADROOM_GIB", "0.35")) * GIB     # desktop / driver slack


def parse_context(text: str) -> int | None:
    """'128k' -> 131072, '32768' -> 32768, 'max' -> None."""
    text = str(text).strip().lower()
    if text == "max":
        return None
    match = re.fullmatch(r"(\d+)\s*([km]?)", text)
    if not match:
        raise ValueError(f"--cpu-fit expects a token count (32768, 128k) or 'max', got {text!r}")
    value = int(match.group(1)) * {"": 1, "k": 1024, "m": 1024 * 1024}[match.group(2)]
    if value < 256:
        raise ValueError("--cpu-fit context must be at least 256 tokens")
    return value


def free_vram(device: str | None) -> tuple[int, int]:
    """(free, total) bytes from nvidia-smi for the selected device (before this server allocates anything)."""
    index = (device or os.getenv("CUDA_VISIBLE_DEVICES") or "0").split(",")[0]
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits",
                          "-i", index], capture_output=True, text=True, check=True).stdout
    total, used = (int(x) for x in out.strip().split(","))
    return (total - used) * MIB, total * MIB


def _gguf_tensor_bytes(path: Path) -> dict:
    """Byte sizes of a GGUF's tensors by name (header only)."""
    from .serving_defaults import read_header

    try:
        _, names = read_header(path)
    except Exception:
        return {}
    # read_header gives names only; sizes come from the gguf reader when available
    try:
        reader_root = Path(__file__).resolve().parents[1] / "gguf-reader"
        if reader_root.is_dir() and str(reader_root) not in sys.path:
            sys.path.insert(0, str(reader_root))
        from gguf import GGUFReader

        reader = GGUFReader(str(path))
        return {t.name: int(t.n_bytes) for t in reader.tensors}
    except Exception:
        return {name: 0 for name in names}


def _mtp_bytes(path: Path, sizes: dict) -> int:
    """Bytes of the trailing MTP (nextn) blocks: blk.N.* for N >= block_count - nextn_predict_layers."""
    from .serving_defaults import read_header

    try:
        metadata, _ = read_header(path)
    except Exception:
        return 0
    arch = metadata.get("general.architecture", "")
    blocks = int(metadata.get(f"{arch}.block_count", 0) or 0)
    nextn = int(metadata.get(f"{arch}.nextn_predict_layers", 0) or 0)
    if not blocks or not nextn:
        return 0
    first = blocks - nextn
    total = 0
    for name, size in sizes.items():
        m = re.match(r"blk\.(\d+)\.", name)
        if m and int(m.group(1)) >= first:
            total += size
    return total


def estimate_weights(model: str, spec: dict | None, env: dict) -> int:
    """GPU bytes of the weights a rung leaves on the GPU (target + drafter), from the files."""
    path = Path(model)
    total = path.stat().st_size if path.is_file() else 0
    sizes = _gguf_tensor_bytes(path) if path.is_file() else {}
    method = (spec or {}).get("method")
    if method != "mtp":
        total -= _mtp_bytes(path, sizes)                                     # MTP layers are not loaded
    if env.get("KAILLM_HOST_EMBEDDING") == "1":
        total -= sum(v for k, v in sizes.items() if k.startswith("token_embd"))
    total = int(total * 1.08)                                                # norms, dense copies, tables (3060: 7%)
    if method == "dflash" and spec.get("model"):
        draft = Path(spec["model"])
        files = list(draft.glob("*.safetensors")) if draft.is_dir() else []
        dbytes = sum(f.stat().st_size for f in files)
        cfg = json.loads((draft / "config.json").read_text()) if (draft / "config.json").is_file() else {}
        bridge = (cfg.get("dflash_config", {}) or {}).get("selected_prefix_code_bridge", {}) or {}
        vocab, code, rank = int(cfg.get("vocab_size", 0)), int(bridge.get("code_size", 0)), int(bridge.get("rank", 0))
        dbytes -= vocab * code * 2                                           # codebook: freed once TSV is built
        int8 = env.get("KAILLM_DSPINE_INT8_TABLES") == "1"
        tables = vocab * 3 * rank * (1 if int8 else 2)                       # TSV [V, 2R] + TG [V, R]
        if not (int8 and env.get("KAILLM_DSPINE_HOST_TABLES") == "1"):
            dbytes += tables
        total += max(dbytes, 0)
    return total


def model_max_context(model: str) -> int:
    """The GGUF's trained context (general.architecture + '.context_length'), the ceiling for 'max'."""
    from .serving_defaults import read_header

    try:
        metadata, _ = read_header(Path(model))
        arch = metadata.get("general.architecture", "")
        return int(metadata.get(f"{arch}.context_length", 0) or 0) or 131072
    except Exception:
        return 131072


def _key(model: str, spec: dict | None, context: int | None, gpu_total: int) -> str:
    path = Path(model)
    size = path.stat().st_size if path.is_file() else 0
    return json.dumps([str(path.resolve()) if path.is_file() else model, size, spec, context, gpu_total],
                      sort_keys=True)


def _load_cache() -> dict:
    try:
        return json.loads(CACHE.read_text())
    except Exception:
        return {}


def _save_cache(key: str, plan: dict) -> None:
    data = _load_cache()
    data[key] = plan
    try:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(data, indent=1))
    except OSError:
        pass


def _set_option(command: list, option: str, value: str) -> list:
    command = list(command)
    if option in command:
        command[command.index(option) + 1] = value
    else:
        command += [option, value]
    return command


def _speculative(command: list) -> dict | None:
    if "--speculative-config" in command:
        return json.loads(command[command.index("--speculative-config") + 1])
    return None


def _attempt(command: list, environment: dict) -> tuple[str, dict, subprocess.Popen | None]:
    """Start vLLM; return ("ok", info, proc) once it serves, ("context", info, None) when the context does not fit
    (info: max_len, needed bytes for the requested context), ("oom", ...) on an out-of-memory start, ("error", ...)
    otherwise.  vLLM's output is echoed."""
    proc = subprocess.Popen(command, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            bufsize=1, start_new_session=True)
    info, verdict = {}, "error"
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        m = re.search(r"max seq len \((\d+)\), \(([\d.]+) GiB KV cache is needed", line)
        if m:
            info["context"], info["needed"] = int(m.group(1)), float(m.group(2)) * GIB
        m = re.search(r"estimated maximum model length is (\d+)", line)
        if m:
            info["max_len"] = int(m.group(1))
            verdict = "context"
        if verdict != "context" and ("CUDA out of memory" in line or "OutOfMemoryError" in line
                                     or "cudaErrorMemoryAllocation" in line):
            verdict = "oom"
        if "Application startup complete" in line:
            return "ok", info, proc
    proc.wait()
    return verdict, info, None


def _stop(proc: subprocess.Popen, device, free_before: int) -> None:
    """Stop a probe server and wait until its GPU memory is back."""
    for sig, wait in ((signal.SIGTERM, 60), (signal.SIGKILL, 30)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            break
        try:
            proc.wait(timeout=wait)
            break
        except subprocess.TimeoutExpired:
            continue
    for _ in range(60):
        if free_vram(device)[0] >= free_before - 512 * MIB:
            return
        time.sleep(2)


def run(prepared: dict, args) -> int:
    """Fit loop.  ``prepared`` = serve.prepare_command output (command + environment)."""
    target = parse_context(args.cpu_fit)
    command, base_env = list(prepared["command"]), dict(prepared["environment"])
    spec = _speculative(command)
    free, total = free_vram(args.device)
    key = _key(args.model, spec, target, total)
    rungs = [r for r in RUNGS if r[0] != "host-draft" or (spec or {}).get("method") == "dflash"]
    if args.max_num_seqs != 1:
        rungs = [(n, {k: v for k, v in e.items() if k != "KAILLM_GDN_SINGLE_SLOT"}) for n, e in rungs]
    names = [n for n, _ in rungs]
    envs, env = [], dict(base_env)
    for _, extra in rungs:
        env = {**env, **extra}
        envs.append(env)
    weights = [estimate_weights(args.model, spec, e) for e in envs]

    def launch(index, kv, context):
        kv = int(kv // MIB * MIB)
        cmd = _set_option(_set_option(command, "--kv-cache-memory-bytes", str(kv)), "--max-model-len", str(context))
        print(f"kaiLLM cpu-fit: trying {' + '.join(names[:index + 1])}: KV {kv / GIB:.2f} GiB, context {context}",
              flush=True)
        return (*_attempt(cmd, envs[index]), kv)

    def serve(index, kv, context, proc):
        _save_cache(key, {"rung": names[index], "kv_bytes": kv, "context": context,
                          "measures": names[:index + 1], "time": time.strftime("%Y-%m-%d %H:%M")})
        print(f"kaiLLM cpu-fit: serving {context} tokens with {' + '.join(names[:index + 1])} "
              f"(KV {kv / GIB:.2f} GiB)", flush=True)
        return _supervise(proc)

    cached = _load_cache().get(key)
    if cached and cached.get("rung") in names:
        index = names.index(cached["rung"])
        print(f"kaiLLM cpu-fit: remembered plan {cached['rung']}, KV {cached['kv_bytes'] / GIB:.2f} GiB, "
              f"context {cached['context']}", flush=True)
        verdict, info, proc, kv = launch(index, cached["kv_bytes"], cached["context"])
        if verdict == "ok":
            return serve(index, kv, cached["context"], proc)
        free = free_vram(args.device)[0]

    overhead, caps, per_token, fixed, best, force_context = RUNTIME_OVERHEAD, {}, None, 0.0, 0, None
    ceiling = model_max_context(args.model)
    if target is not None and target > ceiling:
        print(f"kaiLLM cpu-fit: {target} tokens exceed the model's trained context ({ceiling})", flush=True)
        return 2
    index, ooms = 0, 0
    while index < len(rungs):
        kv = caps.get(index, free - HEADROOM - overhead - weights[index])
        if kv < 256 * MIB:
            print(f"kaiLLM cpu-fit: {names[index]}: no room for a KV cache ({kv / MIB:.0f} MiB)", flush=True)
            index += 1
            continue
        context = target if target is not None else ceiling
        if target is None and per_token:
            context = min(ceiling, max(256, int((kv - fixed) / per_token * 0.98)))
        if force_context:
            context = force_context
        verdict, info, proc, kv = launch(index, kv, context)
        if verdict == "ok":
            return serve(index, kv, context, proc)
        if verdict == "oom":
            ooms += 1
            if ooms > 4:
                print("kaiLLM cpu-fit: repeated out-of-memory starts; giving up", flush=True)
                return 1
            overhead += 256 * MIB
            caps = {j: c - 256 * MIB for j, c in caps.items()}
            continue
        if verdict != "context" or not info.get("max_len"):
            print(f"kaiLLM cpu-fit: vLLM failed at {names[index]} for another reason; see the log above", flush=True)
            return 1
        best = max(best, info["max_len"])
        if info.get("needed") and info.get("context") and info["context"] > info["max_len"]:
            # two points of vLLM's own accounting: `needed` bytes for `context` tokens, `kv` bytes -> `max_len`
            per_token = (info["needed"] - kv) / (info["context"] - info["max_len"])
            fixed = max(kv - per_token * info["max_len"], 0.0)
        if caps:                                   # already calibrated: this rung cannot hold it, try the next
            if target is None and index == len(rungs) - 1 and force_context != info["max_len"]:
                force_context = info["max_len"]    # 'max': vLLM's own maximum at the calibrated reservation
                continue
            index += 1
            continue
        # Calibrate once: start at a context that fits, read the VRAM left after graph capture, stop.
        print(f"kaiLLM cpu-fit: calibrating at {names[index]} with context {info['max_len']}", flush=True)
        verdict2, _, probe, kv2 = launch(index, kv, info["max_len"])
        if verdict2 != "ok":
            index += 1
            continue
        left = free_vram(args.device)[0]
        _stop(probe, args.device, free)
        cap = kv2 + max(left - HEADROOM, 0)
        caps = {j: cap + (weights[index] - weights[j]) for j in range(len(rungs))}
        print("kaiLLM cpu-fit: calibrated KV capacity per rung: " + ", ".join(
            f"{names[j]} {caps[j] / GIB:.2f} GiB (~{int((caps[j] - fixed) / per_token)} tokens)"
            for j in range(len(rungs))),
              flush=True)
        if target is None:
            index = len(rungs) - 1
            continue
        fits = [j for j in range(len(rungs)) if (caps[j] - fixed) / per_token >= target * 1.02]
        if not fits:
            best = max(best, int((caps[len(rungs) - 1] - fixed) / per_token))
            break
        index = fits[0]
    print(f"kaiLLM cpu-fit: {args.cpu_fit} tokens do not fit with every lossless measure (largest context: about "
          f"{best} tokens).  Going further changes outputs or speed (KV-cache quantization, KV offloading), which "
          f"--cpu-fit does not enable.", flush=True)
    return 2


def _supervise(proc: subprocess.Popen) -> int:
    def forward(signum, frame):
        try:
            os.killpg(proc.pid, signum)
        except ProcessLookupError:
            pass
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, forward)
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
    return proc.wait()
