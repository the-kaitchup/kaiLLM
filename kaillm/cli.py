"""Small serving interface; the numerical runtime is an assembled frozen payload."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from . import __version__
from .cuda_host_compiler import check_host_compiler
from .models import resolve_model


def payload_root():
    return Path(__file__).resolve().parent / "_runtime"


def checkout_source():
    stamp = payload_root() / "checkout-source.sha256"
    return stamp.read_text().strip() if stamp.is_file() else None


def query_gpu(device="0"):
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=compute_cap,memory.total,name", "--format=csv,noheader,nounits", "-i", device],
        capture_output=True, text=True, check=True,
    )
    rows = [line.split(",", 2) for line in completed.stdout.splitlines() if line.strip()]
    if not rows or any(len(row) != 3 for row in rows):
        raise ValueError("Missing CUDA device inventory")
    capabilities = {row[0].strip() for row in rows}
    if len(capabilities) != 1:
        raise ValueError("Select GPUs with the same compute capability for a shared runtime")
    capability = capabilities.pop()
    memory = None if capability == "12.1" and any("N/A" in row[1] for row in rows) else min(int(row[1]) for row in rows)
    return {"compute_capability": capability, "total_memory_MiB": memory,
            "name": ", ".join(row[2].strip() for row in rows)}


def parser():
    result = argparse.ArgumentParser(prog="kaillm", description="Batched inference for GGUF and kaiPTQ models.")
    source = checkout_source()
    version = "kaiLLM " + __version__
    if source:
        version += " (checkout " + source[:12] + ")"
    result.add_argument("--version", action="version", version=version)
    commands = result.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="Inspect installation and GPU availability without loading a model")
    doctor.add_argument("--device", default="0")
    serve = commands.add_parser("serve", help="Start an OpenAI-compatible server for a GGUF checkpoint")
    serve.add_argument("model", help="Local GGUF or Hugging Face owner/repository:filename.gguf")
    serve.add_argument("--filename", help="GGUF filename when model is a Hugging Face repository")
    serve.add_argument("--revision", default="main", help="Hugging Face model revision")
    serve.add_argument("--tokenizer", help="Tokenizer override. By default it follows the GGUF: "
                       "LiquidAI/LFM3-1.7B for LFM3, the GGUF's own tokenizer for Qwen3.5/Qwen3.8")
    serve.add_argument("--max-num-seqs", type=int, default=8)
    serve.add_argument("--max-model-len", type=int, default=8192)
    serve.add_argument("--kv-cache-memory-bytes", type=int)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--served-model-name", default="kaillm")
    serve.add_argument("--device", help="Comma-separated CUDA device IDs or UUIDs")
    serve.add_argument("--cuda-home", help="CUDA toolkit directory. The isolated toolkit is used by default.")
    serve.add_argument("--pipeline-parallel-size", type=int, default=1,
                       help="Number of GPUs to use for successive model layers")
    serve.add_argument("--dry-run", action="store_true", help="Prepare a command without starting the runtime")
    serve.add_argument("--compute-capability", choices=("8.6", "12.0", "12.1"), help=argparse.SUPPRESS)
    return result


def prepare_serve(args, forwarded):
    if any(value <= 0 for value in (args.max_num_seqs, args.max_model_len, args.port)) or args.port > 65535:
        raise ValueError("Sequence capacity, context length and port must be positive")
    if args.compute_capability and not args.dry_run:
        raise ValueError("--compute-capability is available only with --dry-run")
    stages = args.pipeline_parallel_size
    if stages < 1:
        raise ValueError("Pipeline size must be positive")
    if args.device is not None:
        device = args.device
    elif stages == 1:
        device = "0"
    else:
        available = os.environ.get("CUDA_VISIBLE_DEVICES")
        ids = available.split(",") if available is not None else list(map(str, range(stages)))
        device = ",".join(ids[:stages])
    selected = [value.strip() for value in device.split(",") if value.strip()]
    if len(set(selected)) < stages:
        raise ValueError("Select at least one distinct CUDA device per pipeline stage")
    gpu = None if args.compute_capability else query_gpu(device)
    capability = args.compute_capability or gpu["compute_capability"]
    if capability not in ("8.6", "12.0", "12.1"):
        raise ValueError("Supported CUDA compute capabilities are 8.6, 12.0 and 12.1")
    # DSpine adds a 1.5 GB drafter. On GPUs with 16 GB or less, a 1 GiB KV cache (8,192 tokens for one sequence) and
    # 512-token prefill chunks keep the server within the GPU's memory: beyond it, decoding gets about 4x slower.
    small_dspine = any(a == "--dspine" or a.startswith("--dspine=") for a in forwarded) and (
        gpu["total_memory_MiB"] is not None and gpu["total_memory_MiB"] <= 16 * 1024 if gpu else capability == "8.6")
    kv = args.kv_cache_memory_bytes
    if kv is None:
        kv = 2**30 if small_dspine else (2 if capability == "8.6" else 4) * 2**30
    if kv <= 0:
        raise ValueError("KV reservation must be positive")
    root = payload_root()
    launcher = root / "deployment/serve.py"
    if not launcher.is_file():
        raise RuntimeError("The runtime payload is missing. Run bash install.sh from the kaiLLM source or release archive.")
    model = resolve_model(args.model, filename=args.filename, revision=args.revision, dry_run=args.dry_run)
    command = [sys.executable, str(launcher), model,
               "--max-num-seqs", str(args.max_num_seqs),
               "--max-model-len", str(args.max_model_len),
               "--kv-cache-memory-bytes", str(kv),
               "--host", args.host, "--port", str(args.port),
               "--served-model-name", args.served_model_name,
               "--device", device, *forwarded]
    if stages > 1:
        command += ["--pipeline-parallel-size", str(stages)]
    cuda_home = getattr(args, "cuda_home", None)
    if cuda_home is not None:
        command.extend(["--cuda-home", cuda_home])
    if args.tokenizer is not None:
        command[3:3] = ["--tokenizer", args.tokenizer]
    if small_dspine and not any(a == "--max-num-batched-tokens" or a.startswith("--max-num-batched-tokens=")
                                for a in forwarded):
        command.extend(["--max-num-batched-tokens", "512"])
    if args.dry_run:
        command.extend(["--dry-run", "--compute-capability", capability])
    environment = dict(os.environ)
    environment.setdefault("OMP_NUM_THREADS", "8")
    assets = root / "runtime-native-cache/native-assets.json"
    if assets.is_file():
        environment["KAILLM_NATIVE_ASSETS"] = str(assets)
    compiler = check_host_compiler(environment, cuda_home=cuda_home)
    if compiler is not None:
        environment.update(compiler["environment"])
        if not args.dry_run and compiler["status"] != "explicit":
            print("kaiLLM: " + compiler["message"], file=sys.stderr)
    return command, environment


def main(argv=None):
    arguments = parser()
    args, forwarded = arguments.parse_known_args(argv)
    try:
        if args.command == "doctor":
            if forwarded:
                arguments.error("Unrecognized doctor arguments: " + " ".join(forwarded))
            status = {"kaiLLM": __version__, "Python": sys.version.split()[0],
                      "runtime_payload": str(payload_root()),
                      "runtime_payload_present": (payload_root() / "deployment/serve.py").is_file()}
            source = checkout_source()
            if source:
                status["checkout_source"] = source
            try:
                status["GPU"] = query_gpu(args.device)
            except (OSError, subprocess.SubprocessError, ValueError) as error:
                status["GPU_error"] = str(error)
            compiler = check_host_compiler(os.environ)
            if compiler is not None:
                status["CUDA_host_compiler"] = compiler
            print(json.dumps(status, indent=2))
            return 0 if status["runtime_payload_present"] and "GPU" in status else 1
        command, environment = prepare_serve(args, forwarded)
        os.execve(command[0], command, environment)
    except (OSError, subprocess.SubprocessError, ValueError, RuntimeError) as error:
        arguments.exit(2, "kaillm: " + str(error) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
