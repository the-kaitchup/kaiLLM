"""Small serving interface; the numerical runtime is an assembled frozen payload."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from . import __version__
from .models import resolve_model


def payload_root():
    return Path(__file__).resolve().parent / "_runtime"


def query_gpu(device="0"):
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=compute_cap,memory.total,name", "--format=csv,noheader,nounits", "-i", device],
        capture_output=True, text=True, check=True,
    )
    fields = completed.stdout.strip().split(",", 2)
    return {"compute_capability": fields[0].strip(),
            "total_memory_MiB": int(fields[1]), "name": fields[2].strip()}


def parser():
    result = argparse.ArgumentParser(prog="kaillm", description="Batched GGUF and YAQA/GGUF inference for LFM3 and Qwen3.5/Qwen3.8.")
    result.add_argument("--version", action="version", version="kaiLLM " + __version__)
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
    serve.add_argument("--device", default="0")
    serve.add_argument("--dry-run", action="store_true", help="Prepare a command without starting the runtime")
    serve.add_argument("--compute-capability", choices=("8.6", "12.0"), help=argparse.SUPPRESS)
    return result


def prepare_serve(args, forwarded):
    if any(value <= 0 for value in (args.max_num_seqs, args.max_model_len, args.port)) or args.port > 65535:
        raise ValueError("Sequence capacity, context length and port must be positive")
    if args.compute_capability and not args.dry_run:
        raise ValueError("--compute-capability is available only with --dry-run")
    capability = args.compute_capability or query_gpu(args.device)["compute_capability"]
    if capability not in ("8.6", "12.0"):
        raise ValueError("This release is qualified for compute capabilities 8.6 and 12.0")
    kv = args.kv_cache_memory_bytes
    if kv is None:
        kv = (2 if capability == "8.6" else 4) * 2**30
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
               "--device", args.device, *forwarded]
    if args.tokenizer is not None:
        command[3:3] = ["--tokenizer", args.tokenizer]
    if args.dry_run:
        command.extend(["--dry-run", "--compute-capability", capability])
    environment = dict(os.environ)
    environment.setdefault("OMP_NUM_THREADS", "8")
    assets = root / "runtime-native-cache/native-assets.json"
    if assets.is_file():
        environment["KAILLM_NATIVE_ASSETS"] = str(assets)
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
            try:
                status["GPU"] = query_gpu(args.device)
            except (OSError, subprocess.SubprocessError, ValueError) as error:
                status["GPU_error"] = str(error)
            print(json.dumps(status, indent=2))
            return 0 if status["runtime_payload_present"] and "GPU" in status else 1
        command, environment = prepare_serve(args, forwarded)
        os.execve(command[0], command, environment)
    except (OSError, subprocess.SubprocessError, ValueError, RuntimeError) as error:
        arguments.exit(2, "kaillm: " + str(error) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
