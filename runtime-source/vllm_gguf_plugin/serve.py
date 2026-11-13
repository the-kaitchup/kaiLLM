# SPDX-License-Identifier: Apache-2.0
"""Serve GGUF models with the installed hardware catalog and stock vLLM CLI."""

import argparse
import json
import os
import sys
from pathlib import Path

from . import serving_defaults, serving_policy as policy


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def nonnegative(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return number


def parser():
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    result.add_argument("model", help="GGUF file or supported GGUF reference")
    result.add_argument("--max-num-seqs", type=positive, default=8)
    result.add_argument("--max-num-batched-tokens", type=positive, default=2048)
    result.add_argument("--max-model-len", type=positive)
    result.add_argument("--kv-cache-memory-bytes", type=positive, required=True)
    result.add_argument("--gpu-memory-utilization", type=float, default=0.65)
    result.add_argument("--maximum-speculative-tokens", type=nonnegative, default=3)
    result.add_argument("--verifier-row-budget", type=positive, default=16)
    result.add_argument("--spec-tokens", type=nonnegative)
    result.add_argument("--lookup-min", type=positive, default=2)
    result.add_argument("--lookup-max", type=positive, default=8)
    result.add_argument("--tokenizer")
    result.add_argument("--hf-config-path")
    result.add_argument("--served-model-name")
    result.add_argument("--host", default="127.0.0.1")
    result.add_argument("--port", type=positive, default=8000)
    result.add_argument("--device", help="CUDA_VISIBLE_DEVICES selection")
    result.add_argument("--cpu-cache-dir")
    result.add_argument(
        "--catalog", type=Path,
        default=Path(__file__).parent / "native/serving-catalog.json",
    )
    result.add_argument(
        "--dry-run", action="store_true",
        help="Print the verified command without starting vLLM",
    )
    result.add_argument(
        "--compute-capability",
        help="Explicit target for a CPU-only dry run, for example 12.0",
    )
    return result


def prepare_command(args, selected, executable=sys.executable, family=None):
    """Build CLI arguments without importing the model runtime.

    ``family`` holds the GGUF serving defaults (serving_defaults.inspect): GGUFs
    with MTP layers default to MTP drafts, LFM3 to the catalog's n-gram drafts,
    other families to no drafts unless --spec-tokens is given.
    """
    if not 0 < args.gpu_memory_utilization <= 1 or args.port > 65535:
        raise ValueError("Require memory utilization in (0,1] and a valid port")
    family = family or {}
    speculation = family.get("speculation", "ngram")
    if speculation == "mtp":
        tokens = policy.choose_mtp_tokens(args.max_num_seqs)
        if args.spec_tokens is not None:
            tokens = args.spec_tokens
        spec = policy.mtp_config(tokens)
    elif speculation == "none" and args.spec_tokens is None:
        tokens, spec = 0, None
    else:
        tokens = policy.choose_speculative_tokens(
            args.max_num_seqs, args.maximum_speculative_tokens, args.verifier_row_budget
        )
        if args.spec_tokens is not None:
            tokens = args.spec_tokens
        spec = policy.speculative_config(
            tokens, selected["conditional_adapter"], args.lookup_min, args.lookup_max
        )
    captures = policy.capture_family(
        args.max_num_seqs, tokens, args.max_num_batched_tokens
    )
    graph = {
        "mode": 0, "cudagraph_mode": "FULL_AND_PIECEWISE",
        "cudagraph_capture_sizes": captures,
        "max_cudagraph_capture_size": max(captures),
    }
    command = [
        executable, "-m", "vllm.entrypoints.cli.main", "serve", args.model,
        "--host", args.host, "--port", str(args.port),
        "--quantization", "gguf", "--load-format", "gguf", "--dtype", "bfloat16",
        "--attention-config", json.dumps({"backend": "TRITON_ATTN"}),
        "--kv-cache-dtype", "auto", "--mamba-cache-dtype", "auto",
        "--max-num-seqs", str(args.max_num_seqs),
        "--max-num-batched-tokens", str(args.max_num_batched_tokens),
        "--async-scheduling", "--long-prefill-token-threshold", "0",
        "--no-enable-prefix-caching",
        "--kv-cache-memory-bytes", str(args.kv_cache_memory_bytes),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--compilation-config", json.dumps(graph),
        "--additional-config", json.dumps({"cudagraph_specialize_prefill": True}),
    ]
    for option, value in (
        ("--max-model-len", args.max_model_len), ("--tokenizer", args.tokenizer),
        ("--hf-config-path", args.hf_config_path),
        ("--served-model-name", args.served_model_name),
    ):
        if value is not None:
            command += [option, str(value)]
    if spec is not None:
        command += ["--speculative-config", json.dumps(spec)]
    environment = dict(selected["environment"])
    environment.update(family.get("environment", {}))
    if args.device is not None:
        environment["CUDA_VISIBLE_DEVICES"] = args.device
    if args.cpu_cache_dir is not None:
        environment["VLLM_GGUF_YAQA_PREFILL_CPU_CACHE_DIR"] = args.cpu_cache_dir
    return {
        "command": command, "environment": environment,
        "num_speculative_tokens": tokens, "capture_sizes": captures,
        "speculative_method": spec["method"] if spec else None,
        "model_architecture": family.get("architecture"),
    }


def main(argv=None):
    arguments = parser()
    args = arguments.parse_args(argv)
    if args.compute_capability is not None and not args.dry_run:
        arguments.error("--compute-capability is only available with --dry-run")
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    try:
        capability = args.compute_capability or policy.discover_capability()
        selected = policy.load_target(
            args.catalog, capability, Path(__file__).parent, os.environ
        )
        core = policy.verify_core_contract(selected["core_contract"], sys.path)
        prepared = prepare_command(args, selected, family=serving_defaults.inspect(args.model))
    except (OSError, ValueError, RuntimeError) as error:
        arguments.error(str(error))
    if args.dry_run:
        visible = {
            key: value for key, value in prepared["environment"].items()
            if os.environ.get(key) != value
        }
        print(json.dumps({
            **prepared, "environment": visible,
            "cleared_environment_keys": sorted(
                set(os.environ) - set(prepared["environment"])
            ),
            "compute_capability": capability,
            "verified_native_assets": selected["verified_native_assets"],
            "core_contract": core, "catalog_scope": selected["catalog_scope"],
            "target_scope": selected["target_scope"], "runtime_started": False,
        }, indent=2))
        return
    os.execve(prepared["command"][0], prepared["command"], prepared["environment"])


if __name__ == "__main__":
    main()
