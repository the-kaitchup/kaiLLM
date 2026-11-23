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
        "--cpu-fit", metavar="CONTEXT",
        help="Fit CONTEXT tokens (e.g. 32768, 128k, max) by enabling lossless memory measures one at a time, "
             "including lookup tables in pinned host memory; without it everything stays on the GPU (see cpu_fit.py)",
    )
    result.add_argument(
        "--kv-cache-dtype", default="auto",
        help="KV-cache dtype (vLLM names, e.g. turboquant_k8v4 / turboquant_4bit_nc: changes outputs, opt-in only)",
    )
    result.add_argument(
        "--kv-cache-dtype-skip-layers", metavar="LIST",
        help="comma-separated layer indices or attention types kept at the model dtype (e.g. sliding_window)",
    )
    result.add_argument(
        "--extra-vllm-args", metavar="JSON",
        help='raw vLLM arguments appended to the command, as a JSON list (e.g. ["--reasoning-parser", "qwen3"])',
    )
    result.add_argument(
        "--speculative-config", metavar="JSON",
        help="vLLM speculative config to use instead of the GGUF's default drafts (e.g. a DSpine / dflash drafter)",
    )
    result.add_argument(
        "--dspine", metavar="DRAFTER",
        help="DSpine speculative decoding, with --max-num-seqs 1: a local drafter directory, a Hugging Face repository, "
             "or REPOSITORY:FOLDER (e.g. kaitchup/Qwen3.8-27B-chimachima-kaiPTQ:dspine-b16-int4); a drafter in "
             "Tencent's format is converted once to 4-bit layers (dspine/prepare.py); --spec-tokens sets the draft "
             "tokens per step (default 7)",
    )
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


def dspine_speculative_config(args):
    """--dspine: vLLM's DFlash method with a DSpine drafter prepared by dspine/prepare.py (not during a dry run)."""
    if getattr(args, "speculative_config", None):
        raise ValueError("Use either --dspine or --speculative-config")
    if args.max_num_seqs != 1:
        raise ValueError("--dspine needs --max-num-seqs 1 in this release")
    tokens = 7 if args.spec_tokens is None else args.spec_tokens
    from .dspine.prepare import resolve

    drafter = resolve(args.dspine, prepare=not getattr(args, "dry_run", False))
    config = drafter / "config.json"
    block = json.loads(config.read_text()).get("block_size", 16) if config.is_file() else 16
    if not 1 <= tokens < block:
        raise ValueError(f"--dspine drafts 1 to {block - 1} tokens per step with this drafter (--spec-tokens)")
    spec = {"method": "dflash", "model": str(drafter), "num_speculative_tokens": tokens,
            "draft_load_config": {"load_format": "auto"}, "quantization": "gptq_marlin"}
    return spec, tokens


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
    if getattr(args, "dspine", None):
        spec, tokens = dspine_speculative_config(args)
    elif speculation == "mtp":
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
    if getattr(args, "speculative_config", None):
        spec = json.loads(args.speculative_config)
        tokens = int(spec.get("num_speculative_tokens", tokens))
    captures = policy.capture_family(
        args.max_num_seqs, tokens, args.max_num_batched_tokens
    )
    # families whose attention replays decode batches only (FULL_DECODE_ONLY): prefill and mixed steps run eagerly
    graph_mode = family.get("cudagraph_mode", "FULL_AND_PIECEWISE")
    prefill_graphs = os.getenv("KAILLM_PREFILL_GRAPHS", "1") != "0" and graph_mode != "FULL_DECODE_ONLY"
    if not prefill_graphs:
        # decode / verification graphs only: prefill chunks run eagerly (their graphs hold the largest activation
        # pool; ~1 % of a chunk's time on an RTX 3060), which leaves that memory to the KV cache
        rows = args.max_num_seqs * (tokens + 1)
        captures = [c for c in captures if c <= 2 * rows] or [rows]
        if graph_mode == "FULL_DECODE_ONLY":
            # decode graphs only: finer sizes (a step runs in the next captured size, padded rows cost MoE reads)
            captures = sorted({c for c in range(1, min(rows, 8) + 1)} | set(range(8, rows + 1, 4)) | {rows})
    graph = {
        "mode": 0, "cudagraph_mode": graph_mode,
        "cudagraph_capture_sizes": captures,
        "max_cudagraph_capture_size": max(captures),
    }
    command = [
        executable, "-m", "vllm.entrypoints.cli.main", "serve", args.model,
        "--host", args.host, "--port", str(args.port),
        "--quantization", "gguf", "--load-format", "gguf", "--dtype", "bfloat16",
        *(() if str(getattr(args, "kv_cache_dtype", "auto")).startswith(("turboquant_", "ultraquant_"))
          else ("--attention-config", json.dumps({"backend": "TRITON_ATTN"}))),   # packed KV: vLLM picks its backend
        "--kv-cache-dtype", getattr(args, "kv_cache_dtype", "auto") or "auto", "--mamba-cache-dtype", "auto",
        "--max-num-seqs", str(args.max_num_seqs),
        "--max-num-batched-tokens", str(args.max_num_batched_tokens),
        "--async-scheduling", "--long-prefill-token-threshold", "0",
        "--no-enable-prefix-caching",
        "--kv-cache-memory-bytes", str(args.kv_cache_memory_bytes),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        *(("--enforce-eager",) if family.get("eager") else
          ("--compilation-config", json.dumps(graph),
           "--additional-config", json.dumps({"cudagraph_specialize_prefill": prefill_graphs}))),
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
    if getattr(args, "kv_cache_dtype_skip_layers", None):
        command += ["--kv-cache-dtype-skip-layers", *args.kv_cache_dtype_skip_layers.split(",")]
    if getattr(args, "extra_vllm_args", None):
        command += [str(x) for x in json.loads(args.extra_vllm_args)]
    environment = dict(selected["environment"])
    environment.update(family.get("environment", {}))
    if getattr(args, "dspine", None):
        # The paths of the DSpine measurements, unless set: one GDN state per request with the deferred commit
        # kernel, CUDA split-KV verification attention, and KV groups sized for the drafter's layers.
        environment.setdefault("KAILLM_GDN_SINGLE_SLOT", "1")
        environment.setdefault("KAILLM_VERIFY_ATTN_CUDA", "1")
        environment.setdefault("KAILLM_KV_GROUPS", "auto")
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
