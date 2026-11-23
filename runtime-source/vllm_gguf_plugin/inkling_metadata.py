# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 Mini (`inkling2_loop`, llama.cpp master layout) metadata and tokenizer from GGUF.

config.json uses Inkling2LoopConfig's fields (vllm-overrides/.../configs/inkling2_loop.py); tokenizer.json is the
GGUF's byte-level BPE (tokenizer.ggml.pre == "inkling2", llama.cpp's GPT-4o pre-tokenizer regex)."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

import gguf
import numpy as np
from filelock import FileLock

from .qwen35_metadata import read_gguf_metadata, resolve_gguf_shards

INKLING_METADATA_VERSION = 2          # 2: generation_config.json (the GGUF's recommended sampling)

# GGUF general.sampling.* -> Hugging Face generation config (llama.cpp applies the same keys as its defaults)
SAMPLING_KEYS = {"general.sampling.temp": "temperature", "general.sampling.top_k": "top_k",
                 "general.sampling.top_p": "top_p", "general.sampling.min_p": "min_p",
                 "general.sampling.penalty_repeat": "repetition_penalty"}
ARCH = "inkling2_loop"

# llama.cpp LLAMA_VOCAB_PRE_TYPE_GPT4O (selected for tokenizer.ggml.pre "inkling2"), original tokenizer.json regex
GPT4O_REGEX = (r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"
               r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"
               r"\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+")


def _req(md, name):
    key = f"{ARCH}.{name}"
    if key not in md:
        raise ValueError(f"Missing required GGUF metadata: {key}")
    return md[key]


def _first(v):
    return int(v[0]) if isinstance(v, (list, tuple, np.ndarray)) else int(v)


def inkling_config_dict(md: dict) -> dict:
    if md.get("general.architecture") != ARCH:
        raise ValueError("Expected an inkling2_loop GGUF checkpoint (llama.cpp master layout)")
    n_layers = int(_req(md, "block_count"))
    heads, heads_kv = _req(md, "attention.head_count"), _req(md, "attention.head_count_kv")
    for v in (heads, heads_kv):
        if isinstance(v, (list, tuple, np.ndarray)) and len(set(int(x) for x in v)) != 1:
            raise ValueError("kaiLLM's Inkling port expects the same head counts in every layer")
    if bool(md.get(f"{ARCH}.loop.varying", False)):
        raise ValueError("Variable loop counts (inkling2_loop.loop.varying) are not supported")
    return {
        "architectures": ["Inkling2LoopForCausalLM"], "model_type": ARCH,
        "vocab_size": int(_req(md, "vocab_size")), "vocab_size_unpadded": int(_req(md, "vocab_size_unpadded")),
        "hidden_size": int(_req(md, "embedding_length")), "num_hidden_layers": n_layers,
        "num_attention_heads": _first(heads), "num_key_value_heads": _first(heads_kv),
        "head_dim": int(_req(md, "attention.key_length")), "num_experts": int(_req(md, "expert_count")),
        "num_experts_per_tok": int(_req(md, "expert_used_count")),
        "moe_intermediate_size": int(_req(md, "expert_feed_forward_length")),
        "intermediate_size": int(md.get(f"{ARCH}.feed_forward_length", _req(md, "expert_feed_forward_length"))),
        "sliding_window": int(_req(md, "attention.sliding_window")),
        "sliding_window_pattern": [bool(x) for x in _req(md, "attention.sliding_window_pattern")],
        "rms_norm_eps": float(_req(md, "attention.layer_norm_rms_epsilon")),
        "max_position_embeddings": int(_req(md, "context_length")),
        "relative_length": int(_req(md, "attention.relative_length")),
        "relative_extent": int(_req(md, "attention.relative_extent")),
        "relative_clip": float(_req(md, "attention.relative_clip")), "qk_clip": float(_req(md, "attention.qk_clip")),
        "log_scale_floor": int(_req(md, "attention.log_scale_floor")),
        "log_scale_alpha": float(_req(md, "attention.log_scale_alpha")),
        "conv_L_cache": int(_req(md, "shortconv.l_cache")), "attn_res_block_size": int(_req(md, "attn_res.block_size")),
        "situ_beta": float(_req(md, "activation.situ_beta")),
        "situ_linear_beta": float(md.get(f"{ARCH}.activation.situ_linear_beta", 0.0)),
        "expert_weights_scale": float(md.get(f"{ARCH}.expert_weights_scale", 0.0)),
        "final_logit_softcapping": float(md.get(f"{ARCH}.final_logit_softcapping", 0.0)),
        "loop_start": int(_req(md, "loop.start")), "loop_end": int(_req(md, "loop.end")),
        "loop_count": int(_req(md, "loop.count")), "loop_gate_rank": int(_req(md, "loop.gate_rank")),
        "tie_word_embeddings": False,
        "eos_token_id": md.get("tokenizer.ggml.eos_token_id"),
        "pad_token_id": md.get("tokenizer.ggml.padding_token_id"),
    }


def inkling_tokenizer_dicts(md: dict) -> tuple[dict, dict]:
    """tokenizer.json + tokenizer_config.json from the GGUF vocabulary, merges and chat template."""
    if md.get("tokenizer.ggml.model") != "gpt2" or md.get("tokenizer.ggml.pre") != "inkling2":
        raise ValueError("Unsupported Inkling GGUF tokenizer (expected gpt2 BPE with pre-tokenizer inkling2)")
    tokens = list(md["tokenizer.ggml.tokens"])
    types = list(md["tokenizer.ggml.token_type"])
    merges = list(md["tokenizer.ggml.merges"])
    added = [{"id": i, "content": t, "single_word": False, "lstrip": False, "rstrip": False,
              "normalized": False, "special": ty == 3}
             for i, (t, ty) in enumerate(zip(tokens, types)) if ty in (3, 4)]
    vocab = {t: i for i, (t, ty) in enumerate(zip(tokens, types)) if ty not in (3, 4)}
    tok = {
        "version": "1.0", "truncation": None, "padding": None, "added_tokens": added,
        "normalizer": None,
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": GPT4O_REGEX}, "behavior": "Isolated", "invert": False},
            {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": False}]},
        "post_processor": None,
        "decoder": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True, "use_regex": True},
        "model": {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": None,
                  "end_of_word_suffix": None, "fuse_unk": False, "byte_fallback": False, "ignore_merges": False,
                  "vocab": vocab, "merges": [m.split(" ", 1) for m in merges]},
    }
    eos, pad = md.get("tokenizer.ggml.eos_token_id"), md.get("tokenizer.ggml.padding_token_id")
    cfg = {"tokenizer_class": "PreTrainedTokenizerFast",
           "model_max_length": int(md.get(f"{ARCH}.context_length", 1 << 20)),
           "add_bos_token": False, "add_eos_token": False, "clean_up_tokenization_spaces": False,
           "eos_token": tokens[eos] if eos is not None else None,
           "pad_token": tokens[pad] if pad is not None else None}
    if md.get("tokenizer.chat_template"):
        cfg["chat_template"] = md["tokenizer.chat_template"]
    return tok, cfg


def inkling_generation_config(md: dict) -> dict:
    """The GGUF's recommended sampling (general.sampling.*) as generation_config.json fields: vLLM serves them as
    request defaults (`--generation-config auto`; `vllm` ignores them), as llama.cpp does with these keys."""
    out = {}
    for key, name in SAMPLING_KEYS.items():
        if key in md:
            v = md[key]
            v = v[0] if isinstance(v, (list, tuple, np.ndarray)) else v
            out[name] = int(v) if name == "top_k" else float(v)
    return out


def prepare_inkling_metadata(model, cache_dir=None):
    """Write config.json, generation_config.json and tokenizer files for an inkling2_loop GGUF; None when `model` is
    something else."""
    shards = resolve_gguf_shards(model)
    reader = gguf.GGUFReader(str(shards[0]))
    md = read_gguf_metadata(reader)
    if md.get("general.architecture") != ARCH:
        return None
    key = hashlib.sha256(json.dumps([str(Path(shards[0]).resolve()), os.path.getsize(shards[0]),
                                     INKLING_METADATA_VERSION]).encode()).hexdigest()[:20]
    root = Path(cache_dir or os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "kaillm" / "inkling-metadata" / key
    with FileLock(str(root) + ".lock"):
        if (root / "config.json").is_file():
            return root
        root.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix=".inkling-", dir=root.parent))
        tok, tok_cfg = inkling_tokenizer_dicts(md)
        (tmp / "config.json").write_text(json.dumps(inkling_config_dict(md), indent=2))
        (tmp / "tokenizer.json").write_text(json.dumps(tok, ensure_ascii=False))
        (tmp / "tokenizer_config.json").write_text(json.dumps(tok_cfg, ensure_ascii=False, indent=2))
        gen = inkling_generation_config(md)
        if gen:
            (tmp / "generation_config.json").write_text(json.dumps(gen, indent=2))
        tmp.replace(root)
    return root
