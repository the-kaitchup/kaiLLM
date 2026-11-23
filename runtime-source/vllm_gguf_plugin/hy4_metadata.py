# SPDX-License-Identifier: Apache-2.0
"""Hy4 flash (`hy_v4_flash`) metadata and tokenizer conversion from GGUF (llama.cpp PR 32517 layout).

The GGUF keys are mapped back to Tencent's config.json field names, so the config that results is
the one the released checkpoint ships (text model only). Engram's hash inputs are kept exactly as
the GGUF stores them: the token map goes to `engram_vocab_map.npz` next to config.json and the
per-layer multipliers to `engram_multipliers`.
"""

import hashlib
import json
import os
import tempfile
from pathlib import Path

import gguf
import numpy as np
from filelock import FileLock

from .qwen35_metadata import read_gguf_metadata, resolve_gguf_shards

HY4_METADATA_VERSION = 1
ARCH = "hy_v4_flash"

# Tencent tokenizer.json pre-tokenizer (tokenizer.ggml.pre == "hy_v4")
HY4_PRETOKENIZERS = [
    r"\p{N}{1,3}",
    "[一-龥぀-ゟ゠-ヿ]+",
    r"[!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+|"
    r" ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
]


def _req(md, name):
    key = f"{ARCH}.{name}"
    if key not in md:
        raise ValueError(f"Missing required GGUF metadata: {key}")
    return md[key]


def _as_list(v, n):
    return list(v) if isinstance(v, (list, tuple, np.ndarray)) else [v] * n


def hy4_config_dict(md: dict, tensor_names: set[str]) -> dict:
    if md.get("general.architecture") != ARCH:
        raise ValueError("Expected a hy_v4_flash GGUF checkpoint")
    n_layers = int(_req(md, "block_count"))
    heads = [int(h) for h in _as_list(_req(md, "attention.head_count"), n_layers)]
    swa = [bool(x) for x in _as_list(_req(md, "attention.sliding_window_pattern"), n_layers)]
    idx_types = [bool(x) for x in _as_list(md.get(f"{ARCH}.attention.indexer.types", [not s for s in swa]), n_layers)]
    sparse_heads = {h for h, s in zip(heads, swa) if not s}
    swa_heads = {h for h, s in zip(heads, swa) if s}
    if len(sparse_heads) != 1 or len(swa_heads) > 1:
        raise ValueError(f"Unexpected per-layer head counts {sorted(set(heads))}")
    rope_dim = int(_req(md, "rope.dimension_count"))
    qk_head = int(_req(md, "attention.key_length_mla"))
    v_head = int(_req(md, "attention.value_length_mla"))
    dense = int(md.get(f"{ARCH}.leading_dense_block_count", 1))
    hc = int(_req(md, "hyper_connection.count"))
    # Engram: llama.cpp stores 0-based layers and a flat per-head bucket list; Tencent's config uses
    # 1-based layer ids and [ngram order][head] nesting.
    ple_layers = [int(x) for x in _as_list(md.get(f"{ARCH}.ple.layers", []), 0)]
    n_gram = int(md.get(f"{ARCH}.ple.ngram_size", 3))
    n_head = int(md.get(f"{ARCH}.ple.heads_per_ngram", 8))
    sizes = [int(x) for x in md.get(f"{ARCH}.ple.head_vocab_sizes", [])]
    per_layer = (n_gram - 1) * n_head
    if ple_layers and len(sizes) != per_layer * len(ple_layers):
        raise ValueError("Engram bucket counts do not match the Engram layers")
    vocab_sizes = {}
    for k, lid in enumerate(ple_layers):
        flat = sizes[k * per_layer:(k + 1) * per_layer]
        vocab_sizes[str(lid + 1)] = [flat[o * n_head:(o + 1) * n_head] for o in range(n_gram - 1)]
    multipliers = [int(x) for x in md.get(f"{ARCH}.ple.layer_multipliers", [])]
    theta_sparse = float(_req(md, "rope.freq_base"))
    theta_swa = float(md.get(f"{ARCH}.rope.freq_base_swa", theta_sparse))
    clamp = md.get(f"{ARCH}.swiglu_clamp_exp")
    swiglu_limit = float(_as_list(clamp, 1)[0]) if clamp is not None else 10.0
    top_k = int(md.get(f"{ARCH}.attention.indexer.top_k", 2048))
    kpool = int(md.get(f"{ARCH}.attention.indexer.kpool", 4))
    vocab = int(md.get(f"{ARCH}.vocab_size", len(md.get("tokenizer.ggml.tokens", []))))
    cfg = {
        "architectures": ["HYV4FlashForCausalLM"],
        "model_type": ARCH,
        "vocab_size": vocab,
        "hidden_size": int(_req(md, "embedding_length")),
        "intermediate_size": int(_req(md, "feed_forward_length")),
        "moe_intermediate_size": int(_req(md, "expert_feed_forward_length")),
        "routed_expert_hidden_size": int(md.get(f"{ARCH}.expert_latent_length", 0)) or None,
        "num_hidden_layers": n_layers,
        "num_attention_heads": sparse_heads.pop(),
        "num_key_value_heads": None,
        "swa_num_attention_heads": swa_heads.pop() if swa_heads else None,
        "q_lora_rank": int(_req(md, "attention.q_lora_rank")),
        "kv_lora_rank": int(_req(md, "attention.kv_lora_rank")),
        "qk_rope_head_dim": rope_dim, "swa_qk_rope_head_dim": rope_dim,
        "qk_nope_head_dim": qk_head - rope_dim, "swa_qk_nope_head_dim": qk_head - rope_dim,
        "qk_head_dim": qk_head, "swa_qk_head_dim": qk_head,
        "v_head_dim": v_head, "swa_v_head_dim": v_head,
        "head_dim": rope_dim,
        "n_routed_experts": int(_req(md, "expert_count")),
        "num_experts_per_tok": int(_req(md, "expert_used_count")),
        "n_shared_experts": int(md.get(f"{ARCH}.expert_shared_count", 1)),
        "routed_scaling_factor": float(md.get(f"{ARCH}.expert_weights_scale", 1.0)),
        "norm_topk_prob": bool(md.get(f"{ARCH}.expert_weights_norm", True)),
        "moe_router_use_sigmoid": int(md.get(f"{ARCH}.expert_gating_func", 2)) == 2,
        "n_group": 1, "topk_group": 1,
        "first_k_dense_replace": dense,
        "mlp_layer_types": ["dense"] * dense + ["sparse"] * (n_layers - dense),
        "layer_types": ["sliding_attention" if s else "indexed_attention" for s in swa],
        "swa_pattern": [int(s) for s in swa],
        "indexer_types": ["full" if x else "none" for x in idx_types],
        "sliding_window": int(md.get(f"{ARCH}.attention.sliding_window", 512)),
        "compressed_indexer_num_attention_heads": int(md.get(f"{ARCH}.attention.indexer.head_count", 16)),
        "compressed_indexer_head_dim": int(md.get(f"{ARCH}.attention.indexer.key_length", 128)),
        "compressed_indexer_candidate_topk": top_k,
        "compressed_indexer_compress_ratio": kpool,
        "compressed_indexer_group_topk": top_k // kpool,
        "compressed_indexer_select_tail": bool(md.get(f"{ARCH}.attention.indexer.kpool_select_tail", True)),
        "rope_parameters": {
            "indexed_attention": {"rope_type": "default", "rope_theta": theta_sparse},
            "sliding_attention": {"rope_type": "default", "rope_theta": theta_swa},
        },
        "rms_norm_eps": float(_req(md, "attention.layer_norm_rms_epsilon")),
        "max_position_embeddings": int(_req(md, "context_length")),
        "swiglu_limit": swiglu_limit,
        "hc_mult": hc,
        "hc_eps": float(md.get(f"{ARCH}.hyper_connection.epsilon", 1e-6)),
        "hc_magnitude": float(md.get(f"{ARCH}.hyper_connection.magnitude", 2.0)),
        "enable_ihc": True, "gated_mla": True, "learnable_sink": True, "sandwich_norm": True,
        "engram_layer_ids": [lid + 1 for lid in ple_layers] or None,
        "engram_max_ngram_size": n_gram,
        "engram_n_head_per_ngram": n_head,
        "engram_n_embed_per_ngram": int(md.get(f"{ARCH}.embedding_length_per_layer_input", 96)) * n_head,
        "engram_layer_head_vocab_sizes": vocab_sizes or None,
        "engram_pad_id": int(md.get(f"{ARCH}.engram.pad_token_id", 120002)),
        "engram_multipliers": multipliers or None,
        "num_nextn_predict_layers": int(md.get(f"{ARCH}.nextn_predict_layers", 0)),
        "tie_word_embeddings": "output.weight" not in tensor_names,
        "bos_token_id": md.get("tokenizer.ggml.bos_token_id"),
        "eos_token_id": md.get("tokenizer.ggml.eos_token_id"),
        "pad_token_id": md.get("tokenizer.ggml.padding_token_id"),
        "torch_dtype": "bfloat16",
    }
    cfg["num_key_value_heads"] = cfg["num_attention_heads"]
    return cfg


def hy4_tokenizer_dicts(md: dict) -> tuple[dict, dict]:
    """tokenizer.json + tokenizer_config.json from the GGUF vocabulary, merges and chat template."""
    if md.get("tokenizer.ggml.pre") != "hy_v4":
        raise ValueError("Unsupported Hy4 GGUF pre-tokenizer")
    tokens = list(md["tokenizer.ggml.tokens"])
    types = list(md["tokenizer.ggml.token_type"])
    merges = list(md["tokenizer.ggml.merges"])
    added = [{"id": i, "content": t, "single_word": False, "lstrip": False, "rstrip": False,
              "normalized": False, "special": ty == 3}
             for i, (t, ty) in enumerate(zip(tokens, types)) if ty in (3, 4)]
    vocab = {t: i for i, (t, ty) in enumerate(zip(tokens, types)) if ty not in (3, 4)}
    tok = {
        "version": "1.0", "truncation": None, "padding": None, "added_tokens": added,
        "normalizer": {"type": "Sequence", "normalizers": []},
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            *({"type": "Split", "pattern": {"Regex": rx}, "behavior": "Isolated", "invert": False}
              for rx in HY4_PRETOKENIZERS),
            {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": False}]},
        "post_processor": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": False, "use_regex": True},
        "decoder": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True, "use_regex": True},
        "model": {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": None,
                  "end_of_word_suffix": None, "fuse_unk": False, "byte_fallback": False, "ignore_merges": False,
                  "vocab": vocab, "merges": [m.split(" ", 1) for m in merges]},
    }
    bos, eos, pad = (md.get(f"tokenizer.ggml.{k}_token_id") for k in ("bos", "eos", "padding"))
    cfg = {"tokenizer_class": "PreTrainedTokenizerFast", "model_max_length": int(md.get(f"{ARCH}.context_length", 1 << 20)),
           "add_bos_token": bool(md.get("tokenizer.ggml.add_bos_token", False)), "clean_up_tokenization_spaces": False,
           "bos_token": tokens[bos] if bos is not None else None, "eos_token": tokens[eos] if eos is not None else None,
           "pad_token": tokens[pad] if pad is not None else None}
    if md.get("tokenizer.chat_template"):
        cfg["chat_template"] = md["tokenizer.chat_template"]
    return tok, cfg


def prepare_hy4_metadata(model, cache_dir=None):
    """Write config.json, tokenizer files and the Engram vocabulary map for a hy_v4_flash GGUF.

    Returns the directory, or None when `model` is not a hy_v4_flash GGUF."""
    shards = resolve_gguf_shards(model)
    reader = gguf.GGUFReader(str(shards[0]))
    md = read_gguf_metadata(reader)
    if md.get("general.architecture") != ARCH:
        return None
    names = {t.name for t in reader.tensors}
    for s in shards[1:]:
        names.update(t.name for t in gguf.GGUFReader(str(s)).tensors)
    key = hashlib.sha256(json.dumps([str(Path(shards[0]).resolve()), os.path.getsize(shards[0]),
                                     HY4_METADATA_VERSION]).encode()).hexdigest()[:20]
    root = Path(cache_dir or os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "kaillm" / "hy4-metadata" / key
    with FileLock(str(root) + ".lock"):
        if (root / "config.json").is_file():
            return root
        root.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix=".hy4-", dir=root.parent))
        cfg = hy4_config_dict(md, names)
        tok, tok_cfg = hy4_tokenizer_dicts(md)
        (tmp / "config.json").write_text(json.dumps(cfg, indent=2))
        (tmp / "tokenizer.json").write_text(json.dumps(tok, ensure_ascii=False))
        (tmp / "tokenizer_config.json").write_text(json.dumps(tok_cfg, ensure_ascii=False, indent=2))
        token_map = np.asarray(md.get(f"{ARCH}.engram.token_map", []), dtype=np.int32)
        if token_map.size:
            np.save(tmp / "engram_token_map.npy", token_map)
        tmp.replace(root)
    return root
