# SPDX-License-Identifier: Apache-2.0
"""Standalone LFM3 metadata and tokenizer conversion from GGUF."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

import gguf
import numpy as np
from filelock import FileLock

from .qwen35_metadata import (
    qwen35_tokenizer_dicts,
    read_gguf_metadata,
    resolve_gguf_shards,
)
from .yaqa_gguf import read_yaqa_profile

LFM3_METADATA_VERSION = 5

LFM3_PRETOKENIZER = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|"
    r"\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|"
    r"\s+(?!\S)|\s+"
)


def lfm3_config_dict(metadata, tensor_names, rope_factors=None, *, yaqa=False):
    if metadata.get("general.architecture") != "lfm3":
        raise ValueError("Expected an LFM3 GGUF checkpoint")

    def required(name):
        key = "lfm3." + name
        if key not in metadata:
            raise ValueError(f"Missing required GGUF metadata: {key}")
        return metadata[key]

    layers = int(required("block_count"))
    heads_kv = required("attention.head_count_kv")
    if isinstance(heads_kv, int):
        heads_kv = [
            0 if f"blk.{i}.shortconv.in_proj.weight" in tensor_names else heads_kv
            for i in range(layers)
        ]
    swa = required("attention.sliding_window_pattern")
    if len(heads_kv) != layers or len(swa) != layers:
        raise ValueError("LFM3 layer metadata has inconsistent lengths")
    active_heads = {n for n in heads_kv if n}
    if len(active_heads) != 1:
        raise ValueError("LFM3 requires a common attention KV head count")
    layer_types = [
        "conv" if not n else "sliding_attention" if window else "full_attention"
        for n, window in zip(heads_kv, swa)
    ]
    head_dim = int(required("attention.key_length"))
    full_rope = {
        "rope_type": "default",
        "rope_theta": required("rope.freq_base"),
    }
    if rope_factors is not None:
        if len(rope_factors) != head_dim // 2:
            raise ValueError("LFM3 RoPE factor tensor disagrees with the head size")
        # llama.cpp exports zero-tail rotation as enormous frequency divisors.
        rotated = sum(float(factor) < 1e20 for factor in rope_factors)
        if any(float(x) != 1.0 for x in rope_factors[:rotated]) or any(
            float(x) < 1e20 for x in rope_factors[rotated:]
        ):
            raise ValueError("Unsupported LFM3 RoPE frequency factor layout")
        if rotated < head_dim // 2:
            full_rope.update(
                rope_type="proportional", partial_rotary_factor=rotated * 2 / head_dim
            )
    return {
        "model_type": "lfm3",
        "architectures": ["Lfm3ForCausalLM"],
        "dtype": "bfloat16",
        "vocab_size": len(metadata["tokenizer.ggml.tokens"]),
        "hidden_size": required("embedding_length"),
        "intermediate_size": required("feed_forward_length"),
        "num_hidden_layers": layers,
        "num_attention_heads": required("attention.head_count"),
        "num_key_value_heads": active_heads.pop(),
        "head_dim": head_dim,
        "layer_types": layer_types,
        "max_position_embeddings": required("context_length"),
        "norm_eps": required("attention.layer_norm_rms_epsilon"),
        "conv_L_cache": required("shortconv.l_cache"),
        "conv_bias": any(name.endswith("shortconv.conv.bias") for name in tensor_names),
        "use_sandwich_norm": "blk.0.post_attention_norm.weight" in tensor_names,
        "use_attention_gate": any(
            name.endswith("attn_gate.weight") for name in tensor_names
        ),
        "tie_word_embeddings": "output.weight" not in tensor_names,
        "sliding_window": required("attention.sliding_window"),
        "rope_parameters": {
            "full_attention": full_rope,
            "sliding_attention": {
                "rope_type": "default",
                "rope_theta": required("rope.freq_base_swa"),
            },
        },
        "eos_token_id": metadata.get("tokenizer.ggml.eos_token_id"),
        "bos_token_id": metadata.get("tokenizer.ggml.bos_token_id"),
        "pad_token_id": metadata.get("tokenizer.ggml.padding_token_id"),
        "gguf_lfm3_norm_dtype": "float32",
        "gguf_yaqa_profile": bool(yaqa),
        "gguf_lfm3_norm_impl": (
            "activation_precast_f32_gain_v1" if yaqa else "f32_gain_v1"
        ),
    }


def lfm3_tokenizer_dicts(metadata):
    if metadata.get("tokenizer.ggml.pre") != "lfm2":
        raise ValueError("Unsupported LFM3 GGUF tokenizer pre-tokenizer")
    adjusted = dict(metadata)
    adjusted["tokenizer.ggml.pre"] = "qwen35"
    adjusted["qwen35.context_length"] = metadata["lfm3.context_length"]
    add_bos = adjusted.pop("tokenizer.ggml.add_bos_token", True)
    add_eos = adjusted.pop("tokenizer.ggml.add_eos_token", False)
    tokenizer, config = qwen35_tokenizer_dicts(adjusted)
    tokenizer["model"]["ignore_merges"] = True
    split = tokenizer["pre_tokenizer"]["pretokenizers"][0]
    split["pattern"]["Regex"] = LFM3_PRETOKENIZER
    specials = []
    if add_bos:
        specials.append(("bos", metadata["tokenizer.ggml.bos_token_id"]))
    if add_eos:
        specials.append(("eos", metadata["tokenizer.ggml.eos_token_id"]))
    if specials:
        tokens = metadata["tokenizer.ggml.tokens"]
        bos = [tokens[i] for role, i in specials if role == "bos"]
        eos = [tokens[i] for role, i in specials if role == "eos"]
        single = bos + ["$A"] + eos
        pair = bos + ["$A"] + eos + bos + ["$B:1"] + [t + ":1" for t in eos]
        # Tokenizers' public builder serializes the schema consistently.
        from tokenizers.processors import TemplateProcessing

        processor = TemplateProcessing(
            single=" ".join(single),
            pair=" ".join(pair),
            special_tokens=[(tokens[i], i) for _, i in specials],
        )
        tokenizer["post_processor"] = json.loads(processor.__getstate__())
    config.update(add_bos_token=add_bos, add_eos_token=add_eos)
    return tokenizer, config


def prepare_lfm3_metadata(model, cache_dir=None):
    shards = resolve_gguf_shards(model)
    fingerprint = repr(
        (
            LFM3_METADATA_VERSION,
            [
                (str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns)
                for p in shards
            ],
        )
    )
    key = hashlib.sha256(fingerprint.encode()).hexdigest()
    root = Path(cache_dir or os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    destination = root / "vllm-gguf-lfm3" / key
    if (destination / "complete.json").is_file():
        return str(destination)
    reader = gguf.GGUFReader(str(shards[0]))
    arch = reader.get_field("general.architecture")
    if arch is None or arch.contents() != "lfm3":
        return None
    metadata = read_gguf_metadata(reader)
    tensors = {t.name: t for t in reader.tensors}
    profile = read_yaqa_profile(reader)
    has_profile = profile is not None
    logical = set(profile["tensors"]) if has_profile else set()
    for shard in shards[1:]:
        shard_reader = gguf.GGUFReader(str(shard))
        shard_profile = read_yaqa_profile(shard_reader)
        has_profile |= shard_profile is not None
        shard_logical = (
            set(shard_profile["tensors"]) if shard_profile is not None else set()
        )
        if logical & shard_logical:
            raise ValueError(f"Duplicate logical YAQA weights in GGUF shard {shard}")
        logical.update(shard_logical)
        for tensor in shard_reader.tensors:
            if tensor.name in tensors:
                raise ValueError(f"Duplicate GGUF tensor: {tensor.name}")
            tensors[tensor.name] = tensor
    factors = tensors.get("rope_freqs.weight")
    rope_factors = (
        np.asarray(factors.data).reshape(-1).tolist() if factors is not None else None
    )
    config = lfm3_config_dict(
        metadata, set(tensors) | logical, rope_factors, yaqa=has_profile
    )
    tokenizer, tokenizer_config = lfm3_tokenizer_dicts(metadata)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(destination) + ".lock"):
        if (destination / "complete.json").is_file():
            return str(destination)
        with tempfile.TemporaryDirectory(dir=destination.parent) as tmp:
            path = Path(tmp)
            for name, value in {
                "config.json": config,
                "tokenizer.json": tokenizer,
                "tokenizer_config.json": tokenizer_config,
                "complete.json": {
                    "source": str(Path(model).resolve()),
                    "version": LFM3_METADATA_VERSION,
                    "yaqa_profile": has_profile,
                },
            }.items():
                (path / name).write_text(
                    json.dumps(value, ensure_ascii=False), encoding="utf-8"
                )
            path.rename(destination)
    return str(destination)
