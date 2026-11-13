# SPDX-License-Identifier: Apache-2.0
"""Standalone Qwen3.5 GGUF metadata conversion, without GPU imports."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

import gguf
from filelock import FileLock

_CACHE_VERSION = 7
QWEN35_GDN_NORM_DTYPE = "float32"
QWEN35_GDN_NORM_IMPL = "fla"
QWEN35_PRETOKENIZER = (
    r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|"
    r"[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|"
    r"\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)


def configure_qwen35_gguf_norm(config) -> bool:
    """Include GGUF norm storage and CUDA dispatch in compilation hashes."""
    text_config = config.get_text_config()
    if text_config.model_type != "qwen3_5_text":
        return False
    # This is implementation metadata, not a user-selectable precision option.
    # AOT graphs capture parameter dtype and implementation. Both must be in
    # the config to invalidate graphs built with the previous norm policy.
    text_config.gguf_gdn_norm_dtype = QWEN35_GDN_NORM_DTYPE
    text_config.gguf_gdn_norm_impl = QWEN35_GDN_NORM_IMPL
    return True


def qwen35_eog_ids(metadata: dict) -> list[int]:
    """Match llama.cpp's Qwen end-of-generation token detection."""
    names = {
        "<|im_end|>",
        "<|endoftext|>",
        "<|fim_pad|>",
        "<|repo_name|>",
        "<|file_sep|>",
    }
    ids = {
        i for i, token in enumerate(metadata["tokenizer.ggml.tokens"]) if token in names
    }
    if (eos := metadata.get("tokenizer.ggml.eos_token_id")) is not None:
        ids.add(eos)
    return sorted(ids)


def resolve_gguf_shards(model: str | Path) -> list[Path]:
    """Resolve a local GGUF shard set and reject incomplete downloads."""
    path = Path(model)
    match = re.fullmatch(r"(.*)-(\d+)-of-(\d+)\.gguf", path.name)
    if match is None:
        if not path.is_file():
            raise FileNotFoundError(path)
        return [path]
    prefix, index, count = match.groups()
    total = int(count)
    if not 1 <= int(index) <= total:
        raise ValueError(f"Invalid GGUF shard number: {path}")
    shards = [
        path.with_name(f"{prefix}-{i:0{len(index)}d}-of-{count}.gguf")
        for i in range(1, total + 1)
    ]
    missing = [str(shard) for shard in shards if not shard.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing GGUF shards: {', '.join(missing)}")
    return shards


def read_gguf_metadata(reader: gguf.GGUFReader) -> dict:
    return {
        name: field.contents()
        for name, field in reader.fields.items()
        if not name.startswith("GGUF.")
    }


def qwen35_config_dict(metadata: dict, tensor_names: set[str]) -> dict:
    """Build a text config from the checkpoint, including its MTP convention."""
    if metadata.get("general.architecture") != "qwen35":
        raise ValueError("Expected a dense qwen35 GGUF checkpoint")
    prefix = "qwen35."

    def required(name):
        key = prefix + name
        if key not in metadata:
            raise ValueError(f"Missing required GGUF metadata: {key}")
        return metadata[key]

    layers = required("block_count") - metadata.get(prefix + "nextn_predict_layers", 0)
    interval = required("full_attention_interval")
    value_heads = required("ssm.time_step_rank")
    inner_size = required("ssm.inner_size")
    head_dim = required("attention.key_length")
    if layers <= 0 or interval <= 0 or inner_size % value_heads:
        raise ValueError("Inconsistent Qwen3.5 layer or linear-attention dimensions")
    config = {
        "model_type": "qwen3_5_text",
        "architectures": ["Qwen3_5ForCausalLM"],
        "dtype": "float16",
        "vocab_size": len(metadata["tokenizer.ggml.tokens"]),
        "hidden_size": required("embedding_length"),
        "intermediate_size": required("feed_forward_length"),
        "num_hidden_layers": layers,
        "num_attention_heads": required("attention.head_count"),
        "num_key_value_heads": required("attention.head_count_kv"),
        "head_dim": head_dim,
        "max_position_embeddings": required("context_length"),
        "rms_norm_eps": required("attention.layer_norm_rms_epsilon"),
        "linear_conv_kernel_dim": required("ssm.conv_kernel"),
        "linear_key_head_dim": required("ssm.state_size"),
        "linear_value_head_dim": inner_size // value_heads,
        "linear_num_key_heads": required("ssm.group_count"),
        "linear_num_value_heads": value_heads,
        # Qwen3.5 checkpoints retain recurrent state in FP32. Without this
        # field, vLLM's automatic cache dtype inherits the FP16 activations.
        "mamba_ssm_dtype": "float32",
        "gguf_gdn_norm_dtype": QWEN35_GDN_NORM_DTYPE,
        "gguf_gdn_norm_impl": QWEN35_GDN_NORM_IMPL,
        "layer_types": [
            "full_attention" if (i + 1) % interval == 0 else "linear_attention"
            for i in range(layers)
        ],
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": required("rope.freq_base"),
            "partial_rotary_factor": required("rope.dimension_count") / head_dim,
            "mrope_section": required("rope.dimension_sections")[:3],
            "mrope_interleaved": True,
        },
        "hidden_act": "silu",
        "attention_bias": False,
        "tie_word_embeddings": "output.weight" not in tensor_names,
        "eos_token_id": metadata.get("tokenizer.ggml.eos_token_id"),
        "bos_token_id": metadata.get("tokenizer.ggml.bos_token_id"),
        "pad_token_id": metadata.get("tokenizer.ggml.padding_token_id"),
    }
    return config


def qwen35_tokenizer_dicts(metadata: dict) -> tuple[dict, dict]:
    """Preserve GGUF token IDs, control tokens, merges and chat template."""
    if metadata.get("tokenizer.ggml.pre") != "qwen35":
        raise ValueError("Unsupported Qwen3.5 GGUF tokenizer pre-tokenizer")
    tokens = metadata["tokenizer.ggml.tokens"]
    types = metadata["tokenizer.ggml.token_type"]
    if len(tokens) != len(types) or len(set(tokens)) != len(tokens):
        raise ValueError("GGUF tokenizer token IDs are not a unique vocabulary")
    added = [
        {
            "id": i,
            "content": token,
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": token_type == 3,
        }
        for i, (token, token_type) in enumerate(zip(tokens, types))
        if token_type in (3, 4)
    ]
    bytelevel = {
        "type": "ByteLevel",
        "add_prefix_space": False,
        "trim_offsets": False,
        "use_regex": False,
    }
    tokenizer = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added,
        # llama.cpp's byte-level BPE preserves decomposed Unicode.
        "normalizer": None,
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {
                    "type": "Split",
                    "pattern": {"Regex": QWEN35_PRETOKENIZER},
                    "behavior": "Isolated",
                    "invert": False,
                },
                bytelevel,
            ],
        },
        "post_processor": bytelevel,
        "decoder": bytelevel,
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": "",
            "end_of_word_suffix": "",
            "fuse_unk": False,
            "byte_fallback": False,
            "ignore_merges": False,
            "vocab": {token: i for i, token in enumerate(tokens)},
            "merges": [merge.split(" ") for merge in metadata["tokenizer.ggml.merges"]],
        },
    }
    tokenizer_config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "model_max_length": metadata["qwen35.context_length"],
        "clean_up_tokenization_spaces": False,
        "fix_mistral_regex": False,
        "add_bos_token": metadata.get("tokenizer.ggml.add_bos_token", False),
        "add_eos_token": metadata.get("tokenizer.ggml.add_eos_token", False),
        "added_tokens_decoder": {
            str(token["id"]): {
                key: value for key, value in token.items() if key != "id"
            }
            for token in added
        },
        "additional_special_tokens": [
            token["content"] for token in added if token["special"]
        ],
    }
    for role, key in (
        ("bos", "bos"),
        ("eos", "eos"),
        ("pad", "padding"),
        ("unk", "unknown"),
    ):
        token_id = metadata.get(f"tokenizer.ggml.{key}_token_id")
        if token_id is not None:
            tokenizer_config[f"{role}_token"] = tokens[token_id]
    if template := metadata.get("tokenizer.chat_template"):
        tokenizer_config["chat_template"] = template
    if tokenizer_config["add_bos_token"] or tokenizer_config["add_eos_token"]:
        raise ValueError("Automatic BOS/EOS insertion is not supported for qwen35")
    return tokenizer, tokenizer_config


def prepare_qwen35_metadata(
    model: str | Path, cache_dir: str | Path | None = None
) -> str | None:
    """Create local HF metadata; return None for architectures other than qwen35."""
    shards = resolve_gguf_shards(model)
    fingerprint = repr(
        (
            _CACHE_VERSION,
            [
                (str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns)
                for p in shards
            ],
        )
    )
    key = hashlib.sha256(fingerprint.encode()).hexdigest()
    root = Path(cache_dir or os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    destination = root / "vllm-gguf" / key
    if (destination / "complete.json").is_file():
        return str(destination)
    reader = gguf.GGUFReader(str(shards[0]))
    architecture = reader.get_field("general.architecture")
    if architecture is None or architecture.contents() != "qwen35":
        return None
    metadata = read_gguf_metadata(reader)
    tensor_names = {tensor.name for tensor in reader.tensors}
    for shard in shards[1:]:
        shard_reader = gguf.GGUFReader(str(shard))
        names = {tensor.name for tensor in shard_reader.tensors}
        if tensor_names & names:
            raise ValueError(f"Duplicate tensors in GGUF shard {shard}")
        tensor_names.update(names)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(destination) + ".lock"):
        if (destination / "complete.json").is_file():
            return str(destination)
        config = qwen35_config_dict(metadata, tensor_names)
        tokenizer, tokenizer_config = qwen35_tokenizer_dicts(metadata)
        with tempfile.TemporaryDirectory(dir=destination.parent) as tmp:
            path = Path(tmp)
            files = {
                "config.json": config,
                "tokenizer.json": tokenizer,
                "tokenizer_config.json": tokenizer_config,
                "generation_config.json": {
                    "eos_token_id": qwen35_eog_ids(metadata),
                    "pad_token_id": config["pad_token_id"],
                },
                "complete.json": {
                    "source": str(Path(model).resolve()),
                    "version": _CACHE_VERSION,
                },
            }
            for name, value in files.items():
                (path / name).write_text(
                    json.dumps(value, ensure_ascii=False), encoding="utf-8"
                )
            path.rename(destination)
    return str(destination)
