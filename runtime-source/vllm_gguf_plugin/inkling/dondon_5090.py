# SPDX-License-Identifier: Apache-2.0
"""RTX 5090 projection schedules for the released Inkling dondon GGUFs."""
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path

import torch


_RELEASES = {
    "Inkling-2-Mini-dondon-Q3_K_M.gguf": ("M", 11663310784, "b25d34613efb53a890ac8d197ac1cbac51cbea8b12bcc4cc5d38c227761405ae"),
    "Inkling-2-Mini-dondon-Q3_K_S.gguf": ("S", 9997961376, "ff6f8da69caffafb80ad8eab0c52f5d8ea1e14653325379f356edfe395ee4864"),
}
_XS_BASENAME = "Inkling-2-Mini-dondon-Q2_K_XS.gguf"
_XS_WEIGHT_BYTES = 7998287520
_XS_WEIGHT_SHA256 = "d2e18b505ed0885e620e4f54c8d7d9734ecdaf5d07c857a32aca069043a862d1"

# Quant types measured for each (GGUF output rows, input columns) projection.
_ATTENTION = {
    "M": {(3072, 2304): (8, 12, 13, 14), (1024, 2304): (8, 13, 14),
          (2304, 3072): (8, 14), (384, 2304): (8, 13, 14)},
    "S": {(3072, 2304): (8, 12, 13, 14, 18), (1024, 2304): (8, 12, 13, 14),
          (2304, 3072): (8, 13, 14), (384, 2304): (8, 12, 13, 14)},
}
_PROJECTIONS = {"q_proj": (3072, 2304), "gate_proj": (3072, 2304), "k_proj": (1024, 2304),
                "v_proj": (1024, 2304), "o_proj": (2304, 3072), "rel_proj": (384, 2304)}


def _identity(stat):
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


@lru_cache(maxsize=2)
def _digest(path, identity):
    with path.open("rb") as stream:
        if _identity(os.fstat(stream.fileno())) != identity:
            return None
        result = hashlib.file_digest(stream, "sha256").hexdigest()
        return result if _identity(os.fstat(stream.fileno())) == identity else None


def released_tier(config, gpu, visible_devices, *, slots=1):
    if (gpu != "NVIDIA GeForce RTX 5090" or visible_devices != 1 or config is None
            or getattr(config, "speculative_config", None) is not None
            or getattr(getattr(config, "parallel_config", None), "world_size_across_dp", None) != 1
            or getattr(getattr(config, "scheduler_config", None), "max_num_seqs", None) != slots):
        return None
    model = getattr(config, "model_config", None)
    if (getattr(getattr(model, "hf_config", None), "model_type", None) != "inkling2_loop"
            or getattr(model, "max_model_len", None) != 73728 or getattr(model, "dtype", None) != torch.bfloat16):
        return None
    # The GGUF survives the HF metadata-directory substitution in model_weights.
    weight = getattr(model, "model_weights", None) or getattr(model, "model", None)
    return released_gguf_tier(weight, slots=slots)


def released_gguf_tier(weight, *, slots=1):
    """Identify a published Dondon checkpoint by size and SHA-256."""
    if not isinstance(weight, (str, Path)):
        return None
    path = Path(weight)
    if slots == 1:
        release = _RELEASES.get(path.name)
    elif slots == 4 and path.name == _XS_BASENAME:
        release = ("XS", _XS_WEIGHT_BYTES, _XS_WEIGHT_SHA256)
    else:
        return None
    if release is None:
        return None
    tier, size, digest = release
    try:
        identity = _identity(path.stat())
        return tier if identity[2] == size and _digest(path, identity) == digest else None
    except OSError:
        return None


def is_dondon_m(config, gpu, visible_devices):
    return released_tier(config, gpu, visible_devices) == "M"


def is_dondon_xs_b4(config, gpu, visible_devices):
    scheduler = getattr(config, "scheduler_config", None)
    return (getattr(scheduler, "max_num_batched_tokens", None) == 2048
            and released_tier(config, gpu, visible_devices, slots=4) == "XS")


def attention_decode_tuning(layer, quant_type, prefix, device):
    parts = prefix.split(".")
    shape = tuple(layer.shape)
    if len(parts) < 4 or parts[-4] != "layers" or not parts[-3].isdigit() or parts[-2] != "attn" or _PROJECTIONS.get(parts[-1]) != shape:
        return None
    if not any(quant_type in inventory[shape] for inventory in _ATTENTION.values()):
        return None
    policy = json.loads(layer.compute_config)
    if not policy["use_batch_invariant"] or policy["use_f16_accum"] or layer.dtype != torch.bfloat16:
        return None
    from vllm.config import get_current_vllm_config_or_none

    tier = released_tier(get_current_vllm_config_or_none(), torch.cuda.get_device_name(device), torch.cuda.device_count())
    if quant_type not in _ATTENTION.get(tier, {}).get(shape, ()):
        return None
    from vllm.utils.humming import get_heuristics_config

    base = get_heuristics_config(layer.layer_config, shape_m=1, **policy, device=device.index)
    expected = {"block_shape": [16, 64, 32], "warp_shape": [16, 32, 32], "use_stream_k": False,
                "use_f16_accum": False, "num_sms": layer.layer_config.shape_n // 64, "num_stages": 3,
                "num_ctas_per_sm": 1, "output_chunk_rows": 0, "use_pdl": True, "mma_type": "mma", "raster_group_m": 1}
    if not all(isinstance(base.get(key), (list, tuple)) for key in ("block_shape", "warp_shape")):
        return None
    normalized = dict(base, block_shape=list(base["block_shape"]), warp_shape=list(base["warp_shape"]))
    if normalized != expected:
        return None
    return json.dumps(dict(base, block_shape=[16, 64, 64], warp_shape=[16, 32, 64]))
