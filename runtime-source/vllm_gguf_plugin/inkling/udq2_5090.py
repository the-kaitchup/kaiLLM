# SPDX-License-Identifier: Apache-2.0
"""Measured Inkling B1 projection schedules for a single RTX 5090."""
from pathlib import Path

GATE_TILES = {16: {1: (1, 16, 128, 4), 2048: (8, 64, 16, 4)},
 17: {1: (1, 16, 64, 4), 2048: (8, 64, 16, 4)},
 19: {1: (1, 16, 64, 4), 2048: (8, 64, 16, 4)},
 22: {1: (1, 16, 64, 4), 2048: (8, 64, 16, 4)}}
ATTENTION_SMS = {(8, 384, 2304): 8,
 (8, 1024, 2304): 16,
 (8, 2304, 3072): 36,
 (8, 3072, 2304): 48,
 (12, 384, 2304): 8,
 (12, 1024, 2304): 16,
 (12, 3072, 2304): 48,
 (13, 384, 2304): 8,
 (13, 1024, 2304): 16,
 (13, 2304, 3072): 36,
 (13, 3072, 2304): 48,
 (14, 384, 2304): 8,
 (14, 1024, 2304): 16,
 (14, 2304, 3072): 36,
 (14, 3072, 2304): 48,
 (18, 3072, 2304): 48}


def supported(serving, gpu, visible, *, include_dondon_xs=False):
    if (gpu != "NVIDIA GeForce RTX 5090" or visible != 1 or serving is None
            or serving.speculative_config is not None or serving.parallel_config.world_size_across_dp != 1
            or serving.scheduler_config.max_num_seqs != 1):
        return False
    model = getattr(serving, "model_config", None)
    path = getattr(model, "model_weights", None) or getattr(model, "model", None)
    if (getattr(getattr(model, "hf_config", None), "model_type", None) != "inkling2_loop"
            or not isinstance(path, (str, Path))):
        return False
    name = Path(path).name
    return name == "Inkling-2-Mini-UD-Q2_K_XL.gguf" or (include_dondon_xs and name == "Inkling-2-Mini-dondon-Q2_K_XS.gguf")


DONDON_XS_B2 = {
    16: (1, 32, 64, 4),
    17: (1, 16, 64, 4),
    19: (1, 32, 64, 4),
    22: (1, 32, 64, 4),
}


def supports_dondon_xs_b2(serving, gpu, visible):
    if (gpu != "NVIDIA GeForce RTX 5090" or visible != 1 or serving is None
            or serving.speculative_config is not None or serving.parallel_config.world_size_across_dp != 1
            or serving.scheduler_config.max_num_seqs != 2):
        return False
    import torch
    model = getattr(serving, "model_config", None)
    path = getattr(model, "model_weights", None) or getattr(model, "model", None)
    return (isinstance(path, (str, Path)) and Path(path).name == "Inkling-2-Mini-dondon-Q2_K_XS.gguf"
            and getattr(model, "max_model_len", None) == 73728 and getattr(model, "dtype", None) == torch.bfloat16
            and getattr(getattr(model, "hf_config", None), "model_type", None) == "inkling2_loop")
