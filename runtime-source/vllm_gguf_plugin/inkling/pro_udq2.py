"""Single-slot schedules measured on original UD-Q2 weights on the 96 GB workstation GPU."""
from pathlib import Path

GPU = 'NVIDIA RTX PRO 6000 Blackwell Workstation Edition'
GATE_TILES = {16: {1: (1, 16, 128, 4), 2048: (8, 64, 16, 4)},
 17: {1: (1, 16, 64, 4), 2048: (8, 64, 16, 4)},
 19: {1: (1, 32, 64, 8), 2048: (8, 64, 16, 4)},
 22: {1: (1, 16, 64, 4), 2048: (16, 32, 16, 4)}}
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
PARALLEL = True


def supported(serving, gpu_name, visible):
    if (serving is None or gpu_name != GPU or visible != 1 or serving.speculative_config is not None
            or serving.scheduler_config.max_num_seqs != 1 or serving.parallel_config.world_size_across_dp != 1):
        return False
    model = serving.model_config
    source = getattr(model, "model_weights", None) or getattr(model, "model", None)
    return (getattr(model.hf_config, "model_type", None) == "inkling2_loop"
            and isinstance(source, (str, Path)) and Path(source).name == "Inkling-2-Mini-UD-Q2_K_XL.gguf")
