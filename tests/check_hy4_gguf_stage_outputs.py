"""Compare CPU pipeline partitions with the full kaiLLM model on one small Hy4 GGUF.

This drives the actual model loader, quantized weights, Engram table and paged attention.
The in-process transport isolates pipeline numerics from distributed device communication.
Run in the assembled runtime with the Hy4 plugin and its vLLM core on PYTHONPATH.
"""
import argparse
from contextlib import ExitStack
from copy import deepcopy
import gc
from pathlib import Path
import tempfile
from unittest.mock import patch

import torch
from test_hy4_vllm_cpu import Runner
from vllm.config import set_current_vllm_config
from vllm.distributed import parallel_state
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.models import hy_v4_flash
from vllm.plugins import load_general_plugins
from vllm_gguf_plugin.hy4.partition import from_gguf


class RankView:
    def __init__(self, group, rank, size):
        self.group = group
        self.rank_in_group = rank
        self.world_size = size
        self.is_first_rank = rank == 0
        self.is_last_rank = rank == size - 1

    def __getattr__(self, name):
        return getattr(self.group, name)


def cpu_config(gguf):
    config = EngineArgs(model=str(gguf), dtype="float32", enforce_eager=True, max_model_len=128,
                        max_num_batched_tokens=128, max_num_seqs=2, block_size=16).create_engine_config()
    config.device_config.device = torch.device("cpu")
    config.device_config.device_type = "cpu"
    config.compilation_config.custom_ops = ["none"]
    return config


def evaluate(gguf, size, prototype):
    import os
    stack = ExitStack()
    runners = []
    layers = from_gguf(gguf, size)
    stack.enter_context(patch.dict(os.environ, VLLM_PP_LAYER_PARTITION=",".join(map(str, layers))))
    with stack, torch.inference_mode():
        for rank in range(size):
            config = deepcopy(prototype)
            group = RankView(parallel_state.get_pp_group(), rank, size)
            with (set_current_vllm_config(config), patch.object(hy_v4_flash, "get_pp_group", return_value=group),
                  patch.object(parallel_state, "get_pp_group", return_value=group)):
                model = get_model_loader(config.load_config).load_model(vllm_config=config, model_config=config.model_config)
                assert model.model.end_layer - model.model.start_layer == layers[rank]
                runners.append(Runner(config, model, 16, num_blocks=12))
        answers = []
        # Two requests with distinct token histories, a chunk boundary, then decode.
        for batch in ([(0, [15, 43, 27, 80, 34], 0), (1, [12, 31, 77], 0)],
                      [(0, [91, 55, 143], 5), (1, [18, 14, 27, 52, 172], 3)],
                      [(0, [108], 8), (1, [33], 8)]):
            hidden = None
            for runner in runners:
                with set_current_vllm_config(runner.vc):
                    hidden = runner.step(batch, hidden)
            answers.append(hidden)
        return torch.cat(answers)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gguf", type=Path)
    args = parser.parse_args()
    load_general_plugins()
    with tempfile.TemporaryDirectory() as tmp:
        parallel_state.init_distributed_environment(world_size=1, rank=0, local_rank=0, backend="gloo",
                                                     distributed_init_method=Path(tmp, "gloo").as_uri())
        prototype = cpu_config(args.gguf)
        with set_current_vllm_config(prototype):
            parallel_state.ensure_model_parallel_initialized(1, 1)
        try:
            baseline = evaluate(args.gguf, 1, prototype)
            for size in (2, 4):
                gc.collect()
                candidate = evaluate(args.gguf, size, prototype)
                torch.testing.assert_close(candidate, baseline, rtol=0, atol=0)
                print(f"{size} stages: {candidate.shape[0]} positions × {candidate.shape[1]} logits equal", flush=True)
        finally:
            parallel_state.destroy_model_parallel()
            parallel_state.destroy_distributed_environment()


if __name__ == "__main__":
    main()
