"""Hy4 pipeline block ownership, token history and exact residual transport."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from vllm.distributed.utils import get_pp_indices
from vllm.model_executor.models.utils import PPMissingLayer
from vllm_gguf_plugin.hy4 import blocks
from vllm_gguf_plugin.weights_adapter.hy4 import GLOBAL, hy4_stage_weight_map

SOURCE = (Path(__file__).resolve().parents[1] / "runtime-source/vllm-overrides"
          / "vllm/model_executor/models/hy_v4_flash.py")
spec = importlib.util.spec_from_file_location("hy4_pipeline_test_model", SOURCE)
hy4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hy4)


class TestHy4Pipeline(unittest.TestCase):
    def test_every_weight_owned_by_one_rank(self):
        layers = 42
        mapping = GLOBAL | {f"blk.{i}.{tail}": f"model.layers.{i}.{tail}"
                            for i in range(layers) for tail in ("attn_sinks.weight", "ffn_gate_exps.weight")}
        for stages in (1, 2, 4):
            selected = []
            for rank in range(stages):
                start, end = get_pp_indices(layers, rank, stages)
                stage = hy4_stage_weight_map(mapping, start, end, layers)
                self.assertEqual("token_embd.weight" in stage, rank == 0)
                self.assertEqual("output.weight" in stage, rank == stages - 1)
                self.assertEqual("output_hc_fn.weight" in stage, rank == stages - 1)
                selected.extend(stage)
            self.assertCountEqual(selected, list(mapping))

    def test_forward_chains_all_streams_and_engram_history(self):
        # Keep arithmetic deterministic and independent of process groups and paged CUDA caches.
        class Block(nn.Module):
            def __init__(self, index):
                super().__init__()
                self.index = index

            def forward(self, positions, streams, history):
                delta = positions if history is None else history[:, self.index % history.shape[1]]
                return streams.roll(1, dims=1) + delta[:, None, None].to(streams.dtype) * .125 + self.index

        class ColdCache:
            def metadata(self):
                return None

        ids = torch.tensor([503, 7, 93, 212, 411], dtype=torch.int32)
        positions = torch.arange(len(ids))
        count, streams, hidden = 9, 4, 8
        embedding = nn.Embedding(512, hidden)
        embedding.weight.data.copy_(torch.arange(512 * hidden).reshape(512, hidden) % 13)

        def forward(stages, with_engram):
            transfer = None
            for rank in range(stages):
                model = hy4.HYV4FlashModel.__new__(hy4.HYV4FlashModel)
                nn.Module.__init__(model)
                model.first_stage, model.last_stage = rank == 0, rank == stages - 1
                model.config = SimpleNamespace(engram_layer_ids=[1, 6] if with_engram else [],
                                               engram_max_ngram_size=3, engram_pad_id=0,
                                               hc_mult=streams, hidden_size=hidden, rms_norm_eps=1e-6, hc_eps=1e-6)
                model.topk_buf = torch.empty((len(ids), 1), dtype=torch.int32)
                model.embed_tokens = embedding if model.first_stage else PPMissingLayer()
                model.token_cache = ColdCache() if model.first_stage and with_engram else None
                model.start_layer, model.end_layer = get_pp_indices(count, rank, stages)
                model.layers = nn.ModuleList(Block(i) if model.start_layer <= i < model.end_layer else PPMissingLayer()
                                             for i in range(count))
                if model.last_stage:
                    model.hc_head_fn = torch.eye(streams, streams * hidden)
                    model.hc_head_base = torch.arange(streams).float()
                    model.hc_head_scale = torch.tensor([.25])
                    model.norm = nn.Identity()
                result = model(ids if rank == 0 else None, positions, transfer)
                if not model.last_stage:
                    transfer = hy4.HYV4FlashForCausalLM.make_empty_intermediate_tensors(
                        SimpleNamespace(model=model), len(ids), torch.float32, torch.device("cpu"))
                    self.assertEqual(set(result.tensors), set(transfer.tensors))
                    for key, values in result.tensors.items():
                        self.assertEqual(values.shape, transfer[key].shape)
                        self.assertEqual(values.dtype, transfer[key].dtype)
                        transfer[key].copy_(values)
                    if with_engram:
                        torch.testing.assert_close(transfer["ngram"], blocks.ngram_tokens(ids, None, 3, 0))
            return result

        for with_engram in (False, True):
            single = forward(1, with_engram)
            for stages in (2, 4):
                with self.subTest(stages=stages, engram=with_engram):
                    torch.testing.assert_close(forward(stages, with_engram), single, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
