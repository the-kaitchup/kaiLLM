"""Select the exact PRO 6000 donor, physical geometry and routed projection."""
import json
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from vllm_gguf_plugin.inkling.iq4_prefill import pro_projection_tiles as qualified_tiles
from vllm_gguf_plugin.quantization.humming import _inkling_q8_decode_tuning


class ProInkling(unittest.TestCase):
    def test_exact_shapes_and_fallbacks(self):
        obj = SimpleNamespace
        card = "NVIDIA RTX PRO 6000 Blackwell Workstation Edition"
        model = obj(model_weights="/weights/Inkling-2-Mini-IQ4_NL.gguf", hf_config=obj(model_type="inkling2_loop"))
        config = obj(speculative_config=None, parallel_config=obj(world_size_across_dp=1),
                     scheduler_config=obj(max_num_seqs=1), model_config=model)
        self.assertEqual(qualified_tiles(config,card,1,1152), {1:(16,64,2),2048:(32,64,2)})
        self.assertEqual(qualified_tiles(config,card,1,2304), {1:(8,64,4),2048:(32,64,2)})
        self.assertIsNone(qualified_tiles(config,card,2,2304))
        self.assertIsNone(qualified_tiles(config,"NVIDIA GeForce RTX 5090",1,2304))
        with patch.object(config.scheduler_config, "max_num_seqs", 2):
            self.assertIsNone(qualified_tiles(config,card,1,1152))
        original = dict(block_shape=(16,64,32), warp_shape=(16,32,32), use_stream_k=False,
                        use_f16_accum=False, num_stages=3, num_ctas_per_sm=1, mma_type="mma")
        mocks = {"vllm.config": obj(get_current_vllm_config_or_none=lambda:config),
                 "vllm.utils.humming": obj(get_heuristics_config=lambda *a, **kw:original)}
        def select(shape):
            return _inkling_q8_decode_tuning(None, {"use_batch_invariant": True}, shape, 8,
                                             torch.bfloat16, torch.device("cuda:0"))
        with patch.dict(sys.modules,mocks), patch.object(torch.cuda,"device_count",return_value=1), patch.object(torch.cuda,"get_device_name",return_value=card):
            for shape,sms in [((24,2304),4),((384,2304),8),((1024,2304),16),((2304,3072),36),((3072,2304),48)]:
                original["num_sms"]=sms
                self.assertEqual(json.loads(select(shape)),{**original,"block_shape":[16,64,64],"warp_shape":[16,32,64]})
                with patch.dict(original,num_sms=sms+1):
                    self.assertIsNone(select(shape))
                with patch.object(model,"model_weights","unqualified-Inkling.gguf"):
                    self.assertIsNone(select(shape))
                    self.assertIsNone(qualified_tiles(config,card,1,2304))

    def test_routed_geometry_survives_dispatch(self):
        from vllm_gguf_plugin.inkling import iq4_prefill
        from vllm_gguf_plugin.inkling.moe import ExpertStage

        align_module = "vllm.model_executor.layers.fused_moe.fused_moe"
        def align(routes, block_m, experts):
            return routes, block_m, experts
        for cols, shared in [(1152, False), (2304, True)]:
            stage = ExpertStage(20, torch.empty(1, 1, 18, dtype=torch.uint8), 1, cols, None)
            stage.pro_iq4_tiles = {1: (8, 64, 4) if shared else (16, 64, 2), 2048: (32, 64, 2)}
            for tokens, tile in stage.pro_iq4_tiles.items():
                x = torch.empty((tokens if shared else tokens * 8, 1), dtype=torch.bfloat16)
                ids = torch.zeros((tokens, 8), dtype=torch.int64)
                with patch.dict(sys.modules, {align_module: SimpleNamespace(moe_align_block_size=align)}), \
                     patch.object(iq4_prefill, "get_iq_table_tensors", return_value={"kvalues_iq4nl": 9}), \
                     patch.object(iq4_prefill, "run_triton_fused_moe_kernel", return_value=True) as launch:
                    self.assertTrue(stage.run(x, ids, shared))
                    args = launch.call_args
                    self.assertEqual(args.args[3].shape, (tokens, 8) if shared else (tokens * 8, 1))
                    self.assertEqual(args.kwargs, dict(extra_args=(9,), block_m=tile[0],
                                                       block_n=tile[1], block_k_blocks=tile[2], ORIGINAL_TABLE=True))
            with patch.object(stage, "_kernel", side_effect=RuntimeError("ordinary dispatch")):
                with self.assertRaisesRegex(RuntimeError, "ordinary dispatch"):
                    stage.run(x[:8], ids[:8], shared)
