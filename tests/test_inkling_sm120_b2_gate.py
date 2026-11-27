"""Check independent per-request activations and packed expert routes."""
import os
import unittest
from unittest.mock import patch

import numpy as np
import torch


@unittest.skipUnless(os.environ.get("TRITON_INTERPRET") == "1", "Triton CPU interpreter required")
class IndependentRequestGate(unittest.TestCase):
    def test_shared_experts_use_their_request_row(self):
        from gguf.quants import IQ4_NL
        from vllm_gguf_plugin.inkling.iq4_decode import routed_iq4_b1_kernel

        rng = np.random.default_rng(5900)
        width, rows = 2304, 5
        raw = rng.integers(0,256,(12,rows,width//32,18),dtype=np.uint8)
        scales = (rng.choice([-1.,1.],raw.shape[:-1])*np.exp2(rng.integers(-6,-2,size=raw.shape[:-1]))).astype('<f2')
        raw[:,:,:,:2] = scales.view(np.uint8).reshape(*scales.shape,2)
        n=raw[:,:,:,2:];codes=np.concatenate((n&15,n>>4),axis=-1)
        dense=(np.array(IQ4_NL.kvalues,dtype=np.float32)[codes]*scales[...,None]).reshape(12,rows,width)
        table = torch.tensor(IQ4_NL.kvalues, dtype=torch.int8)
        ids = np.array([np.arange(8), np.arange(4,12), np.arange(8)[::-1], np.roll(np.arange(4,12), 3)])
        for batch, tile in ((2, 64), (4, 32)):
            activations = rng.integers(-3, 4, (batch, width)).astype(np.float16)
            routes = ids[:batch]
            out = torch.empty((batch * 8, rows), dtype=torch.float32)
            routed_iq4_b1_kernel[(1, batch * 8)](
                torch.from_numpy(activations), torch.from_numpy(raw), torch.from_numpy(routes),
                out, table, rows, width, tile, True, num_warps=4, num_stages=1, enable_fp_fusion=False)
            expected = np.einsum('tnrk,tk->tnr', dense[routes], activations.astype(np.float32), optimize=False)
            np.testing.assert_array_equal(expected.reshape(batch * 8, rows), out.numpy())
            self.assertFalse(np.array_equal(out[4:8].numpy(), out[8:12].numpy()))


class MultiRequestSelection(unittest.TestCase):
    def test_selects_measured_gate_and_down_tiles_for_this_donor_and_gpu(self):
        from types import SimpleNamespace as Record
        from vllm_gguf_plugin.inkling.iq4_prefill import qualified_projection_tiles
        config=Record(speculative_config=None, scheduler_config=Record(max_num_seqs=2),
                      parallel_config=Record(world_size_across_dp=1),
                      model_config=Record(model_weights='/weights/Inkling-2-Mini-IQ4_NL.gguf',
                                          hf_config=Record(model_type='inkling2_loop')))
        for batch in (2, 4):
            config.scheduler_config.max_num_seqs = batch
            gate = 64 if batch == 2 else 32
            for kind, width, tiles in [(8,2304,(None,64)), (8,1152,(None,16)), (20,2304,(32,gate)), (20,1152,(32,16))]:
                self.assertEqual(qualified_projection_tiles(config,'NVIDIA GeForce RTX 5090',1,False,width,kind),tiles)
                self.assertEqual(qualified_projection_tiles(config,'NVIDIA GeForce RTX 3090',1,False,width,kind),(None,None))
            with patch.object(config.model_config, 'model_weights', 'Inkling-2-Mini-UD-Q2_K_XL.gguf'):
                self.assertEqual(qualified_projection_tiles(config,'NVIDIA GeForce RTX 5090',1,False,2304,8),(None,None))

    def test_eight_routes_per_token_keep_their_position_in_the_joined_projection(self):
        from vllm_gguf_plugin.inkling import q8_decode
        from vllm_gguf_plugin.inkling.iq4_decode import down_route
        from vllm_gguf_plugin.inkling.moe import ExpertStage
        layer=ExpertStage(8,torch.zeros(1,1,34,dtype=torch.uint8),1,1152,None)
        for batch in (2, 4):
            rows = batch * 8
            layer.q8_decode_tile = 16
            layer.decode_rows = batch
            layer.decode_route = down_route('cpu', batch)
            order, padded = layer.decode_route
            self.assertEqual(padded.tolist(), [rows * 4])
            self.assertEqual(order.view(rows,4)[:,0].tolist(), list(range(rows)))
            self.assertTrue((order.view(rows,4)[:,1:] == rows).all())
            x = torch.arange(rows,dtype=torch.bfloat16)[:,None].expand(rows,1152).contiguous()
            ids = torch.arange(8,dtype=torch.int32).repeat(batch,1)
            with patch.object(q8_decode,'run_triton_fused_moe_kernel',return_value='qualified') as call:
                self.assertEqual(layer.run(x,ids,False),'qualified')
                args = call.call_args.args
                self.assertIs(args[2],x)
                self.assertIs(args[3],order)
                self.assertIs(args[5],padded)
                self.assertTrue(torch.equal(args[4],ids.flatten()))
                self.assertEqual(args[7:], (1, rows, 8))
            with patch.object(layer,'_kernel',side_effect=LookupError):
                with self.assertRaises(LookupError):
                    layer.run(x[:8],ids[:1],False)
