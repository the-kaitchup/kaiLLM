"""Verify IQ4_NL expert assignment and row-tile parity on the CPU Triton interpreter."""
import os
import unittest
from unittest.mock import patch

import numpy as np
import torch


class Tiles(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('TRITON_INTERPRET') == '1', 'requires Triton CPU interpreter')
    def test_actual_iq4_kernel_both_real_input_widths(self):
        from vllm_gguf_plugin.triton.fused_moe.iq_quant.iq4_nl import iq4_nl_moe_kernel
        from gguf.quants import IQ4_NL

        rng = np.random.default_rng(92314)
        n, experts, tokens, k = 19, 11, 39, 8
        ids = np.stack([rng.choice(experts - 1, k, replace=False) for _ in range(tokens)])
        self.assertGreater(np.bincount(ids.ravel()).max(), 32)
        table = torch.tensor(IQ4_NL.kvalues, dtype=torch.int8)
        for width, shared in ((2304, True), (1152, False)):
            # Triton's CPU interpreter stores BF16 in uint16 and its dot treats
            # those bits as integers. These binary inputs/weights are exact in
            # both FP16 and BF16; physical validation uses BF16 and real scales.
            x = torch.from_numpy((rng.integers(-4, 5, (tokens if shared else tokens*k, width)) / 4).astype(np.float16))
            weights = rng.integers(256, size=(experts, n, width//32, 18), dtype=np.uint8)
            scales = (rng.choice([-1., 1.], weights.shape[:-1]) * np.exp2(
                rng.integers(-8, -2, weights.shape[:-1]))).astype('<f2')
            weights[..., :2] = scales.view(np.uint8).reshape(*scales.shape, 2)
            w = torch.from_numpy(weights.reshape(experts, n, -1))
            outputs = []
            for bm in (4, 16, 32):
                slots, groups = [], []
                for e in range(experts):
                    routes = np.flatnonzero(ids.ravel() == e).tolist()
                    pad = -len(routes) % bm
                    slots.extend(routes + [ids.size] * pad)
                    groups.extend([e] * ((len(routes) + pad) // bm))
                valid = len(slots)
                # The aligner returns a worst-case grid with unused expert IDs set to -1.
                slots.extend([ids.size] * bm)
                groups.append(-1)
                out = torch.zeros((ids.size, n), dtype=torch.float32)
                iq4_nl_moe_kernel[(len(groups), 1)](x, w, out, torch.tensor(slots, dtype=torch.int32),
                    torch.tensor(groups, dtype=torch.int32), torch.tensor([valid], dtype=torch.int32),
                    ids.size, k if shared else 1, n, width // 32, *x.stride(), *w.stride(), *out.stride(), table,
                    BLOCK_M=bm, BLOCK_N=128, BLOCK_K_BLOCKS=4)
                self.assertTrue(torch.isfinite(out).all())
                self.assertGreater(out.float().abs().max().item(), 1.)
                outputs.append(out)
            # Independently unpack GGML's 16 low nibbles followed by 16 high.
            packed = weights[..., 2:]
            codes = np.concatenate([packed & 15, packed >> 4], axis=-1)
            dense = (np.array(IQ4_NL.kvalues, dtype=np.float32)[codes] * scales[..., None]).reshape(experts, n, width)
            rr = np.arange(ids.size) // k if shared else np.arange(ids.size)
            golden = np.einsum('rnk,rk->rn', dense[ids.ravel()], x.float().numpy()[rr], optimize=False)
            np.testing.assert_array_equal(outputs[0].numpy(), golden)
            if shared:
                from vllm_gguf_plugin.inkling.iq4_decode import routed_iq4_b1_kernel
                direct = torch.empty((k,n),dtype=torch.float32)
                routed_iq4_b1_kernel[(1,k)](x[:1],w,torch.from_numpy(ids[:1]),direct,table,n,width,64,
                                           num_warps=4,num_stages=1,enable_fp_fusion=False)
                np.testing.assert_array_equal(direct.numpy(),golden[:k])
            for candidate in outputs[1:]:
                self.assertTrue(torch.equal(candidate, outputs[0]))

    def test_hardware_and_request_scope(self):
        from types import SimpleNamespace
        from vllm_gguf_plugin.inkling.iq4_prefill import qualified_projection_tiles
        for name, batch, devices, world, native, spec, expected in [
            ('NVIDIA GeForce RTX 5090', 1, 1, 1, False, None, 32),
            ('NVIDIA GeForce RTX 5080', 1, 1, 1, False, None, None),
            ('NVIDIA GeForce RTX 5090', 2, 1, 1, False, None, None),
            ('NVIDIA GeForce RTX 5090', 1, 2, 1, False, None, None),
            ('NVIDIA GeForce RTX 5090', 1, 1, 2, False, None, None),
            ('NVIDIA GeForce RTX 5090', 1, 1, 1, True, None, None),
            ('NVIDIA GeForce RTX 5090', 1, 1, 1, False, object(), None),
        ]:
            config=SimpleNamespace(scheduler_config=SimpleNamespace(max_num_seqs=batch),
                                   parallel_config=SimpleNamespace(world_size_across_dp=world),
                                   speculative_config=spec)
            for width in (1152,2304):
                self.assertEqual(qualified_projection_tiles(config,name,devices,native,width)[0],expected)
            self.assertEqual(qualified_projection_tiles(config,name,devices,native,256),(None,None))

    def test_down_prefill_capability_and_fallback(self):
        from types import SimpleNamespace
        from vllm_gguf_plugin.inkling.iq4_prefill import supports_down_prefill_tile
        from vllm_gguf_plugin.inkling.moe import ExpertStage

        model = SimpleNamespace(hf_config=SimpleNamespace(model_type='inkling2_loop'), dtype=torch.bfloat16)
        config = SimpleNamespace(model_config=model, speculative_config=None,
                                 parallel_config=SimpleNamespace(world_size_across_dp=1))
        option = 'KAILLM_INKLING_IQ4_DOWN_PREFILL'
        with patch.dict(os.environ, {option: '1'}), patch.object(torch.cuda, 'get_device_capability') as cc:
            for capability, expected in [((7, 5), False), ((8, 6), True), ((12, 0), True)]:
                cc.return_value = capability
                self.assertEqual(supports_down_prefill_tile(config, 'cuda:0'), expected)
            with patch.object(model, 'dtype', torch.float16):
                self.assertFalse(supports_down_prefill_tile(config, 'cuda:0'))
            with patch.object(config, 'speculative_config', object()):
                self.assertFalse(supports_down_prefill_tile(config, 'cuda:0'))
            with patch.object(config.parallel_config, 'world_size_across_dp', 2):
                self.assertFalse(supports_down_prefill_tile(config, 'cuda:0'))
        with patch.dict(os.environ, {option: '0'}):
            self.assertFalse(supports_down_prefill_tile(config, 'cuda:0'))

        packed = torch.empty(1, 2304, 1152 // 32 * 18, dtype=torch.uint8)
        stage = ExpertStage(20, packed, 2304, 1152, None)
        stage.iq4_down_prefill_2048 = True
        stage.iq4_prefill_tile = 16
        with patch('vllm_gguf_plugin.inkling.iq4_prefill.tiled_prefill') as dispatch:
            for length, expected in [(512, 16), (2048, 32)]:
                ids = torch.zeros((length, 8), dtype=torch.int32)
                x = torch.empty(length * 8, 1152, dtype=torch.bfloat16)
                stage.run(x, ids, False)
                self.assertEqual(dispatch.call_args.args[-2], expected)
            stage.iq4_down_prefill_2048 = False
            stage.run(x, ids, False)
            self.assertEqual(dispatch.call_args.args[-2], 16)

    def test_qualified_decode_tiles_by_weight_type_and_gpu(self):
        from types import SimpleNamespace
        from vllm_gguf_plugin.inkling.iq4_prefill import qualified_projection_tiles
        config = SimpleNamespace(scheduler_config=SimpleNamespace(max_num_seqs=1),
                                 parallel_config=SimpleNamespace(world_size_across_dp=1), speculative_config=None)
        for board, quant, gate, down in (
            ('RTX 3090', 20, (32,64), (16,16)), ('RTX 3090', 8, (32,64), (16,16)),
            ('RTX 5090', 20, (32,64), (32,16)), ('RTX 5090', 8, (None,32), (None,16)),
            ('RTX 5080', 8, (None,None), (None,None)), ('RTX 5090', 13, (None,None), (None,None)),
        ):
            name = 'NVIDIA GeForce ' + board
            self.assertEqual(qualified_projection_tiles(config,name,1,False,2304,quant),gate)
            self.assertEqual(qualified_projection_tiles(config,name,1,False,1152,quant),down)
            self.assertEqual(qualified_projection_tiles(config,name,1,True,2304,quant),(None,None))
        config.scheduler_config.max_num_seqs=2
        self.assertEqual(qualified_projection_tiles(config,'NVIDIA GeForce RTX 5090',1,False,2304,8),(None,None))

    def test_qualified_ampere_single_token_projections(self):
        from types import SimpleNamespace
        from vllm_gguf_plugin.inkling.iq4_prefill import qualified_projection_tiles
        from vllm_gguf_plugin.inkling.moe import ExpertStage
        config = SimpleNamespace(speculative_config=None, scheduler_config=SimpleNamespace(max_num_seqs=1),
                                 parallel_config=SimpleNamespace(world_size_across_dp=1))
        self.assertEqual(qualified_projection_tiles(config,'NVIDIA GeForce RTX 3090',1,False,2304),(32,64))
        self.assertEqual(qualified_projection_tiles(config,'NVIDIA GeForce RTX 3090',1,False,1152),(16,16))
        layer = ExpertStage(20,torch.empty(1,1,18,dtype=torch.uint8),1,2304,None)
        layer.iq4_decode_tile = 64
        with patch.object(layer,'_kernel',side_effect=RuntimeError('ordinary projection')), patch(
                'vllm_gguf_plugin.inkling.iq4_decode.direct_gate_up',return_value=True) as optimized:
            for t,k,shared,dtype,eligible in [(1,8,True,torch.bfloat16,True),(2,8,True,torch.bfloat16,False),
                    (1,8,False,torch.bfloat16,False),(1,8,True,torch.float16,False),(1,4,True,torch.bfloat16,False)]:
                x=torch.empty((t,2304),dtype=dtype);ids=torch.zeros((t,k),dtype=torch.int64)
                if eligible:
                    self.assertTrue(layer.run(x,ids,shared))
                else:
                    with self.assertRaisesRegex(RuntimeError,'ordinary projection'):
                        layer.run(x,ids,shared)
            self.assertEqual(optimized.call_count,1)

    def test_down_route_masks_three_lanes_per_expert(self):
        from vllm_gguf_plugin.inkling.iq4_decode import down_route
        from vllm_gguf_plugin.inkling.moe import ExpertStage
        order, padded = down_route('cpu')
        expected = torch.full((8,4),8,dtype=torch.int32)
        expected[:,0] = torch.arange(8,dtype=torch.int32)
        self.assertTrue(torch.equal(order,expected.flatten()))
        self.assertEqual(padded.tolist(),[32])
        layer = ExpertStage(20,torch.empty(1,1,18,dtype=torch.uint8),1,1152,None)
        layer.iq4_decode_tile = 32
        layer.decode_route = order, padded
        x = torch.ones(8,1152,dtype=torch.bfloat16)
        ids = torch.arange(8).view(1,8)
        with patch('vllm_gguf_plugin.inkling.iq4_decode.grouped_down',return_value=True) as down:
            self.assertTrue(layer.run(x,ids,False))
            self.assertEqual(down.call_args.args[3],32)
            self.assertIs(down.call_args.args[4],layer.decode_route)

    def test_q8_decode_dispatch_preserves_route_and_projection_geometry(self):
        from vllm_gguf_plugin.inkling.iq4_decode import down_route
        from vllm_gguf_plugin.inkling.moe import ExpertStage
        from vllm_gguf_plugin.inkling import q8_decode
        for cols, shared, tile in ((2304,True,64),(2304,True,32),(1152,False,16)):
            layer=ExpertStage(8,torch.empty(1,1,34,dtype=torch.uint8),1,cols,None)
            layer.q8_decode_tile=tile
            layer.decode_route=down_route('cpu')
            x=torch.zeros((1 if shared else 8,cols),dtype=torch.bfloat16)
            ids=torch.tensor([[11,2,4,6,12,1,7,3]])
            with patch.object(q8_decode,'run_triton_fused_moe_kernel',return_value=True) as call:
                self.assertTrue(layer.run(x,ids,shared))
                args=call.call_args.args
                self.assertTrue(torch.equal(args[4],ids[0].int()))
                self.assertIs(args[3],layer.decode_route[0])
                self.assertIs(args[5],layer.decode_route[1])
                self.assertEqual(tuple(args[7:]),(8 if shared else 1,x.shape[0],8))
                self.assertEqual(call.call_args.kwargs,{'block_m':4,'block_n':tile})
            with patch.object(layer,'_kernel',side_effect=RuntimeError('fallback')):
                with self.assertRaisesRegex(RuntimeError,'fallback'):
                    layer.run(x.half(),ids,shared)

    def test_q8_prefill_eligible_tokens_and_tile(self):
        from vllm_gguf_plugin.inkling import q8_prefill
        from vllm_gguf_plugin.inkling.moe import ExpertStage
        import sys
        from types import SimpleNamespace
        for width, shared in [(1152, False), (2304, True)]:
            layer = ExpertStage(8,torch.empty(1,1,34,dtype=torch.uint8),1,width,None)
            layer.q8_prefill = True
            for t in [1,511,512,2048]:
                ids = torch.zeros((t,8),dtype=torch.int64)
                x = torch.zeros((t if shared else 8*t,1),dtype=torch.bfloat16)
                if t not in (512,2048):
                    with patch.object(layer,'_kernel',side_effect=RuntimeError('default kernel')):
                        with self.assertRaisesRegex(RuntimeError,'default kernel'):
                            layer.run(x,ids,shared)
                    continue
                block_m = 16 if shared and t==512 else 32
                def align(routes, block_size, experts):
                    self.assertEqual(routes.shape,(t,8) if shared else (8*t,1))
                    self.assertEqual(block_size,block_m)
                    return 1,2,3
                mod = SimpleNamespace(moe_align_block_size=align)
                path = 'vllm.model_executor.layers.fused_moe.fused_moe'
                with patch.dict(sys.modules,{path:mod}), patch.object(
                        q8_prefill,'run_triton_fused_moe_kernel',return_value=True) as launch:
                    self.assertTrue(layer.run(x,ids,shared))
                    self.assertEqual(launch.call_args.kwargs,dict(block_m=block_m,block_n=64))
                    self.assertEqual(launch.call_args.args[-1],8)

    def test_only_certified_prefill_shapes_dispatch(self):
        from vllm_gguf_plugin.inkling.moe import ExpertStage
        class OriginalDispatch(Exception):
            pass
        for width, shared in ((2304,True),(1152,False)):
            layer = ExpertStage(20,torch.empty(1,1,18,dtype=torch.uint8),1,width,None)
            for tile,t,k,right_projection,expected in [(None,512,8,True,False), (32,1,8,True,False),
                    (32,511,8,True,False), (32,1024,8,True,False), (32,512,8,True,True),
                    (32,2048,8,True,True), (32,512,1,True,False), (32,512,8,False,False)]:
                layer.iq4_prefill_tile = tile
                x = torch.empty((t if shared else t*k,1),dtype=torch.bfloat16)
                ids = torch.zeros((t,k),dtype=torch.int64)
                with patch.object(layer,'_kernel',side_effect=OriginalDispatch), patch(
                        'vllm_gguf_plugin.inkling.iq4_prefill.tiled_prefill',return_value=True) as tiled:
                    if expected:
                        self.assertTrue(layer.run(x,ids,shared))
                        self.assertEqual(tiled.call_args.args[5],tile)
                    else:
                        with self.assertRaises(OriginalDispatch):
                            layer.run(x,ids,shared if right_projection else not shared)


if __name__ == '__main__':
    unittest.main()
