"""Scope and preserve the measured two-token attention projections on RTX 5080."""
from contextlib import nullcontext
import json
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch
from vllm_gguf_plugin.inkling import projection_streams as streams
from vllm_gguf_plugin.inkling.iq_gate import BLOCKS, measured_tiles
from vllm_gguf_plugin.quantization.humming import GGUFHummingLinear, _inkling_q8_decode_tuning


def serving():
    return NS(speculative_config=None, parallel_config=NS(world_size_across_dp=1),
              scheduler_config=NS(max_num_seqs=2),
              model_config=NS(model_weights='/weights/Inkling-2-Mini-UD-Q2_K_XL.gguf', max_model_len=73728, dtype=torch.bfloat16,
                              hf_config=NS(model_type='inkling2_loop')))


class TwoRowProjections(unittest.TestCase):
    def test_exact_packed_gate_geometry(self):
        card = 'NVIDIA GeForce RTX 5080'
        run = serving()
        for kind, (block_bytes, _) in BLOCKS.items():
            tiles = measured_tiles(card, kind, run, 1, 9 * block_bytes)
            prefill = (8, 64, 16, 4) if kind in (16, 22) else (8, 32, 32, 4)
            self.assertEqual(tiles, {1:(1,16,64,4), 2:(1,16,64,4), 2048:prefill})
            self.assertIsNone(measured_tiles(card, kind, run, 1, 9 * block_bytes + 1))
            with patch.object(run.model_config, 'model_weights', 'Inkling-2-Mini-IQ4_NL.gguf'):
                self.assertIsNone(measured_tiles(card, kind, run, 1, 9 * block_bytes))

    def test_only_full_context_plain_original_donor_on_one_5080(self):
        card = 'NVIDIA GeForce RTX 5080'
        run = serving()
        self.assertTrue(streams.supports_5080_b2(run, card, 1))
        self.assertFalse(streams.supports_5080_b2(run, card, 2))
        self.assertFalse(streams.supports_5080_b2(run, 'NVIDIA GeForce RTX 5090', 1))
        for owner, key, value in ((run, 'speculative_config', object()), (run.parallel_config, 'world_size_across_dp', 2),
                                  (run.scheduler_config, 'max_num_seqs', 1), (run.model_config, 'max_model_len', 4096),
                                  (run.model_config, 'dtype', torch.float16), (run.model_config, 'model_weights', 'Inkling-2-Mini-IQ4_NL.gguf')):
            with patch.object(owner, key, value):
                self.assertFalse(streams.supports_5080_b2(run, card, 1))

    def test_k64_uses_all_sixteen_measured_weight_types_shapes_and_sm_counts(self):
        groups = {(384,2304): (8, (8,12,13,14)), (1024,2304): (16, (8,12,13,14)),
                  (2304,3072): (36, (8,13,14)), (3072,2304): (48, (8,12,13,14,18))}
        src = dict(block_shape=[16,64,32], warp_shape=[16,32,32], num_stages=3, num_ctas_per_sm=1,
                   mma_type='mma', use_stream_k=False, use_f16_accum=False, use_pdl=True, output_chunk_rows=0,
                   raster_group_m=1)
        heuristic = Mock(return_value=src)
        modules = {'vllm.config':NS(get_current_vllm_config_or_none=serving),
                   'vllm.utils.humming':NS(get_heuristics_config=heuristic)}
        def select(shape, kind):
            return _inkling_q8_decode_tuning(None, {'use_batch_invariant':True}, shape, kind,
                                             torch.bfloat16, torch.device('cuda:0'), 2)
        with patch.dict(sys.modules,modules), patch.object(torch.cuda,'device_count',return_value=1), \
             patch.object(torch.cuda,'get_device_name',return_value='NVIDIA GeForce RTX 5080'):
            for shape,(sms,types) in groups.items():
                for kind in types:
                    src['num_sms']=sms
                    expected=dict(src,block_shape=[16,64,64],warp_shape=[16,32,64])
                    self.assertEqual(json.loads(select(shape,kind)),expected)
                    self.assertEqual(heuristic.call_args.kwargs['shape_m'],2)
                    src['num_sms']=sms+1
                    self.assertIsNone(select(shape,kind))
            self.assertIsNone(select((2304,3072),12))
            self.assertIsNone(select((512,2304),8))

    def test_actual_row_count_dispatch(self):
        def forward(config, **kwargs):
            chosen.append(kwargs['tuning_config'])
            return torch.zeros((kwargs['inputs'].shape[0],4))
        chosen=[]
        layer=NS(shape=(4,8),dtype=torch.float32,layer_config=None,weight=None,weight_scale=None,locks=None,
                 compute_config=None,decode_tuning='one',two_row_tuning='two',four_row_tuning=None,_humming_forward=forward)
        for rows in (1,2,3,4,2048):
            self.assertEqual(GGUFHummingLinear.forward(layer,torch.ones(rows,8)).shape,(rows,4))
        self.assertEqual(chosen,['one','two',None,None,None])

    def test_full_and_sliding_calls_preserve_input_ownership_and_stream_lifetime(self):
        run=serving()
        vllm=NS(envs=NS(VLLM_BATCH_INVARIANT=True))
        modules={'vllm':vllm,'vllm.config':NS(get_current_vllm_config_or_none=lambda:run)}
        main=Mock(name='main');side=[Mock(name=f'side{i}') for i in range(3)]
        inputs=[NS(device='cuda',index=i) for i in range(3)]
        def linear(layer, x):
            self.assertEqual(x.index,streams.OWNERS[layer])
            return Mock(name=layer)
        with patch.dict(sys.modules,modules),patch.object(torch.cuda,'is_available',return_value=True), \
             patch.object(torch.cuda,'get_device_name',return_value='NVIDIA GeForce RTX 5080'), \
             patch.object(torch.cuda,'device_count',return_value=1),patch.object(torch.cuda,'current_device',return_value=0), \
             patch.object(torch.cuda,'stream',side_effect=lambda _:nullcontext()), \
             patch.object(torch.cuda,'current_stream',return_value=main),patch.object(streams,'pool',return_value=tuple(side)):
            for full in (True,False):
                names={n for n in streams.OWNERS if full or '_conv_' not in n}
                layer=NS(**{n:n for n in names})
                plan=streams.prepare(layer)
                self.assertEqual(plan[0],2)
                result=streams.project(plan[1],linear,inputs)
                self.assertEqual(set(result),names)
                for group, stream in zip(streams.SCHEDULE[1:],side,strict=True):
                    stream.wait_stream.assert_called_with(main)
                    main.wait_stream.assert_any_call(stream)
                    for key in set(group)&names:
                        result[key].record_stream.assert_called_once_with(main)
