"""Keep the measured single-token Q8 schedule scoped to the tested serving configuration."""
import json
import sys
import types
import unittest
from unittest.mock import patch

import torch

from vllm_gguf_plugin.quantization.humming import _inkling_q8_decode_tuning


class DecodeSchedule(unittest.TestCase):
    def test_only_measured_projection_selects_k64(self):
        obj = types.SimpleNamespace
        request = obj(speculative_config=None, parallel_config=obj(world_size_across_dp=1),
                      scheduler_config=obj(max_num_seqs=1), model_config=obj(hf_config=obj(model_type="inkling2_loop")))
        original = dict(block_shape=(16, 64, 32), warp_shape=(16, 32, 32), use_stream_k=False,
                        use_f16_accum=False, num_stages=3, num_ctas_per_sm=1, mma_type="mma", num_sms=36)
        mock_modules = {"vllm.config": obj(get_current_vllm_config_or_none=lambda: request),
                        "vllm.utils.humming": obj(get_heuristics_config=lambda *a, **kw: original)}
        def select(shape=(2304, 3072), kind=8):
            return _inkling_q8_decode_tuning(None, {"use_batch_invariant": True}, shape, kind,
                                             torch.bfloat16, torch.device("cuda:0"))
        with patch.dict(sys.modules, mock_modules), patch.object(torch.cuda, "device_count", return_value=1), \
             patch.object(torch.cuda, "get_device_name") as card:
            for name in ("NVIDIA GeForce RTX 5070 Ti", "NVIDIA GeForce RTX 5080"):
                card.return_value = name
                self.assertEqual(json.loads(select())["block_shape"], [16, 64, 64])
            card.return_value = "NVIDIA GeForce RTX 5090"
            self.assertIsNone(select())
            card.return_value = "NVIDIA GeForce RTX 5080"
            self.assertEqual(json.loads(select()), {**original, "block_shape":[16, 64, 64], "warp_shape":[16, 32, 64]})
            for shape, kind in (((384,2304),8), ((384,2304),14), ((3072,2304),18), ((1024,2304),12)):
                self.assertEqual(json.loads(select(shape,kind))["block_shape"], [16,64,64])
            self.assertIsNone(select((2304,3072),12))
            self.assertIsNone(select((24,2304),8))
            self.assertIsNone(select((512, 2304)))
            card.return_value = "NVIDIA GeForce RTX 5070 Ti"
            self.assertIsNone(select(kind=13))
            request.speculative_config = obj()
            self.assertIsNone(select())

    def test_5090_iq4_donor_and_original_geometry(self):
        obj = types.SimpleNamespace
        weights = '/cached/Inkling-2-Mini-IQ4_NL.gguf'
        model = obj(model_weights=weights, hf_config=obj(model_type='inkling2_loop'))
        request = obj(speculative_config=None, parallel_config=obj(world_size_across_dp=1),
                      scheduler_config=obj(max_num_seqs=1), model_config=model)
        original = dict(block_shape=(16, 64, 32), warp_shape=(16, 32, 32), use_stream_k=False,
                        use_f16_accum=False, num_stages=3, num_ctas_per_sm=1, mma_type='mma')
        modules = {'vllm.config': obj(get_current_vllm_config_or_none=lambda: request),
                   'vllm.utils.humming': obj(get_heuristics_config=lambda *a, **kw: original)}
        def select(shape, q=8):
            return _inkling_q8_decode_tuning(None, {'use_batch_invariant': True}, shape, q,
                                             torch.bfloat16, torch.device('cuda:0'))
        with patch.dict(sys.modules, modules), patch.object(torch.cuda, 'device_count', return_value=1), \
             patch.object(torch.cuda, 'get_device_name', return_value='NVIDIA GeForce RTX 5090') as sku:
            for shape, sms in [((24,2304),4), ((384,2304),8), ((1024,2304),16), ((2304,3072),36), ((3072,2304),48)]:
                original['num_sms'] = sms
                self.assertEqual(json.loads(select(shape)),
                                 {**original, 'block_shape':[16,64,64], 'warp_shape':[16,32,64]})
                with patch.dict(original, num_sms=sms+1):
                    self.assertIsNone(select(shape))
                with patch.object(model, 'model_weights', 'Inkling-2-Mini-UD-Q2_K_XL.gguf'):
                    if shape == (24,2304):
                        self.assertIsNone(select(shape))
                    else:
                        self.assertEqual(json.loads(select(shape))['block_shape'],[16,64,64])
                self.assertIsNone(select(shape, 13))
            sku.return_value = 'NVIDIA GeForce RTX 3090'
            original['warp_shape'] = (16,64,32)
            for shape in [(24,2304), (384,2304), (2304,3072)]:
                self.assertEqual(json.loads(select(shape))['warp_shape'], [16,32,64])
            original['warp_shape'] = (16,32,32)
            self.assertIsNone(select((2304,3072)))

    def test_multirow_selection_covers_the_measured_attention_inputs(self):
        obj=types.SimpleNamespace
        model=obj(model_weights='Inkling-2-Mini-IQ4_NL.gguf',hf_config=obj(model_type='inkling2_loop'))
        serving=obj(model_config=model,speculative_config=None,scheduler_config=obj(max_num_seqs=2),
                    parallel_config=obj(world_size_across_dp=1))
        baseline=dict(mma_type='mma',block_shape=[16,64,32],warp_shape=[16,32,32],num_stages=3,
                      num_ctas_per_sm=1,use_stream_k=False,use_f16_accum=False)
        def base(config,shape_m,**kwargs):
            self.assertEqual(shape_m, serving.scheduler_config.max_num_seqs)
            return baseline
        modules={'vllm.config':obj(get_current_vllm_config_or_none=lambda:serving),
                 'vllm.utils.humming':obj(get_heuristics_config=base)}
        def selected(shape):
            return _inkling_q8_decode_tuning(None,{'use_batch_invariant':True},shape,8,torch.bfloat16,torch.device('cuda:0'), serving.scheduler_config.max_num_seqs)
        with patch.dict(sys.modules,modules),patch.object(torch.cuda,'device_count',return_value=1), \
             patch.object(torch.cuda,'get_device_name',return_value='NVIDIA GeForce RTX 5090') as sku:
            for slots in (2, 4):
                serving.scheduler_config.max_num_seqs = slots
                sku.return_value = 'NVIDIA GeForce RTX 5090'
                for shape,sms in [((24,2304),4),((32,2304),4),((384,2304),8),((1024,2304),16),((3072,2304),48)]:
                    baseline['num_sms']=sms
                    self.assertEqual(json.loads(selected(shape))['block_shape'],[16,64,64])
                self.assertIsNone(selected((2304,3072)))
                with patch.object(model, 'model_weights', 'Inkling-2-Mini-UD-Q2_K_XL.gguf'):
                    self.assertIsNone(selected((3072,2304)))
                for card in ('NVIDIA GeForce RTX 5080', 'NVIDIA GeForce RTX 5070 Ti', 'NVIDIA GeForce RTX 3090',
                             'NVIDIA RTX PRO 6000 Blackwell Workstation Edition'):
                    sku.return_value = card
                    self.assertIsNone(selected((1024, 2304)))

    def test_forward_only_passes_the_schedule_for_its_actual_rows(self):
        from unittest.mock import Mock
        from vllm_gguf_plugin.quantization.humming import GGUFHummingLinear

        def ordinary(layer, *, inputs, **kwargs):
            return torch.zeros(inputs.shape[0], 8)

        runner = Mock(side_effect=ordinary)
        mocked = types.SimpleNamespace(humming_forward=runner, create_humming_locks=lambda device: torch.empty(0))
        with patch.dict(sys.modules, {"vllm.utils.humming": mocked}):
            layer = GGUFHummingLinear(None, {"weight": torch.empty(0), "weight_scale": torch.empty(0)},
                                      shape=(5, 16), dtype=torch.bfloat16, compute_config="{}",
                                      decode_tuning="single", two_row_tuning="double", four_row_tuning="four")
            for count, expected in ((1, "single"), (2, "double"), (3, None), (4, "four"), (5, None)):
                self.assertEqual(layer(torch.zeros(count, 16)).shape, (count, 5))
                self.assertEqual(runner.call_args.kwargs["tuning_config"], expected)
