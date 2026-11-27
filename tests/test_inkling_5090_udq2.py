"""Keep the original donor’s RTX 5090 schedules within their measured inputs."""
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as Namespace
from unittest.mock import patch

import torch

from vllm_gguf_plugin.inkling import projection_streams, udq2_5090
from vllm_gguf_plugin.inkling.iq_gate import BLOCKS, measured_tiles
from vllm_gguf_plugin.quantization.humming import _inkling_q8_decode_tuning


class Original5090Schedules(unittest.TestCase):
    gpu = 'NVIDIA GeForce RTX 5090'

    def setUp(self):
        self.model = Namespace(model_weights=Path('/models/Inkling-2-Mini-UD-Q2_K_XL.gguf'),
                               hf_config=Namespace(model_type='inkling2_loop'))
        self.serving = Namespace(model_config=self.model, speculative_config=None,
                                 scheduler_config=Namespace(max_num_seqs=1),
                                 parallel_config=Namespace(world_size_across_dp=1))

    def test_qualified_operations(self):
        self.assertTrue(udq2_5090.supported(self.serving, self.gpu, 1))
        self.assertFalse(projection_streams.supported(self.serving, self.gpu, 1))
        for obj, name, value in ((self.serving.scheduler_config, 'max_num_seqs', 2),
                                 (self.serving.parallel_config, 'world_size_across_dp', 2),
                                 (self.serving, 'speculative_config', Namespace()),
                                 (self.model, 'model_weights', 'Inkling-2-Mini-IQ4_NL.gguf'),
                                 (self.model.hf_config, 'model_type', 'inkling2')):
            with self.subTest(name=name), patch.object(obj, name, value):
                self.assertFalse(udq2_5090.supported(self.serving, self.gpu, 1))
                self.assertIsNone(measured_tiles(self.gpu, 16, self.serving, 1, 9 * BLOCKS[16][0]))
        self.assertFalse(udq2_5090.supported(self.serving, self.gpu, 2))
        self.assertFalse(udq2_5090.supported(self.serving, 'NVIDIA GeForce RTX 5080', 1))
        self.assertFalse(udq2_5090.supported(None, self.gpu, 1))

    def test_packed_gate_and_up_widths(self):
        for ggml_type in (16, 17, 19, 22):
            row_bytes = 9 * BLOCKS[ggml_type][0]
            with self.subTest(ggml_type=ggml_type):
                self.assertEqual(measured_tiles(self.gpu, ggml_type, self.serving, 1, row_bytes),
                                 {1: (1, 16, 128 if ggml_type == 16 else 64, 4), 2048: (8, 64, 16, 4)})
                self.assertIsNone(measured_tiles(self.gpu, ggml_type, self.serving, 1, row_bytes - 1))
        self.assertIsNone(measured_tiles(self.gpu, 8, self.serving, 1, 2304))

    def test_published_dondon_xs_packed_gate_scope(self):
        self.model.model_weights = Path('/models/Inkling-2-Mini-dondon-Q2_K_XS.gguf')
        self.assertFalse(udq2_5090.supported(self.serving, self.gpu, 1))
        self.assertFalse(projection_streams.supported(self.serving, self.gpu, 1))
        for kind, tiles in udq2_5090.GATE_TILES.items():
            with self.subTest(ggml_type=kind):
                size = 9 * BLOCKS[kind][0]
                self.assertEqual(measured_tiles(self.gpu, kind, self.serving, 1, size), tiles)
                self.assertIsNone(measured_tiles(self.gpu, kind, self.serving, 1, size - 1))
        for obj, attr, value in (
            (self.serving.scheduler_config, 'max_num_seqs', 2),
            (self.serving.parallel_config, 'world_size_across_dp', 2),
            (self.serving, 'speculative_config', Namespace()),
            (self.model.hf_config, 'model_type', 'inkling2'),
            (self.model, 'model_weights', 'Inkling-2-Mini-dondon-Q3_K_S.gguf'),
            (self.model, 'model_weights', 'Inkling-2-Mini-chimachima-kaiPTQ-Q2_K_XS.gguf'),
        ):
            with self.subTest(attr=attr, value=value), patch.object(obj, attr, value):
                self.assertIsNone(measured_tiles(self.gpu, 16, self.serving, 1, 9 * BLOCKS[16][0]))
        self.assertIsNone(measured_tiles(self.gpu, 16, self.serving, 2, 9 * BLOCKS[16][0]))

    def test_published_dondon_xs_two_slot_packed_gates(self):
        self.model.model_weights = Path('/models/Inkling-2-Mini-dondon-Q2_K_XS.gguf')
        self.model.max_model_len = 73728
        self.model.dtype = torch.bfloat16
        self.serving.scheduler_config.max_num_seqs = 2
        expected = {16: (1, 32, 64, 4), 17: (1, 16, 64, 4), 19: (1, 32, 64, 4), 22: (1, 32, 64, 4)}
        def select(kind, **kwargs):
            return measured_tiles(kwargs.get('gpu', self.gpu), kind, self.serving,
                                  kwargs.get('visible', 1), 9 * BLOCKS[kind][0] + kwargs.get('extra_bytes', 0))
        for kind, tile in expected.items():
            self.assertEqual(select(kind), {**udq2_5090.GATE_TILES[kind], 2: tile})
            self.assertIsNone(select(kind, extra_bytes=1))
            self.assertIsNone(select(kind, visible=2))
            self.assertIsNone(select(kind, gpu='NVIDIA GeForce RTX 3090'))
        for obj, name, value in ((self.model, 'model_weights', 'Inkling-2-Mini-dondon-Q3_K_S.gguf'),
                                 (self.model, 'model_weights', 'Inkling-2-Mini-UD-Q2_K_XL.gguf'),
                                 (self.model, 'max_model_len', 8192), (self.model, 'dtype', torch.float16),
                                 (self.model.hf_config, 'model_type', 'inkling2'),
                                 (self.serving.scheduler_config, 'max_num_seqs', 4),
                                 (self.serving, 'speculative_config', Namespace()),
                                 (self.serving.parallel_config, 'world_size_across_dp', 2)):
            with self.subTest(name=name, value=value), patch.object(obj, name, value):
                self.assertIsNone(select(16))
        self.serving.scheduler_config.max_num_seqs = 1
        self.assertEqual(select(16), udq2_5090.GATE_TILES[16])

    def test_dense_schedules_check_format_shape_and_sms(self):
        self.check_dense_schedules()

    def test_published_dondon_xs_dense_schedules(self):
        self.model.model = '/cache/inkling-metadata'
        self.model.model_weights = Path('/models/Inkling-2-Mini-dondon-Q2_K_XS.gguf')
        self.check_dense_schedules()
        for name in ('Inkling-2-Mini-dondon-Q3_K_S.gguf', 'Inkling-2-Mini-dondon-Q3_K_M.gguf',
                     'Inkling-2-Mini-chimachima-kaiPTQ-Q2_K_XS.gguf'):
            with self.subTest(name=name), patch.object(self.model, 'model_weights', name):
                self.assertFalse(udq2_5090.supported(self.serving, self.gpu, 1, include_dondon_xs=True))
        for attr, value in (('speculative_config', Namespace()), ('parallel_config', Namespace(world_size_across_dp=2)),
                            ('scheduler_config', Namespace(max_num_seqs=2))):
            with self.subTest(attr=attr), patch.object(self.serving, attr, value):
                self.assertFalse(udq2_5090.supported(self.serving, self.gpu, 1, include_dondon_xs=True))

    def check_dense_schedules(self):
        expected = {(384, 2304): (8, (8, 12, 13, 14)), (1024, 2304): (16, (8, 12, 13, 14)),
                    (2304, 3072): (36, (8, 13, 14)), (3072, 2304): (48, (8, 12, 13, 14, 18))}
        baseline = dict(block_shape=(16, 64, 32), warp_shape=(16, 32, 32), num_sms=0,
                        use_stream_k=False, use_f16_accum=False, num_stages=3, num_ctas_per_sm=1, mma_type='mma')
        modules = {'vllm.config': Namespace(get_current_vllm_config_or_none=lambda: self.serving),
                   'vllm.utils.humming': Namespace(get_heuristics_config=lambda *args, **kwargs: baseline)}

        def tuning(shape, ggml_type):
            return _inkling_q8_decode_tuning(None, {'use_batch_invariant': True}, shape, ggml_type,
                                             torch.bfloat16, torch.device('cuda:0'))

        with patch.dict(sys.modules, modules), patch.object(torch.cuda, 'device_count', return_value=1), \
                patch.object(torch.cuda, 'get_device_name', return_value=self.gpu):
            for shape, (sms, ggml_types) in expected.items():
                baseline['num_sms'] = sms
                for ggml_type in (8, 12, 13, 14, 18):
                    with self.subTest(shape=shape, ggml_type=ggml_type):
                        if ggml_type in ggml_types:
                            selected = json.loads(tuning(shape, ggml_type))
                            self.assertEqual(selected['warp_shape'], [16, 32, 64])
                            self.assertEqual(selected['block_shape'], [16, 64, 64])
                            with patch.dict(baseline, num_sms=sms + 1):
                                self.assertIsNone(tuning(shape, ggml_type))
                        else:
                            self.assertIsNone(tuning(shape, ggml_type))
            self.assertIsNone(tuning((24, 2304), 8))
