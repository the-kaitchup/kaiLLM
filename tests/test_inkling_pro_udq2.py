"""Guard the original donor, serving configuration, selected projection and real token count."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch

from vllm_gguf_plugin.inkling import pro_udq2, projection_streams

class OriginalMixedDonor(unittest.TestCase):
    def test_hardware_and_checkpoint(self):
        obj=SimpleNamespace
        model=obj(hf_config=obj(model_type='inkling2_loop'),model_weights='/weights/Inkling-2-Mini-UD-Q2_K_XL.gguf')
        config=obj(model_config=model,parallel_config=obj(world_size_across_dp=1),
                   scheduler_config=obj(max_num_seqs=1),speculative_config=None)
        self.assertTrue(pro_udq2.supported(config,pro_udq2.GPU,1))
        self.assertFalse(projection_streams.supported(config,pro_udq2.GPU,1))
        for gpu,devices in [('NVIDIA GeForce RTX 5080',1),(pro_udq2.GPU,2)]:
            self.assertFalse(pro_udq2.supported(config,gpu,devices))
        for o,k,v in [(model,'model_weights','Inkling-2-Mini-IQ4_NL.gguf'),(config,'speculative_config',obj()),
                      (config.scheduler_config,'max_num_seqs',2)]:
            with patch.object(o,k,v):self.assertFalse(pro_udq2.supported(config,pro_udq2.GPU,1))

    def test_original_packed_width_selects_the_measured_gate(self):
        from vllm_gguf_plugin.inkling.iq_gate import BLOCKS, measured_tiles
        model=SimpleNamespace(hf_config=SimpleNamespace(model_type='inkling2_loop'),
                              model_weights='Inkling-2-Mini-UD-Q2_K_XL.gguf')
        serving=SimpleNamespace(model_config=model,parallel_config=SimpleNamespace(world_size_across_dp=1),
                                scheduler_config=SimpleNamespace(max_num_seqs=1),speculative_config=None)
        for kind,tiles in pro_udq2.GATE_TILES.items():
            size=9*BLOCKS[kind][0]
            self.assertEqual(measured_tiles(pro_udq2.GPU,kind,serving,1,size),tiles)
            self.assertIsNone(measured_tiles(pro_udq2.GPU,kind,serving,1,size+1))
            self.assertIsNone(measured_tiles(pro_udq2.GPU,kind,serving,2,size))
        with patch.object(model,'model_weights','Inkling-2-Mini-IQ4_NL.gguf'):
            self.assertIsNone(measured_tiles(pro_udq2.GPU,16,serving,1,9*BLOCKS[16][0]))

    def test_routing_uses_frozen_tile_only_at_supported_width(self):
        from vllm_gguf_plugin.inkling.moe import ExpertStage
        from vllm_gguf_plugin.inkling import iq_gate
        for kind,shapes in pro_udq2.GATE_TILES.items():
            stage=ExpertStage(kind,torch.empty(1,1,18,dtype=torch.uint8),1,2304,None)
            stage.iq_gate_tiles=shapes
            for length,tile in shapes.items():
                x=torch.empty(length,1,dtype=torch.bfloat16);ids=torch.zeros(length,8,dtype=torch.int64)
                with patch.object(iq_gate,'run_projection',return_value=True) as kernel:
                    self.assertTrue(stage.run(x,ids,True))
                    self.assertEqual(kernel.call_args.args[-2:],(kind,tile))
            with patch.object(stage,'_kernel',side_effect=RuntimeError('ordinary forward')):
                with self.assertRaisesRegex(RuntimeError,'ordinary forward'):
                    stage.run(torch.empty(2,1,dtype=torch.bfloat16),torch.zeros(2,8,dtype=torch.int64),True)
