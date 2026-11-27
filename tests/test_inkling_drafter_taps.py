"""Exercise tap selection through the three passes of the Inkling target forward."""
from importlib.util import find_spec
from types import SimpleNamespace
import unittest

import torch


@unittest.skipUnless(find_spec("vllm"), "requires the compatible kaiLLM runtime")
class InklingDrafterTapTest(unittest.TestCase):
    def test_taps_use_visit_order_and_end_with_normalized_output(self):
        from vllm.model_executor.models.inkling2_loop import InklingModel
        from vllm.transformers_utils.configs.inkling2_loop import Inkling2LoopConfig

        config = Inkling2LoopConfig(num_hidden_layers=24, loop_start=6, loop_end=18, loop_count=3)
        trace = []

        class Layer:
            msg_query = torch.zeros(1, 4)

            def __init__(self, layer):
                self.layer = layer

            def __call__(self, x, blocks, positions, tau, attn, conv, messages, pass_index):
                trace.append((self.layer, pass_index, x.clone()))
                y = x + ((self.layer + 1) * (pass_index + 1)) * torch.tensor([.01, -.02, .03, .04])
                blocks.append(y)
                return y

        taps = (5, 23, 41, 48)
        model = SimpleNamespace(config=config, layers=[Layer(i) for i in range(config.num_hidden_layers)],
                                slots=[SimpleNamespace(attn=None, conv=None) for _ in range(config.num_slots)],
                                aux_hidden_state_layers=taps, embed_norm=torch.ones(4), norm=torch.ones(4),
                                output_res_query=torch.ones(1, 4), model_dtype=torch.float32)
        x = torch.tensor([[1., 2., -3., -1.], [-.3, .1, .6, -.7]])
        positions = torch.tensor([0, 1])
        hidden, features = InklingModel.forward(model, None, positions, inputs_embeds=x)
        self.assertEqual(len(trace), 48)
        self.assertEqual([(trace[i][0], trace[i][1]) for i in taps[:-1]], [(5, 0), (11, 1), (17, 2)])
        self.assertEqual(len(features), 4)
        for feature, visit in zip(features[:-1], taps[:-1]):
            torch.testing.assert_close(feature, trace[visit][2])
        torch.testing.assert_close(features[-1], hidden)
        model.aux_hidden_state_layers = ()
        ordinary = InklingModel.forward(model, None, positions, inputs_embeds=x)
        torch.testing.assert_close(ordinary, hidden)
