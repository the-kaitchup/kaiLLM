from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

import torch


def load_source(name, filename):
    base = Path(__file__).resolve().parents[1] / 'runtime-source/vllm_gguf_plugin'
    spec = spec_from_file_location(name, base / filename)
    obj = module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


class DSpineStaging(unittest.TestCase):
    def test_spill_and_partial_block_match_projection_and_leave_query_intact(self):
        aux = load_source('dspine_graph', 'dspine/context_graph.py')
        context = load_source('draft_context', 'draft_context_graph.py')
        rows = 11
        taps = [torch.arange(rows * n).reshape(rows, n).float() for n in (2, 3)]
        weight = torch.tensor([[1., 0], [2, -1], [3, -2], [4, 3], [5, 4]])
        expected = torch.cat(taps, dim=-1) @ weight
        slots = torch.tensor([[7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]])
        proposer = NS(_aux_staging=torch.zeros(3, 5), hidden_states=torch.zeros(rows, 2),
                      context_positions=torch.arange(rows), _context_slot_mappings=slots, max_num_tokens=rows)
        aux_graph = aux.AuxiliaryProjectionGraph(proposer, 3)
        replayed = []
        observer = NS(draft_replayed=lambda: replayed.append(1))
        aux_graph.outputs = torch.zeros(3, 2)
        aux_graph.graph = NS(replay=lambda: aux_graph.outputs.copy_(aux_graph.inputs @ weight))
        aux_graph.stage(taps, 0, 3, proposer._aux_staging)
        aux_graph.replay(taps, 3, rows, observer)
        torch.testing.assert_close(proposer._aux_staging, torch.cat(taps, dim=1)[:3])
        torch.testing.assert_close(proposer.hidden_states[3:], expected[3:])
        self.assertEqual(len(replayed), 3)

        kv = context.DraftContextGraph(proposer, 3)
        cache = torch.zeros(20, 2)
        def project():
            live = kv.slots[0] >= 0
            cache[kv.slots[0, live]] = kv.features[live] * 2 + kv.positions[live, None]
        kv.graph = NS(replay=project)
        kv.replay(3, rows, observer)
        torch.testing.assert_close(cache[slots[0, 3:]], expected[3:] * 2 + torch.arange(3, rows)[:, None])
        self.assertEqual(torch.count_nonzero(cache[:10]), 0)
        self.assertEqual(len(replayed), 6)


class OutputHeadGraphTest(unittest.TestCase):
    def test_ranked_projection_padded_rows_and_empty_steps(self):
        cls = load_source('output_head', 'benchmark_output_head.py').OutputHeadGraphs
        head = cls.__new__(cls)
        head.inputs = torch.zeros(4,3)
        head.sizes = [1,2,4]
        weights = torch.arange(15).reshape(3,5).float()
        def item(n):
            out = torch.zeros(n,5)
            return NS(replay=lambda: out.copy_(head.inputs[:n] @ weights)), out
        head.graphs = {n:item(n) for n in head.sizes}
        count = []
        observer = NS(head_scheduled=lambda p: count.append(('begin',p)), head_replayed=lambda p: count.append(('end',p)))
        hidden = torch.arange(9).reshape(3,3).float()
        torch.testing.assert_close(head.project(hidden, (0,1), observer), hidden @ weights)
        empty = head.project(hidden[:0], (0,), observer)
        self.assertEqual(empty.shape, (0,5))
        self.assertEqual(count,[('begin',(0,1)),('end',(0,1))])
        with self.assertRaises(RuntimeError):
            head.project(hidden.double(), (1,), observer)


if __name__ == '__main__':
    unittest.main()
