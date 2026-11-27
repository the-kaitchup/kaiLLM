from itertools import combinations
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime-source"))
from vllm_gguf_plugin.hy4.partition import layer_counts


class PipelineBalanceTest(unittest.TestCase):
    def test_optimal_contiguous_partitions(self):
        for weights in ([1, 1, 12, 12, 12, 12], [13, 6, 7, 7, 3, 4, 15], [4] * 14):
            for embed, head in ((1, 1), (9, 4), (4, 9)):
                for stages in (1, 2, 4):
                    def largest(cuts):
                        starts = (0, *cuts)
                        ends = (*cuts, len(weights))
                        costs = [sum(weights[a:b]) for a, b in zip(starts, ends)]
                        costs[0] += embed
                        costs[-1] += head
                        return max(costs)
                    sizes = layer_counts(weights, embed, head, stages)
                    self.assertEqual(sum(sizes), len(weights))
                    self.assertEqual(len(sizes), stages)
                    self.assertTrue(all(n > 0 for n in sizes))
                    cuts = tuple(sum(sizes[:i]) for i in range(1, stages))
                    self.assertEqual(largest(cuts), min(largest(c) for c in combinations(range(1, len(weights)), stages-1)))
