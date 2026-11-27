"""Full graph capture for the target vocabulary projection used by the benchmark."""
from bisect import bisect_left
import torch


class OutputHeadGraphs:
    def __init__(self, hidden, max_rows, project):
        self.inputs = hidden.new_zeros((max_rows, *hidden.shape[1:]))
        self.sizes = sorted({max_rows, *(1 << i for i in range(max_rows.bit_length()) if 1 << i <= max_rows)})
        self.graphs = {}
        from vllm.distributed.parallel_state import get_tp_group
        with torch.cuda.device(hidden.device), get_tp_group().graph_capture() as context:
            for n in reversed(self.sizes):
                project(self.inputs[:n])
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=context.stream):
                    logits = project(self.inputs[:n])
                if logits is None or logits.shape[0] != n:
                    raise RuntimeError('Target head returned no capturable vocabulary rows on this rank')
                self.graphs[n] = (graph, logits)

    def project(self, hidden, phases, observer):
        rows = hidden.shape[0]
        if not rows:
            return self.graphs[self.sizes[0]][1][:0]
        index = bisect_left(self.sizes, rows)
        if index == len(self.sizes) or tuple(hidden.shape[1:]) != tuple(self.inputs.shape[1:]) or hidden.dtype != self.inputs.dtype:
            raise RuntimeError('Blocked an uncaptured target output-head shape')
        n = self.sizes[index]
        self.inputs[:rows].copy_(hidden)
        graph, logits = self.graphs[n]
        observer.head_scheduled(phases)
        graph.replay()
        observer.head_replayed(phases)
        return logits[:rows]
