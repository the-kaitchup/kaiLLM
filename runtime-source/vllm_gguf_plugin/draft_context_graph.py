# SPDX-License-Identifier: Apache-2.0
"""Replay a fixed-size drafter context projection for arbitrarily long prompt chunks."""
import torch


class DraftContextGraph:
    def __init__(self, proposer, chunk_size=256):
        self.proposer = proposer
        self.size = min(chunk_size, proposer.max_num_tokens)
        self.features = proposer.hidden_states.new_zeros((self.size, proposer.hidden_states.shape[-1]))
        self.positions = proposer.context_positions.new_zeros(self.size)
        self.slots = proposer._context_slot_mappings.new_full((proposer._context_slot_mappings.shape[0], self.size), -1)
        self.graph = None

    def capture(self):
        p = self.proposer
        layer_groups = p._layer_group_idx
        slots = self.slots[0] if layer_groups is None else [self.slots[i] for i in layer_groups]
        graph = torch.cuda.CUDAGraph()
        # Invalid slots isolate both warmup and capture from the model's real cache.
        with torch.cuda.device(self.features.device):
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                p.model.precompute_and_store_context_kv(self.features, self.positions, slots)
            torch.cuda.current_stream().wait_stream(stream)
            with torch.cuda.graph(graph, stream=stream):
                p.model.precompute_and_store_context_kv(self.features, self.positions, slots)
        self.graph = graph

    def replay(self, start, end, observer=None):
        if self.graph is None:
            if observer is not None:
                observer.draft_fallback('No captured drafter context projection')
            raise RuntimeError('No captured drafter context projection')
        p = self.proposer
        for i in range(start, end, self.size):
            stop = min(i + self.size, end)
            n = stop - i
            self.features[:n].copy_(p.hidden_states[i:stop])
            self.positions[:n].copy_(p.context_positions[i:stop])
            self.slots[:, :n].copy_(p._context_slot_mappings[:, i:stop])
            # The final partial block can reuse its existing feature values. Invalid
            # slots stop its projection from modifying any live request's KV cache.
            if n < self.size:
                self.slots[:, n:].fill_(-1)
            self.graph.replay()
            if observer is not None:
                observer.draft_replayed()
