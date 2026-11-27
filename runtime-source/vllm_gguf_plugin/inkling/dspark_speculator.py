# SPDX-License-Identifier: Apache-2.0
"""Run the Inkling recurrent head with vLLM's draft sampler and target rejection sampler."""

import os

import torch

from ..draft_context_graph import DraftContextGraph
from torch.nn import functional as F
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator


class InklingDSparkSpeculator(DSparkSpeculator):
    _speculator_name = 'Inkling DSpark'

    def __init__(self, vllm_config, device):
        super().__init__(vllm_config, device)
        c = self.draft_model_config.hf_config
        self.hidden_states = torch.zeros(self.max_num_tokens, len(c.dflash_config['layer_ids']) * c.hidden_size,
                                        dtype=self.dtype, device=device)
        self.anchor_context = torch.zeros(self.max_num_reqs, c.hidden_size, dtype=self.dtype, device=device)

        # InputBatch rejection counts are gathered into this persistent graph input.
        self._num_rejected = torch.zeros(self.max_num_reqs, dtype=torch.int32, device=device)
        self._context_graph = None
        self._capturing_query = False
        self._captured_markov = False
        self._graph_observer = None

    def capture(self):
        if self._graph_observer is not None:
            self._graph_observer.draft_fallback('Cannot recapture during Inkling benchmark execution')
        self._captured_markov = False
        self._context_graph = DraftContextGraph(self)
        self._context_graph.capture()
        self._capturing_query = True
        try:
            super().capture()
        finally:
            self._capturing_query = False

    def install_benchmark_graphs(self, observer):
        if self._context_graph is None or self._context_graph.graph is None or not self._captured_markov:
            raise RuntimeError('Incomplete Inkling context, query or Markov graph capture')
        manager = self.query_cudagraph_manager
        if manager is None or not manager._graphs_captured:
            raise RuntimeError('Inkling draft attention did not capture its query graphs')
        for n in range(1, self.max_num_reqs + 1):
            desc = manager.dispatch(n, n * self.num_query_per_req, self.num_query_per_req, 0)
            if desc not in manager.graphs:
                raise RuntimeError(f'Missing Inkling draft query graph for {n} requests')
        count = observer.observe_manager(manager)
        self._graph_observer = observer
        return count + 1

    def _precompute_context_kv(self, start, end, dummy_run=False):
        if self._capturing_query or self._context_graph is None:
            if self._graph_observer is not None:
                self._graph_observer.draft_fallback('Uncaptured Inkling context projection')
            return super()._precompute_context_kv(start, end, dummy_run)
        if dummy_run:
            self._context_slot_mappings[:, start:end].fill_(-1)
        self._context_graph.replay(start, end, self._graph_observer)

    def _generate_draft(self, *args, **kwargs):
        if self._graph_observer is not None:
            self._graph_observer.draft_fallback('Inkling draft query selected an uncaptured shape')
        return super()._generate_draft(*args, **kwargs)

    def load_draft_model(self, target_model, target_attn_layer_names):
        model = DFlashSpeculator.load_draft_model(self, target_model, target_attn_layer_names)
        if not model.config.folded_markov:
            model.model.embed_norm = target_model.model.embed_norm
            if (self.num_speculative_steps == 5 and self.max_num_reqs == 1
                    and os.environ.get('KAILLM_SPARK_DSPARK5_DRAFT_WINDOW') == '1'):
                from .spark_verify_router import supported

                if supported(self.vllm_config, self.device):
                    from .spark_draft_local_kv import prepare_window

                    for layer in model.model.layers:
                        if layer.window == 1024:
                            layer.self_attn._inkling_prepare_window = prepare_window
        return model

    def prepare_context_anchor(self, num_reqs, query_start_loc, num_rejected):
        if self._graph_observer is not None:
            self._graph_observer.draft_fallback('Uncaptured Inkling Markov anchor')
        last = query_start_loc[1:num_reqs + 1].long() - 1
        if num_rejected is not None:
            last = last - num_rejected[:num_reqs]
        rows = self.hidden_states[last.clamp(0, self.hidden_states.shape[0] - 1)]
        c = self.model.config
        self.anchor_context[:num_reqs].copy_(self.model.markov_context(rows.reshape(num_reqs, -1, c.hidden_size)))

    def _sample_sequential(self, num_reqs, head_hidden):
        steps = self.num_speculative_steps
        n = num_reqs * steps
        h = head_hidden[self.sample_indices[:n]].reshape(num_reqs, steps, -1)
        prev = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]]
        c = self.model.config
        state = torch.zeros(num_reqs, c.markov_rank, device=h.device, dtype=torch.float32)
        positions = self.sample_pos[:n].reshape(num_reqs, steps)
        indices = self.sample_idx_mapping[:n].reshape(num_reqs, steps)
        head = self.model.markov
        if head.folded:
            base = self.model.compute_logits(h.reshape(n, -1)).reshape(num_reqs, steps, -1)
        for i in range(steps):
            hidden = h[:, i]
            state, delta = head.step(prev, state, hidden, self.anchor_context[:num_reqs])
            if head.folded:
                logits = base[:, i, :c.unpadded_vocab_size].float() + delta.float()
                if head.softcap:
                    logits = head.softcap * (logits / head.softcap).tanh()
            else:
                logits = head.logits(hidden, delta, self.model.compute_logits)[:, :c.unpadded_vocab_size]
            logits = F.pad(logits, (0, c.vocab_size - c.unpadded_vocab_size), value=-float('inf'))
            prev = self._sample_logits(logits, indices[:, i], positions[:, i], i)
            self.draft_tokens[:num_reqs, i] = prev
        if self._capturing_query and torch.cuda.is_current_stream_capturing():
            self._captured_markov = True
