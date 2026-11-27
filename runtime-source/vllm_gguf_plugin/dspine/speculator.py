"""DSpine speculator for vLLM's GPU model runner (v2): the DFlash parallel draft plus DSpine's exact-sender
refinement of the last write (dspine_mlx/propose.py RefinementTail.greedy).

Per proposal row i (block positions 1..T-1) the head's top-K candidates c are rescored for every predecessor
candidate p (row i-1's candidates; the anchor for row 0):

    score[i, p, c] = base[i, c] + rms_i * < sigmoid(q_i + TSV_g[p] + b) * TSV_v[p], TG[c] >
    base[i, c]     = < raw_i * norm_w, head[c] >        (the last layer's stream before its write)
    TG[c]          = head[c] * norm_w @ W_write[last]    (vocabulary-wide, built once from the target head)

and the chain is walked greedily (temperature 0) or with Gumbel noise over the realized scores (temperature > 0),
by DFlash2's CandidateSampler, whose realized scores also feed the rejection sampler and adaptive verification.

The head is the target's (shared by load_dflash_model), so with a quantized target (kaiLLM ITQ / GGUF) both base
and TG are computed through the target's own head forward: base by one extra head pass on raw * norm_w, TG by one
head pass over the R columns of diag(norm_w) W_write at load.
"""
from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import CandidateSampler

from .bridge import _F32_TINY, Int8Rows, dspine_head_topk, int8_tables, refine_scores

logger = init_logger(__name__)


class DSpineSpeculator(DFlashSpeculator):
    _speculator_name = "DSpine"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        self.top_k = dspine_head_topk(self.draft_model_config.hf_config)
        self.candidate_sampler = CandidateSampler(self.max_num_reqs, self.num_speculative_steps, self.top_k, device)
        self.tg: torch.Tensor | None = None
        self._graph_observer = None
        self._context_graph = None
        self._auxiliary_graph = None
        self._refinement_captured = False

    def capture(self):
        super().capture()
        if os.environ.get('KAILLM_BENCHMARK_GRAPH_DIR'):
            from ..draft_context_graph import DraftContextGraph
            from .context_graph import AuxiliaryProjectionGraph

            if self._aux_staging is None:
                raise RuntimeError('DSpine needs the target auxiliary tap staging buffer')
            self._context_graph = DraftContextGraph(self)
            self._context_graph.capture()
            self._auxiliary_graph = AuxiliaryProjectionGraph(self)
            self._auxiliary_graph.capture()

    def install_benchmark_graphs(self, observer):
        manager = self.query_cudagraph_manager
        if (not self._refinement_captured or self._context_graph is None or self._context_graph.graph is None
                or self._auxiliary_graph is None or self._auxiliary_graph.graph is None or manager is None):
            raise RuntimeError('Incomplete DSpine auxiliary, context or refinement graph capture')
        for batch in range(1, self.max_num_reqs + 1):
            desc = manager.dispatch(batch, batch * self.num_query_per_req, self.num_query_per_req, 0)
            if desc not in manager.graphs:
                raise RuntimeError(f'No captured DSpine query for {batch} requests')
        captures = observer.observe_manager(manager)
        self._graph_observer = observer
        return captures + 2

    def _store_hidden_states(self, num_tokens, last_hidden_states, aux_hidden_states, num_graph_tokens=0):
        if self._graph_observer is None:
            return super()._store_hidden_states(num_tokens, last_hidden_states, aux_hidden_states, num_graph_tokens)
        if not aux_hidden_states:
            self._graph_observer.draft_fallback('DSpine received no target auxiliary states', unclassified=True)
        covered = min(num_tokens, num_graph_tokens)
        if covered:
            self._auxiliary_graph.stage(aux_hidden_states, 0, covered, self._aux_staging)
        self._auxiliary_graph.replay(aux_hidden_states, covered, num_tokens, self._graph_observer)

    def _precompute_context_kv(self, start, end, dummy_run=False):
        if self._graph_observer is None:
            return super()._precompute_context_kv(start, end, dummy_run)
        if dummy_run:
            self._graph_observer.draft_fallback('Unexpected DSpine dummy request', unclassified=True)
        self._context_graph.replay(start, end, self._graph_observer)

    def draft_logits_spec(self, vllm_config: VllmConfig) -> tuple[torch.dtype, float]:
        # As DFlash2: the walk and the rejection that checks it read the same fp32 distribution over K candidates.
        return torch.float32, -float("inf")

    def load_draft_model(self, target_model: nn.Module, target_attn_layer_names: set[str]) -> nn.Module:
        model = super().load_draft_model(target_model, target_attn_layer_names)
        self.tg = self._build_tg(model)
        return model

    @torch.inference_mode()
    def _build_tg(self, model: nn.Module, chunk: int = 64) -> torch.Tensor:
        bridge = model.model.selected_prefix_bridge
        norm_w = model.model.norm.weight.float()
        write = bridge.feature_write_weights[bridge.num_layers - 1].float()            # [H, R]
        x = (norm_w[:, None] * write).T.contiguous().to(self.dtype)                      # [R, H]
        parts = []
        for start in range(0, x.shape[0], chunk):
            logits = model.compute_logits(x[start:start + chunk])                       # [rows, V] via the target head
            parts.append(logits[:, : self.vocab_size].to(self.dtype))
        tg = torch.cat(parts, 0).T.contiguous()                                          # [V, R]
        del parts
        if int8_tables():
            tg = Int8Rows(tg)
        torch.cuda.empty_cache()      # the BF16 table and the logits chunks were transient
        logger.info("DSpine: refinement table TG %s built through the target head%s", tuple(tg.shape),
                    " (int8 rows)" if int8_tables() else "")
        return tg

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        if self._graph_observer is not None:
            self._graph_observer.draft_fallback('DSpine query selected an uncovered shape')
        out = self._run_model(num_tokens_padded, attn_metadata, slot_mappings, num_tokens_across_dp,
                              cudagraph_runtime_mode)
        steps, K = self.num_speculative_steps, self.top_k
        num_sample = num_reqs * steps
        H = self.hidden_size
        bridge = self.model.model.selected_prefix_bridge
        R = bridge.rank
        rows = out[self.sample_indices[:num_sample]]                                    # [n, 2H + R]
        final, raw, q = rows.split([H, H, R], dim=-1)
        rawf = raw.float()
        rms = rawf.square().mean(-1).clamp_min(_F32_TINY).sqrt()                         # [n]
        lifted = (rawf * self.model.model.norm.weight.float()).to(self.dtype)
        # One head pass for both: candidates (top-k of the final rows) and the base logits of the lifted raw rows
        # (the head's rows are independent: same values as two passes, ~one 16-row head pass less per step).
        logits = self.model.compute_logits(torch.cat([final.to(self.dtype), lifted], 0)).float()
        _, cand = logits[:num_sample].topk(self.model.head_topk, dim=-1)                 # [n, K]
        base = logits[num_sample:].gather(1, cand)                                       # [n, K]
        cand = cand.view(num_reqs, steps, K)
        anchors = self.input_buffers.input_ids[: num_reqs * self.num_query_per_req: self.num_query_per_req]
        scores = refine_scores(q, rms, cand, base.view(num_reqs, steps, K), anchors, bridge.tsv, self.tg,
                               bridge.gate.bias)
        self.candidate_sampler.sample(
            cand,
            scores,
            num_reqs,
            self.sample_pos,
            self.sample_idx_mapping,
            self.temperature,
            self.seeds,
            self.draft_tokens,
            self.draft_logits,
            self.use_fp64_gumbel,
        )
        if self.enable_adaptive_verification:
            self._maybe_predict_acceptance(
                self.candidate_sampler.scores[:num_reqs].flatten(0, 1),
                self.sample_idx_mapping[:num_sample],
                self.sample_col[:num_sample],
            )
        if os.environ.get('KAILLM_BENCHMARK_GRAPH_DIR') and torch.cuda.is_current_stream_capturing():
            self._refinement_captured = True
