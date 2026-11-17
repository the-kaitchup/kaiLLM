"""DSpine drafter for vLLM's DFlash path (Zhang et al. 2026, "Draft in Parallel, Condition Through Depth:
Adjacent Causal Injection for Speculative Decoding", arXiv 2609.36173; reference code tencent/dspine).

The backbone is the DFlash drafter (target features of layers 1/16/31/46/61 -> fc -> context K/V of five Qwen3
layers; a block of anchor + mask tokens, causal inside the block).  After every layer a code bridge writes a message
from each block position into the next one: the layer's residual stream is RMS-normalized, read into a 1024-wide
unit code (the anchor uses its token's codebook row), packed by [G_q; G_p; A] into (q, s, v); position t >= 1 receives
sigmoid(q_t + s_{t-1} + b) * v_{t-1} through the layer's write matrix, scaled by the stream RMS.  The numerics follow
dspine_mlx/bridge.py (itself a rounding-point mirror of the SGLang plugin): BF16 storage, FP32 reductions.

forward() returns [final_hidden | last-layer raw stream | last-layer q] (BF16, width 2H + R) so the speculator's
refinement tail (exact-sender rescoring of the head's top-k, see speculator.py) gets the stash without side buffers.
The codebook is only needed through TSV = [G_p; A] applied to sqrt(C) * codebook rows (the anchor's s / v and the
refinement's predecessor terms); it is built at load and the codebook is dropped (-508 MB).
"""
from __future__ import annotations

from collections.abc import Iterable

import torch

from vllm.config import VllmConfig
from vllm.model_executor.models.qwen3_dflash import DFlashQwen3ForCausalLM, DFlashQwen3Model

from .bridge import _VERSIONS, DSpineBridge, dspine_head_topk

class DSpineQwen3Model(DFlashQwen3Model):
    def __init__(self, *, vllm_config: VllmConfig, start_layer_id: int = 0, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, start_layer_id=start_layer_id, prefix=prefix)
        self.block_tokens = 1 + int(vllm_config.speculative_config.num_speculative_tokens)
        self.selected_prefix_bridge = DSpineBridge(self.config, vllm_config.model_config.dtype)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor,
                input_embeds: torch.Tensor | None = None) -> torch.Tensor:
        if input_embeds is None:
            input_embeds = self.embed_input_ids(input_ids)
        rows = torch.arange(input_ids.shape[0], device=input_ids.device)
        is_anchor = rows % self.block_tokens == 0
        hidden, residual = input_embeds, None
        raw = q = None
        for i, layer in enumerate(self.layers):
            hidden, residual = layer(positions=positions, hidden_states=hidden, residual=residual)
            hidden, raw, q = self.selected_prefix_bridge.write(i, hidden, residual, input_ids, is_anchor)
            residual = None                       # the next layer normalizes the written stream itself
        final = self.norm(hidden)
        return torch.cat([final, raw.to(final.dtype), q.to(final.dtype)], dim=-1)


class DSpineQwen3ForCausalLM(DFlashQwen3ForCausalLM):
    model_cls = DSpineQwen3Model

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.head_topk = dspine_head_topk(self.config)
        self.feature_dtype = vllm_config.model_config.dtype

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # kaiLLM's Qwen3.5/3.8 targets keep the residual stream in FP32; the drafter was trained on BF16 features
        # (model dtype, not fc.weight's: a quantized fc has no weight tensor).
        return super().combine_hidden_states(hidden_states.to(self.feature_dtype))

    def compute_candidates(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Top-k token ids / logits of the (shared target) head on the final draft rows."""
        logits = self.compute_logits(hidden_states)
        values, ids = logits.float().topk(self.head_topk, dim=-1)
        return ids, values

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        kept = []
        for name, tensor in weights:
            short = name.split(".")[-1]
            if short in _VERSIONS:
                if int(tensor.item()) != 1:
                    raise ValueError(f"unsupported DSpine checkpoint: {short} = {int(tensor.item())}")
                continue
            kept.append((name, tensor))
        super().load_weights(kept)
        self.model.selected_prefix_bridge.prepare()


EntryClass = DSpineQwen3ForCausalLM
