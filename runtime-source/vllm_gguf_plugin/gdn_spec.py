# SPDX-License-Identifier: Apache-2.0
"""Fused GDN decode for long speculative verifications (KAILLM_GDN_SPEC_TRITON=0 disables).

vLLM's fused GDN spec-decode path (gated delta-rule update + gated RMSNorm reading q/k/v and a/b in place, after the
causal conv) runs the CUDA op fused_gdn_decode_post_conv_mtp, which takes at most MAX_FUSED_GDN_MTP_TOKENS = 8
tokens per request.  A 16-token DSpine verification falls back to the unfused path (sigmoid-gated delta-rule kernel,
separate gated RMSNorm, copies of b / a and the core output).  vLLM ships a Triton port of the same fused kernel
(written for ROCm; any number of tokens, same state reads and writes, outputs rounded like the unfused path).  This
raises the limit to 16 and sends requests of more than 8 tokens to the Triton port; up to 8 keep the CUDA op.
KAILLM_GDN_DEFERRED=1 (servers with --max-num-seqs 1 only): those 9-16-token verifications run gdn_deferred instead,
which stores one state per step and layer instead of one per token (bit-identical outputs); up to 8 tokens the CUDA op
stays faster than either Triton kernel (RTX 3060, per layer: 4 tokens 32 vs 81-84 us).
KAILLM_GDN_SINGLE_SLOT=1 (--max-num-seqs 1 only; implies the deferred kernel for every verification length): vLLM
allocates ONE state block per GDN layer and request instead of 1 + num_speculative_tokens (MambaSpec
num_speculative_blocks = 0), the verification's state indices name that block in every column, and gdn_deferred
recomputes the accepted state in place.  Same outputs; frees the per-token state blocks for the attention KV cache
(DSpine 7 on the RTX 3060: 80 of 110 blocks).
"""
from __future__ import annotations

import os

import torch

MAX_TOKENS = 16


def _install_single_slot(layer_cls) -> None:
    """One GDN state block per request: no per-token state blocks in the KV-cache plan, and verification state
    indices widened back to 1 + num_spec columns (all naming the request's block) where the block table is narrower,
    so the causal conv still sees the verification length."""
    from dataclasses import replace

    from vllm.v1.attention.backends import gdn_attn
    from vllm.v1.kv_cache_interface import MambaSpec

    original_spec = layer_cls.get_kv_cache_spec

    def get_kv_cache_spec(self, vllm_config):
        spec = original_spec(self, vllm_config)
        if isinstance(spec, MambaSpec) and spec.num_speculative_blocks:
            spec = replace(spec, num_speculative_blocks=0)
        return spec

    builder = gdn_attn.GDNAttentionMetadataBuilder
    original_build, original_update = builder.build, builder.update_block_table

    def widen(self, md):
        idx = md.spec_state_indices_tensor
        width = self.num_spec + 1
        if idx is not None and idx.dim() == 2 and idx.shape[1] < width:
            md = replace(md, spec_state_indices_tensor=idx[:, :1].expand(-1, width))
        return md

    def build(self, *args, **kwargs):
        return widen(self, original_build(self, *args, **kwargs))

    def update_block_table(self, *args, **kwargs):
        return widen(self, original_update(self, *args, **kwargs))

    layer_cls.get_kv_cache_spec = get_kv_cache_spec
    builder.build, builder.update_block_table = build, update_block_table


def install() -> bool:
    if os.getenv("KAILLM_GDN_SPEC_TRITON", "1") != "1":
        return False
    try:
        from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn
        from vllm.third_party.flash_linear_attention.ops.fused_gdn_decode_post_conv_mtp import (
            fused_gdn_decode_post_conv_mtp_triton,
        )
    except ImportError:
        return False
    if getattr(gdn, "_kai_gdn_spec", False) or not hasattr(gdn, "MAX_FUSED_GDN_MTP_TOKENS"):
        return False
    cuda_limit = gdn.MAX_FUSED_GDN_MTP_TOKENS
    original_impl = gdn.fused_mtp_decode_impl
    single = os.getenv("KAILLM_GDN_SINGLE_SLOT", "0") == "1"
    deferred = single or os.getenv("KAILLM_GDN_DEFERRED", "0") == "1"
    current = [None]                                  # the GDN layer whose fused core is running
    if deferred:
        from . import gdn_deferred

        cls = gdn.QwenGatedDeltaNetAttention
        original_init, original_fused_norm = cls.__init__, cls._forward_core_fused_norm

        def init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            from vllm.config import get_current_vllm_config_or_none

            cfg = get_current_vllm_config_or_none()
            max_seqs = getattr(getattr(cfg, "scheduler_config", None), "max_num_seqs", None)
            maxt = int(getattr(self, "num_spec", 0) or 0) + 1
            self._kai_deferred = None
            if single and maxt > 1 and max_seqs != 1:
                raise RuntimeError("KAILLM_GDN_SINGLE_SLOT=1 needs --max-num-seqs 1")
            if single and maxt > MAX_TOKENS:
                raise RuntimeError(f"KAILLM_GDN_SINGLE_SLOT=1 supports up to {MAX_TOKENS} verification tokens")
            lo = 1 if single else cuda_limit
            if max_seqs == 1 and lo < maxt <= MAX_TOKENS and torch.cuda.is_available():
                dtype = cfg.model_config.dtype if cfg is not None else torch.bfloat16
                self._kai_deferred = gdn_deferred.make_context(
                    self.num_k_heads // self.tp_size, self.head_k_dim, self.num_v_heads // self.tp_size,
                    self.head_v_dim, maxt, dtype, torch.device("cuda", torch.cuda.current_device()),
                    single_slot=single)

        def fused_norm(self, mixed_qkv, b, a, output_gate, core_attn_out):
            from vllm.forward_context import get_forward_context

            raw = get_forward_context().attn_metadata
            md = raw.get(self.prefix) if isinstance(raw, dict) else None
            if md is not None and not (self._can_use_fused_gdn_mtp_decode(md) and md.num_prefills == 0):
                if single and self._kai_deferred is not None and md.num_spec_decodes:
                    # the per-token reference kernels would overwrite the single slot with unaccepted states
                    raise RuntimeError("KAILLM_GDN_SINGLE_SLOT: a verification left the fused GDN path")
                gdn_deferred.invalidate()        # prefill / plain decode: the side buffers no longer apply
            current[0] = self
            try:
                return original_fused_norm(self, mixed_qkv, b, a, output_gate, core_attn_out)
            finally:
                current[0] = None

        cls.__init__, cls._forward_core_fused_norm = init, fused_norm
        if single:
            _install_single_slot(cls)

    def dispatch(**kwargs):
        ctx = getattr(current[0], "_kai_deferred", None) if deferred else None
        if single and ctx is not None:
            if not gdn_deferred.supports(
                    ctx, kwargs["mixed_qkv"], kwargs["a"], kwargs["b"], kwargs["state"], kwargs["state_indices"],
                    kwargs["cu_seqlens"], kwargs["out"]):
                raise RuntimeError("KAILLM_GDN_SINGLE_SLOT: verification shape not supported by gdn_deferred")
            return gdn_deferred.fused_gdn_decode_deferred(ctx, **kwargs)
        if kwargs["state_indices"].shape[1] > cuda_limit:
            if ctx is not None and gdn_deferred.supports(
                    ctx, kwargs["mixed_qkv"], kwargs["a"], kwargs["b"], kwargs["state"], kwargs["state_indices"],
                    kwargs["cu_seqlens"], kwargs["out"]):
                return gdn_deferred.fused_gdn_decode_deferred(ctx, **kwargs)
            return fused_gdn_decode_post_conv_mtp_triton(**kwargs)
        return original_impl()(**kwargs)

    gdn.fused_mtp_decode_impl = lambda: dispatch
    gdn.MAX_FUSED_GDN_MTP_TOKENS = max(cuda_limit, MAX_TOKENS)
    gdn._kai_gdn_spec = True
    return True
