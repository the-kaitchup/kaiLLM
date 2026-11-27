# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 glue on the GPU: depth mixes, loop controllers, norms, clips, routing and the MoE epilogues.

Each function is the reference op (ref.py / moe.py) or a pure-torch composition of reference ops; CUDA calls go
through torch.compile (inductor) by default, so a function's dozen elementwise / reduction kernels become a few
fused ones.  The CUDA graphs capture the generated kernels like any other.  Shapes are dynamic in the token
dimension (one compilation per function and list length, not per batch size).  Cost: 15-45 s more at server start
with a warm inductor cache, more on a cold one.  CPU calls (tests) and KAILLM_INKLING_COMPILE=0 (opt-out) run the
reference code unchanged.

Dtypes around the linears: the norms feeding the projections return the model dtype directly (rms_to), the
projections' bf16 outputs are widened inside the consumer (post_projection, gate_attn, the convs), and the MoE
combine returns float32 (no bf16 round trip of the expert output).  The loop controllers of up to CONTROL_MAX_T
tokens and the depth mixes of up to MIX_MAX_T tokens run as two Triton kernels each (triton_glue.py;
KAILLM_INKLING_CONTROL_KERNEL=0 / KAILLM_INKLING_MIX_KERNEL=0: the torch path).
"""

from __future__ import annotations

import os

import torch

from . import ref as R
from . import chima_verifier, spark_verify_router
from .moe import route as _route
from .moe import route_from_logits as _route_from_logits
from .moe import situ as _situ

ENABLED = os.environ.get("KAILLM_INKLING_COMPILE", "1") == "1"
ROW_POST = os.environ.get("KAILLM_SPARK_DSPARK5_ROW_POST") == "1"
ROW_ATTENTION_RMS = os.environ.get("KAILLM_SPARK_DSPARK5_ROW_ATTN_RMS") == "1"
ROW_EMBEDDING = os.environ.get("KAILLM_SPARK_DSPARK5_ROW_EMBEDDING") == "1"
SPARK_ROW_FULL_ROUTER = os.environ.get("KAILLM_SPARK_DSPARK5_ROW_FULL_ROUTER") == "1"
SPARK_MIXTURE_FMA4 = os.environ.get("KAILLM_SPARK_DSPARK5_MIXTURE_FMA4") == "1"
SPARK_ROW_MIXTURE = os.environ.get("KAILLM_SPARK_DSPARK5_ROW_MIXTURE") == "1"
CONTROL_KERNEL = os.environ.get("KAILLM_INKLING_CONTROL_KERNEL", "1") == "1"
CONTROL_MAX_T = int(os.environ.get("KAILLM_INKLING_CONTROL_MAX_T", "32"))
MIX_KERNEL = os.environ.get("KAILLM_INKLING_MIX_KERNEL", "1") == "1"
MIX_MAX_T = int(os.environ.get("KAILLM_INKLING_MIX_MAX_T", "64"))
# functions kept eager up to a token count where the compiled kernel measured slower (name:T,...; e.g. gate_attn:1)
EAGER_MAX_T = {k: int(v) for k, v in (item.split(":") for item in
                                         os.environ.get("KAILLM_INKLING_EAGER_MAX_T", "").split(",") if item)}


def _post_projection(q, k, v, rel, q_clip, rel_clip, tau, qk_clip: float, rel_clip_thr: float, eps: float,
                     H: int, Hkv: int, hd: int, nrel: int, full: bool):
    """q / k / rel row clips, v RMS, and the full-attention log-scale tau, after the projections."""
    T = q.shape[0]
    q = R.clip_rows(q.view(T, H, hd).to(R.DT), q_clip, qk_clip)
    k = R.clip_rows(k.view(T, Hkv, hd).to(R.DT), None, qk_clip)
    v = R.rms(v.view(T, Hkv, hd), None, eps)
    rel = R.clip_rows(rel.view(T, H, nrel).to(R.DT), rel_clip, rel_clip_thr)
    if full:
        q = q * tau.view(T, 1, 1)
        rel = rel * tau.view(T, 1, 1)
    return q, k, v, rel


def _rms_to(x, w, eps: float, out_dtype: torch.dtype):
    """rms(x) * w (float32 math) in out_dtype: the input of the projections."""
    return R.rms(x, w, eps).to(out_dtype)


def _rms_pair(x, w, eps: float, out_dtype: torch.dtype):
    """rms(x) * w in float32 (the router's input) and in out_dtype (the experts' input), one pass."""
    y = R.rms(x, w, eps)
    return y, y.to(out_dtype)


def _gate_attn(att, gate, out_dtype: torch.dtype):
    """Attention output [T, H, hd] x sigmoid(gate projection [T, H * hd]) in out_dtype (the output projection's
    input); float32 math."""
    T = gate.shape[0]
    return (att.reshape(T, -1).to(R.DT) * torch.sigmoid(gate.to(R.DT))).to(out_dtype)


def _route_gpu(x, router, bias, top_k: int, w_scale: float):
    """Routing with int32 expert ids (what the expert kernels read)."""
    ids, w = _route(x, router, bias, top_k, w_scale)
    return ids.to(torch.int32), w


def _router_weights(weights):
    """Preserve the ordinary GB10 graph's pair fold and rounded reciprocal."""
    p0, p1, p2, p3, p4, p5, p6, p7 = weights.unbind(-1)
    total = ((p0 + p1) + (p4 + p5)) + ((p2 + p3) + (p6 + p7))
    reciprocal = (total + 1e-20).to(torch.float64).reciprocal().to(torch.float32)
    return weights * reciprocal.unsqueeze(-1)


def _router_tail(logits, bias, top_k: int, w_scale: float):
    if top_k != 8 or w_scale != 8.0:
        ids, weights = _route_from_logits(logits, bias, top_k, w_scale)
        return ids.to(torch.int32), weights
    probs = torch.sigmoid(logits)
    selection = probs + bias.float() if bias is not None else probs
    ids = selection.topk(top_k, dim=-1).indices
    weights = _router_weights(probs.gather(1, ids)) * w_scale
    return ids.to(torch.int32), weights


def _gate_up_epilogue(gu, scale, ids, F: int, beta: float, lb: float, out_dtype: torch.dtype):
    """Per-expert gate_up scale, SITU, cast for the down projection: gu [T * k, 2F] -> [T * k, F]."""
    gu = gu.float()
    if scale is not None:
        gu = gu * scale[ids.reshape(-1)][:, None]
    h = _situ(gu[:, :F], gu[:, F:], beta, lb)
    return h.to(out_dtype) if out_dtype != torch.float32 else h


def _combine(y, weights, scale, ids, out_dtype: torch.dtype):
    """Routing weights (x the per-expert down scale) applied to the down outputs [T * k, D], summed over the k
    slots -> [T, D]."""
    T, k = weights.shape
    w = weights.float()
    if scale is not None:
        w = w * scale[ids]
    y = y.float().reshape(T, k, y.shape[-1]) * w[..., None]
    return y.sum(1).to(out_dtype)


_REF = {"mix": R.mix, "control": R.control, "rms": R.rms, "rms_to": _rms_to, "rms_pair": _rms_pair, "router_tail": _router_tail,
        "clip_rows": R.clip_rows, "route": _route, "route_gpu": _route_gpu, "post_projection": _post_projection,
        "gate_attn": _gate_attn, "gate_up_epilogue": _gate_up_epilogue, "combine": _combine}
_COMPILED: dict = {}


def _fn(name: str, x: torch.Tensor):
    if not (ENABLED and x.is_cuda) or (EAGER_MAX_T and x.shape[0] <= EAGER_MAX_T.get(name, 0)):
        return _REF[name]
    f = _COMPILED.get(name)
    if f is None:
        # one graph per list length (depth mixes see 2-6 states) and per 0/1-specialised token count: room for all
        import torch._dynamo
        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
        torch._dynamo.config.accumulated_cache_size_limit = max(torch._dynamo.config.accumulated_cache_size_limit, 512)
        f = _COMPILED[name] = torch.compile(_REF[name], dynamic=True, fullgraph=True)
    return f


def _mix_rows_ok(blocks, cur) -> bool:
    st = cur.stride()
    return (len(blocks) < 8 and cur.dtype == torch.float32 and st[-1] == 1
            and all(b.dtype == torch.float32 and b.stride() == st and b.shape == cur.shape for b in blocks))


def mix(blocks, cur, query):
    if (MIX_KERNEL and cur.is_cuda and cur.shape[0] <= MIX_MAX_T and R.DT == torch.float32
            and _mix_rows_ok(blocks, cur)):
        from .triton_glue import mix_fused
        return mix_fused(blocks, cur, query)
    return _fn("mix", cur)(blocks, cur, query)


def control(cur, message, down, gain, forget):
    if (CONTROL_KERNEL and cur.is_cuda and cur.shape[0] <= CONTROL_MAX_T and R.DT == torch.float32
            and cur.dtype == torch.float32 and message.dtype == torch.float32):
        from .triton_glue import control_fused
        return control_fused(cur, message, down, gain, forget)
    return _fn("control", cur)(cur, message, down, gain, forget)


def rms(x, w=None, eps: float = 1e-6, out_dtype=None):
    """rms(x) * w; out_dtype (e.g. the model dtype for a projection's input) casts in the same kernel."""
    if out_dtype is None or out_dtype == R.DT:
        return _fn("rms", x)(x, w, eps)
    return _fn("rms_to", x)(x, w, eps, out_dtype)


def embedding_rms(x, w, eps):
    if chima_verifier.active() or (
        ROW_EMBEDDING and x.dtype == torch.bfloat16 and 2 <= x.shape[0] <= 6 and spark_verify_router.active()
    ):
        return torch.cat([rms(row, w, eps) for row in x.split(1, dim=0)], dim=0)
    return rms(x, w, eps)


def attention_rms(x, w, eps, out_dtype, *, chima_rows=False):
    if (chima_rows and chima_verifier.active()) or (
        ROW_ATTENTION_RMS and 2 <= x.shape[0] <= 6 and spark_verify_router.active()
    ):
        return torch.cat([rms(row, w, eps, out_dtype) for row in x.split(1, dim=0)], dim=0)
    return rms(x, w, eps, out_dtype)


def rms_pair(x, w, eps: float, out_dtype):
    return _fn("rms_pair", x)(x, w, eps, out_dtype)


def gate_attn(att, gate, out_dtype):
    return _fn("gate_attn", gate)(att, gate, out_dtype)


def clip_rows(x, scale, threshold: float):
    return _fn("clip_rows", x)(x, scale, threshold)


def route(x, router, bias, top_k: int, w_scale: float):
    """(ids [T, k], weights [T, k] float32); ids int32 on the GPU (the expert kernels' index type)."""
    spark = spark_verify_router.active()
    if chima_verifier.active() or (spark and SPARK_ROW_FULL_ROUTER):
        # Reuse the one-token decode's compiled projection and routing arithmetic.
        name = "route_gpu" if spark or x.is_cuda else "route"
        route_row = _fn(name, x[:1])
        pairs = [route_row(row, router, bias, top_k, w_scale) for row in x.split(1)]
        ids, weights = zip(*pairs, strict=True)
        return torch.cat(ids, dim=0), torch.cat(weights, dim=0)
    if spark:
        logits = spark_verify_router.project(x, router)
        return _fn("router_tail", logits)(logits, bias, top_k, w_scale)
    if x.is_cuda:
        return _fn("route_gpu", x)(x, router, bias, top_k, w_scale)
    return _fn("route", x)(x, router, bias, top_k, w_scale)


def post_projection(q, k, v, rel, q_clip, rel_clip, tau, *config, **kwargs):
    if chima_verifier.active():
        return chima_verifier.projected(_fn("post_projection", q[:1]), q, k, v, rel,
                                       q_clip, rel_clip, tau, config, kwargs)
    function = _fn("post_projection", q)
    if ROW_POST and 2 <= q.shape[0] <= 6 and spark_verify_router.active():
        # Match the ordinary single-token clip and RMS reduction during DSpark verification.
        rows = []
        for i in range(q.shape[0]):
            row = slice(i, i + 1)
            rows.append(function(q[row], k[row], v[row], rel[row], q_clip, rel_clip, tau[row], *config, **kwargs))
        return tuple(torch.cat(component, dim=0) for component in zip(*rows))
    return function(q, k, v, rel, q_clip, rel_clip, tau, *config, **kwargs)


def gate_up_epilogue(gu, scale, ids, F: int, beta: float, lb: float, out_dtype):
    return _fn("gate_up_epilogue", gu)(gu, scale, ids, F, beta, lb, out_dtype)


def combine(y, weights, scale, ids, out_dtype):
    tokens, experts = weights.shape
    if (SPARK_ROW_MIXTURE and spark_verify_router.active() and 2 <= tokens <= 6 and experts == 8
            and y.shape == (tokens * experts, 2304) and y.dtype == out_dtype == torch.float32):
        ordinary = _fn("combine", y[:experts])
        rows = [ordinary(y[row * experts:(row + 1) * experts], weights[row:row + 1],
                         scale, ids[row:row + 1], out_dtype) for row in range(tokens)]
        return torch.cat(rows, dim=0)
    if SPARK_MIXTURE_FMA4 and spark_verify_router.mixture_supported(y, weights, scale, ids, out_dtype):
        from .spark_mixture_fma4 import combine_fma4
        result = combine_fma4(y, weights, scale, ids)
        return result
    return _fn("combine", y)(y, weights, scale, ids, out_dtype)
