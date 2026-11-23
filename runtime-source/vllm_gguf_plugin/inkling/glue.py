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
from .moe import route as _route
from .moe import situ as _situ

ENABLED = os.environ.get("KAILLM_INKLING_COMPILE", "1") == "1"
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


_REF = {"mix": R.mix, "control": R.control, "rms": R.rms, "rms_to": _rms_to, "rms_pair": _rms_pair,
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


def rms_pair(x, w, eps: float, out_dtype):
    return _fn("rms_pair", x)(x, w, eps, out_dtype)


def gate_attn(att, gate, out_dtype):
    return _fn("gate_attn", gate)(att, gate, out_dtype)


def clip_rows(x, scale, threshold: float):
    return _fn("clip_rows", x)(x, scale, threshold)


def route(x, router, bias, top_k: int, w_scale: float):
    """(ids [T, k], weights [T, k] float32); ids int32 on the GPU (the expert kernels' index type)."""
    if x.is_cuda:
        return _fn("route_gpu", x)(x, router, bias, top_k, w_scale)
    return _fn("route", x)(x, router, bias, top_k, w_scale)


def post_projection(*args, **kwargs):
    return _fn("post_projection", args[0])(*args, **kwargs)


def gate_up_epilogue(gu, scale, ids, F: int, beta: float, lb: float, out_dtype):
    return _fn("gate_up_epilogue", gu)(gu, scale, ids, F, beta, lb, out_dtype)


def combine(y, weights, scale, ids, out_dtype):
    return _fn("combine", y)(y, weights, scale, ids, out_dtype)
