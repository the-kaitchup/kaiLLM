# SPDX-License-Identifier: Apache-2.0
"""Weight loading for HYV4FlashForCausalLM (names produced by weights_adapter/hy4.py)."""

from __future__ import annotations

import torch

_GGML_FROM_DTYPE = {torch.float32: 0, torch.float16: 1, torch.bfloat16: 30}


def _dequant(d: dict, part: str, shape: tuple[int, int, int], device) -> torch.Tensor:
    """[H, a, b] dense float32 from a GGUF payload (qweight + qweight_type) or a plain weight."""
    if (part, "weight") in d:
        return d[(part, "weight")].to(device=device, dtype=torch.float32).reshape(shape)
    w = d[(part, "qweight")]
    t = int(d[(part, "qweight_type")].item())
    H, a, b = shape
    if torch.device(device).type != "cuda":   # CPU (tests): gguf-py's reference dequantization
        import gguf
        from gguf.quants import dequantize
        arr = dequantize(w.reshape(H * a, -1).numpy(), gguf.GGMLQuantizationType(t))
        return torch.from_numpy(arr).to(torch.float32).view(H, a, b)
    from .. import ops
    w = w.to(device)
    return ops.ggml_dequantize(w.reshape(H * a, -1).contiguous(), t, H * a, b, torch.float32).view(H, a, b)


def _ready(d: dict, part: str) -> bool:
    return (part, "weight") in d or ((part, "qweight") in d and (part, "qweight_type") in d)


def load_hy4_weights(model, weights) -> set[str]:
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader

    params = dict(model.named_parameters())
    mods = dict(model.named_modules())
    loaded: set[str] = set()
    kb: dict[str, dict] = {}
    for name, w in weights:
        head, part, kind = (name.rsplit(".", 2) + ["", ""])[:3] if name.count(".") >= 2 else (name, "", "")
        if head.endswith(".mlp.experts") and part in ("gate", "up", "down"):
            if kind == "weight":        # unquantized GGUF tensor: no qweight_type, derive it from the dtype
                mods[head].offer(part, "type", torch.tensor(_GGML_FROM_DTYPE[w.dtype]))
            mods[head].offer(part, "type" if kind == "qweight_type" else "data", w)
            loaded.add(name)
            continue
        if head.endswith(".self_attn") and part in ("k_b", "v_b"):
            d = kb.setdefault(head, {})
            d[(part, kind)] = w
            if _ready(d, "k_b") and _ready(d, "v_b"):
                attn = mods[head]
                H, nope, v, kv = attn.num_heads, attn.nope, attn.v_dim, attn.kv_lora
                dev = attn.kv_b_proj.weight.device
                k_b = _dequant(d, "k_b", (H, kv, nope), dev)          # GGUF attn_k_b: [H, kv, nope]
                v_b = _dequant(d, "v_b", (H, v, kv), dev)             # GGUF attn_v_b: [H, v, kv]
                kv_b = torch.cat([k_b.transpose(1, 2), v_b], dim=1).reshape(H * (nope + v), kv)
                attn.kv_b_proj.weight.data.copy_(kv_b.to(attn.kv_b_proj.weight.dtype))
                loaded.add(f"{head}.kv_b_proj.weight")
                del kb[head]
            loaded.add(name)
            continue
        p = params.get(name)
        if p is None:
            raise KeyError(f"Unexpected Hy4 weight: {name}")
        store = getattr(p, "_store", None)
        if store is not None:
            # kaiLLM's GGUF parameters (qweight / qweight_type): vLLM's ReplicatedLinear has no
            # weight_loader_v2, and its base weight_loader cannot materialize them
            store(w)
        else:
            getattr(p, "weight_loader", default_weight_loader)(p, w)
        loaded.add(name)
    if kb:
        raise ValueError(f"Incomplete attn_k_b / attn_v_b for {sorted(kb)}")
    for name, m in mods.items():
        if name.endswith(".mlp.experts") and m.type13 < 0:
            raise ValueError(f"Missing expert tensors for {name}")
    return loaded
