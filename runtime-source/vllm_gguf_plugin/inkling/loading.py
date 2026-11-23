# SPDX-License-Identifier: Apache-2.0
"""Weight loading for Inkling2LoopForCausalLM (names from weights_adapter/inkling.py)."""

from __future__ import annotations

import torch

_GGML_FROM_DTYPE = {torch.float32: 0, torch.float16: 1, torch.bfloat16: 30}


def load_inkling_weights(model, weights) -> set[str]:
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader

    from .moe import InklingExperts

    params = dict(model.named_parameters())
    mods = dict(model.named_modules())
    loaded: set[str] = set()
    for name, w in weights:
        if ".experts." in name:
            head, part, kind = name.rsplit(".", 2)
            experts = mods[head]
            if kind == "weight":
                if part.endswith("_scale"):
                    experts.offer(part, "data", w)
                else:
                    experts.offer(part, "type", torch.tensor(_GGML_FROM_DTYPE[w.dtype]))
                    experts.offer(part, "data", w)
            else:
                experts.offer(part, "type" if kind == "qweight_type" else "data", w)
            loaded.add(name)
            continue
        p = params.get(name)
        if p is None:
            raise KeyError(f"Unexpected Inkling weight: {name}")
        store = getattr(p, "_store", None)
        if store is not None:
            store(w)                       # kaiLLM GGUF parameters (qweight / qweight_type)
        elif isinstance(p, torch.nn.Parameter) and not hasattr(p, "weight_loader"):
            if p.numel() != w.numel():
                raise ValueError(f"{name}: {tuple(w.shape)} does not fit {tuple(p.shape)}")
            p.data.copy_(w.reshape(p.shape).to(p.dtype))
        else:
            getattr(p, "weight_loader", default_weight_loader)(p, w)
        loaded.add(name)
    device = next(model.parameters()).device
    load_itq_experts(model, device)
    for name, m in mods.items():
        if isinstance(m, InklingExperts):
            m.finalize(device)
    return loaded


def load_itq_experts(model, device) -> int:
    """Routed experts stored as itq-moe (yaqa.manifest): one ITQExpertStage per logical tensor, split gate / up
    joined; any expert tensor not in the manifest stays a GGML stage (formats mix per tensor)."""
    import re

    import gguf

    from ..yaqa_gguf import load_yaqa_components, read_yaqa_manifest
    from .itq_moe import ITQExpertStage, SplitGateUp

    found: dict = {}
    for path in getattr(model, "gguf_paths", []):
        reader = gguf.GGUFReader(str(path))
        for name, desc in read_yaqa_manifest(reader).items():
            if desc.get("codec") != "itq-moe":
                continue
            m = re.match(r"blk\.(\d+)\.ffn_(gate_up|gate|up|down)_exps\.weight$", name)
            if m is None:
                raise ValueError(f"itq-moe tensor with an unknown name: {name}")
            found[(int(m.group(1)), m.group(2))] = ITQExpertStage(desc, load_yaqa_components(reader, desc), device)
    layers = model.model.layers
    for (L, part), stage in found.items():
        ex = layers[L].experts
        if part == "gate_up":
            ex.itq_gate_up = stage
        elif part == "down":
            ex.itq_down = stage
    for L in {L for (L, part) in found if part in ("gate", "up")}:
        if (L, "gate") not in found or (L, "up") not in found:
            raise ValueError(f"layer {L}: split itq-moe gate / up needs both tensors")
        layers[L].experts.itq_gate_up = SplitGateUp(found[(L, "gate")], found[(L, "up")])
    for L, layer in enumerate(layers):
        ex = layer.experts
        if ex.itq_gate_up is not None and (ex.itq_gate_up.rows != 2 * ex.F or getattr(ex.itq_gate_up, "n", ex.D) != ex.D):
            raise ValueError(f"layer {L}: itq-moe gate_up shape does not match [2 x {ex.F}, {ex.D}]")
        if ex.itq_down is not None and (ex.itq_down.rows != ex.D or ex.itq_down.n != ex.F):
            raise ValueError(f"layer {L}: itq-moe down shape does not match [{ex.D}, {ex.F}]")
    return len(found)
