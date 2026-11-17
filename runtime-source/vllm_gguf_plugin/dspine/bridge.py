"""DSpine code bridge and refinement scores in plain PyTorch (no vLLM imports; see model.py for the
algorithm and dspine_mlx/bridge.py + propose.py for the reference)."""
from __future__ import annotations

import math
import os

import torch
from torch import nn

_F32_TINY = torch.finfo(torch.float32).tiny
_VERSIONS = ("draft_block_causal_version", "last_write_refine_version", "independent_write_version",
             "terminal_disabled_version")


def dspine_bridge_config(hf_config) -> dict:
    cfg = (getattr(hf_config, "dflash_config", None) or {}).get("selected_prefix_code_bridge")
    if not cfg:
        raise ValueError("not a DSpine checkpoint: dflash_config.selected_prefix_code_bridge is missing")
    return cfg


def dspine_head_topk(hf_config) -> int:
    dflash = getattr(hf_config, "dflash_config", None) or {}
    for key in ("selected_prefix_last_write_refine", "code_bridge_second_pass"):
        value = dflash.get(key) or getattr(hf_config, key, None)
        if value and "head_topk" in value:
            return int(value["head_topk"])
    raise ValueError("DSpine checkpoint names no refinement width (head_topk)")


class DSpineBridge(nn.Module):
    """Parameter names / shapes of the HF export under ``model.selected_prefix_bridge.``."""

    def __init__(self, config, dtype: torch.dtype):
        super().__init__()
        bridge = dspine_bridge_config(config)
        H, V, L = int(config.hidden_size), int(config.vocab_size), int(config.num_hidden_layers)
        C, R = int(bridge["code_size"]), int(bridge["rank"])
        self.hidden_size, self.code_size, self.rank, self.num_layers = H, C, R, L
        self.eps = float(bridge.get("eps", 1e-6))
        self.sqrt_c = math.sqrt(C)
        p = lambda *shape: nn.Parameter(torch.zeros(*shape, dtype=dtype), requires_grad=False)
        self.prev = nn.Module()
        self.prev.weight = p(R, C)
        self.gate = nn.Module()
        self.gate.weight = p(R, C)
        self.gate.bias = p(R)
        self.predecessor_gate = nn.Module()
        self.predecessor_gate.weight = p(R, C)
        self.out = nn.Module()
        self.out.weight = p(C, R)
        self.read_weight = p(C, H)
        self.early_read_weights = p(L - 1, C, H)
        self.feature_write_weights = p(L, H, R)
        self.codebook = p(V, C)
        # Derived at load (prepare): packed [G_q; G_p; A] [3R, C] and TSV [V, 2R] (predecessor gate / value terms).
        self.register_buffer("packed_weight", torch.zeros(3 * R, C, dtype=dtype), persistent=False)
        self.register_buffer("tsv", torch.zeros(V, 2 * R, dtype=dtype), persistent=False)
        self.prepared = False

    @torch.no_grad()
    def prepare(self) -> None:
        if bool((self.out.weight != 0).any()):
            raise ValueError("terminal-disabled DSpine checkpoint has a nonzero terminal projection")
        self.packed_weight.copy_(torch.cat([self.gate.weight, self.predecessor_gate.weight, self.prev.weight], 0))
        pv = torch.cat([self.predecessor_gate.weight, self.prev.weight], 0)               # [2R, C]
        for start in range(0, self.codebook.shape[0], 32768):
            rows = (self.sqrt_c * self.codebook[start:start + 32768].float()).to(self.codebook.dtype)
            self.tsv[start:start + 32768] = rows @ pv.T
        self.codebook = nn.Parameter(self.codebook.new_empty(0, self.code_size), requires_grad=False)
        if int8_tables():
            table = self._buffers.pop("tsv")
            self.tsv = Int8Rows(table)
            del table
        torch.cuda.empty_cache()      # the BF16 codebook / TSV were transient: hand their blocks back to the driver
        self.prepared = True

    def read(self, layer: int) -> torch.Tensor:
        return self.read_weight if layer == self.num_layers - 1 else self.early_read_weights[layer]

    def write(self, layer: int, hidden: torch.Tensor, residual: torch.Tensor, input_ids: torch.Tensor,
              is_anchor: torch.Tensor):
        """One write after ``layer``: hidden = MLP output, residual = pre-MLP stream, rows in DFlash query order
        (per request: anchor, then mask tokens).  Returns (out, raw, q)."""
        R = self.rank
        raw = (hidden + residual).to(torch.bfloat16)
        f = raw.float()
        rms = f.square().mean(-1).clamp_min(_F32_TINY).sqrt()                                     # [N]
        normalized = (f / (rms[:, None] + self.eps)).to(torch.bfloat16)
        projected = (normalized @ self.read(layer).T).float()                                   # [N, C]
        codes = (self.sqrt_c * projected / (projected.norm(dim=-1, keepdim=True) + self.eps)).to(torch.bfloat16)
        packed = codes @ self.packed_weight.T                                                  # [N, 3R] bf16
        q, s, v = packed.float().split(R, dim=-1)
        anchor = is_anchor[:, None]
        tsv = self.tsv[input_ids].float()                                                      # anchor rows: codebook code
        s = torch.where(anchor, tsv[:, :R], s)
        v = torch.where(anchor, tsv[:, R:], v)
        gate = torch.sigmoid(q + torch.roll(s, 1, 0) + self.gate.bias.float())
        message = torch.where(anchor, 0.0, gate * torch.roll(v, 1, 0)).to(torch.bfloat16)      # from position t-1
        delta = (message @ self.feature_write_weights[layer].T).float()                       # [N, H]
        out = torch.where(anchor, f, f + rms[:, None] * delta).to(torch.bfloat16)
        return out, raw, packed[:, :R]


class Int8Rows:
    """Row-quantized lookup table (int8, one F32 scale per row) for TSV / TG (KAILLM_DSPINE_INT8_TABLES=1, small GPUs):
    half the memory of BF16; only gathered rows are dequantized (t[idx] -> F32 rows)."""

    def __init__(self, table: torch.Tensor, chunk: int = 16384):
        self.shape = table.shape
        self.q = torch.empty(table.shape, dtype=torch.int8, device=table.device)
        self.s = torch.empty(table.shape[0], dtype=torch.float32, device=table.device)
        for start in range(0, table.shape[0], chunk):
            t = table[start:start + chunk].float()
            s = t.abs().amax(-1).clamp_min(1e-12) / 127.0
            self.q[start:start + chunk] = torch.round(t / s[:, None]).clamp_(-127, 127).to(torch.int8)
            self.s[start:start + chunk] = s
        if host_tables() and self.q.is_cuda:
            # rows stay in pinned host memory; the GPU gathers them over PCIe (same bytes, same kernels)
            from ..host_tables import host_alias

            q, self.q = self.q, None
            self.q = host_alias(q)
            del q

    def __getitem__(self, idx):
        return self.q[idx].float() * self.s[idx].unsqueeze(-1)


def int8_tables() -> bool:
    return os.getenv("KAILLM_DSPINE_INT8_TABLES", "0") == "1"


def host_tables() -> bool:
    """KAILLM_DSPINE_HOST_TABLES=1 (with int8 tables): TSV / TG rows in pinned host memory (zero-copy gathers)."""
    return int8_tables() and os.getenv("KAILLM_DSPINE_HOST_TABLES", "0") == "1"


def refine_scores(q: torch.Tensor, rms: torch.Tensor, cand: torch.Tensor, base: torch.Tensor,
                  anchors: torch.Tensor, tsv: torch.Tensor, tg: torch.Tensor, gate_bias: torch.Tensor) -> torch.Tensor:
    """Exact-sender scores [B, S, K(predecessor), K(candidate)] of the last write (dspine_mlx RefinementTail.greedy).

    q [B*S, R] (last write's gate input), rms [B*S] (stream RMS before it), cand / base [B, S, K] (head top-K ids and
    <raw * norm_w, head[c]>), anchors [B], tsv [V, 2R], tg [V, R]."""
    B, S, K = cand.shape
    R = tg.shape[1]
    sid = torch.cat((anchors[:, None, None].expand(-1, 1, K), cand[:, :-1]), dim=1)        # predecessors
    t = tsv[sid].float()                                                                  # [B, S, K, 2R]
    m = torch.sigmoid(q.float().view(B, S, 1, R) + t[..., :R] + gate_bias.float()) * t[..., R:]
    s = torch.einsum("bspr,bscr->bspc", m, tg[cand].float())
    return base.view(B, S, 1, K) + rms.view(B, S, 1, 1) * s
