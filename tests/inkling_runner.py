# SPDX-License-Identifier: Apache-2.0
"""Steps Inkling2LoopForCausalLM through vLLM's real KV planner, allocator and metadata builders, like the
scheduler would (decode requests first), on CPU or CUDA."""

from __future__ import annotations

import torch


class Runner:
    def __init__(self, vc, model, num_blocks: int, device="cpu"):
        from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
        from vllm.v1.attention.backends.utils import get_supported_kv_cache_layouts, resolve_kv_cache_layout
        from vllm.v1.core.kv_cache_planning import get_kv_cache_config_from_groups, get_kv_cache_groups
        from vllm.v1.kv_cache_interface import MambaSpec
        from vllm.v1.worker.utils import allocate_kv_cache, bind_kv_cache, customize_attention_spec

        self.vc, self.model, self.device = vc, model, torch.device(device)
        layers = {n: m for n, m in vc.compilation_config.static_forward_context.items()
                  if isinstance(m, AttentionLayerBase)}
        specs = {n: customize_attention_spec(m.get_attn_backend(), m.get_kv_cache_spec(vc)) for n, m in layers.items()}
        uniq = {m.get_attn_backend() for m in layers.values()}
        self.layout = resolve_kv_cache_layout(vc, [[x.name for x in get_supported_kv_cache_layouts([be])]
                                                   for be in uniq], specs.values())
        groups = get_kv_cache_groups(vc, dict(specs))
        self.group_info = [(type(g.kv_cache_spec).__name__, len(g.layer_names), g.kv_cache_spec.block_size)
                           for g in groups]
        backends = [layers[g.layer_names[0]].get_attn_backend() for g in groups]
        page = max(g.kv_cache_spec.page_size_bytes for g in groups) * max(len(g.layer_names) for g in groups)
        kc = get_kv_cache_config_from_groups(vc, groups, page * num_blocks)
        caches = allocate_kv_cache(kc, self.device, self.layout)
        self.caches = caches
        self.kv_bytes = sum(t.size for t in kc.kv_cache_tensors)
        bind_kv_cache(caches, vc.compilation_config.static_forward_context, [], kv_cache_groups=kc.kv_cache_groups)
        self.groups = []
        for g, be in zip(kc.kv_cache_groups, backends):
            spec = g.kv_cache_spec
            builder = be.get_builder_cls()(spec, g.layer_names, vc, self.device)
            self.groups.append((g.layer_names, spec, builder, isinstance(spec, MambaSpec)))
        self.free = list(range(1, kc.num_blocks))[::-1]
        self.tables: dict[tuple[int, int], list[int]] = {}

    def _table(self, gi, req, end, bs, mamba):
        t = self.tables.setdefault((gi, req), [])
        need = 1 if mamba else -(-end // bs)
        while len(t) < need:
            t.append(self.free.pop())
        return t

    def prepare(self, batch):
        """(input ids, positions, per-layer metadata, tokens) of one scheduler step; batch: list of (req, tokens,
        start, prefilling), decodes first."""
        from vllm.v1.attention.backend import CommonAttentionMetadata

        dev = self.device
        ids, pos, qsl, seq = [], [], [0], []
        for req, toks, start, _ in batch:
            ids.append(torch.as_tensor(toks, dtype=torch.long))
            pos.append(torch.arange(start, start + len(toks)))
            qsl.append(qsl[-1] + len(toks))
            seq.append(start + len(toks))
        T = qsl[-1]
        q_cpu = torch.tensor(qsl, dtype=torch.int32)
        s_cpu = torch.tensor(seq, dtype=torch.int32)
        attn_md = {}
        for gi, (names, spec, builder, mamba) in enumerate(self.groups):
            bs = spec.block_size
            tabs = [self._table(gi, req, start + len(toks), bs, mamba) for req, toks, start, _ in batch]
            width = max(len(t) for t in tabs)
            bt = torch.zeros(len(batch), width, dtype=torch.int32)
            slots = []
            for i, ((req, toks, start, _), t) in enumerate(zip(batch, tabs)):
                bt[i, :len(t)] = torch.tensor(t, dtype=torch.int32)
                p = torch.arange(start, start + len(toks))
                slots.append(torch.tensor(t)[p // bs] * bs + p % bs if not mamba else torch.full_like(p, -1))
            cm = CommonAttentionMetadata(
                query_start_loc=q_cpu.to(dev), query_start_loc_cpu=q_cpu, seq_lens=s_cpu.to(dev),
                num_reqs=len(batch), num_actual_tokens=T, max_query_len=max(len(t) for _, t, _, _ in batch),
                max_seq_len=max(seq), block_table_tensor=bt.to(dev), slot_mapping=torch.cat(slots).long().to(dev),
                is_prefilling=torch.tensor([b[3] for b in batch]), seq_lens_cpu_upper_bound=s_cpu.clone())
            md = builder.build(0, cm)
            for n in names:
                attn_md[n] = md
        return torch.cat(ids).to(dev), torch.cat(pos).to(dev), attn_md, T

    def hidden(self, prepared):
        """Final hidden states of a prepared step (CUDA-graph capturable)."""
        from vllm.forward_context import set_forward_context
        ids, pos, attn_md, T = prepared
        with set_forward_context(attn_md, self.vc, num_tokens=T):
            return self.model(ids, pos)

    def step(self, batch, logits_rows=None):
        """Float32 logits for every scheduled token of batch (or for the token indices in logits_rows)."""
        h = self.hidden(self.prepare(batch))
        if logits_rows is not None:
            h = h[logits_rows]
        return self.model.compute_logits(h).float()

    def release(self, req):
        for key in [k for k in self.tables if k[1] == req]:
            self.free.extend(self.tables.pop(key))
