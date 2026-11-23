# SPDX-License-Identifier: Apache-2.0
"""ITQ vocabulary codec (yaqa.manifest codec "itqvocab", version 1) for a tied embedding / LM head.

Rows are stored in the rotated basis r = R (e * su) (R = orthonormal block Hadamard) and split into rate
classes: int8 rows, NF4 nibbles with 16 levels, and ITQ trellis rows (tile16 layout).  The embedding
decodes the requested rows and rotates them back.  The head rotates the hidden state once (itq_prep, int8
per 32 columns, the same preparation as the ITQ linears) and runs one integer GEMV per class, writing
each class's logits into its token columns.  Nothing is dequantized ahead of time: VRAM holds the codes.
"""

from __future__ import annotations

import ctypes
import os

import torch
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase

from ..yaqa_gguf import load_yaqa_components
from .itq import (MAX_DECODE_ROWS, _check, block_size, grid_blocks, ksplit_for, launch_params, library,
                  max_gemv_rows, pad_rows)

KINDS = {"int8": 0, "nf4": 1, "itq": 2}
VOCAB_MMA_ROWS = 32     # tensor-core head (itq_vocab_mma_kernel): up to 32 activation rows per call


def vocab_chunk_rows(device_index: int) -> int:
    """Activation rows per head call: 32 with the tensor-core kernels (sm_80+, ITQ_MMA / ITQ_VOCAB_MMA != 0), else 16."""
    if max_gemv_rows(device_index) > MAX_DECODE_ROWS and os.getenv("ITQ_VOCAB_MMA", "1") != "0":
        return VOCAB_MMA_ROWS
    return MAX_DECODE_ROWS


def _bind(lib):
    if getattr(lib, "_itq_vocab_bound", False):
        return lib
    p, i = ctypes.c_void_p, ctypes.c_int
    lib.itq_vocab_embed.argtypes = [p, i, p, i, p, p, p, p, p, p, p, p, p, i, i, i, i, p]
    lib.itq_vocab_gemv.argtypes = [p, p, p, p, p, p, p, i, i, i, i, i, p]
    lib.itq_vocab_scatter.argtypes = [p, p, p, p, i, i, i, i, p]
    for fn in (lib.itq_vocab_embed, lib.itq_vocab_gemv, lib.itq_vocab_scatter):
        fn.restype = ctypes.c_int
    lib._itq_vocab_bound = True
    return lib


class ITQVocab(torch.nn.Module):
    def __init__(self, descriptor, components, params_dtype, *, device, host=False):
        """host=True (an untied embedding, KAILLM_HOST_EMBEDDING=1): the class row data stays in pinned host memory
        and the embedding kernel reads the requested rows over PCIe (host_tables); everything else is on the GPU."""
        super().__init__()
        self.host = bool(host)
        self.vocab, self.dim = descriptor["shape"]
        rotation = descriptor["rotation"]
        self.block = block_size(rotation, self.dim)
        self.out_dtype = params_dtype
        self.register_buffer("su", components["su"].to(torch.int8).to(device), persistent=False)
        self.register_buffer("map", components["map"].to(torch.int32).to(device), persistent=False)
        table = None
        if "table" in components:
            t = components["table"].to(torch.int32) & 0xFF
            table = (t[:, 0] | (t[:, 1] << 8) | (t[:, 2] << 16) | (t[:, 3] << 24)).to(torch.int32).contiguous()
        self.register_buffer("table", None if table is None else table.to(device), persistent=False)
        self.classes = []
        for k, item in enumerate(descriptor["classes"]):
            kind = item["kind"]
            rows = int(item["rows"])
            entry = {"kind": KINDS[kind], "rows": rows, "bps": int(item.get("bps", 0)), "wscale": 1.0}
            prefix = f"c{k}_"
            ids = components[f"c{k}.rows"].to(torch.int32)
            scale = components[f"c{k}.scale"].float()
            self.register_buffer(prefix + "rows", ids.to(device), persistent=False)
            self.register_buffer(prefix + "scale", scale.to(device), persistent=False)
            levels = None
            head_scale = scale
            lv8 = None
            if kind == "int8":
                data = components[f"c{k}.q"].to(torch.int8).contiguous()
            elif kind == "nf4":
                data = components[f"c{k}.codes"].to(torch.int8).contiguous()
                levels = components[f"c{k}.levels"].float()
                peak = float(levels.abs().max())
                lv8 = torch.round(levels / peak * 127).clamp(-127, 127).to(torch.int32)
                head_scale = scale * (peak / 127.0)    # head: int8 levels; embedding: float levels
            else:
                data = components[f"c{k}.stream"].to(torch.int8).contiguous()
                entry["wscale"] = float(components[f"c{k}.wscale"].reshape(-1)[0])
                entry["padded"] = (rows + 15) // 16 * 16
            if self.host:
                from ..host_tables import host_alias

                self.register_buffer(prefix + "data", host_alias(data), persistent=False)
            else:
                self.register_buffer(prefix + "data", data.to(device), persistent=False)
            self.register_buffer(prefix + "levels", None if levels is None else levels.to(device), persistent=False)
            self.register_buffer(prefix + "head_scale", head_scale.to(device), persistent=False)
            self.register_buffer(prefix + "lv8", None if lv8 is None else lv8.to(device), persistent=False)
            self.classes.append(entry)
        self._lib = None
        self._args = None

    def _buf(self, k, name):
        return getattr(self, f"c{k}_{name}")

    def _setup(self, device):
        if self._lib is not None:
            return
        self._lib = _bind(library(device))
        self._sms = torch.cuda.get_device_properties(device).multi_processor_count
        self._major = torch.cuda.get_device_capability(device)[0]
        n = len(self.classes)
        vp, cf, ci = ctypes.c_void_p, ctypes.c_float, ctypes.c_int
        levels = [self._buf(k, "levels") for k in range(n)]
        self._args = (
            n,
            (vp * n)(*[self._buf(k, "data").data_ptr() for k in range(n)]),
            (vp * n)(*[self._buf(k, "scale").data_ptr() for k in range(n)]),
            (vp * n)(*[0 if lv is None else lv.data_ptr() for lv in levels]),
            (cf * n)(*[c["wscale"] for c in self.classes]),
            (ci * n)(*[c["kind"] for c in self.classes]),
            (ci * n)(*[c["bps"] for c in self.classes]),
        )
        for c in self.classes:
            if c["kind"] == 2:
                c["ksplit"] = ksplit_for(c["padded"], self.dim, self._sms)
        self._ybuf_rows = max([c.get("padded", 0) for c in self.classes] + [0])

    def embedding(self, ids):
        shape = ids.shape
        flat = ids.reshape(-1)
        if flat.dtype not in (torch.int32, torch.int64):
            flat = flat.long()
        flat = flat.contiguous()
        self._setup(flat.device)
        out = torch.empty((flat.numel(), self.dim), dtype=torch.bfloat16, device=flat.device)
        if flat.numel():
            stream = torch.cuda.current_stream(flat.device).cuda_stream
            n, data, scales, levels, wscales, kinds, bpss = self._args
            _check(self._lib.itq_vocab_embed(
                flat.data_ptr(), 1 if flat.dtype == torch.int64 else 0, self.map.data_ptr(), n, data, scales, levels,
                wscales, kinds, bpss, None if self.table is None else self.table.data_ptr(), self.su.data_ptr(),
                out.data_ptr(), flat.numel(), self.dim, self.block, self.vocab, stream), "vocab_embed")
        out = out.reshape(*shape, self.dim)
        return out if self.out_dtype == torch.bfloat16 else out.to(self.out_dtype)

    def logits(self, x):
        shape = x.shape
        x2 = x.reshape(-1, self.dim)
        if x2.dtype not in (torch.bfloat16, torch.float32):
            x2 = x2.float()
        if not x2.is_contiguous():
            x2 = x2.contiguous()
        self._setup(x2.device)
        rows = x2.shape[0]
        out = torch.empty((rows, self.vocab), dtype=torch.float32, device=x2.device)
        if rows == 0:
            return out.reshape(*shape[:-1], self.vocab)
        lib, stream = self._lib, torch.cuda.current_stream(x2.device).cuda_stream
        bf16 = 1 if x2.dtype == torch.bfloat16 else 0
        chunk = vocab_chunk_rows(x2.device.index if x2.device.index is not None else torch.cuda.current_device())
        nb_max = pad_rows(min(chunk, rows))
        xq = torch.empty((nb_max, self.dim // 4), dtype=torch.int32, device=x2.device)
        xs = torch.empty((nb_max, self.dim // 32), dtype=torch.float32, device=x2.device)
        ybuf = torch.empty((nb_max * self._ybuf_rows,), dtype=torch.float32, device=x2.device) if self._ybuf_rows else None
        for start in range(0, rows, chunk):
            r = min(chunk, rows - start)
            xc = x2[start:start + r]
            logits = out[start:start + r]
            _check(lib.itq_prep(xc.data_ptr(), bf16, None, self.su.data_ptr(), xq.data_ptr(), xs.data_ptr(), None, r,
                                self.dim, self.dim, self.block, 1, None, 0, stream), "vocab_prep")
            for k, c in enumerate(self.classes):
                if c["kind"] in (0, 1):
                    lv8 = self._buf(k, "lv8")
                    _check(lib.itq_vocab_gemv(self._buf(k, "data").data_ptr(), None if lv8 is None else lv8.data_ptr(),
                                              self._buf(k, "rows").data_ptr(), self._buf(k, "head_scale").data_ptr(),
                                              xq.data_ptr(), xs.data_ptr(), logits.data_ptr(), c["rows"], self.dim,
                                              self.vocab, r, 1 if c["kind"] == 1 else 0, stream), "vocab_gemv")
                else:
                    m = c["padded"]
                    y = ybuf[: nb_max * m]
                    _check(lib.itq_gemv(self._buf(k, "data").data_ptr(), self.table.data_ptr(), xq.data_ptr(),
                                        xs.data_ptr(), y.data_ptr(), m, self.dim, c["bps"], 12, r, c["ksplit"],
                                        c["wscale"], grid_blocks(lib, self._sms, r, c["bps"], self.dim, self._major,
                                                                 (m // 16) * c["ksplit"]),
                                        launch_params(r, c["bps"], self.dim, self._major)[0], stream), "vocab_itq_gemv")
                    _check(lib.itq_vocab_scatter(y.data_ptr(), self._buf(k, "rows").data_ptr(),
                                                 self._buf(k, "scale").data_ptr(), logits.data_ptr(), c["rows"], m,
                                                 self.vocab, r, stream), "vocab_scatter")
        return out.reshape(*shape[:-1], self.vocab)


class ITQVocabMethod(QuantizeMethodBase):
    """Embedding + tied head of a VocabParallelEmbedding whose GGUF weight is an "itqvocab" manifest entry."""

    def __init__(self, quant_config, part, host=False):
        self.quant_config = quant_config
        self.part = part
        self.host = host

    def create_weights(self, layer, input_size_per_partition, output_partition_sizes, input_size, output_size,
                       params_dtype, **extra_weight_attrs):
        del input_size, output_size, extra_weight_attrs
        if getattr(layer, "tp_size", 1) != 1:
            raise ValueError("The ITQ vocabulary currently requires tensor_parallel_size=1")
        name, (path, tensor_name, descriptor) = self.part
        reader = self.quant_config.yaqa_readers[path]
        module = ITQVocab(descriptor, load_yaqa_components(reader, descriptor), params_dtype,
                          device=torch.empty(0).device, host=self.host)
        if module.dim != input_size_per_partition or module.vocab > sum(output_partition_sizes):
            raise ValueError(f"ITQ vocabulary shape mismatch for {name}")
        if self.host:
            import sys

            mib = sum(module._buf(k, "data").numel() for k in range(len(module.classes))) / 2**20
            print(f"kaiLLM: {name} vocabulary rows in pinned host memory ({mib:.1f} MiB, zero-copy gathers)",
                  file=sys.stderr, flush=True)
        layer.itq_vocab = module
        self.quant_config.yaqa_consumed_names.add(name)
        self.quant_config.yaqa_consumed_tensors.add(tensor_name)

    def process_weights_after_loading(self, layer):
        del layer

    def embedding(self, layer, x):
        return layer.itq_vocab.embedding(x)

    def apply(self, layer, x, bias=None):
        logits = layer.itq_vocab.logits(x)
        return logits if bias is None else logits + bias

    def tie_weights(self, layer, embed_tokens):
        del layer
        if getattr(getattr(embed_tokens, "itq_vocab", None), "host", False):
            raise RuntimeError("KAILLM_HOST_EMBEDDING=1: a tied vocabulary is the LM head too; keep it in VRAM")
        embed_tokens._gguf_tied_lm_head = True
        return embed_tokens
