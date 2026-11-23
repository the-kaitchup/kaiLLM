# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5 dense GGUF tensor mappings and reversible export transforms."""

from __future__ import annotations

from collections.abc import Iterable

import os
import sys

import gguf
import torch
from vllm.logger import init_logger

from ..qwen35_metadata import (
    QWEN35_GDN_NORM_DTYPE,
    QWEN35_GDN_NORM_IMPL,
    configure_qwen35_gguf_norm,
    qwen35_config_dict,
    read_gguf_metadata,
    resolve_gguf_shards,
)
from ..weight_utils import gguf_quant_weights_iterator_multi
from ..yaqa_gguf import read_yaqa_profile
from .base import BaseGGUFWeightsAdapter, GGUFLoadSpec

logger = init_logger(__name__)


def qwen35_name_map(num_layers: int) -> dict[str, str]:
    mapping = {
        "token_embd.weight": "model.embed_tokens.weight",
        "output_norm.weight": "model.norm.weight",
        "output.weight": "lm_head.weight",
    }
    suffixes = {
        "attn_norm.weight": "input_layernorm.weight",
        "post_attention_norm.weight": "post_attention_layernorm.weight",
        "ffn_gate.weight": "mlp.gate_proj.weight",
        "ffn_up.weight": "mlp.up_proj.weight",
        "ffn_down.weight": "mlp.down_proj.weight",
        "attn_q.weight": "self_attn.q_proj.weight",
        "attn_k.weight": "self_attn.k_proj.weight",
        "attn_v.weight": "self_attn.v_proj.weight",
        "attn_output.weight": "self_attn.o_proj.weight",
        "attn_q_norm.weight": "self_attn.q_norm.weight",
        "attn_k_norm.weight": "self_attn.k_norm.weight",
        "attn_qkv.weight": "linear_attn.in_proj_qkv.weight",
        "attn_gate.weight": "linear_attn.in_proj_z.weight",
        "ssm_alpha.weight": "linear_attn.in_proj_a.weight",
        "ssm_beta.weight": "linear_attn.in_proj_b.weight",
        "ssm_conv1d.weight": "linear_attn.conv1d.weight",
        "ssm_dt.bias": "linear_attn.dt_bias",
        "ssm_a": "linear_attn.A_log",
        "ssm_norm.weight": "linear_attn.norm.weight",
        "ssm_out.weight": "linear_attn.out_proj.weight",
    }
    for layer in range(num_layers):
        for original, target in suffixes.items():
            mapping[f"blk.{layer}.{original}"] = f"model.layers.{layer}.{target}"
    return mapping


MTP_NORMS = (
    "pre_fc_norm_embedding.weight",
    "pre_fc_norm_hidden.weight",
)


def qwen35_mtp_name_map(num_layers: int) -> dict[str, str]:
    """GGUF nextn/MTP layer (blk.<num_layers>) -> Qwen3_5MTP module names."""
    mtp = f"blk.{num_layers}."
    # The drafter's unquantized embed_tokens is replaced by the target's (vLLM
    # shares MTP embeddings and LM heads after loading), so it is not loaded.
    mapping = {
        "output.weight": "lm_head.weight",
        mtp + "nextn.eh_proj.weight": "model.fc.weight",
        mtp + "nextn.enorm.weight": "model.pre_fc_norm_embedding.weight",
        mtp + "nextn.hnorm.weight": "model.pre_fc_norm_hidden.weight",
        mtp + "nextn.shared_head_norm.weight": "model.norm.weight",
    }
    for original, target in qwen35_name_map(1).items():
        if original.startswith("blk.0.") and ".linear_attn." not in target:
            mapping[mtp + original.removeprefix("blk.0.")] = target
    return mapping


def qwen35_v_head_permutations(text_config) -> dict[str, torch.Tensor] | None:
    """Indices that undo llama.cpp's grouped->tiled linear-attention V-head order.

    When num_v_heads != num_k_heads, convert_hf_to_gguf stores the V heads of
    in_proj_qkv, in_proj_z, in_proj_a/b, A_log, dt_bias, the conv1d V channels
    and the out_proj input columns in tiled order (stored head j*K + k is group k,
    member j) so that ggml broadcasting pairs V head t with K head t % K. vLLM's
    GDN kernels pair V head g with K head g // (HV / K) (grouped order), so the
    loaded data must be stored[index] with these indices.
    """
    k = int(text_config.linear_num_key_heads)
    hv = int(text_config.linear_num_value_heads)
    if hv == k:
        return None
    if hv % k:
        raise ValueError("Qwen3.5 linear_num_value_heads must be a multiple of the key heads")
    per = hv // k
    grouped = torch.arange(hv, dtype=torch.long, device="cpu")
    heads = (grouped % per) * k + grouped // per
    dv = int(text_config.linear_value_head_dim)
    v = (heads[:, None] * dv + torch.arange(dv, dtype=torch.long, device="cpu")[None, :]).reshape(-1)
    qk = 2 * k * int(text_config.linear_key_head_dim)
    return {"heads": heads, "v": v, "qkv": torch.cat([torch.arange(qk, dtype=torch.long, device="cpu"), qk + v])}


def is_qwen35_mtp(config) -> bool:
    return "Qwen3_5MTP" in (getattr(config, "architectures", None) or [])


class Qwen35GGUFAdapter(BaseGGUFWeightsAdapter):
    @classmethod
    def matches(cls, config) -> bool:
        return config.get_text_config().model_type in ("qwen3_5_text", "qwen3_5_mtp")

    def prepare_model(self, model: torch.nn.Module) -> None:
        from vllm.model_executor.model_loader.reload import (
            record_metadata_for_reloading,
        )

        text_config = self.config.get_text_config()
        if not getattr(self, "is_mtp", False) and (
            getattr(text_config, "gguf_gdn_norm_dtype", None) != QWEN35_GDN_NORM_DTYPE
            or getattr(text_config, "gguf_gdn_norm_impl", None) != QWEN35_GDN_NORM_IMPL
        ):
            raise ValueError("GGUF GDN norm policy disagrees with its config markers")

        # GGUF stores these weights in F32, and gated RMS norm already computes
        # with F32 weights independently of its input/output dtype. Allocate F32
        # storage before loading so copying does not discard the original bits.
        self._parameter_dtypes = {}
        from vllm.model_executor.layers.layernorm import GemmaRMSNorm

        from ..quantization import kai_gemma_norm

        for module_name, module in model.named_modules():
            if (
                isinstance(module, GemmaRMSNorm)
                and module_name.endswith(
                    ("input_layernorm", "post_attention_layernorm", "model.norm",
                     "pre_fc_norm_embedding", "pre_fc_norm_hidden")
                )
                and kai_gemma_norm.bind(module)
            ):
                # GGUF stores the exact F32 multiplier (1 + w): keep w in F32.
                module.weight.data = module.weight.data.to(dtype=torch.float32)
                with module.weight.device:
                    record_metadata_for_reloading(module)
        import types as _types

        def _itq_mlp(mlp):
            # ((gate, up), down) when both projections are ITQ and the decode MLP can run in forward_mlp.
            if os.getenv("KAILLM_ITQ_FUSED_MLP", "1") != "1":
                return False
            gu, down = mlp.gate_up_proj, mlp.down_proj
            if getattr(gu, "_itq_shared_input", None) is not True or getattr(down, "_itq_shared_input", None) is None:
                return False
            if getattr(down, "_kai_input_gather", None) is not None:
                return False
            from ..quantization.itq import mlp_parts
            return mlp_parts(gu.yaqa_parts, down.yaqa_parts) or False

        def _itq_rows(x):
            from ..quantization.itq import max_gemv_rows
            return max_gemv_rows(x.device.index if x.device.index is not None else torch.cuda.current_device())

        def _mlp_forward(mlp, x):
            fused = mlp.__dict__.get("_itq_mlp")
            if fused is None:                       # decided on the first (profiling) forward, before graph capture
                fused = mlp.__dict__["_itq_mlp"] = _itq_mlp(mlp)
            if fused and x.is_cuda and 0 < x.numel() // x.shape[-1] <= _itq_rows(x):
                from ..quantization.itq import forward_mlp
                return forward_mlp(fused[0], fused[1], x, raw=mlp.__dict__.get("_itq_raw_out", False))
            gate_up, _ = mlp.gate_up_proj(x)
            method = getattr(mlp.down_proj, "quant_method", None)
            if hasattr(method, "apply_swiglu"):
                return method.apply_swiglu(mlp.down_proj, gate_up, mlp.act_fn)
            out, _ = mlp.down_proj(mlp.act_fn(gate_up))
            return out

        for module_name, module in model.named_modules():
            if (
                module_name.endswith(".mlp")
                and all(hasattr(module, a) for a in ("gate_up_proj", "down_proj", "act_fn"))
                and getattr(module, "expert_gate", None) is None
            ):
                module.forward = _types.MethodType(_mlp_forward, module)
            if module_name.endswith(("input_layernorm", "post_attention_layernorm")) and getattr(
                module, "_kai_original_forward", None
            ) is not None:
                parent = model.get_submodule(module_name.rsplit(".", 1)[0])
                if module_name.endswith("input_layernorm"):
                    owners = ((getattr(parent, "linear_attn", None), ("in_proj_qkvz", "in_proj_ba")),
                              (getattr(parent, "self_attn", None), ("qkv_proj",)))
                else:
                    owners = ((getattr(parent, "mlp", None), ("gate_up_proj",)),)
                consumers = [getattr(owner, name, None) for owner, names in owners if owner is not None
                             for name in names]
                consumers = [c for c in consumers if c is not None]
                if consumers:
                    # The norm writes its output's Q8_1 activation; these projections reuse it.
                    module.__dict__["_kai_np_consumers"] = True
                    for consumer in consumers:
                        consumer.__dict__["_kai_act_producer"] = id(module)
        self._wire_itq_residual_stream(model)
        self._wire_itq_gated_norm(model)
        self._wire_small_gguf_dense(model)
        for name, parameter in model.named_parameters():
            if name.endswith(".linear_attn.norm.weight"):
                parameter.data = parameter.data.to(dtype=torch.float32)
                norm = model.get_submodule(name.rsplit(".", 1)[0])
                if parameter.device.type == "cuda":
                    # Use vLLM's existing FLA CUDA norm explicitly for these
                    # GGUF layers. CPU dispatch and other norms are unchanged.
                    norm._forward_method = norm.forward_cuda
                # initialize_model captured the original model dtype. Refresh
                # only this norm, without recapturing processed quantized layers.
                with parameter.device:
                    record_metadata_for_reloading(norm)
            self._parameter_dtypes[name] = parameter.data.dtype

    @staticmethod
    def _wire_small_gguf_dense(model):
        """Multi-row calls (verification, batches) of the tiny GDN a/b projections (GGUF Q8_0, 5120 -> 2 x 48 in the
        ITQ files): one GEMM on a dense copy (BF16 1 MB or F32 2 MB per layer) instead of llama.cpp MMQ + stream-K
        fixup (2.9 ms per 16-row step over 48 layers).  1-4 rows (KAILLM_SMALL_GGUF_DENSE_MIN_ROWS) and prefill chunks
        (> KAILLM_SMALL_GGUF_DENSE_MAX_ROWS, 64) keep the GGUF kernels.  The GGUF parts exist only after loading, so the first multi-row call (the profiling
        run, before graph capture) builds the copy and checks it against the GGUF path; a mismatch keeps GGUF.
        KAILLM_SMALL_GGUF_DENSE=0 disables."""
        if os.getenv("KAILLM_SMALL_GGUF_DENSE", "1") != "1":
            return
        max_rows = int(os.getenv("KAILLM_SMALL_GGUF_DENSE_MAX_ROWS", "64"))
        # Up to 4 rows (MTP 3 verification) the GGUF kernels are cheaper than any cuBLAS shape for [rows, 5120] x
        # [5120, 96] (sm_120: BF16 4 rows 31 us, F32 8 us per call); BF16 rows are padded to multiples of 16, where
        # cuBLAS picks a tensor-core kernel (~7.5 us at 16 rows vs 2.9 ms per 48-layer step for MMQ + stream-K).
        min_rows = int(os.getenv("KAILLM_SMALL_GGUF_DENSE_MIN_ROWS", "5"))

        def matrices(layer):
            """(wire bytes, GGML type, rows, columns) per output block, in output order, or None."""
            parts = list(getattr(layer, "gguf_compact", []) or [])
            if parts:
                if not all(p.is_quantized for p in parts):
                    return None
                return [(p.weight, int(p.weight_type), int(p.shape[0]), int(p.shape[1])) for p in parts]
            qweight, qtype = getattr(layer, "qweight", None), getattr(layer, "qweight_type", None)
            if qweight is None or qtype is None or not hasattr(qtype, "weight_type"):
                return None
            k = int(layer.input_size_per_partition)
            fallback = int(qtype.weight_type)
            shards = list(getattr(qweight, "shard_id", None) or [])
            if not shards:
                return [(qweight.data, fallback, int(qweight.shape[0]), k)]
            offsets = qweight.shard_offset_map
            types = getattr(qtype, "shard_weight_type", {}) or {}
            return [(qweight.data[offsets[i][0]:offsets[i][1], : offsets[i][2]], int(types.get(i, fallback)),
                     offsets[i][1] - offsets[i][0], k) for i in shards]

        def dequant(data, t, n, k):
            if t == 8:                                # Q8_0 blocks: fp16 scale + 32 int8
                b = data.contiguous().reshape(n, k // 32, 34)
                return (b[..., :2].contiguous().view(torch.float16).float()
                        * b[..., 2:].contiguous().view(torch.int8).float()).reshape(n, k)
            if t in (0, 1):                           # F32 / F16 stored densely
                return data.reshape(n, k).float()
            from vllm import _custom_ops as ops
            return ops.ggml_dequantize(data.contiguous(), t, n, k, torch.float32)

        _dense_reported = []

        def probe(original, k, like, chunk=1024):
            """The layer's effective matrix from its own forward on 127 x identity rows: a one-hot row quantizes to Q8_1
            exactly at scale 1.0 (127 / 127), WNA16 / dense kernels are linear; outputs come back in the activation dtype
            (bf16: a 2^-9 relative rounding of each weight, far below the GGUF path's own activation quantization)."""
            cols = []
            for start in range(0, k, chunk):
                eye = torch.zeros(min(chunk, k - start), k, dtype=like.dtype, device=like.device)
                eye[torch.arange(eye.shape[0], device=like.device), torch.arange(start, start + eye.shape[0],
                                                                                 device=like.device)] = 127.0
                out, _ = original(eye)
                cols.append(out.float() / 127.0)
            return torch.cat(cols).t().contiguous()       # [N, K]

        def build(layer, x2, original):
            mats = matrices(layer)
            how = "dequantized GGUF"
            if mats and sum(m[2] for m in mats) <= 512:
                w = torch.cat([dequant(*m) for m in mats])
            elif mats is None and int(getattr(layer, "output_size_per_partition", 1 << 20)) <= 512:
                w, how = probe(original, x2.shape[-1], x2), "probed (identity rows)"
            else:
                return False
            ref, _ = original(x2)
            scale = ref.float().norm().clamp_min(1e-12)
            err = ((x2.float() @ w.t() - ref.float()).norm() / scale).item()
            # BF16 copy (one tensor-core GEMM, no conversions; weights rounded to 2^-9, about the GGUF path's own
            # Q8_1 activation rounding) when it stays as close to GGUF as the F32 copy.
            wb = w.to(torch.bfloat16)
            err_b = ((x2.to(torch.bfloat16) @ wb.t()).float().sub(ref.float()).norm() / scale).item()
            use_b = os.getenv("KAILLM_SMALL_GGUF_DENSE_BF16", "1") == "1" and err_b < max(1.5 * err, 5e-3)
            if not _dense_reported:
                _dense_reported.append(err)
                print(f"kaiLLM: GDN a/b projections {tuple(w.shape)} ({how}) use a dense {'BF16' if use_b else 'F32'} copy "
                      f"for 2-{max_rows} rows (relative difference vs GGUF: F32 {err:.1e}, BF16 {err_b:.1e}; "
                      f"{x2.shape[0]} rows)" + ("" if err < 0.02 else ": REJECTED"), file=sys.stderr, flush=True)
            layer.__dict__["_kai_dense_dtype"] = ref.dtype
            return (wb if use_b else w) if err < 0.02 else False

        def wrap(layer):
            original = layer.forward

            def forward(x, *args, _original=original, _layer=layer, **kwargs):
                rows = x.numel() // x.shape[-1]
                state = _layer.__dict__.get("_kai_dense")
                if rows <= 1 or state is False or args or kwargs:
                    return _original(x, *args, **kwargs)
                x2 = x.reshape(-1, x.shape[-1])
                if state is None:
                    if torch.cuda.is_current_stream_capturing():
                        return _original(x)
                    try:
                        state = build(_layer, x2, _original)
                    except Exception as e:  # noqa: BLE001
                        print(f"kaiLLM: dense GDN a/b projection not used: {e!r}", file=sys.stderr, flush=True)
                        state = False
                    _layer.__dict__["_kai_dense"] = state
                if state is False or rows > max_rows or rows < min_rows:
                    return _original(x)
                if state.dtype == torch.bfloat16:
                    xb = x2.to(torch.bfloat16)
                    if rows % 16:
                        xb = torch.nn.functional.pad(xb, (0, 0, 0, 16 - rows % 16))
                    out = (xb @ state.t())[:rows].to(_layer.__dict__.get("_kai_dense_dtype", x.dtype))
                else:
                    out = (x2.float() @ state.t()).to(_layer.__dict__.get("_kai_dense_dtype", x.dtype))
                return out.reshape(*x.shape[:-1], out.shape[-1]), None

            layer.forward = forward
            layer.__dict__["_kai_dense"] = None

        wrapped = 0
        for _, module in model.named_modules():
            ba = getattr(module, "in_proj_ba", None)
            if ba is not None and hasattr(module, "in_proj_qkvz") and "_kai_dense" not in ba.__dict__:
                wrap(ba)
                wrapped += 1
        print(f"kaiLLM: dense GDN a/b path wired on {wrapped} layers", file=sys.stderr, flush=True)

    @staticmethod
    def _wire_itq_gated_norm(model):
        """Decode GDN output projection (KAILLM_ITQ_FUSED_GATED_NORM=0 disables): RMSNormGated(core, z) and out_proj's
        int8 prep run as one itq_gated_norm_prep launch when out_proj is a single all-ITQ part (decided on the first
        call, before graph capture); out_proj then runs only its GEMV and epilogue (raw output included)."""
        if os.getenv("KAILLM_ITQ_FUSED_GATED_NORM", "1") != "1":
            return
        import types as _types

        def fused(self, core, z, _original):
            ok = self.__dict__.get("_itq_gated")
            if ok is None:
                from ..quantization.itq import gated_norm_supported
                ok = self.__dict__["_itq_gated"] = gated_norm_supported(self.out_proj, self.norm)
            if ok and core.is_cuda and core.dtype == torch.bfloat16 and z.dtype == torch.bfloat16:
                from ..quantization.itq import gated_norm_prep, max_gemv_rows
                rows = core.shape[0]
                index = core.device.index if core.device.index is not None else torch.cuda.current_device()
                if 0 < rows <= max_gemv_rows(index):
                    output, _ = self.out_proj(gated_norm_prep(self.out_proj, core, z, self.norm.weight, self.norm.eps))
                    return output
            return _original(core, z)

        for _, module in model.named_modules():
            if all(hasattr(module, a) for a in ("_output_projection", "norm", "out_proj", "in_proj_qkvz")):
                original = module._output_projection
                module._output_projection = _types.MethodType(
                    lambda self, core, z, _o=original: fused(self, core, z, _o), module)

    @staticmethod
    def _wire_itq_residual_stream(model):
        """Decode residual-stream fusion (KAILLM_ITQ_FUSED_NORM=0 disables): an all-ITQ single-matrix producer
        (o_proj / out_proj, down_proj) whose output goes straight into a bound Gemma norm returns its raw F32 output;
        the norm consumes it with itq.add_norm and prepares the next all-ITQ projection group (qkv / qkvz, gate_up)."""
        if os.getenv("KAILLM_ITQ_FUSED_NORM", "1") != "1":
            return
        from ..quantization import kai_gemma_norm

        if not kai_gemma_norm.f32_residual():
            return

        def single_itq(layer):
            parts = getattr(layer, "yaqa_parts", None)
            return (layer is not None and parts is not None and len(parts) == 1
                    and getattr(layer, "_itq_shared_input", None) is not None)

        def itq_group(layer):
            return layer if layer is not None and getattr(layer, "_itq_shared_input", None) is True else None

        def bound(norm):
            return norm is not None and getattr(norm, "_kai_original_forward", None) is not None

        for name, module in model.named_modules():
            if not isinstance(module, torch.nn.ModuleList) or not len(module):
                continue
            layers = list(module)
            if not all(all(hasattr(l, a) for a in ("input_layernorm", "post_attention_layernorm", "mlp"))
                       for l in layers):
                continue
            parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
            final_norm = getattr(parent, "norm", None)
            for i, layer in enumerate(layers):
                if getattr(layer, "layer_scale", False):
                    continue
                attn = getattr(layer, "linear_attn", None)
                if attn is None:
                    attn = getattr(layer, "self_attn", None)
                if attn is None:
                    continue
                out = getattr(attn, "out_proj", None)
                if out is None:
                    out = getattr(attn, "o_proj", None)
                nxt = layers[i + 1].input_layernorm if i + 1 < len(layers) else final_norm
                mlp = layer.mlp
                down = getattr(mlp, "down_proj", None)
                if single_itq(out) and bound(layer.post_attention_layernorm):
                    out.__dict__["_itq_raw_out"] = True
                if single_itq(down) and bound(nxt):
                    down.__dict__["_itq_raw_out"] = True
                    mlp.__dict__["_itq_raw_out"] = True
                qkv = getattr(attn, "in_proj_qkvz", None)
                if qkv is None:
                    qkv = getattr(attn, "qkv_proj", None)
                layer.input_layernorm.__dict__["_itq_consumer"] = itq_group(qkv)
                layer.post_attention_layernorm.__dict__["_itq_consumer"] = itq_group(getattr(mlp, "gate_up_proj", None))

    def prepare_loading(self, model_path, model_config) -> GGUFLoadSpec:
        # Direct loader callers may bypass EngineArgs and the GGUF config
        # parser. This still runs before initialize_model creates AOT wrappers.
        self.is_mtp = is_qwen35_mtp(model_config.hf_config)
        configure_qwen35_gguf_norm(model_config.hf_config)
        paths = resolve_gguf_shards(model_path)
        readers = [gguf.GGUFReader(str(path)) for path in paths]
        tensors = {}
        for reader in readers:
            for tensor in reader.tensors:
                if tensor.name in tensors:
                    raise ValueError(f"Duplicate GGUF tensor: {tensor.name}")
                tensors[tensor.name] = tensor
        # yaqa.manifest logical matrices (e.g. ITQ) are absent from the tensor list; their components are
        # ordinary tensors that no module maps.  ITQ matrices are stored in vLLM (grouped V-head) order.
        logical, components = {}, set()
        for reader in readers:
            profile = read_yaqa_profile(reader)
            descriptors = {} if profile is None else profile["tensors"]
            if logical.keys() & descriptors.keys():
                raise ValueError("Duplicate logical YAQA weights across GGUF shards")
            logical.update(descriptors)
            for descriptor in descriptors.values():
                components.update(descriptor["components"].values())
        names = (set(tensors) - components) | set(logical)
        metadata = read_gguf_metadata(readers[0])
        config_dict = qwen35_config_dict(metadata, names)
        text_config = model_config.hf_config.get_text_config()
        for key in ("num_hidden_layers", "hidden_size", "vocab_size", "layer_types"):
            if getattr(text_config, key) != config_dict[key]:
                raise ValueError(f"HF config {key} disagrees with GGUF metadata")
        text_config.tie_word_embeddings = config_dict["tie_word_embeddings"]
        if self.is_mtp:
            mapping = qwen35_mtp_name_map(text_config.num_hidden_layers)
            missing = sorted(
                name for name in mapping
                if name not in names
                and not (name == "output.weight" and text_config.tie_word_embeddings)
            )
            if missing:
                raise ValueError(f"Missing Qwen3.5 MTP GGUF tensors: {missing}")
            mapping = {name: target for name, target in mapping.items() if name in names}
            unquantized = [
                mapping[name].removesuffix(".weight")
                for name, tensor in tensors.items()
                if name in mapping
                and name.endswith(".weight")
                and tensor.tensor_type.name in ("F32", "F16", "BF16")
            ]
            self.load_spec = GGUFLoadSpec(
                weights_source=[str(path) for path in paths],
                unquantized_modules=unquantized,
                gguf_to_hf_name_map=mapping,
                yaqa_subset=True,
            )
            return self.load_spec
        mapping = qwen35_name_map(text_config.num_hidden_layers)
        expected = {
            "token_embd.weight",
            "output_norm.weight",
        }
        for name in mapping:
            if not name.startswith("blk."):
                continue
            layer = int(name.split(".")[1])
            is_linear = text_config.layer_types[layer] == "linear_attention"
            target = mapping[name]
            if (".linear_attn." in target and not is_linear) or (
                ".self_attn." in target and is_linear
            ):
                continue
            expected.add(name)
        if not text_config.tie_word_embeddings:
            expected.add("output.weight")
        missing = sorted(expected - names)
        if missing:
            raise ValueError(f"Missing Qwen3.5 GGUF tensors: {missing}")
        main_names = {
            name
            for name in names
            if not name.startswith("blk.")
            or int(name.split(".")[1]) < text_config.num_hidden_layers
        }
        unexpected = sorted(main_names - mapping.keys())
        if unexpected:
            raise ValueError(f"Unmapped Qwen3.5 GGUF tensors: {unexpected}")
        unquantized = [
            mapping[name].removesuffix(".weight")
            for name, tensor in tensors.items()
            if name in mapping
            and name.endswith(".weight")
            and tensor.tensor_type.name in ("F32", "F16", "BF16")
        ]
        rows, columns = {}, {}
        self._row_permutations, self._column_permutations = {}, {}
        perms = qwen35_v_head_permutations(text_config)
        if perms is not None:
            if any(key.startswith("prism.hadamard") for key in readers[0].fields):
                raise ValueError("Hadamard-rotated Qwen3.5 GGUFs are not supported")
            for layer, kind in enumerate(text_config.layer_types):
                if kind != "linear_attention":
                    continue
                blk = f"blk.{layer}."
                for target, name, index in (
                    (rows, "attn_qkv.weight", perms["qkv"]), (rows, "attn_gate.weight", perms["v"]),
                    (rows, "ssm_alpha.weight", perms["heads"]), (rows, "ssm_beta.weight", perms["heads"]),
                    (rows, "ssm_conv1d.weight", perms["qkv"]), (rows, "ssm_dt.bias", perms["heads"]),
                    (rows, "ssm_a", perms["heads"]), (columns, "ssm_out.weight", perms["v"]),
                ):
                    if blk + name not in logical:      # logical matrices are already in vLLM order
                        target[blk + name] = index
            # Tensors that reach prepare_weights (F32 and non-compact matrices).
            self._row_permutations = {mapping[name]: index for name, index in rows.items()}
            self._column_permutations = {mapping[name]: index for name, index in columns.items()}
        self.load_spec = GGUFLoadSpec(
            weights_source=[str(path) for path in paths],
            unquantized_modules=unquantized,
            gguf_to_hf_name_map=mapping,
            gguf_row_permutations=rows or None,
            gguf_column_permutations=columns or None,
        )
        return self.load_spec

    def prepare_weights(self, model_config) -> Iterable[tuple[str, torch.Tensor]]:
        del model_config
        for name, weight in gguf_quant_weights_iterator_multi(
            self.load_spec.weights_source, self.load_spec.gguf_to_hf_name_map
        ):
            is_type = name.endswith(".qweight_type")
            base = name.removesuffix(".qweight") + ".weight" if name.endswith(".qweight") else name
            row_index = None if is_type else getattr(self, "_row_permutations", {}).get(base)
            column_index = None if is_type else getattr(self, "_column_permutations", {}).get(base)
            if name.endswith(".qweight") and (row_index is not None or column_index is not None):
                # Compact GGUF layers apply these permutations at construction.
                raise ValueError(f"V-head reordered Qwen3.5 GGUF matrices need the compact backend: {name}")
            if row_index is not None:
                weight = weight[row_index]
            if column_index is not None:
                weight = weight[:, column_index]
            if name.endswith(".A_log"):
                if not torch.all(torch.isfinite(weight) & (weight < 0)):
                    raise ValueError(f"Expected finite negative GGUF SSM A: {name}")
                weight = torch.log(-weight.float())
            elif name.endswith(".conv1d.weight"):
                weight = weight.unsqueeze(1)
            elif name == "model.norm.weight" or name.endswith(
                (
                    ".input_layernorm.weight",
                    ".post_attention_layernorm.weight",
                    ".self_attn.q_norm.weight",
                    ".self_attn.k_norm.weight",
                )
                + MTP_NORMS
            ):
                weight = weight.float() - 1.0
            # Match the copy into known floating destinations, after all F32
            # export transforms. Fused shards without an exact parameter name
            # retain their source dtype for their specialized weight loader.
            target_dtype = getattr(self, "_parameter_dtypes", {}).get(name)
            if (
                weight.is_floating_point()
                and target_dtype is not None
                and target_dtype.is_floating_point
            ):
                weight = weight.to(dtype=target_dtype)
            if getattr(self, "is_mtp", False) and name.startswith("model.") and "embed_tokens" not in name:
                # Qwen3_5MTP.load_weights accepts mtp.* names and maps them to model.*.
                name = "mtp." + name.removeprefix("model.")
            yield name, weight
