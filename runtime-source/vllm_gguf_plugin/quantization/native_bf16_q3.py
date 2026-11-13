# SPDX-License-Identifier: Apache-2.0
"""Default-off BF16 input adapter for a source-bound native Q3_K head.

BF16 inputs cast exactly to F32. The unchanged guarded native backend operates
on original GGUF wire bytes and returns F32 logits. Its decoded-weight/Q8 RHS
arithmetic differs from canonical BF16 compact multiplication; whole-model
quality remains required. This module does not change embeddings or bodies.
"""

import hashlib
import importlib
import os
from pathlib import Path

import torch

ENV = "VLLM_GGUF_NATIVE_BF16_Q3_HEAD"
MODES = ("none", "sm86", "sm120")
SELECTED_ROWS = (1, 4)
ROWS_ENV = "VLLM_GGUF_NATIVE_BF16_Q3_HEAD_ROWS"
ROW_MODES = ("legacy", "m8", "generic")
TARGETS = {"sm86": (8, 6), "sm120": (12, 0)}
_WARMED = set()


def mode():
    value = os.getenv(ENV, "none")
    if value not in MODES:
        raise ValueError(f"{ENV} must be one of {MODES}")
    if value != "none" and os.getenv("VLLM_GGUF_LFM3_HEAD", "compact") not in (
        "compact",
        "auto-native",
    ):
        raise ValueError(
            "Explicit BF16 Q3 adapter requires LFM3_HEAD=compact or auto-native"
        )
    return value


def rows_mode():
    value = os.getenv(ROWS_ENV, "legacy")
    if value not in ROW_MODES:
        raise ValueError(f"{ROWS_ENV} must be one of {ROW_MODES}")
    if value == "m8" and mode() != "sm86":
        raise ValueError("Explicit Q3 row8 admission currently requires SM86 mode")
    return value


def approved_native_rows(selected):
    """Backend ABI family; SM120 retains its independently proved BF16 family."""
    available = tuple(sorted(set(int(v) for v in backend_for(selected).SELECTED_ROWS)))
    if selected == "sm120":
        available = tuple(v for v in available if v in (1, 4))
    if not available or available[0] != 1 or any(v < 1 for v in available):
        raise ValueError("Native Q3 family must include a positive single-row tile")
    return available


def selected_rows():
    requested = rows_mode()
    if requested == "generic" and mode() != "none":
        return approved_native_rows(mode())
    return (1, 4, 8) if requested == "m8" else SELECTED_ROWS


def row_tiles(rows, native_rows):
    """Greedy positive rows over the existing ABI; no checkpoint/batch table."""
    if rows <= 0 or not native_rows or min(native_rows) != 1:
        raise ValueError("Positive rows and a backend single-row tile are required")
    remaining = rows
    for tile in sorted(native_rows, reverse=True):
        while remaining >= tile:
            yield tile
            remaining -= tile
    if remaining:
        raise ValueError("Native row family did not cover the requested rows")


def backend_for(selected):
    if selected not in TARGETS:
        raise ValueError("A target-specific BF16 Q3 mode is required")
    suffix = "native_q3_head" if selected == "sm86" else "native_q3_head_sm120"
    return importlib.import_module(f"{__package__}.{suffix}")


def _run_tiled_guard(x, packed, qtype, k, library, selected):
    """Sanitize once and preserve the original whole-input repair reduction."""
    import triton

    backend = backend_for(selected)
    clean, unsafe = backend.sanitize(x)
    rows, n = x.shape[0], packed.shape[0]
    direct = getattr(backend, "direct_rows_supported", None)
    if direct is not None and direct(library, x.device.index, rows):
        output = backend.native_raw(clean, packed, qtype, k, library)
    else:
        output = x.new_empty((rows, n))
        offset = 0
        for tile in row_tiles(rows, approved_native_rows(selected)):
            output[offset : offset + tile].copy_(
                backend.native_raw(clean[offset : offset + tile], packed, qtype, k, library)
            )
            offset += tile
    if rows <= 8:
        backend._repair_gemv[(triton.cdiv(n, 4),)](
            x,
            packed,
            output,
            unsafe,
            M=rows,
            N=n,
            K=k,
            QTYPE=qtype,
            BLOCK_M=triton.next_power_of_2(rows),
            BLOCK_N=4,
            BLOCK_K=256,
            num_warps=4,
            enable_fp_fusion=False,
        )
    else:
        backend._repair_gemm[(triton.cdiv(rows, 16), triton.cdiv(n, 32))](
            x,
            packed,
            output,
            unsafe,
            M=rows,
            N=n,
            K=k,
            QTYPE=qtype,
            BLOCK_M=16,
            BLOCK_N=32,
            BLOCK_K=128,
            num_warps=4,
            num_stages=2,
            enable_fp_fusion=False,
        )
    return output


@torch.library.custom_op("gguf_yaqa::native_q3_head_tiled", mutates_args=())
def guarded_tiled(
    x: torch.Tensor,
    packed: torch.Tensor,
    qtype: int,
    k: int,
    library: str,
    selected: str,
) -> torch.Tensor:
    if (
        x.ndim != 2
        or x.dtype != torch.float32
        or not x.is_cuda
        or not x.is_contiguous()
        or x.shape[0] <= 0
        or x.shape[1] != k
        or packed.dtype != torch.uint8
        or packed.device != x.device
        or not packed.is_contiguous()
        or torch.cuda.get_device_capability(x.device) != TARGETS.get(selected)
    ):
        raise ValueError(
            "Tiled Q3 requires target-matched CUDA F32 rows and original wire"
        )
    backend = backend_for(selected)
    backend.validate_layout(
        qtype,
        packed.shape,
        k,
        packed.data_ptr(),
        packed.untyped_storage().nbytes() - packed.storage_offset(),
    )
    return _run_tiled_guard(x, packed, qtype, k, library, selected)


@guarded_tiled.register_fake
def _fake_tiled(x, packed, qtype, k, library, selected):
    return x.new_empty((x.shape[0], packed.shape[0]), dtype=torch.float32)


def eligible(qtype, shape, dtype):
    return qtype == 11 and tuple(shape) == (128000, 2048) and dtype == torch.bfloat16


class PreparedBF16Q3Head:
    """Original packed pointer, exact input cast, and original native F32 math."""

    SELECTED_ROWS = SELECTED_ROWS

    def __init__(self, original, *, selected_mode=None):
        selected = mode() if selected_mode is None else selected_mode
        if selected not in TARGETS or mode() != selected:
            raise ValueError("BF16 Q3 head requires its explicit target opt-in")
        if not eligible(original.weight_type, original.shape, original.dtype):
            raise ValueError("Require the original BF16 Q3_K vocabulary head")
        if not original.canonical_rounding:
            raise ValueError("BF16 Q3 adapter requires canonical RCO head storage")
        if (
            torch.cuda.get_device_capability(original.weight.device)
            != TARGETS[selected]
        ):
            raise ValueError("BF16 Q3 adapter target does not match this GPU")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Prepare the BF16 Q3 head before CUDA graph capture")
        self.original, self.mode = original, selected
        self.rows_mode = rows_mode()
        self.SELECTED_ROWS = selected_rows()
        self.native_rows = self.SELECTED_ROWS
        self.backend = backend_for(selected)
        self.prepared = self.backend.PreparedNativeQ3Head(
            original.weight, original.weight_type, original.shape[1]
        )
        self.packed = self.prepared.packed
        if self.packed is not original.weight:
            raise ValueError("Native Q3 adapter must reuse original packed bytes")
        self.library = self.prepared.library
        self._warm()

    def _warm(self):
        key = (
            self.mode,
            self.rows_mode,
            self.library,
            self.packed.device,
            tuple(self.original.shape),
        )
        if key in _WARMED:
            return
        with torch.inference_mode():
            for rows in self.SELECTED_ROWS:
                x = torch.ones(
                    rows,
                    self.original.shape[1],
                    dtype=torch.bfloat16,
                    device=self.packed.device,
                )
                self(x, output_dtype=torch.float32)
                self(x, output_dtype=torch.bfloat16)
        torch.cuda.synchronize(self.packed.device)
        _WARMED.add(key)

    def supports_rows(self, rows):
        return rows > 0 if self.rows_mode == "generic" else rows in self.SELECTED_ROWS

    def __call__(self, x, bias=None, *, output_dtype=None):
        if (
            x.dtype != torch.bfloat16
            or x.device != self.packed.device
            or x.shape[-1] != self.original.shape[1]
        ):
            raise ValueError("BF16 Q3 head requires original BF16 input width/device")
        rows = x.numel() // self.original.shape[1]
        if not self.supports_rows(rows):
            raise ValueError("Unsupported Q3 rows must use the original compact router")
        output_dtype = torch.bfloat16 if output_dtype is None else output_dtype
        if output_dtype not in (torch.bfloat16, torch.float32):
            raise ValueError("BF16 Q3 output must be BF16 or F32")
        source = x.reshape(-1, self.original.shape[1]).to(torch.float32).contiguous()
        if self.rows_mode != "generic" or rows in self.native_rows:
            output = self.prepared(source)
        else:
            output = guarded_tiled(
                source,
                self.packed,
                self.original.weight_type,
                self.original.shape[1],
                str(self.library),
                self.mode,
            )
        output = output.to(output_dtype)
        output = output.reshape(*x.shape[:-1], self.original.shape[0])
        return output if bias is None else output + bias

    def marker(self):
        return {
            "mode": self.mode,
            "rows_mode": self.rows_mode,
            "supports_all_positive_rows": self.rows_mode == "generic",
            "native_tile_family": list(self.native_rows),
            "arbitrary_rows_policy": (
                "whole-input guard/repair; greedy native clean-row tiles"
            ),
            "activation_rows": list(self.SELECTED_ROWS),
            "packed_pointer_reused": self.packed is self.original.weight
            and self.packed.data_ptr() == self.original.weight.data_ptr(),
            "unchanged_prepared_identity": self.prepared.identity,
            "BF16_input_to_F32_cast_exact": True,
            "native_F32_guard_and_repair_unchanged": True,
            "canonical_BF16_weight_rounding_preserved": False,
            "persistent_additional_weight_bytes": 0,
        }


def prepare(original):
    selected = mode()
    if selected == "none":
        return None
    if original.weight_type != 11 or tuple(original.shape) != (128000, 2048):
        return None
    return PreparedBF16Q3Head(original, selected_mode=selected)


def runtime_marker():
    selected = mode()
    path = Path(__file__).resolve()
    root = path.parents[1]
    source_paths = [
        path,
        path.with_name("compact.py"),
        path.with_name("vocal_embeds.py"),
        root / "plugin.py",
        root / "__init__.py",
    ]
    marker = {
        "contract": "opt-in-original-wire-bf16-exact-cast-q3-head-v2",
        "mode": selected,
        "sources": {
            str(source.resolve()): hashlib.sha256(source.read_bytes()).hexdigest()
            for source in source_paths
        },
        "activation_dtype": "bfloat16",
        "logits_dtype": "float32",
        "activation_rows": list(selected_rows()),
        "rows_mode": rows_mode(),
        "supports_all_positive_rows": selected != "none" and rows_mode() == "generic",
        "native_tile_family": list(selected_rows()),
        "arbitrary_rows_policy": (
            "whole-input guard/repair; greedy native clean-row tiles"
        ),
        "head_shape": [128000, 2048],
        "qtype": 11,
        "persistent_additional_weight_bytes": 0,
        "canonical_BF16_weight_rounding_preserved": selected == "none",
        "whole_model_EAR_required": selected != "none",
        "selected": False,
    }
    if selected != "none":
        backend = backend_for(selected)
        marker["backend"] = backend.native_q3_head_cache_marker()
        marker["sources"].update(marker["backend"]["sources"])
        marker["target"] = list(TARGETS[selected])
    return marker


def model_status(model, verify=False):
    rows = []
    for name, layer in model.named_modules():
        prepared = getattr(layer, "_native_bf16_q3_head", None)
        if prepared is None:
            continue
        if type(prepared) is not PreparedBF16Q3Head or prepared.original is not layer:
            raise ValueError("Unexpected Q3 prepared adapter object")
        if (
            prepared.packed is not layer.weight
            or prepared.prepared.packed is not layer.weight
        ):
            raise ValueError("Original Q3 packed pointer was replaced")
        row = {
            "name": name,
            "qtype": int(layer.weight_type),
            "shape": list(layer.shape),
            "marker": prepared.marker(),
            "library": str(prepared.library),
            "packed_data_ptr": int(layer.weight.data_ptr()),
            "packed_shape": list(layer.weight.shape),
            "packed_dtype": str(layer.weight.dtype),
        }
        if verify:
            wire = layer.weight.detach().cpu().contiguous().numpy().tobytes()
            row["wire_sha256"] = hashlib.sha256(wire).hexdigest()
        rows.append(row)
    return {
        "mode": mode(),
        "Q3_head_count": len(rows),
        "modules": rows,
        "original_wire_hashes_verified": bool(verify),
        "persistent_additional_weight_bytes": 0,
    }
