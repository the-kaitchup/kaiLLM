# SPDX-License-Identifier: Apache-2.0
"""Unselected BF16 host adapter around unchanged F32 native GGUF operations.

Explicit opt-in product host adapter. Native
activation quantization and F32 decoded-weight arithmetic differ from canonical
BF16 compact multiplication. Whole-model EAR is required before selection.
"""

import hashlib
import os
from pathlib import Path

import torch

ENV = "VLLM_GGUF_NATIVE_BF16_ADAPTER"
PADDING_ENV = "VLLM_GGUF_NATIVE_BF16_PADDING_CAST"
PADDING_ROWS = (512, 514, 516, 1024, 2048)
MODES = ("none", "body", "body-q6")
BODY_TYPES = (2, 3, 6, 7, 8, 12, 13, 14, 18, 21, 22, 23)
BODY_ROWS = (1, 2, 4, 8, 512, 514, 516, 1024, 2048)
Q6_ROWS = (1, 2, 4, 8, 16)
HEAD_ROWS = (1, 2, 4, 8, 16)  # warm-up only; any C-ABI-admitted row count is served
_WARMED = set()


def padding_cast_mode():
    value = os.getenv(PADDING_ENV, "0")
    if value not in ("0", "1"):
        raise ValueError(f"{PADDING_ENV} must be0or1")
    return value == "1"


def _forward_padding(x, rows):
    from vllm.forward_context import get_forward_context, is_forward_context_available

    if not is_forward_context_available():
        return None
    padding = get_forward_context().is_padding
    if (
        padding is None
        or padding.dtype != torch.bool
        or padding.device != x.device
        or padding.ndim != 1
        or padding.shape[0] != rows
        or not padding.is_contiguous()
    ):
        return None
    return padding


def self_half_power2_types():
    """Q5_K keeps its dedicated half-power2 route when that route is enabled."""
    return (13,) if os.getenv("VLLM_GGUF_NATIVE_BODY_HALF_POWER2", "0") == "1" else ()


def mode():
    value = os.getenv(ENV, "none")
    if value not in MODES:
        raise ValueError(f"{ENV} must be one of {MODES}")
    return value


def eligible(scope, qtype, shape, dtype):
    if dtype != torch.bfloat16 or len(shape) != 2:
        return False
    if scope == "body":
        if qtype in (2, 3, 6, 7) and os.getenv("KAILLM_LEGACY_NATIVE", "1") != "1":
            return False
        return (
            qtype in BODY_TYPES and 128 <= shape[0] <= 7680 and shape[1] in (2048, 7680)
        )
    if scope == "q6-head":
        return qtype == 14 and tuple(shape) == (128000, 2048)
    if scope == "generic-head":
        from .native_body_extended import FORMATS

        return (
            qtype in (2, 3, 6, 7, 8, 12)
            and shape[0] >= 128
            and shape[1] % FORMATS[qtype][0] == 0
        )
    return False


def _make_prepared(original, scope):
    if scope in ("body", "generic-head"):
        from . import native_body_extended

        return native_body_extended.PreparedNativeGGUFBody(
            original.weight,
            original.weight_type,
            original.shape[1],
            values=original.values,
            grid=original.grid,
            signs=original.signs,
        )
    from . import native_q6_head

    return native_q6_head.PreparedNativeQ6Head(
        original.weight, original.weight_type, original.shape[1]
    )


class PreparedBF16NativeLinear:
    """Exact input cast, unchanged prepared F32 op, requested final output cast."""

    def __init__(self, original, *, scope, selected_mode=None):
        selected_mode = mode() if selected_mode is None else selected_mode
        if selected_mode not in MODES or selected_mode == "none":
            raise ValueError("BF16 native adapter requires an explicit opt-in mode")
        if scope == "q6-head" and selected_mode != "body-q6":
            raise ValueError("Q6 vocabulary adapter needs explicit body-q6 mode")
        if not eligible(scope, original.weight_type, original.shape, original.dtype):
            raise ValueError("BF16 original compact layer is outside adapter scope")
        from . import native_bf16_q8q6_power2

        self.body_q8q6_power2 = (
            scope in ("body", "generic-head")
            and native_bf16_q8q6_power2.power2_type(original.weight_type, self_half_power2_types())
            and native_bf16_q8q6_power2.enabled()
        )
        self.original, self.scope = original, scope
        self.mode = selected_mode
        self.padding_cast = padding_cast_mode() and scope == "body"
        half_mode = os.getenv("VLLM_GGUF_NATIVE_BODY_HALF_POWER2", "0")
        if half_mode not in ("0", "1"):
            raise ValueError("VLLM_GGUF_NATIVE_BODY_HALF_POWER2 must be0or1")
        self.body_half_power2 = half_mode == "1" and scope == "body"
        from . import native_q6_mmvq_mma

        self.q6_mma_m4 = scope == "q6-head" and native_q6_mmvq_mma.mode() in (
            "m4",
            "m4-m8",
            "m4-m8-m16",
        )
        self.q6_mma_m8 = scope == "q6-head" and native_q6_mmvq_mma.mode() in (
            "m4-m8",
            "m4-m8-m16",
        )
        self.q6_mma_m16 = (
            scope == "q6-head" and native_q6_mmvq_mma.mode() == "m4-m8-m16"
        )
        if self.q6_mma_m4:
            native_q6_mmvq_mma.prepare(original.weight.device)
        self.prepared = _make_prepared(original, scope)
        self.packed = self.prepared.packed
        if self.packed is not original.weight:
            raise ValueError("Prepared native operation copied the original wire")
        self.SELECTED_ROWS = (
            BODY_ROWS if scope == "body" else HEAD_ROWS if scope == "generic-head" else Q6_ROWS
        )
        self.library = self.prepared.library
        self._warm()

    def supports_rows(self, rows):
        if self.scope in ("body", "generic-head"):
            generic = getattr(self.prepared, "supports_rows", None)
            if generic is not None:
                return generic(rows)
            if self.body_q8q6_power2:
                from . import native_bf16_q8q6_power2

                if native_bf16_q8q6_power2.supports(self.prepared, rows):
                    return True
        return rows in self.SELECTED_ROWS

    def _warm(self):
        # The unchanged prepared ctor already rejects capture and prepares the
        # C library. Warm the added BF16 cast/output shapes before any capture.
        key = (
            self.scope,
            self.original.weight_type,
            tuple(self.original.shape),
            self.packed.device,
            self.library,
            self.padding_cast,
            self.body_q8q6_power2,
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
                self(x)
                if self.scope == "q6-head":
                    self(x, output_dtype=torch.float32)
        if self.body_q8q6_power2:
            from . import native_bf16_q8q6_power2

            native_bf16_q8q6_power2.warm(self.prepared)
        if self.padding_cast:
            from .native_bf16_padding_cast import warm

            warm(self.prepared, [r for r in PADDING_ROWS if r in self.SELECTED_ROWS])
        torch.cuda.synchronize(self.packed.device)
        _WARMED.add(key)

    def __call__(self, x, bias=None, *, output_dtype=None):
        if (
            x.dtype != torch.bfloat16
            or x.device != self.packed.device
            or x.shape[-1] != self.original.shape[1]
        ):
            raise ValueError("BF16 native adapter requires original BF16 inputs")
        output_dtype = torch.bfloat16 if output_dtype is None else output_dtype
        if output_dtype not in (torch.bfloat16, torch.float32):
            raise ValueError("BF16 adapter output must be BF16 or F32")
        rows = x.numel() // x.shape[-1]
        if not self.supports_rows(rows):
            return self.original(x, bias, output_dtype=output_dtype)
        if self.body_q8q6_power2:
            from . import native_bf16_q8q6_power2

            if native_bf16_q8q6_power2.supports(self.prepared, rows):
                output = (
                    native_bf16_q8q6_power2.apply(
                        x.reshape(-1, x.shape[-1]).contiguous(), self.prepared
                    )
                    .to(output_dtype)
                    .reshape(*x.shape[:-1], self.original.shape[0])
                )
                return output if bias is None else output + bias
        padding = (
            _forward_padding(x, rows)
            if self.padding_cast and rows in PADDING_ROWS
            else None
        )
        if padding is None:
            source = x.reshape(-1, x.shape[-1]).to(torch.float32).contiguous()
        else:
            from .native_bf16_padding_cast import apply

            source = apply(
                x.reshape(-1, x.shape[-1]).contiguous(), self.prepared, padding
            )
        # Preserve the existing Q8 MMQ row policy exactly for irregular prefills.
        # Extra zero rows never reach attention, convolution or the output head.
        if (
            self.scope == "body"
            and self.original.weight_type == 8
            and rows in (514, 516)
        ):
            source = torch.nn.functional.pad(source, (0, 0, 0, 1024 - rows))
        if (
            (self.q6_mma_m4 and rows == 4)
            or (self.q6_mma_m8 and rows == 8)
            or (self.q6_mma_m16 and rows == 16)
        ):
            from . import native_q6_mmvq_mma

            output = native_q6_mmvq_mma.guarded(source, self.packed).to(output_dtype)
        elif self.body_half_power2 and (
            (
                self.original.weight_type == 13
                and rows in (1, 2, 4, 8, 512, 514, 516, 1024, 2048)
            )
            or (self.original.weight_type == 14 and rows in (1, 2, 4, 8))
        ):
            from . import native_body_half_power2

            output = native_body_half_power2.apply(source, self.prepared).to(
                output_dtype
            )
        else:
            output = self.prepared(source).to(output_dtype)
        output = output[:rows].reshape(*x.shape[:-1], self.original.shape[0])
        return output if bias is None else output + bias

    def marker(self):
        path = Path(__file__).resolve()
        return {
            "contract": "prototype-bf16-exact-cast-native-f32-op-v1",
            "mode": self.mode,
            "scope": self.scope,
            "padding_cast": self.padding_cast,
            "padding_cast_activation_rows": list(PADDING_ROWS),
            "source": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()},
            "unchanged_prepared_identity": self.prepared.identity,
            "activation_rows": list(self.SELECTED_ROWS),
            "packed_pointer_reused": self.packed is self.original.weight
            and self.packed.data_ptr() == self.original.weight.data_ptr(),
            "BF16_input_to_F32_cast_exact": True,
            "native_F32_guard_and_repair_unchanged": not (
                self.body_half_power2 or self.q6_mma_m4 or self.body_q8q6_power2
            ),
            "Q6_MMVQ_MMA_M4": self.q6_mma_m4,
            "Q6_MMVQ_MMA_M8": self.q6_mma_m8,
            "body_half_power2": self.body_half_power2,
            "body_q8q6_power2": self.body_q8q6_power2,
            "canonical_BF16_weight_rounding_preserved": False,
            "persistent_extra_weight_bytes": 0,
            "selected": False,
            "whole_model_EAR_required": True,
        }


def prepare(original, *, scope):
    selected = mode()
    if selected == "none" or (scope == "q6-head" and selected != "body-q6"):
        return None
    if not eligible(scope, original.weight_type, original.shape, original.dtype):
        return None
    return PreparedBF16NativeLinear(original, scope=scope, selected_mode=selected)


def runtime_marker():
    """Bind adapter source/mode into the model/AOT cache even when default off."""
    path = Path(__file__).resolve()
    selected = mode()
    marker = {
        "contract": "opt-in-bf16-exact-cast-native-f32-v1",
        "mode": selected,
        "sources": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()},
        "activation_dtype": "bfloat16",
        "hardware_scope": "SM86",
        "canonical_RCO_required": True,
        "body_types": list(BODY_TYPES),
        "body_rows": list(BODY_ROWS),
        "Q6_head_rows": list(Q6_ROWS),
        "persistent_extra_weight_bytes": 0,
        "precision_change": selected != "none",
        "canonical_BF16_weight_rounding_preserved": selected == "none",
        "whole_model_EAR_required": selected != "none",
    }
    from . import native_bf16_q8q6_power2

    marker["Q8Q6_generic_power2"] = native_bf16_q8q6_power2.marker()
    marker["sources"].update(marker["Q8Q6_generic_power2"]["sources"])
    marker["padding_cast"] = padding_cast_mode()
    if marker["padding_cast"]:
        from . import native_bf16_padding_cast

        marker["padding_input"] = native_bf16_padding_cast.runtime_marker()
        marker["sources"].update(marker["padding_input"]["sources"])
    half_mode = os.getenv("VLLM_GGUF_NATIVE_BODY_HALF_POWER2", "0")
    if half_mode not in ("0", "1"):
        raise ValueError("VLLM_GGUF_NATIVE_BODY_HALF_POWER2 must be0or1")
    marker["body_half_power2"] = half_mode == "1"
    from . import native_q6_mmvq_mma

    marker["Q6_MMVQ_MMA_M4"] = native_q6_mmvq_mma.mode() in ("m4", "m4-m8", "m4-m8-m16")
    marker["Q6_MMVQ_MMA_M8"] = native_q6_mmvq_mma.mode() in ("m4-m8", "m4-m8-m16")
    marker["Q6_MMVQ_MMA_M16"] = native_q6_mmvq_mma.mode() == "m4-m8-m16"
    if marker["Q6_MMVQ_MMA_M4"]:
        marker["Q6_MMVQ_MMA"] = native_q6_mmvq_mma.marker()
        marker["sources"].update(marker["Q6_MMVQ_MMA"]["sources"])
    if half_mode == "1":
        half_source = Path(__file__).with_name("native_body_half_power2.py")
        marker["sources"][str(half_source)] = hashlib.sha256(
            half_source.read_bytes()
        ).hexdigest()
    if selected != "none":
        from . import native_body_extended

        marker["body"] = native_body_extended.marker()
        marker["sources"].update(marker["body"]["sources"])
        if selected == "body-q6":
            from . import native_q6_head

            marker["Q6_head"] = native_q6_head.native_q6_head_cache_marker()
            marker["sources"].update(marker["Q6_head"]["sources"])
    return marker
