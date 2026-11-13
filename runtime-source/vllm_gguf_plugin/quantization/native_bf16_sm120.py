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

ENV = "VLLM_GGUF_NATIVE_SM120_B1"
MODES = ("none", "body", "body-q6")
BODY_TYPES = (8, 12, 13, 14, 23)
_WARMED = set()
COMPONENT_SHA256 = "68e6faa6f368646f823abc7e6dfea5e9038c3ac3f7c7e723dc49effab5c614db"
POLICY = "sm120-qtype-wire-size-row-regime-generic-v1"


def b2_enabled():
    # Compatibility API only. No logical-batch admission flag affects dispatch.
    return False


def validate_profile(compilation_mode, activation_dtype, max_num_seqs):
    # Serving capacity is validated by vLLM; operators dispatch on actual rows.
    del max_num_seqs
    if mode() == "none":
        return
    if compilation_mode != 0 or activation_dtype != "torch.bfloat16":
        raise ValueError("SM120 native requires mode0 and BF16 activations")


def selected_rows(scope, qtype, shape):
    # Historical name: this bounded family is warmup metadata, not eligibility.
    from .native_geometry import warm_rows

    return warm_rows() if eligible(scope, qtype, shape, torch.bfloat16) else ()


def row_supported(scope, qtype, shape, rows):
    from .native_body_sm120 import FORMATS, geometry_supported

    if not geometry_supported(qtype, *tuple(shape), rows):
        return False
    if scope == "q6-head":
        return qtype == 14
    if scope != "body":
        return False
    n, k = shape
    block, size = FORMATS[qtype]
    weight_bytes = n * (k // block) * size
    # Small broadcast GEMVs can beat native quantization/launch overhead.
    # These are architecture/type/size/row regimes, never exact matrix tuples.
    if rows <= 8 and weight_bytes < 256 * 1024:
        return False
    if qtype == 8 and rows == 1 and weight_bytes <= 8 * 1024**2:
        return False
    if qtype == 8 and rows <= 2 and weight_bytes < 2 * 1024**2:
        return False
    if qtype in (13, 14) and rows == 1 and weight_bytes < 2 * 1024**2:
        return False
    if qtype == 14 and rows <= 2 and weight_bytes < 2 * 1024**2:
        return False
    return not (qtype == 14 and 6 < rows <= 8 and weight_bytes < 2 * 1024**2)


def mode():
    value = os.getenv(ENV, "none")
    if value not in MODES:
        raise ValueError(f"{ENV} must be one of {MODES}")
    return value


def eligible(scope, qtype, shape, dtype):
    from .native_body_sm120 import geometry_supported

    return (
        dtype == torch.bfloat16
        and len(shape) == 2
        and geometry_supported(qtype, *tuple(shape))
        and (scope == "body" or (scope == "q6-head" and qtype == 14))
    )


def _make_prepared(original, scope):
    if scope == "body":
        from . import native_body_sm120 as native_body_extended

        return native_body_extended.PreparedNativeGGUFBody(
            original.weight,
            original.weight_type,
            original.shape[1],
            values=original.values,
        )
    from . import native_q6_head_sm120 as native_q6_head

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
        self.original, self.scope = original, scope
        self.mode = selected_mode
        self.prepared = _make_prepared(original, scope)
        self.packed = self.prepared.packed
        if self.packed is not original.weight:
            raise ValueError("Prepared native operation copied the original wire")
        self.SELECTED_ROWS = selected_rows(scope, original.weight_type, original.shape)
        if not self.SELECTED_ROWS:
            raise ValueError("No supported native warm family for this geometry")
        self.library = self.prepared.library
        self._warm()

    def _warm(self):
        # The unchanged prepared ctor already rejects capture and prepares the
        # C library. Warm the added BF16 cast/output shapes before any capture.
        key = (
            self.scope,
            self.original.weight_type,
            tuple(self.original.shape),
            self.packed.device,
            self.library,
            tuple(self.SELECTED_ROWS),
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
        torch.cuda.synchronize(self.packed.device)
        _WARMED.add(key)

    def supports_rows(self, rows):
        if not row_supported(
            self.scope, self.original.weight_type, self.original.shape, rows
        ):
            return False
        from .native_body_sm120 import _native_rows_supported

        return _native_rows_supported(
            self.library,
            self.packed.device.index,
            self.original.weight_type,
            *self.original.shape,
            rows,
        )

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
        source = x.reshape(-1, x.shape[-1]).to(torch.float32).contiguous()
        output = self.prepared(source).to(output_dtype)
        output = output.reshape(*x.shape[:-1], self.original.shape[0])
        return output if bias is None else output + bias

    def marker(self):
        path = Path(__file__).resolve()
        return {
            "contract": "prototype-sm120-bf16-exact-cast-native-f32-op-v1",
            "mode": self.mode,
            "scope": self.scope,
            "policy": POLICY,
            "logical_batch_independent": True,
            "component_sha256": COMPONENT_SHA256,
            "source": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()},
            "unchanged_prepared_identity": self.prepared.identity,
            "constructor_warm_rows": list(self.SELECTED_ROWS),
            "activation_rows": "runtime geometry and cost predicate",
            "packed_pointer_reused": self.packed is self.original.weight
            and self.packed.data_ptr() == self.original.weight.data_ptr(),
            "BF16_input_to_F32_cast_exact": True,
            "native_F32_guard_and_repair_unchanged": True,
            "canonical_BF16_weight_rounding_preserved": False,
            "persistent_additional_dense_weight_bytes": 0,
            "selected": False,
            "whole_model_EAR_required": True,
        }


def prepare(original, *, scope):
    selected = mode()
    if selected == "none" or (scope == "q6-head" and selected != "body-q6"):
        return None
    if scope == "body" and getattr(original, "role", None) != "body":
        return None
    if not eligible(scope, original.weight_type, original.shape, original.dtype):
        return None
    if not selected_rows(scope, original.weight_type, original.shape):
        return None
    if not original.canonical_rounding:
        raise ValueError("SM120 native BF16 mode requires canonical GGUF weights")
    from . import native_body_sm120

    if not original.weight.is_cuda or not native_body_sm120._is_sm120(
        original.weight.device
    ):
        raise ValueError("Explicit SM120 native mode requires an SM120 CUDA tensor")
    return PreparedBF16NativeLinear(original, scope=scope, selected_mode=selected)


def runtime_marker():
    """Bind every host admission/lifecycle source, mode and static measured map."""
    path = Path(__file__).resolve()
    files = [
        path,
        *[
            path.with_name(name)
            for name in (
                "native_body_sm120.py",
                "native_q6_head_sm120.py",
                "compact.py",
                "compact_decode.py",
                "native_head.py",
                "native_geometry.py",
                "native_wire.py",
                "linear.py",
                "yaqa_linear.py",
                "vocal_embeds.py",
            )
        ],
        path.parents[1] / "plugin.py",
        path.parents[1] / "__init__.py",
    ]
    selected = mode()
    result = {
        "contract": "opt-in-SM120-generic-BF16-native-ordinary-Q6-v1",
        "mode": selected,
        "target_SM": 120,
        "activation_dtype": "bfloat16",
        "policy": POLICY,
        "component_sha256": COMPONENT_SHA256,
        "sources": {
            str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
        },
        "body_policy": "qtype/wire-size/actual-row-regime; no model or MNK tuple list",
        "constructor_warm_rows": [1, 4, 16],
        "activation_rows": "all positive native-C-ABI-supported tensor rows",
        "logical_batch_independent": True,
        "partial_row_component_sha256": COMPONENT_SHA256,
        "required_compilation_mode": 0,
        "compiled_graph_routing_qualified": False,
        "persistent_additional_dense_weight_bytes": 0,
        "original_GGUF_buffers_retained": True,
        "canonical_BF16_weight_rounding_preserved": selected == "none",
        "selected": False,
        "same8_graph_EAR_required": selected != "none",
    }
    if selected != "none":
        from . import native_body_sm120

        result["body"] = native_body_sm120.marker()
        result["sources"].update(result["body"]["sources"])
        if selected == "body-q6":
            from . import native_q6_head_sm120

            result["Q6_head"] = native_q6_head_sm120.native_q6_head_cache_marker()
            result["sources"].update(result["Q6_head"]["sources"])
    return result


def model_status(model, verify=False):
    """Read-only, untimed; packed-wire GPU reads happen only with verify=True."""
    from . import compact

    rows = []
    for name, layer in model.named_modules():
        if not isinstance(layer, compact.PackedGGUFLinear):
            continue
        for field, scope in (
            ("_native_sm120_body", "body"),
            ("_native_sm120_q6_head", "q6-head"),
        ):
            prepared = getattr(layer, field, None)
            if prepared is None:
                continue
            native = prepared.prepared
            if (
                prepared.original is not layer
                or prepared.packed is not layer.weight
                or native.packed is not layer.weight
                or prepared.packed.data_ptr() != layer.weight.data_ptr()
                or native.library != prepared.library
            ):
                raise ValueError(
                    "SM120 prepared object does not retain original wire/library"
                )
            row = {
                "name": name,
                "scope": scope,
                "qtype": int(layer.weight_type),
                "shape": list(layer.shape),
                "constructor_warm_rows": list(prepared.SELECTED_ROWS),
                "activation_rows": "generic-capability-and-cost-predicate",
                "packed_pointer": int(layer.weight.data_ptr()),
                "native_wire_padding": getattr(layer, "native_wire_padding", {}),
                "original_packed_object_reused": True,
                "library": str(prepared.library),
                "marker": prepared.marker(),
                "native_identity": native.identity,
                "wire_GPU_hash_verified": bool(verify),
            }
            if verify:
                row["wire_sha256"] = hashlib.sha256(
                    layer.weight.detach().cpu().numpy().tobytes()
                ).hexdigest()
            rows.append(row)
    return {
        "mode": mode(),
        "policy": POLICY,
        "logical_batch_independent": True,
        "modules": rows,
        "extra_resident_quant_padding_bytes": sum(
            row["native_wire_padding"].get("extra_resident_quant_bytes", 0)
            for row in rows
        ),
        "body_count": sum(row["scope"] == "body" for row in rows),
        "Q6_head_count": sum(row["scope"] == "q6-head" for row in rows),
        "persistent_additional_dense_weight_bytes": 0,
        "default_status_GPU_sync": False,
        "GPU_hashes_verified": bool(verify and rows),
        "torch": str(torch.__version__),
    }
