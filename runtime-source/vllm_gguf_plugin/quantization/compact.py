# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Direct GGUF GEMV/GEMM with unchanged packed weight storage.

Decode reads the GGUF wire bytes inside the consuming kernel. No floating-point
weight matrix, expanded IQ integers, or rounded scale array is retained. Decode
and accumulation use FP32; batched prefill approximates FP32 products with
three-pass TF32 Tensor Cores on Ampere and newer.
"""

import torch
import triton
import triton.language as tl

from ..triton.gemm.iq_quant.iq_tables import get_iq_table_tensors
from .compact_decode import load_gguf
from .gguf_repack import FORMATS

SUPPORTED_COMPACT_TYPES = frozenset(FORMATS)
DENSE_GGUF_TYPES = frozenset({0, 1, 24, 25, 26, 27, 28, 30})
_GRID_NAMES = {
    16: "iq2xxs_grid",
    17: "iq2xs_grid",
    18: "iq3xxs_grid",
    19: "iq1s_grid",
    21: "iq3s_grid",
    22: "iq2s_grid",
    29: "iq1s_grid",
}


@triton.jit(
    do_not_specialize=["M"],
    do_not_specialize_on_alignment=["M"],
)
def _compact_gemv(
    x_ptr,
    w_ptr,
    y_ptr,
    grid_ptr,
    sign_ptr,
    values_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    STRIDE_XM: tl.constexpr,
    STRIDE_XK: tl.constexpr,
    QTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    CANONICAL_ROUNDING: tl.constexpr = False,
):
    rows = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    batches = tl.arange(0, BLOCK_M)
    columns = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for start in range(tl.cdiv(K, BLOCK_K)):
        k = start * BLOCK_K + columns
        w = load_gguf(
            w_ptr,
            rows[:, None] * K + k[None, :],
            (rows[:, None] < N) & (k[None, :] < K),
            grid_ptr,
            sign_ptr,
            values_ptr,
            QTYPE,
        )
        if CANONICAL_ROUNDING and x_ptr.dtype.element_ty != tl.float32:
            w = w.to(x_ptr.dtype.element_ty, fp_downcast_rounding="rtne").to(tl.float32)
        x = tl.load(
            x_ptr + batches[:, None] * STRIDE_XM + k[None, :] * STRIDE_XK,
            (batches[:, None] < M) & (k[None, :] < K),
            other=0,
        ).to(tl.float32)
        acc += tl.sum(x[:, None, :] * w[None, :, :], axis=2)
    tl.store(
        y_ptr + batches[:, None] * N + rows[None, :],
        acc,
        (batches[:, None] < M) & (rows[None, :] < N),
    )


@triton.jit(
    do_not_specialize=["M"],
    do_not_specialize_on_alignment=["M"],
)
def _compact_gemm(
    x_ptr,
    w_ptr,
    y_ptr,
    grid_ptr,
    sign_ptr,
    values_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    STRIDE_XM: tl.constexpr,
    STRIDE_XK: tl.constexpr,
    QTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    CANONICAL_ROUNDING: tl.constexpr = False,
):
    batches = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    rows = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    columns = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for start in range(tl.cdiv(K, BLOCK_K)):
        k = start * BLOCK_K + columns
        w = load_gguf(
            w_ptr,
            rows[:, None] * K + k[None, :],
            (rows[:, None] < N) & (k[None, :] < K),
            grid_ptr,
            sign_ptr,
            values_ptr,
            QTYPE,
        )
        x = tl.load(
            x_ptr + batches[:, None] * STRIDE_XM + k[None, :] * STRIDE_XK,
            (batches[:, None] < M) & (k[None, :] < K),
            other=0,
        ).to(tl.float32)
        if CANONICAL_ROUNDING and x_ptr.dtype.element_ty != tl.float32:
            w = w.to(x_ptr.dtype.element_ty, fp_downcast_rounding="rtne")
            acc = tl.dot(x.to(w.dtype), tl.trans(w), acc)
        else:
            acc = tl.dot(x, tl.trans(w), acc, input_precision="tf32x3")
    tl.store(
        y_ptr + batches[:, None] * N + rows[None, :],
        acc,
        (batches[:, None] < M) & (rows[None, :] < N),
    )


@triton.jit(
    do_not_specialize=["TOTAL"],
    do_not_specialize_on_alignment=["TOTAL"],
)
def _compact_gather(
    ids_ptr,
    w_ptr,
    y_ptr,
    grid_ptr,
    sign_ptr,
    values_ptr,
    TOTAL,
    N: tl.constexpr,
    K: tl.constexpr,
    QTYPE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = offs < TOTAL
    rows = tl.load(ids_ptr + offs // K, active, other=0)
    valid = active & (rows >= 0) & (rows < N)
    w = load_gguf(
        w_ptr, rows * K + offs % K, valid, grid_ptr, sign_ptr, values_ptr, QTYPE
    )
    tl.store(y_ptr + offs, w, active)


def _native_mma_b16_eligible(
    head, rows, weight_type, shape, output_dtype, activation_dtype=None
):
    """Reuse the identity already validated by PreparedNativeGGUFHead."""
    return (
        (rows == 16 or (9 <= rows < 16 and activation_dtype == torch.float32))
        and weight_type == 13
        and tuple(shape) == (128000, 2048)
        and output_dtype == torch.float32
        and head is not None
        and getattr(head, "mode", None) == "native-mma"
        and bool(getattr(head, "mma_identity", None))
    )


def _native_body_rows_supported(prepared, rows):
    capability = getattr(prepared, "supports_rows", None)
    return (
        capability(rows)
        if capability is not None
        else rows in getattr(prepared, "SELECTED_ROWS", (1, 2, 8, 16, 512))
    )


class PackedGGUFLinear(torch.nn.Module):
    """Batched linear or embedding over the original GGUF row bytes.

    Args:
        data: Contiguous row-major wire bytes, after any projection permutation
            or tensor-parallel slicing. Scalar GGUF types accept numeric data.
        weight_type: Numeric ``GGMLQuantizationType``.
        dtype: Activation and default output dtype.
        canonical_rounding: Round each decoded weight to ``dtype`` before
            multiplication, matching a dense reconstruction in that dtype.
    """

    def __init__(
        self,
        data: torch.Tensor,
        weight_type: int,
        dtype: torch.dtype,
        *,
        canonical_rounding: bool = False,
        role: str | None = None,
    ):
        super().__init__()
        self.weight_type = int(weight_type)
        self.dtype = dtype
        self.canonical_rounding = canonical_rounding
        if role not in (None, "body", "vocabulary"):
            raise ValueError("Unknown GGUF matrix role")
        self.role = role
        self._native_q6_head = None
        self._native_head = None
        self._native_bf16_q3_head = None
        self._native_body = None
        self._native_bf16_body = None
        self._native_bf16_head = None
        self._native_sm120_body = None
        self._native_sm120_q6_head = None
        if dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("Compact GGUF requires FP16, BF16, or FP32 activations")
        if data.ndim != 2:
            raise ValueError("Compact GGUF weights must be a two-dimensional tensor")
        self.is_quantized = self.weight_type not in DENSE_GGUF_TYPES
        if self.is_quantized:
            if self.weight_type not in SUPPORTED_COMPACT_TYPES:
                raise ValueError(f"Unsupported compact GGUF type: {self.weight_type}")
            if data.dtype != torch.uint8:
                raise ValueError("Quantized GGUF data must contain uint8 wire bytes")
            if not data.is_cuda or torch.version.hip is not None:
                raise ValueError("Compact GGUF requires an NVIDIA CUDA device")
            if torch.cuda.get_device_capability(data.device) < (8, 0):
                raise ValueError("Compact GGUF requires NVIDIA SM80 or newer")
            _, _, block_elements, block_bytes = FORMATS[self.weight_type]
            if data.shape[1] == 0 or data.shape[1] % block_bytes:
                raise ValueError("GGUF row width must be a whole number of wire blocks")
            self.shape = (data.shape[0], data.shape[1] // block_bytes * block_elements)
            weight = data.detach().contiguous()
        else:
            integer_storage = self.weight_type in (24, 25, 26, 27) and data.dtype in (
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            )
            if not data.is_floating_point() and not integer_storage:
                raise ValueError(
                    "Dense GGUF weights must contain numeric scalar values"
                )
            self.shape = tuple(data.shape)
            weight = data.detach().to(dtype=dtype).contiguous()
        self.native_wire_padding = {"extra_resident_quant_bytes": 0}
        sm120 = weight.device.type == "cuda" and torch.cuda.get_device_capability(
            weight.device
        ) == (12, 0)
        if (
            self.is_quantized
            and self.role == "body"
            and dtype == torch.bfloat16
            and sm120
            and self.canonical_rounding
        ):
            from .native_bf16_sm120 import mode as native_sm120_mode

            if native_sm120_mode() != "none":
                from .native_body_sm120 import FORMATS as native_formats
                from .native_wire import align_native_wire

                weight, self.native_wire_padding = align_native_wire(
                    weight, self.weight_type, self.shape[1], native_formats
                )
        self.register_buffer("weight", weight)
        # Missing tables alias one byte of the packed matrix; constexpr dispatch
        # removes every unused pointer load. Shared IQ tables remain cached once.
        dummy = weight.reshape(-1)[:1]
        grid = signs = values = dummy
        if self.weight_type in _GRID_NAMES or self.weight_type in (20, 23):
            tables = get_iq_table_tensors(data.device)
            grid = tables.get(_GRID_NAMES.get(self.weight_type), dummy)
            signs = tables["ksigns_iq2xs"]
            values = tables["kvalues_iq4nl"]
        if self.weight_type in (39, 40):
            values = torch.tensor(
                [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12],
                device=data.device,
                dtype=torch.int8,
            )
        self.register_buffer("grid", grid, persistent=False)
        self.register_buffer("signs", signs, persistent=False)
        self.register_buffer("values", values, persistent=False)
        if self.role == "body" and not sm120:
            from .native_body import prepare_native_body

            self._native_body = prepare_native_body(
                self.weight,
                self.weight_type,
                self.shape,
                self.dtype,
                values=self.values,
            )
            if self.dtype == torch.bfloat16 and self.canonical_rounding:
                from .native_bf16 import prepare

                self._native_bf16_body = prepare(self, scope="body")

        if self.role == "body" and sm120 and self.canonical_rounding:
            from .native_bf16_sm120 import prepare as prepare_sm120

            self._native_sm120_body = prepare_sm120(self, scope="body")

    def prepare_native_sm120_q6_head(self, selected_mode=None):
        """Prepare an SM120 output backend independently of embedding lookup."""
        if self.dtype != torch.bfloat16 or not self.canonical_rounding:
            return
        from .native_bf16_sm120 import PreparedBF16NativeLinear, prepare

        if selected_mode is None:
            self._native_sm120_q6_head = prepare(self, scope="q6-head")
        else:
            self._native_sm120_q6_head = PreparedBF16NativeLinear(
                self, scope="q6-head", selected_mode=selected_mode
            )

    def prepare_native_bf16_q6_head(self):
        """Prepare only an explicit BF16 vocabulary output, not embedding lookup."""
        if self.dtype == torch.bfloat16 and self.canonical_rounding:
            from .native_bf16 import prepare

            self._native_bf16_head = prepare(self, scope="q6-head")

    def prepare_native_q6_head(self):
        """Opt into the separately scoped F32 Q6 head over the same bytes."""
        if self.weight.is_cuda and torch.cuda.get_device_capability(
            self.weight.device
        ) == (12, 0):
            if self.dtype == torch.bfloat16:
                self.prepare_native_sm120_q6_head(selected_mode="body-q6")
            return
        from .native_q6_head import prepare_native_q6_head

        self._native_q6_head = prepare_native_q6_head(
            self.weight, self.weight_type, self.shape, self.dtype
        )

    def prepare_native_bf16_q3_head(self):
        """Load and prewarm an explicitly selected original-wire Q3 adapter."""
        from .native_bf16_q3 import prepare

        self._native_bf16_q3_head = prepare(self)

    def prepare_native_head(self, mode="native"):
        """Opt into a warmed native head without copying the packed weights."""
        from .native_head import PreparedNativeGGUFHead

        self._native_head = PreparedNativeGGUFHead(
            self.weight, self.weight_type, self.shape[1], self.dtype, mode=mode
        )

    def forward(
        self,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        *,
        output_dtype: torch.dtype | None = None,
    ):
        state = getattr(self, "_ordinary_marlin_physical", None)
        if state is not None:
            from .ordinary_marlin_physical import apply_prepared

            return apply_prepared(x, state, bias, output_dtype=output_dtype)
        return self._forward_original(x, bias, output_dtype=output_dtype)

    def _forward_original(
        self,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        *,
        output_dtype: torch.dtype | None = None,
    ):
        """Apply the packed matrix, optionally retaining FP32 logits.

        Default compact weight reconstruction follows the activation dtype.
        The explicit BF16 native adapter retains native F32 weight arithmetic;
        its output cast only controls the final store. Packed storage is shared.
        """
        if x.shape[-1] != self.shape[1]:
            raise ValueError("GGUF activation width does not match the weight matrix")
        if x.dtype != self.dtype or x.device != self.weight.device:
            raise ValueError("GGUF activations and weights must share dtype and device")
        output_dtype = self.dtype if output_dtype is None else output_dtype
        if output_dtype not in (self.dtype, torch.float32):
            raise ValueError("Compact GGUF output dtype must match activations or FP32")
        if self.role == "vocabulary" and output_dtype in (torch.float32, torch.bfloat16):
            fused = getattr(self, "_kai_fused", None)
            if fused is None:
                from .native_fused import prepare as prepare_fused

                fused = prepare_fused([self]) or False
                self._kai_fused = fused
            if fused and fused.supports(x):
                output = fused(x, output_dtype)
                return output if bias is None else output + bias
        if getattr(self, "body_prefill_weight", None) is not None:
            from .shared_body_prefill_cache import apply as apply_body_prefill
            from .shared_body_prefill_cache import selected as select_body_prefill

            if select_body_prefill(x, self):
                return apply_body_prefill(x, self, bias, output_dtype=output_dtype)
        q3_adapter = self._native_bf16_q3_head
        if q3_adapter is not None and q3_adapter.supports_rows(
            x.numel() // self.shape[1]
        ):
            return q3_adapter(x, bias, output_dtype=output_dtype)
        if self.dtype == torch.bfloat16:
            adapter = self._native_bf16_head
            if adapter is None:
                adapter = self._native_bf16_body
            # Check the exact per-object scope before calling the host adapter.
            # Unsupported rows continue below through the original compact path.
            if adapter is not None and adapter.supports_rows(
                x.numel() // self.shape[1]
            ):
                return adapter(x, bias, output_dtype=output_dtype)
        native_sm120 = self._native_sm120_body or self._native_sm120_q6_head
        if native_sm120 is not None and native_sm120.supports_rows(
            x.numel() // self.shape[1]
        ):
            return native_sm120(x, bias, output_dtype=output_dtype)
        if (
            self._native_q6_head is not None
            and output_dtype == torch.float32
            and x.numel() // self.shape[1] in (1, 2, 4, 8, 16)
        ):
            output = self._native_q6_head(x).reshape(*x.shape[:-1], self.shape[0])
            return output if bias is None else output + bias
        if not self.is_quantized:
            if output_dtype == self.dtype:
                return torch.nn.functional.linear(x, self.weight, bias)
            if x.is_cuda:
                output = torch.mm(
                    x.reshape(-1, self.shape[1]),
                    self.weight.T,
                    out_dtype=output_dtype,
                ).reshape(*x.shape[:-1], self.shape[0])
            else:
                output = torch.nn.functional.linear(x.float(), self.weight.float())
            return output if bias is None else output + bias
        inputs = x.reshape(-1, self.shape[1])
        rows = inputs.shape[0]
        if (
            self._native_body is not None
            and output_dtype == torch.float32
            and _native_body_rows_supported(self._native_body, rows)
        ):
            output = self._native_body(inputs).reshape(*x.shape[:-1], self.shape[0])
            return output if bias is None else output + bias
        if (
            self._native_head is not None
            and output_dtype == torch.float32
            and 0 < rows <= 8
        ):
            output = self._native_head(inputs, self.weight).reshape(
                *x.shape[:-1], self.shape[0]
            )
            return output if bias is None else output + bias
        if _native_mma_b16_eligible(
            self._native_head,
            rows,
            self.weight_type,
            self.shape,
            output_dtype,
            self.dtype,
        ):
            tail = inputs[8:]
            if rows < 16:
                tail = torch.nn.functional.pad(tail, (0, 0, 0, 16 - rows))
            output = torch.cat(
                [
                    self._native_head(inputs[:8], self.weight),
                    self._native_head(tail, self.weight)[: rows - 8],
                ],
                dim=0,
            ).reshape(*x.shape[:-1], self.shape[0])
            return output if bias is None else output + bias
        if (
            self._native_head is not None
            and output_dtype == torch.float32
            and self._native_head.selects_large_rows(
                rows, inputs.dtype, self.canonical_rounding
            )
        ):
            output = self._native_head.project_large_rows(
                inputs,
                self.weight,
                self.grid,
                self.signs,
                self.values,
                canonical_rounding=self.canonical_rounding,
                canonical_dtype=self.dtype,
            ).reshape(*x.shape[:-1], self.shape[0])
            return output if bias is None else output + bias
        output = torch.empty((rows, self.shape[0]), dtype=output_dtype, device=x.device)
        if rows and self.shape[0]:
            common = (inputs, self.weight, output, self.grid, self.signs, self.values)
            args = dict(
                M=rows,
                N=self.shape[0],
                K=self.shape[1],
                STRIDE_XM=inputs.stride(0),
                STRIDE_XK=inputs.stride(1),
                QTYPE=self.weight_type,
                CANONICAL_ROUNDING=self.canonical_rounding,
                enable_fp_fusion=False,
            )
            large_q5_head = (
                self.canonical_rounding
                and self.dtype in (torch.float16, torch.bfloat16)
                and self.weight_type == 13
                and self.shape[0] >= 32768
            )
            if rows <= 8 and not large_q5_head:
                _compact_gemv[(triton.cdiv(self.shape[0], 4),)](
                    *common,
                    **args,
                    BLOCK_M=triton.next_power_of_2(rows),
                    BLOCK_N=4,
                    BLOCK_K=256,
                    num_warps=4,
                )
            else:
                if large_q5_head:
                    if rows <= 16:
                        bm, bn, bk = 16, 64, 64
                    elif rows <= 32:
                        bm, bn, bk = 32, 64, 64
                    elif rows < 128:
                        bm, bn, bk = 64, 64, 64
                    else:
                        bm, bn, bk = 128, 32, 64
                else:
                    bm, bn, bk = 16, 32, 128
                _compact_gemm[(triton.cdiv(rows, bm), triton.cdiv(self.shape[0], bn))](
                    *common,
                    **args,
                    BLOCK_M=bm,
                    BLOCK_N=bn,
                    BLOCK_K=bk,
                    num_warps=4,
                    num_stages=2,
                )
        output = output.reshape(*x.shape[:-1], self.shape[0])
        return output if bias is None else output + bias

    def embedding(self, indices: torch.Tensor):
        if indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("GGUF embedding indices must be int32 or int64")
        if indices.device != self.weight.device:
            raise ValueError("GGUF embedding indices and weights must share a device")
        if not self.is_quantized:
            return torch.nn.functional.embedding(indices, self.weight)
        indices = indices.contiguous()
        output = torch.empty(
            (*indices.shape, self.shape[1]), dtype=self.dtype, device=indices.device
        )
        if output.numel():
            _compact_gather[(triton.cdiv(output.numel(), 256),)](
                indices,
                self.weight,
                output,
                self.grid,
                self.signs,
                self.values,
                TOTAL=output.numel(),
                N=self.shape[0],
                K=self.shape[1],
                QTYPE=self.weight_type,
                BLOCK=256,
                enable_fp_fusion=False,
                num_warps=4,
            )
        return output
