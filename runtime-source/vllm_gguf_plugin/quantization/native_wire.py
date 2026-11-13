# SPDX-License-Identifier: Apache-2.0
"""Load-time native alignment/tail storage over unchanged quantized row bytes."""

from .native_geometry import geometry_supported, required_storage_bytes


def align_native_wire(data, qtype, k, formats):
    """Return readable aligned wire, its unchanged logical shape, and byte cost.

    Unsupported native formats return the original tensor for compact fallback.
    The caller chooses a semantic body role and source-qualified backend first.
    Existing aligned, tail-free LFM matrices are returned without allocation.
    For a nonempty tail this helper owns and zeros new padding rather than
    assuming neighboring tensors in an input storage are safe padding.
    Allocation/copy happens once before module buffers/tables are registered.
    """
    import torch

    info = {
        "copy_performed": False,
        "extra_resident_quant_bytes": 0,
        "logical_wire_bytes": data.numel() * data.element_size(),
        "original_GPU_input_pointer_reused": True,
        "logical_values_preserved": True,
        "additional_dense_weight_bytes": 0,
    }
    if qtype not in formats:
        info["native_format_supported"] = False
        return data, info
    if data.ndim != 2 or data.dtype != torch.uint8:
        raise ValueError("Native wire alignment requires a uint8 row matrix")
    n, row_bytes = map(int, data.shape)
    if not geometry_supported(qtype, n, k, 1, formats):
        info["native_geometry_supported"] = False
        return data, info
    block, block_bytes = formats[qtype]
    if row_bytes != k // block * block_bytes:
        raise ValueError("Native GGUF wire width differs from K/type")
    logical = data.numel()
    required = required_storage_bytes(qtype, n, k, formats)
    tail = required - logical
    info.update(native_format_supported=True, required_tail_quant_bytes=tail)
    if data.is_contiguous() and data.data_ptr() % 128 == 0 and tail == 0:
        return data, info
    # A 127-byte alignment reserve makes this portable to CPU fixtures and any
    # allocator. CUDA's normal aligned allocation also follows this same path.
    storage = torch.zeros(required + 127, dtype=torch.uint8, device=data.device)
    offset = (-storage.data_ptr()) % 128
    result = storage[offset : offset + logical].reshape(n, row_bytes)
    result.copy_(data)
    if result.data_ptr() % 128:
        raise RuntimeError("Native quantized storage alignment failed")
    info.update(
        copy_performed=True,
        original_GPU_input_pointer_reused=False,
        extra_resident_quant_bytes=storage.numel() - logical,
        allocation_quant_bytes=storage.numel(),
        alignment_reserve_quant_bytes=127,
        tail_owned_and_zero_initialized=True,
    )
    return result, info
