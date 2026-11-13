# SPDX-License-Identifier: Apache-2.0
"""Shared native GGUF wire geometry, independent of models and request batches.

Backends supply their source-verified qtype formats, CUDA capability and library
identity. This module never infers a matrix's operator role from its dimensions,
changes numerical guards, or treats a small constructor warm family as an
admission list. A geometry match still requires the backend's real C ABI
workspace/capability query before native execution.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

INT32_MAX = 2**31 - 1
FormatMap = Mapping[int, tuple[int, int]]


def geometry_supported(
    qtype: int, n: int, k: int, rows: int, formats: FormatMap
) -> bool:
    """Validate the native C ABI integer and GGUF block contracts."""
    if qtype not in formats:
        return False
    if any(
        type(value) is not int or not 0 < value <= INT32_MAX for value in (n, k, rows)
    ):
        return False
    block, block_bytes = formats[qtype]
    return block > 0 and block_bytes > 0 and k % block == 0


def required_storage_bytes(qtype: int, n: int, k: int, formats: FormatMap) -> int:
    """GGML padding for K's 512-element load and N's four-row load groups."""
    if not geometry_supported(qtype, n, k, 1, formats):
        raise ValueError("Unsupported native GGUF wire geometry")
    block, block_bytes = formats[qtype]
    row_bytes = k // block * block_bytes
    tail_elements = (-k) % 512 + ((-n) % 4) * k
    return n * row_bytes + tail_elements // block * block_bytes


def storage_supported(
    qtype: int,
    packed_shape: tuple[int, int],
    k: int,
    pointer: int,
    available_bytes: int,
    formats: FormatMap,
) -> bool:
    if len(packed_shape) != 2:
        return False
    n, row_bytes = packed_shape
    if not geometry_supported(qtype, n, k, 1, formats):
        return False
    block, block_bytes = formats[qtype]
    return (
        row_bytes == k // block * block_bytes
        and pointer > 0
        and pointer % 128 == 0
        and available_bytes >= required_storage_bytes(qtype, n, k, formats)
    )


@dataclass(frozen=True)
class NativeBackend:
    """The verified backend identity; tensor capabilities are queried separately."""

    compute_capability: tuple[int, int]
    library_sha256: str
    source_manifest_sha256: str

    def matches(
        self,
        compute_capability: tuple[int, int],
        library_sha256: str,
        source_manifest_sha256: str,
    ) -> bool:
        return (
            tuple(compute_capability) == self.compute_capability
            and library_sha256 == self.library_sha256
            and source_manifest_sha256 == self.source_manifest_sha256
        )


def capability_supported(
    backend: NativeBackend,
    *,
    actual_compute_capability: tuple[int, int],
    actual_library_sha256: str,
    actual_source_manifest_sha256: str,
    qtype: int,
    n: int,
    k: int,
    rows: int,
    formats: FormatMap,
    workspace_query: Callable[[int, int, int, int], int],
) -> bool:
    """Call the backend's true typed-row capability, without a model whitelist.

    `workspace_query(qtype, n, k, rows)` is the already prepared backend's host
    C ABI query. It must be warmed/cached before graph capture as appropriate;
    this helper does not synchronize CUDA or create a new CUDA context.
    """
    return (
        backend.matches(
            actual_compute_capability,
            actual_library_sha256,
            actual_source_manifest_sha256,
        )
        and geometry_supported(qtype, n, k, rows, formats)
        and workspace_query(qtype, n, k, rows) > 0
    )


def warm_rows() -> tuple[int, ...]:
    """Bounded tile families only; never interpreted as supported tensor rows."""
    return (1, 4, 16)
