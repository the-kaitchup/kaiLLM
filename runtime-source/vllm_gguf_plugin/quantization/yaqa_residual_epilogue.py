# SPDX-License-Identifier: Apache-2.0
"""Default-off SM120 RAW folded core plus exact split-sum/unsafe epilogue.

Keep the existing residual identity, original policy, buffers and whole-BM16
repair authoritative. Authenticate the optional binary before graph capture.
"""

import ast
import ctypes
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

ENV = "VLLM_GGUF_YAQA_REGISTER_EPILOGUE"
LIBRARY_ENV = ENV + "_LIBRARY"
METADATA_ENV = ENV + "_METADATA"
SOURCE_SHA = "318e559c66ff4cfbe173822210b6c4e23e42fa2c355d5d00239a331393c8928b"
LIBRARY_SHA = "ce49402cac4bdb8e6628b0943a755ba420ddd6aa4ab0df343743ccae89eb3059"
METADATA_SHA = "0825988d77d2919219660343fa97c0299d35fcc01b724c4f1d767bd83b1d32fa"
ORIGINAL_MMA_SOURCE_SHA = (
    "dd841ab110ba52a1729a6a4cad57f332c142bed6e682e56af401bb1fa1138b56"
)
RESIDUAL_SOURCE_SHA = "59a1290c6d0cc7168bfd8f8bbe22d42e9c653ae15d23810845dbec37180ae384"
MMA_SHA = "04ac533c2157d90dc94d2d2e660da92c25f5090b1e2580f09b1a3c56204d97d0"
NUMERICAL_JIT_SHA = "1b982c6cd0519605df0017a684638887fb048c00f42cc7e8eca58894abf00c98"
FUSED_JIT_AST_SHA = "dca5d4d3da974629df9dd7dc6a4f3969f1283a2ff0e7fc0430e478293e311c58"
PARTIAL_BYTES = 4 << 20
SOURCE = (
    Path(__file__).parents[1]
    / "native/kernels/yaqa_register_mma_residual_folded_raw.cu"
)
_BINDINGS = {}


def mode():
    value = os.getenv(ENV, "none")
    if value not in ("none", "auto"):
        raise ValueError(ENV + " requires none or auto")
    return value


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_contract(mma_source, numerical_source, kernel_source=None):
    if sha(mma_source) != MMA_SHA:
        raise ValueError("Original whole-BM16 unsafe provider differs")
    tree = ast.parse(Path(numerical_source).read_text())
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    numerical = {
        name: hashlib.sha256(
            ast.dump(node, include_attributes=False).encode()
        ).hexdigest()
        for name, node in functions.items()
        if any(ast.unparse(d).startswith("triton.jit") for d in node.decorator_list)
    }
    if (
        hashlib.sha256(json.dumps(numerical, sort_keys=True).encode()).hexdigest()
        != NUMERICAL_JIT_SHA
    ):
        raise ValueError("Original numerical JIT providers differ")
    gemm = functions["_yaqa_gemm_fp16_prepared"]
    for call in ast.walk(gemm):
        if (
            isinstance(call, ast.Call)
            and ast.unparse(call.func) == "tl.load"
            and any(
                isinstance(node, ast.Name) and node.id == "Y" for node in ast.walk(call)
            )
        ):
            raise ValueError("Original unsafe GEMM unexpectedly reads Y")
    kernel_source = kernel_source or Path(__file__).with_name(
        "yaqa_residual_epilogue_kernel.py"
    )
    kernel = next(
        node
        for node in ast.parse(Path(kernel_source).read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "fused_sum_unsafe"
    )
    if (
        hashlib.sha256(ast.dump(kernel, include_attributes=False).encode()).hexdigest()
        != FUSED_JIT_AST_SHA
    ):
        raise ValueError("Qualified split-sum/unsafe fused JIT math differs")
    return {
        "qualified_fused_JIT_AST_sha256": FUSED_JIT_AST_SHA,
        "original_unsafe_module_sha256": MMA_SHA,
        "all_16_numerical_JIT_AST_sha256": NUMERICAL_JIT_SHA,
        "unsafe_GEMM_never_reads_Y": True,
    }


@lru_cache(maxsize=8)
def metadata(library, raw):
    library, raw = Path(library).absolute(), Path(raw).absolute()
    if (
        sha(library) != LIBRARY_SHA
        or sha(raw) != METADATA_SHA
        or sha(SOURCE) != SOURCE_SHA
    ):
        raise ValueError("Qualified RAW residual source/native/metadata differs")
    identity = json.loads(raw.read_text())
    if (
        identity.get("library_sha256") != LIBRARY_SHA
        or identity.get("source_sha256") != SOURCE_SHA
        or identity.get("architecture", {}).get("compute_capability") != [12, 0]
    ):
        raise ValueError("Qualified RAW residual architecture identity differs")
    root = Path(__file__).parent
    contract = source_contract(root / "yaqa_mma.py", root / "yaqa_kernels.py")
    return {
        "library": str(library),
        "library_sha256": LIBRARY_SHA,
        "raw_metadata": str(raw),
        "raw_metadata_sha256": METADATA_SHA,
        "source_sha256": SOURCE_SHA,
        "compute_capability": [12, 0],
        "unsafe_source_contract": contract,
    }


def key_identity(key, source):
    try:
        value = json.loads(key)
    except (TypeError, ValueError):
        return None
    if (
        not isinstance(value, dict)
        or value.get("source_sha256") != source
        or not isinstance(value.get("architecture"), dict)
        or value["architecture"].get("compute_capability") != [12, 0]
    ):
        return None
    return value


@lru_cache(maxsize=1)
def _kernel():
    from . import yaqa_residual_epilogue_kernel as module

    unsafe = module._overwrite_unsafe
    gemm = unsafe.fn.__globals__["_yaqa_gemm_fp16_prepared"]
    source_contract(
        unsafe.fn.__code__.co_filename,
        gemm.fn.__code__.co_filename,
        module.fused_sum_unsafe.fn.__code__.co_filename,
    )
    return module.fused_sum_unsafe


def prepare(device, bits, original_key, residual_key):
    if mode() == "none" or type(bits) is not int or bits not in (2, 3, 4):
        return None
    original = key_identity(original_key, ORIGINAL_MMA_SOURCE_SHA)
    residual = key_identity(residual_key, RESIDUAL_SOURCE_SHA)
    if (
        original is None
        or residual is None
        or residual.get("original_mma_key_sha256")
        != hashlib.sha256(original_key.encode()).hexdigest()
    ):
        raise ValueError("RAW epilogue requires the existing qualified SM120 pair")
    import torch

    device = torch.device(device)
    if device.type != "cuda" or tuple(torch.cuda.get_device_capability(device)) != (
        12,
        0,
    ):
        return None
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Prepare RAW epilogue before graph capture")
    # Bind actual target and physical index once before capture. The existing
    # opaque op continues its original per-call identity/device validation.
    index = device.index if device.index is not None else torch.cuda.current_device()
    identity = metadata(os.environ[LIBRARY_ENV], os.environ[METADATA_ENV])
    key = (residual_key, index)
    if key not in _BINDINGS:
        native = ctypes.CDLL(identity["library"])
        call = native.yaqa_register_mma_residual_folded_raw_run
        call.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 8 + [ctypes.c_void_p]
        call.restype = ctypes.c_int
        _BINDINGS[key] = {
            **identity,
            "device_index": index,
            "original_key": original_key,
            "residual_key": residual_key,
            "library_object": native,
            "call": call,
        }
    _kernel()
    return key


def eligible(rows, m, n, bits, policy, device_index, binding, original_key):
    if any(type(value) is not int for value in (rows, m, n, bits)) or len(policy) != 4:
        return False
    warps, ring, shared, splits = policy
    return (
        2 <= rows <= 8
        and bits in (2, 3, 4)
        and min(m, n) >= 32
        and m % 16 == 0
        and n % 16 == 0
        and n <= 65536
        and warps == 4
        and ring in (4, 8)
        and shared == 1
        and type(splits) is int
        and 1 < splits <= 64
        and splits & (splits - 1) == 0
        and rows * m * splits * 4 <= PARTIAL_BYTES
        and device_index == binding["device_index"]
        and original_key == binding["original_key"]
    )


def maybe_apply(
    source,
    high,
    low,
    safe,
    trellis,
    lut,
    signed_half_lut,
    m,
    n,
    bits,
    original_key,
    residual_key,
    policy,
):
    binding = _BINDINGS.get((residual_key, source.device.index))
    if binding is None or not eligible(
        source.shape[0], m, n, bits, policy, source.device.index, binding, original_key
    ):
        return None
    # This private hook is reached only after the unchanged opaque op validates
    # every dtype/shape/device/alignment and original/residual loaded identity.
    import torch
    import triton

    rows = source.shape[0]
    warps, ring, shared, splits = policy
    partial = torch.empty((splits, rows, m), device=source.device, dtype=torch.float32)
    output = torch.empty((rows, m), device=source.device, dtype=torch.float32)
    with torch.cuda.device(source.device):
        code = binding["call"](
            high.data_ptr(),
            low.data_ptr(),
            trellis.data_ptr(),
            signed_half_lut.data_ptr(),
            partial.data_ptr(),
            output.data_ptr(),
            m,
            n,
            rows,
            bits,
            warps,
            ring,
            shared,
            splits,
            torch.cuda.current_stream(source.device).cuda_stream,
        )
        if code:
            raise RuntimeError("Qualified RAW residual launch failed: " + str(code))
        _kernel()[(triton.cdiv(rows, 16), triton.cdiv(m, 32))](
            partial,
            source,
            high,
            safe,
            trellis,
            lut,
            output,
            M=m,
            N=n,
            BATCH=rows,
            BITS=bits,
            SPLITS=splits,
            BM=16,
            BN=32,
            BK=64,
            num_warps=4,
            num_stages=2,
        )
    return output


def runtime_marker(target=None):
    active = mode() == "auto" and tuple(target or ()) == (12, 0)
    identity = (
        metadata(os.environ[LIBRARY_ENV], os.environ[METADATA_ENV]) if active else {}
    )
    root = Path(__file__).parent
    paths = [
        Path(__file__),
        root / "yaqa_residual_epilogue_kernel.py",
        root / "yaqa_residual_mma.py",
        root / "yaqa_mma.py",
        root / "yaqa_kernels.py",
        root.parent / "plugin.py",
        root.parent / "__init__.py",
        SOURCE,
    ]
    native = (
        {identity["library"]: LIBRARY_SHA, identity["raw_metadata"]: METADATA_SHA}
        if identity
        else {}
    )
    return {
        "mode": mode(),
        "active_for_target": active,
        "scope": "SM120 QTIP2..4 physicalM2..8; W4/R4|8/shared1; S>1; partial<=4MiB",
        "S1_SM86_and_incompatible_resources_keep_original59a": True,
        "existing_residual_admission_only_no_other_head_backend_changed": True,
        "original_policy_QUERY4_and_unsafe_JIT_object_preserved": True,
        "positive_zero_ascending_add_rn_then_original_whole_BM16_repair": True,
        "weight_cache_added_bytes": 0,
        "partial_input_immutable_fresh_Y": True,
        "new_RAW_folded_binary_used_when_selected": True,
        "native_identity": identity,
        "native_sources": native,
        "loaded_bindings": [
            {k: v for k, v in binding.items() if k not in ("library_object", "call")}
            for binding in _BINDINGS.values()
        ],
        "per_token_CC_hash_environment_query_added": False,
        "sources": {str(path): sha(path) for path in paths},
    }
