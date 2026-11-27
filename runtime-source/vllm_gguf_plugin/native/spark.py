"""Identify the native Linux/AArch64 Humming and vLLM host components for GB10."""
import hashlib
import json
import platform
from pathlib import Path


HUMMING = tuple("third-party/humming/_native/aarch64/" + name for name in (
    "libhumming_launcher.so", "_device_info.abi3.so", "libcubinpatch.so", "nvrtc_compile"))
CORE = "compatible-vllm/core/vllm"
CORE_COMMIT = "f3871c3cb"
RECEIPT = "deployment/spark-native.json"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def arm_elf(path):
    with path.open("rb") as stream:
        header = stream.read(20)
    # ELF64, little endian, EM_AARCH64.
    return header.startswith(b"\x7fELF\x02\x01") and header[18:20] == b"\xb7\x00"


def native_files(root):
    core = root / CORE
    files = {str(p.relative_to(root)) for p in core.rglob("*.so*") if p.is_file()}
    if (core / "vllm-rs").is_file():
        files.add(f"{CORE}/vllm-rs")
    for component in ("_C_stable_libtorch", "_moe_C_stable_libtorch", "cumem_allocator"):
        if not any(p.startswith(f"{CORE}/{component}.") for p in files):
            raise ValueError(f"The ARM64 vLLM build does not contain {component}")
    shared = core.parent / "vllm.libs"
    files.update(str(p.relative_to(root)) for p in shared.rglob("*.so*") if p.is_file())
    return sorted(files.union(HUMMING))


def verify_runtime(root):
    if platform.system() != "Linux" or platform.machine().lower() not in ("aarch64", "arm64"):
        raise ValueError("The DGX Spark native runtime requires Linux AArch64")
    root = Path(root)
    record = json.loads((root / RECEIPT).read_text())
    if record.get("vllm_commit") != CORE_COMMIT or record.get("compute_capability") != [12, 1]:
        raise ValueError("The native build does not match the GB10 vLLM ABI")
    expected = record.get("native_sha256", {})
    files = native_files(root)
    if set(files) != set(expected):
        raise ValueError("The installed ARM64 extension inventory changed")
    for name in files:
        path = root / name
        if not arm_elf(path) or digest(path) != expected[name]:
            raise ValueError("Invalid native ARM64 ELF or checksum: " + name)
    manifest = root / "third-party/humming/_native/aarch64/manifest.json"
    if digest(manifest) != record.get("humming_manifest_sha256"):
        raise ValueError("Humming native source/build manifest changed")
    return record
