# SPDX-License-Identifier: Apache-2.0
"""Probe CUDA/host-header declarations without editing the installed toolkit."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

CONTRACT = "stock-first-rsqrt-declaration-probe-v1"
PROBE = """#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>
__global__ void header_probe(float* y, const float* x) {
    y[0] = rsqrtf(x[0]);
}
"""
FLAGS = ("-O3", "-std=c++17", "-Xcompiler=-fPIC", "-gencode=arch=compute_86,code=sm_86")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for value in iter(lambda: stream.read(8 << 20), b""):
            h.update(value)
    return h.hexdigest()


def run(command):
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        env={**os.environ, "LC_ALL": "C", "LANG": "C"},
    )


def compiler_identity(nvcc):
    version = run([str(nvcc), "--version"])
    if version.returncode:
        raise RuntimeError("Cannot query the selected CUDA compiler")
    drivers = []
    names = [os.getenv("NVCC_CCBIN", ""), "g++", "gcc"]
    for variable in ("NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS"):
        options = shlex.split(os.getenv(variable, ""))
        for i, option in enumerate(options):
            if option in ("-ccbin", "--compiler-bindir") and i + 1 < len(options):
                names.append(options[i + 1])
            elif option.startswith(("-ccbin=", "--compiler-bindir=")):
                names.append(option.split("=", 1)[1])
    seen = set()
    for name in filter(None, names):
        path = Path(name)
        if path.is_dir():
            path = path / "g++"
        elif not path.is_file():
            located = shutil.which(name)
            if located is None:
                continue
            path = Path(located)
        path = path.resolve()
        if path in seen:
            continue
        seen.add(path)
        info = {"path": str(path), "sha256": digest(path)}
        implementation = run([str(path), "-print-prog-name=cc1plus"])
        cc1 = Path(implementation.stdout.strip())
        if implementation.returncode == 0 and cc1.is_file():
            info["cc1plus"] = {"path": str(cc1.resolve()), "sha256": digest(cc1)}
        drivers.append(info)
    return {
        "invocation": str(Path(nvcc).absolute()),
        "path": str(Path(nvcc).resolve()),
        "sha256": digest(nvcc),
        "version": version.stdout.strip(),
        "host_compilers": drivers,
        "flags_environment": {
            k: v
            for k, v in os.environ.items()
            if k.startswith("NVCC_") or k == "CUDAHOSTCXX"
        },
    }


def dependency_identity(nvcc, include, work):
    source = work / "dependency-probe.cu"
    source.write_text(PROBE)
    result = run([str(nvcc), *FLAGS, "-I", str(include), "-M", str(source)])
    if result.returncode:
        raise RuntimeError("CUDA header dependency probe failed:\n" + result.stderr)
    tokens = shlex.split(result.stdout.replace("\\\n", " "))
    files = sorted(
        {
            str(Path(token).resolve())
            for token in tokens[1:]
            if Path(token).is_file() and Path(token).resolve() != source.resolve()
        }
    )
    if not files:
        raise RuntimeError("CUDA dependency probe returned no readable headers")
    identities = {path: digest(path) for path in files}
    aggregate = hashlib.sha256(
        json.dumps(identities, sort_keys=True).encode()
    ).hexdigest()
    return identities, aggregate


def patched_declarations(original):
    pattern = re.compile(
        r"\b(rsqrt\s*\(\s*double\s+x\s*\)|rsqrtf\s*\(\s*float\s+x\s*\))\s*;"
    )
    patched, count = pattern.subn(r"\1 noexcept(true);", original)
    if count == 0:
        raise RuntimeError(
            "The failed rsqrt declaration probe has no known declarations to patch"
        )
    return patched, count


def link_headers(source, destination, ancestors=()):
    """Real private directories; only files point into the read-only SDK."""
    source = Path(source).resolve()
    if source in ancestors:
        raise RuntimeError("CUDA include tree contains a directory-symlink cycle")
    destination.mkdir()
    for item in source.iterdir():
        target = destination / item.name
        if item.is_dir():
            link_headers(item, target, (*ancestors, source))
        else:
            target.symlink_to(item.resolve())


def compile_probe(nvcc, include, cache, label):
    source = cache / "probe.cu"
    source.write_text(PROBE)
    output = cache / (label + ".o")
    command = [
        str(nvcc),
        *FLAGS,
        "-I",
        str(include),
        "-c",
        str(source),
        "-o",
        str(output),
    ]
    result = run(command)
    log = cache / (label + ".log")
    log.write_text(result.stdout + result.stderr)
    output.unlink(missing_ok=True)
    return result, {
        "command": command,
        "returncode": result.returncode,
        "log": str(log),
        "log_sha256": digest(log),
    }


@lru_cache(maxsize=8)
def select_cuda_headers(nvcc, cuda_home):
    """Snapshot a process's toolchain; disk choices include actual header hashes."""
    # Preserve the caller's nvcc invocation path: SDK symlinks/wrappers can
    # determine where nvcc finds its profile and device compiler tools.
    nvcc, cuda_home = Path(nvcc).absolute(), Path(cuda_home).resolve()
    include = cuda_home / "include"
    compiler = compiler_identity(nvcc)
    cache_root = Path(os.getenv("XDG_CACHE_HOME", str(Path.home() / ".cache")))
    root = cache_root / "vllm_gguf_plugin/cuda_header_probe"
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="dependencies-", dir=root) as temporary:
        dependencies, headers_hash = dependency_identity(nvcc, include, Path(temporary))
    inputs = {
        "contract": CONTRACT,
        "selector_source_sha256": digest(__file__),
        "compiler": compiler,
        "headers_sha256": headers_hash,
        "probe_source_sha256": hashlib.sha256(PROBE.encode()).hexdigest(),
        "flags": FLAGS,
    }
    key = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    cache = root / key
    cache.mkdir(exist_ok=True)
    with (cache / "probe.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest = cache / "choice.json"
        if manifest.exists():
            saved = json.loads(manifest.read_text())
            selected = Path(saved["include_directory"])
            if selected.is_dir() and (
                saved["identity"]["strategy"] == "stock"
                or digest(selected / "crt/math_functions.h")
                == saved["identity"]["patched_math_sha256"]
            ):
                return saved
            # Only our regenerable private tree is removed; symlinks are never
            # followed into the installed toolkit.
            if selected == cache / "include":
                if selected.is_symlink():
                    selected.unlink()
                elif selected.exists():
                    shutil.rmtree(selected)
            manifest.unlink()
        stock, stock_record = compile_probe(nvcc, include, cache, "stock")
        strategy, chosen, patched_hash = "stock", include, None
        proof = [stock_record]
        if stock.returncode:
            diagnostic = (stock.stdout + stock.stderr).lower()
            if "rsqrt" not in diagnostic or not (
                "exception" in diagnostic or "noexcept" in diagnostic
            ):
                raise RuntimeError(
                    "Stock CUDA declaration probe failed for another reason; "
                    f"see {cache / 'stock.log'}"
                )
            math_source = include / "crt/math_functions.h"
            patched, count = patched_declarations(math_source.read_text())
            chosen = cache / "include"
            if chosen.exists():
                if chosen.is_symlink():
                    chosen.unlink()
                else:
                    shutil.rmtree(chosen)
            link_headers(include, chosen)
            header = chosen / "crt/math_functions.h"
            if not header.parent.resolve().is_relative_to(chosen.resolve()):
                raise RuntimeError(
                    "Refusing to write outside the private CUDA-header overlay"
                )
            header.unlink()
            header.write_text(patched)
            patched_hash = digest(header)
            checked, patched_record = compile_probe(nvcc, chosen, cache, "noexcept")
            proof.append(patched_record)
            if checked.returncode:
                raise RuntimeError(
                    "Private CUDA declaration repair did not compile; "
                    f"see {cache / 'noexcept.log'}"
                )
            strategy = "rsqrt-noexcept"
        identity = {
            "contract": CONTRACT,
            "fingerprint": key,
            "strategy": strategy,
            "headers_sha256": headers_hash,
            "patched_math_sha256": patched_hash,
        }
        result = {
            "identity": identity,
            "include_directory": str(chosen),
            "metadata_file": str(manifest),
            "inputs": inputs,
            "dependencies": dependencies,
            "probes": proof,
        }
        if strategy != "stock":
            result["patched_declarations"] = count
        temporary = manifest.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(manifest)
        return result
