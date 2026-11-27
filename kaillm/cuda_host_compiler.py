"""Resolve CUDA 13.0 host compiler compatibility for the installed entrypoint."""

import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys


HOST_OVERRIDES = ("NVCC_CCBIN", "CC", "CXX", "CUDAHOSTCXX", "PYTORCH_NVCC")
NVCC_FLAG_ENV = ("NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS")
GCC14_ENV = (
    "export NVCC_CCBIN=/usr/bin/g++-14 CC=/usr/bin/gcc-14 "
    "CXX=/usr/bin/g++-14 CUDAHOSTCXX=/usr/bin/g++-14"
)


def _probe(command, environment):
    try:
        completed = subprocess.run(
            command, stdin=subprocess.DEVNULL, capture_output=True, check=True,
            text=True, errors="replace", timeout=5, env={**environment, "LC_ALL": "C"},
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return completed.stdout


def _nvcc(environment, cuda_home):
    home = cuda_home or environment.get("CUDA_HOME")
    if home is None:
        # Match the frozen launcher before falling back to a toolkit on PATH.
        home = next((str(Path(path) / "nvidia/cu13") for path in sys.path
                     if (Path(path) / "nvidia/cu13/bin/nvcc").is_file()), None)
    search = environment.get("PATH", os.defpath)
    if home:
        search = str(Path(home).absolute() / "bin") + os.pathsep + search
    return shutil.which("nvcc", path=search)


def _gnu_major(compiler, language, environment):
    macros = _probe([compiler, "-dM", "-E", "-x", language, "-"], environment)
    # Clang and other compilers may advertise GCC compatibility macros.
    if re.search(r"^#define (?:__clang__|__NVCOMPILER|__INTEL_COMPILER)\b", macros, re.MULTILINE):
        return None
    match = re.search(r"^#define __GNUC__ (\d+)$", macros, re.MULTILINE)
    return int(match[1]) if match else None


def _has_overrides(environment):
    if any(name in environment for name in HOST_OVERRIDES):
        return True
    for name in NVCC_FLAG_ENV:
        try:
            flags = shlex.split(environment.get(name, ""))
        except ValueError:
            return True
        if any(flag.split("=", 1)[0] in ("-ccbin", "--compiler-bindir", "-optf", "--options-file")
               for flag in flags):
            return True
    return False


def check_host_compiler(environment, *, cuda_home=None):
    """Return a diagnostic and process-local settings for CUDA 13.0 and GCC 16+."""
    nvcc = _nvcc(environment, cuda_home)
    if nvcc is None:
        return None
    version = _probe([nvcc, "--version"], environment)
    if not re.search(r"\brelease\s+13\.0(?:\s|,|$)", version):
        return None

    search = environment.get("PATH", os.defpath)
    gcc = shutil.which("gcc", path=search)
    major = _gnu_major(gcc, "c", environment) if gcc else None
    if major is None or major <= 15:
        return None
    problem = f"CUDA 13.0 host compilation supports GCC up to 15. The default gcc is GCC {major}."
    remedy = ("Install a matching GCC and G++ compatibility pair. For GCC 14, set the following "
              "in the shell running kaiLLM and adjust the paths to match the installation.\n" + GCC14_ENV)
    report = {"nvcc": nvcc, "nvcc_release": "13.0", "default_gcc_major": major, "environment": {}}

    if _has_overrides(environment):
        return dict(report, status="explicit", message=problem + " The configured compiler overrides are kept."
                    " If a local build reports an unsupported GNU version, use a compatible pair.\n" + remedy)

    for version in (15, 14, 13):
        cc = shutil.which(f"gcc-{version}", path=search)
        cxx = shutil.which(f"g++-{version}", path=search)
        if (cc and cxx and _gnu_major(cc, "c", environment) == version
                and _gnu_major(cxx, "c++", environment) == version):
            cc, cxx = str(Path(cc).absolute()), str(Path(cxx).absolute())
            return dict(report, status="selected", gcc_major=version,
                        message=f"{problem} kaiLLM will use the installed GCC {version} pair for local builds.",
                        environment={"NVCC_CCBIN": cxx, "CC": cc, "CXX": cxx, "CUDAHOSTCXX": cxx})
    return dict(report, status="needs_compatible_gcc", message=problem + " " + remedy)
