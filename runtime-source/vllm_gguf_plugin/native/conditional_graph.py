# SPDX-License-Identifier: Apache-2.0
"""Build the packaged CUDA adapter for default-off LFM3 conditional decoding."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

SOURCE = Path(__file__).with_name("kernels") / "ngram_conditional.cu"
SOURCE_SHA256 = "5d87ba76f83c6763a362893725341d1c03ff01897ba43fe07aff341f4b7f136a"
BUILD_CONTRACT = "sm86-ngram-conditional-abi2-seqlen-counters-childpreflight-v2"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def build():
    """Build without opening a CUDA device and return the complete identity."""
    if sha256(SOURCE) != SOURCE_SHA256:
        raise ValueError("Packaged conditional CUDA source differs from ABI2")
    nvcc = shutil.which("nvcc")
    if nvcc is None and os.getenv("CUDA_HOME"):
        nvcc = str(Path(os.environ["CUDA_HOME"]) / "bin/nvcc")
    if not nvcc or not Path(nvcc).is_file():
        raise RuntimeError("Conditional decoding requires the CUDA13 nvcc toolkit")
    version = subprocess.check_output([nvcc, "--version"], text=True)
    release = re.search(r"release (\d+)\.(\d+)", version)
    if release is None or int(release[1]) < 13:
        raise RuntimeError("The packaged conditional adapter requires CUDA13 or newer")
    from vllm_gguf_plugin.quantization.yaqa_cuda import build_cuda_library

    # This exact contract/source resolves the already-validated content-addressed
    # library when the compiler identity matches; other toolkits get a new key.
    identity = build_cuda_library(
        SOURCE, BUILD_CONTRACT, "ngram_conditional_engine", architecture=(8, 6)
    )
    return {
        **identity,
        "abi_version": 2,
        "build_helper_sha256": sha256(__file__),
        "packaged_source": str(SOURCE.resolve()),
        "device_used_for_build": False,
    }


def speculative_config(identity):
    """Return the narrow public config; ordinary batch serving omits it."""
    return {
        "method": "ngram_gpu",
        "num_speculative_tokens": 3,
        "prompt_lookup_min": 2,
        "prompt_lookup_max": 4,
        "ngram_gpu_match_only": True,
        "ngram_gpu_match_only_library": identity["library"],
        "rejection_sample_method": "standard",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--library-path", action="store_true")
    output.add_argument("--speculative-config", action="store_true")
    args = parser.parse_args()
    identity = build()
    if args.library_path:
        print(identity["library"])
    elif args.speculative_config:
        print(json.dumps(speculative_config(identity), separators=(",", ":")))
    else:
        print(json.dumps(identity, indent=2))


if __name__ == "__main__":
    main()
