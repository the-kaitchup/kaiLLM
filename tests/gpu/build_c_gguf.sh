#!/usr/bin/env bash
# Build kaiLLM's GGUF CUDA extension (vllm_gguf_plugin/csrc: GGML MMVQ / MMQ / MoE kernels) against a kaiLLM runtime's
# torch, for this GPU's architecture; the dev plugin's setup.py recipe.  The rc5 runtime payload does not ship it.
#   bash tests/gpu/build_c_gguf.sh <runtime venv python> <output dir>     (CPU only; ~20 s for one architecture)
# Use: KAILLM_HY4_C_GGUF=<output dir>/vllm_gguf_plugin/_C_gguf.abi3.so with tests/gpu/hy4_env.sh.
set -eu
PY=${1:?usage: build_c_gguf.sh <python> <outdir>}; OUT=${2:?usage: build_c_gguf.sh <python> <outdir>}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SP=$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
[ -x "$SP/nvidia/cu13/bin/nvcc" ] && export CUDA_HOME=$SP/nvidia/cu13 && export PATH=$CUDA_HOME/bin:$PATH \
  && export LD_LIBRARY_PATH=$CUDA_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PATH=$(dirname "$PY"):$PATH
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)}
export MAX_JOBS=${MAX_JOBS:-4}
rm -rf "$OUT"; mkdir -p "$OUT/vllm_gguf_plugin"
cp -r "$ROOT/runtime-source/vllm_gguf_plugin/csrc" "$OUT/vllm_gguf_plugin/"
cat > "$OUT/setup.py" <<'PY'
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
setup(name="c_gguf", ext_modules=[CUDAExtension(
    name="vllm_gguf_plugin._C_gguf",
    sources=["vllm_gguf_plugin/csrc/torch_bindings.cpp", "vllm_gguf_plugin/csrc/gguf/gguf_kernel.cu"],
    include_dirs=["vllm_gguf_plugin/csrc", "vllm_gguf_plugin/csrc/gguf"], py_limited_api=True,
    extra_compile_args={"cxx": ["-O3", "-std=c++17"],
                        "nvcc": ["-O3", "-std=c++17", "--use_fast_math", "-DUSE_CUDA"]})],
    cmdclass={"build_ext": BuildExtension})
PY
cd "$OUT" && "$PY" setup.py build_ext --inplace > build.log 2>&1
ls "$OUT"/vllm_gguf_plugin/_C_gguf*.so
