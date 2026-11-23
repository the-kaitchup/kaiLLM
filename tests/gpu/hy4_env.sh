# Environment for the Hy4 GPU tests, as kaiLLM's launcher (deployment/serve.py) sets it up.
#   source tests/gpu/hy4_env.sh <python>     # dev venv python, or a kaiLLM release runtime's venv python
# - release runtime (site-packages/kaillm/_runtime): its vLLM core, GGUF reader and third-party
#   packages (humming) on PYTHONPATH, behind this branch's overlay (tests/gpu/site);
# - pip CUDA 13 bundle (site-packages/nvidia/cu13) as CUDA_HOME, so humming's NVRTC finds
#   libnvrtc-builtins.
PY=${1:?usage: source tests/gpu/hy4_env.sh <python>}
export KAILLM_HY4_ROOT=${KAILLM_HY4_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
_sp=$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
_rt=$_sp/kaillm/_runtime
if [ -d "$_rt/compatible-vllm/core" ]; then
  export PYTHONPATH="$KAILLM_HY4_ROOT/tests/gpu/site:$_rt/deployment:$_rt:$_rt/compatible-vllm/core:$_rt/gguf-reader:$_rt/third-party"
else
  export PYTHONPATH="$KAILLM_HY4_ROOT/tests/gpu/site"
fi
if [ -d "$_sp/nvidia/cu13/lib" ]; then
  export CUDA_HOME="$_sp/nvidia/cu13"
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export PY
unset _sp _rt
