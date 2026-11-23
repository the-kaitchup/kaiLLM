#!/usr/bin/env bash
# OpenAI-compatible kaiLLM server for an Inkling 2 Mini GGUF from this branch over a kaiLLM release runtime (vLLM's
# serve CLI with the branch overlay, the flags kaiLLM's launcher passes for inkling2_loop: decode-only CUDA graphs,
# async scheduling, FP8 KV).  The GGUF CUDA extension (tests/gpu/build_c_gguf.sh) is used when C_GGUF points to it.
# Sampling defaults: the GGUF's recommended ones (general.sampling.*, as llama.cpp; GENCFG=vllm for vLLM's).
#   MODEL=x.gguf PORT=8801 NS=32 KVB=10000000000 GMU=0.24 C_GGUF=.../_C_gguf.abi3.so \
#     bash tests/gpu/inkling_serve.sh <runtime venv python> > server.log 2>&1
set -u
PY0=${1:?usage: inkling_serve.sh <runtime venv python>}
export KAILLM_HY4_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
[ -n "${C_GGUF:-}" ] && export KAILLM_HY4_C_GGUF=$C_GGUF
source "$KAILLM_HY4_ROOT/tests/gpu/hy4_env.sh" "$PY0"
NS=${NS:-32}
CAP=$("$PY" -c "n=$NS; print(sorted(set(range(1, min(n, 8) + 1)) | set(range(8, n + 1, 4)) | {n}))")
exec "$PY" -m vllm.entrypoints.cli.main serve "${MODEL:?set MODEL}" --host 127.0.0.1 --port ${PORT:-8801} \
  --served-model-name ${NAME:-inkling} --quantization gguf --load-format gguf --dtype bfloat16 \
  --kv-cache-dtype ${KVDT:-fp8_e4m3} --mamba-cache-dtype auto --max-num-seqs $NS \
  --max-num-batched-tokens ${NBT:-4096} --max-model-len ${MAXLEN:-262144} --no-enable-prefix-caching \
  --compilation-config "{\"mode\": 0, \"cudagraph_mode\": \"FULL_DECODE_ONLY\", \"cudagraph_capture_sizes\": $CAP, \"max_cudagraph_capture_size\": $NS}" \
  --additional-config '{"cudagraph_specialize_prefill": false}' --async-scheduling --long-prefill-token-threshold 0 \
  --gpu-memory-utilization ${GMU:-0.24} --kv-cache-memory-bytes ${KVB:-10000000000} \
  --generation-config ${GENCFG:-auto}
