# Third-party notices

kaiLLM's own code is licensed under Apache-2.0, like vLLM. Third-party
components retain their existing copyright and license terms.

- **vLLM:** Apache-2.0. The compatible base and retained copyright notices are
  included in the runtime wheel; required override source is in `runtime-source`.
- **llama.cpp / ggml:** MIT. The GGUF reader and selected native decoding and
  arithmetic portions use this work. License texts are in `vendor/licenses`.
- **Humming kernels and tabulate:** their original licenses are retained under
  `runtime-source/third-party`, and with the runtime distribution.
- **PyTorch, Triton, NVIDIA CUDA and other installed dependencies:** their own
  licenses apply. The installer retrieves pinned upstream distributions.
- **GNU C/C++ runtime libraries:** their original notices are retained with the
  runtime. The complete release includes matching corresponding source archives
  under `vendor/corresponding-source`.

Support for a quantization file format does not bundle its quantizer. The
Cornell YAQA quantization repository is not included in this runtime. Model
weights and their model cards have separate licenses; the engine's Apache-2.0
license does not change those terms.
