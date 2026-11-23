"""Dev bootstrap for Hy4 tests: kaiLLM plugin + vllm-overrides from the hy4 worktree.

Put this directory on PYTHONPATH with KAILLM_HY4_ROOT=<repository root>; it runs in every
interpreter (vLLM's worker processes included):
- the branch's plugin sources and GGUF reader (F8_E4M3) go first on sys.path;
- the new Hy4 modules are added to vLLM's model / config packages;
- the compiled GGML extension `vllm_gguf_plugin._C_gguf` is not in the source tree: it is taken from
  the installed kaiLLM plugin (or KAILLM_HY4_C_GGUF=<path to _C_gguf*.so>), loaded lazily, so the
  GPU tests run kaiLLM's CUDA kernels instead of the Triton fallbacks. Set VLLM_GGUF_USE_CUDA=0 on
  GPUs the extension was not built for (the release targets sm86 and sm120).
"""
import glob
import importlib.abc
import importlib.util
import os
import sys

_root = os.environ.get("KAILLM_HY4_ROOT")
if _root:
    _so = os.environ.get("KAILLM_HY4_C_GGUF")
    if not _so:
        try:
            _spec = importlib.util.find_spec("vllm_gguf_plugin")   # the installed plugin, before shadowing
            _dirs = list(_spec.submodule_search_locations or []) if _spec else []
            _hits = [p for d in _dirs for p in glob.glob(os.path.join(d, "_C_gguf*.so"))]
            _so = _hits[0] if _hits else None
        except Exception:  # pragma: no cover
            _so = None

    sys.path.insert(0, os.path.join(_root, "runtime-source"))
    sys.path.insert(0, os.path.join(_root, "runtime-source", "gguf-reader"))

    if _so and not glob.glob(os.path.join(_root, "runtime-source", "vllm_gguf_plugin", "_C_gguf*.so")):
        class _CGGUFFinder(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path, target=None):
                if name == "vllm_gguf_plugin._C_gguf":
                    return importlib.util.spec_from_file_location(name, _so)
                return None

        sys.meta_path.insert(0, _CGGUFFinder())

    try:
        import vllm.model_executor.models as _models
        import vllm.transformers_utils.configs as _configs
        _ov = os.path.join(_root, "runtime-source", "vllm-overrides", "vllm")
        for _pkg, _sub in ((_models, ("model_executor", "models")), (_configs, ("transformers_utils", "configs"))):
            _p = os.path.join(_ov, *_sub)
            if _p not in _pkg.__path__:
                _pkg.__path__.append(_p)
    except Exception as _e:  # pragma: no cover
        print(f"kaillm-hy4 sitecustomize: {_e}", file=sys.stderr)
