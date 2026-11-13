# SPDX-License-Identifier: Apache-2.0
"""Diagnostic Qwen3.5 beta/alpha projections using original native float GEMMs.

Installed only by the experimental current-GGML direct adapter. The GGUF
prepare hook replaces the linear method before loading, while matrix copies
are created in process_weights_after_loading after the original F32 data have
arrived. MMF/cuBLAS keep separate N16 matrices. SM86 MMVF can merge rows without
changing its per-row block size, dot products, or reduction order.
Source F16 shards (as in the 2B GGUF) are restored to their original storage
type after loading into the F32 model, preserving native dtype-specific GEMMs.
"""


def _original_float_parts(weight, sizes, source_dtypes):
    import torch

    if len(sizes) != 2 or sizes[0] != sizes[1] or sum(sizes) != weight.shape[0]:
        raise ValueError("Expected separate equally sized beta/alpha matrices")
    if set(source_dtypes) != {0, 1} or any(
        dtype not in (torch.float32, torch.float16) for dtype in source_dtypes.values()
    ):
        raise ValueError("Both original beta/alpha storage dtypes must be F32/F16")
    return [
        part.detach().to(source_dtypes[index]).contiguous()
        for index, part in enumerate(weight.split(sizes, dim=0))
    ]


def _register_beta_alpha_mmvf_weight(layer, merged):
    # Reload copies processed persistent buffers into their original storage,
    # preserving the tensor pointers already referenced by CUDA graphs.
    layer.register_buffer("_gguf_native_ba_mmvf_weight", merged, persistent=True)


def install_current_float_adapter():
    import torch
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    from vllm_gguf_plugin.weights_adapter.qwen35 import Qwen35GGUFAdapter

    from .current_ggml_direct import CurrentGGMLFloatLinear
    from .native_beta_alpha_mmvf import beta_alpha

    if getattr(Qwen35GGUFAdapter, "_parity_current_float", False):
        return

    class OriginalBetaAlphaMethod(UnquantizedLinearMethod):
        def __init__(self):
            super().__init__()
            self.source_dtypes = {}

        def process_weights_after_loading(self, layer):
            if layer.weight.dtype != torch.float32 or not layer.weight.is_cuda:
                raise ValueError(
                    "Native beta/alpha diagnostic requires CUDA F32 weights"
                )
            if layer.tp_size != 1:
                raise ValueError("Native beta/alpha diagnostic currently requires TP=1")
            # vLLM stores beta first and alpha second, matching the Qwen3.5
            # checkpoint mapping. Native MMF/cuBLAS keep these N16 shapes.
            layer.current_ggml_ba = torch.nn.ModuleList(
                CurrentGGMLFloatLinear(part)
                for part in _original_float_parts(
                    layer.weight, layer.output_sizes, self.source_dtypes
                )
            )
            parts = list(layer.current_ggml_ba)
            merged = None
            if (
                len(parts) == 2
                and all(part.shape[0] == 16 for part in parts)
                and parts[0].weight.dtype == parts[1].weight.dtype
                and torch.cuda.get_device_capability(layer.weight.device) == (8, 6)
            ):
                merged = torch.cat([part.weight for part in parts]).contiguous()
            _register_beta_alpha_mmvf_weight(layer, merged)

        def apply(self, layer, x, bias=None):
            if x.dtype != torch.float32:
                raise ValueError(
                    "Native beta/alpha diagnostic requires FP32 activations"
                )
            merged = layer._gguf_native_ba_mmvf_weight
            if merged is None:
                result = torch.cat([part(x) for part in layer.current_ggml_ba], dim=-1)
            else:
                result = beta_alpha(
                    x.reshape(-1, merged.shape[1]).contiguous(), merged
                ).reshape(*x.shape[:-1], 32)
            return result if bias is None else result + bias

    original_prepare = Qwen35GGUFAdapter.prepare_model

    def prepare(self, model):
        original_prepare(self, model)
        self._parity_current_float_layers = {}
        for name, layer in model.named_modules():
            if not name.endswith(".linear_attn.in_proj_ba"):
                continue
            if not isinstance(layer.quant_method, UnquantizedLinearMethod):
                raise TypeError("Expected an unquantized beta/alpha linear method")
            if layer.weight.dtype != torch.float32:
                raise ValueError("Use dtype=float32 with native beta/alpha diagnostics")
            layer.quant_method = OriginalBetaAlphaMethod()
            self._parity_current_float_layers[name] = layer.quant_method

    original_weights = Qwen35GGUFAdapter.prepare_weights

    def weights(self, model_config):
        for name, weight in original_weights(self, model_config):
            for suffix, index in ((".in_proj_b.weight", 0), (".in_proj_a.weight", 1)):
                if not name.endswith(suffix):
                    continue
                target = name.removesuffix(suffix) + ".in_proj_ba"
                method = self._parity_current_float_layers.get(target)
                if method is None:
                    raise ValueError(f"No native beta/alpha destination for {name}")
                if weight.dtype not in (torch.float32, torch.float16):
                    raise ValueError(f"Unsupported original beta/alpha dtype: {name}")
                method.source_dtypes[index] = weight.dtype
            yield name, weight

    Qwen35GGUFAdapter.prepare_model = prepare
    Qwen35GGUFAdapter.prepare_weights = weights
    Qwen35GGUFAdapter._parity_current_float = True
