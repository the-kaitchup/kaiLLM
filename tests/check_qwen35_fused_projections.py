# SPDX-License-Identifier: Apache-2.0
"""Compare packed Qwen3.5 GDN projections with llama.cpp's reference GGML decoder.

    python tests/check_qwen35_fused_projections.py checkpoint.gguf /path/to/libggml-base.so
"""

import argparse
import ctypes
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import gguf
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime-source"))

from vllm_gguf_plugin.quantization.yaqa_linear import _ordinary_tensor, _permute_stored, configure_yaqa_loading
from vllm_gguf_plugin.qwen35_metadata import qwen35_config_dict, read_gguf_metadata
from vllm_gguf_plugin.weights_adapter.qwen35 import Qwen35GGUFAdapter


class GGMLTraits(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char_p), ("block", ctypes.c_int64), ("interleave", ctypes.c_int64),
        ("size", ctypes.c_size_t), ("quantized", ctypes.c_bool), ("to_float", ctypes.c_void_p),
        ("from_float", ctypes.c_void_p),
    ]


def ggml_decoder(lib, qtype):
    traits = lib.ggml_get_type_traits(int(qtype)).contents
    if (traits.block, traits.size) != gguf.GGML_QUANT_SIZES[qtype] or not traits.to_float:
        raise ValueError(f"Incompatible GGML type traits for {qtype.name}")
    convert = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64)(traits.to_float)

    def decode(raw):
        width = raw.shape[1] // traits.size * traits.block
        out = np.empty((len(raw), width), dtype=np.float32)
        convert(raw.ctypes.data, out.ctypes.data, out.size)
        return out

    return decode


def project(stored, vectors, decoder):
    # Limit memory independently of the checkpoint's matrix width and height.
    result = np.empty((len(stored), vectors.shape[1]), dtype=np.float64)
    for start in range(0, len(stored), 128):
        rows = np.ascontiguousarray(stored[start:start + 128])
        result[start:start + len(rows)] = decoder(rows).astype(np.float64) @ vectors
    return result


def compare(checkpoint, library, layer):
    reader = gguf.GGUFReader(str(checkpoint))
    raw = {tensor.name: tensor for tensor in reader.tensors}
    text = SimpleNamespace(**qwen35_config_dict(read_gguf_metadata(reader), set(raw)))
    config = SimpleNamespace(hf_config=SimpleNamespace(get_text_config=lambda: text))
    adapter = Qwen35GGUFAdapter(config.hf_config)
    spec = adapter.prepare_loading(checkpoint, config)
    quant = SimpleNamespace()
    configure_yaqa_loading(quant, spec)
    lib = ctypes.CDLL(str(library))
    lib.ggml_get_type_traits.argtypes = [ctypes.c_int]
    lib.ggml_get_type_traits.restype = ctypes.POINTER(GGMLTraits)
    key_heads = text.linear_num_key_heads
    value_heads = text.linear_num_value_heads
    qk = 2 * key_heads * text.linear_key_head_dim
    value_width = value_heads * text.linear_value_head_dim
    vectors = np.random.default_rng(2718).normal(size=(text.hidden_size, 3)).astype(np.float32)

    def grouped(output, head_dim):
        return (output.reshape(value_heads // key_heads, key_heads, head_dim, -1)
                .transpose(1, 0, 2, 3).reshape(value_heads * head_dim, -1))

    report = {}
    for suffix in ("attn_qkv_gate", "ssm_beta_alpha"):
        physical = f"blk.{layer}.{suffix}.weight"
        if physical not in (spec.gguf_row_splits or {}):
            raise ValueError(f"Expected a fused projection: {physical}")
        source = raw[physical]
        qtype = source.tensor_type
        reference = project(source.data, vectors, ggml_decoder(lib, qtype))
        if suffix == "attn_qkv_gate":
            v = grouped(reference[qk:qk + value_width], text.linear_value_head_dim)
            expected = (np.concatenate((reference[:qk], v)), grouped(reference[qk + value_width:], text.linear_value_head_dim))
        else:
            expected = (grouped(reference[:value_heads], 1), grouped(reference[value_heads:], 1))
        for (name, _), output in zip(spec.gguf_row_splits[physical], expected):
            path, source_name, _ = quant.yaqa_sources[spec.gguf_to_hf_name_map[name]]
            view = quant.yaqa_tensor_maps[path][source_name]
            if view.tensor_type != qtype:
                raise AssertionError(f"Changed GGML type for {name}")
            packed, gather = _permute_stored(quant, name, _ordinary_tensor(view), int(qtype))
            if gather is not None:
                raise AssertionError(f"Unexpected input permutation for {name}")
            actual = project(packed.numpy(), vectors, lambda batch: gguf.dequantize(batch, qtype))
            np.testing.assert_allclose(actual, output, rtol=1e-9, atol=1e-8)
            report[name] = dict(ggml_type=qtype.name, rows=len(output), vectors=vectors.shape[1],
                                max_abs=float(np.abs(actual - output).max()))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("ggml_library", type=Path)
    parser.add_argument("--layer", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(compare(args.checkpoint, args.ggml_library, args.layer), indent=2))
