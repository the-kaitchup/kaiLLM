# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5 fused GDN projections retain GGML bytes and grouped V-head order."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import gguf
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime-source"))

from vllm_gguf_plugin.quantization.yaqa_linear import (
    _ordinary_tensor,
    _permute_stored,
    configure_yaqa_loading,
    get_yaqa_parts,
)
from vllm_gguf_plugin.weight_utils import gguf_tensor_views
from vllm_gguf_plugin.weights_adapter.qwen35 import Qwen35GGUFAdapter, qwen35_name_map


class FusedQwen35GGUFTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "checkpoint.gguf"
        self.config = SimpleNamespace(
            model_type="qwen3_5_text", num_hidden_layers=1, hidden_size=256, vocab_size=2,
            layer_types=["linear_attention"], linear_num_key_heads=2, linear_num_value_heads=6,
            linear_key_head_dim=4, linear_value_head_dim=8,
        )
        self.model = SimpleNamespace(hf_config=SimpleNamespace(get_text_config=lambda: self.config))
        self.types = (gguf.GGMLQuantizationType.IQ2_S, gguf.GGMLQuantizationType.Q8_0)
        rng = np.random.default_rng(15)
        # 16 Q/K + 48 V + 48 Z rows, 6 beta + 6 alpha rows.
        self.raw = []
        for n_rows, qt in zip((112, 12), self.types):
            block, size = gguf.GGML_QUANT_SIZES[qt]
            blocks = rng.integers(0, 256, size=(n_rows, 256 // block, size), dtype=np.uint8)
            blocks[:, :, :2] = np.float16(0.025).reshape(1).view(np.uint8)
            self.raw.append(blocks.reshape(n_rows, -1))

    def write(self, fused=(True, True), extra=False, short=False, floating=False):
        writer = gguf.GGUFWriter(self.path, "qwen35")
        writer.add_array("tokenizer.ggml.tokens", ["a", "b"])
        for key, value in {
            "block_count": 1, "full_attention_interval": 2, "embedding_length": 256,
            "feed_forward_length": 256, "attention.head_count": 2, "attention.head_count_kv": 2,
            "attention.key_length": 4, "context_length": 64, "rope.freq_base": 10000,
            "rope.dimension_count": 4, "ssm.time_step_rank": 6, "ssm.inner_size": 48,
            "ssm.conv_kernel": 4, "ssm.state_size": 4, "ssm.group_count": 2,
        }.items():
            writer.add_uint32("qwen35." + key, value)
        writer.add_float32("qwen35.attention.layer_norm_rms_epsilon", 1e-5)
        writer.add_array("qwen35.rope.dimension_sections", [1, 1, 0])
        pairs = (
            ("attn_qkv_gate", "attn_qkv", "attn_gate", 64),
            ("ssm_beta_alpha", "ssm_beta", "ssm_alpha", 6),
        )
        split_names = set()
        for (joined, left, right, cut), join, raw, qt in zip(pairs, fused, self.raw, self.types):
            if floating:
                raw = gguf.dequantize(raw, qt)
                qt = None
            split_names.update(f"blk.0.{part}.weight" for part in (left, right))
            if join:
                writer.add_tensor(f"blk.0.{joined}.weight", raw[:-1] if short else raw, raw_dtype=qt)
            if not join or (extra and joined == "attn_qkv_gate"):
                writer.add_tensor(f"blk.0.{left}.weight", raw[:cut], raw_dtype=qt)
                writer.add_tensor(f"blk.0.{right}.weight", raw[cut:], raw_dtype=qt)
        for name, target in qwen35_name_map(1).items():
            if name not in split_names and ".self_attn." not in target:
                writer.add_tensor(name, np.ones(1, dtype=np.float32))
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

    def load(self):
        adapter = Qwen35GGUFAdapter(self.model.hf_config)
        spec = adapter.prepare_loading(self.path, self.model)
        quant = SimpleNamespace(packed_modules_mapping={
            "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"], "in_proj_ba": ["in_proj_b", "in_proj_a"],
        })
        with patch.dict(os.environ, {"VLLM_GGUF_BACKEND": "compact"}):
            configure_yaqa_loading(quant, spec)
        return adapter, quant

    def reference(self):
        # The exporter tiles repeated V heads by member, then key head.
        # Recover the grouped HF order independently using matrix transposes.
        def grouped(matrix, dim):
            return matrix.reshape(3, 2, dim, -1).transpose(1, 0, 2, 3).reshape(6 * dim, -1)

        qkvz = self.raw[0]
        beta, alpha = np.split(self.raw[1], 2)
        return (np.concatenate((qkvz[:16], grouped(qkvz[16:64], 8))),
                grouped(qkvz[64:], 8), grouped(beta, 1), grouped(alpha, 1))

    def test_original_and_legacy_storage(self):
        for layout in ((True, True), (True, False), (False, True), (False, False)):
            with self.subTest(layout=layout):
                self.write(fused=layout)
                adapter, quant = self.load()
                spec = adapter.load_spec
                self.assertEqual(len(spec.gguf_row_splits or {}), sum(layout))
                sources = []
                for fused in ("in_proj_qkvz", "in_proj_ba"):
                    sources.extend(get_yaqa_parts(quant, "model.layers.0.linear_attn." + fused))
                for (_, (path, name, _)), expected, qt in zip(
                    sources, self.reference(), (self.types[0], self.types[0], self.types[1], self.types[1])
                ):
                    tensor = quant.yaqa_tensor_maps[path][name]
                    self.assertEqual(tensor.tensor_type, qt)
                    packed, gather = _permute_stored(quant, name, _ordinary_tensor(tensor), int(qt))
                    self.assertIsNone(gather)
                    np.testing.assert_array_equal(packed.numpy(), expected)

                reader = gguf.GGUFReader(self.path)
                for original in reader.tensors:
                    if original.name in (spec.gguf_row_splits or {}):
                        views = list(gguf_tensor_views([original], spec.gguf_row_splits))
                        for view in views:
                            self.assertTrue(np.shares_memory(original.data, view.data))
                        self.assertEqual(views[1].data_offset, original.data_offset + views[0].n_bytes)
                        self.assertEqual(sum(v.n_bytes for v in views), original.n_bytes)
                        self.assertEqual(sum(v.n_elements for v in views), original.n_elements)

    def test_floating_projection_iterator(self):
        self.write(floating=True)
        adapter, _ = self.load()
        # Match the model loader's consumed-name filter so unrelated toy vectors are skipped.
        names = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")
        wanted = {f"model.layers.0.linear_attn.{name}.weight" for name in names}
        spec = adapter.load_spec
        spec.gguf_to_hf_name_map = {key: val for key, val in spec.gguf_to_hf_name_map.items() if val in wanted}
        loaded = dict(adapter.prepare_weights(self.model))
        for name, raw, qt in zip(names, self.reference(), (self.types[0], self.types[0], self.types[1], self.types[1])):
            torch.testing.assert_close(loaded[f"model.layers.0.linear_attn.{name}.weight"],
                                       torch.from_numpy(gguf.dequantize(raw, qt)), rtol=0, atol=0)

    def test_malformed_fusion(self):
        self.write(extra=True)
        with self.assertRaisesRegex(ValueError, "Both fused and split"):
            self.load()
        self.write(short=True)
        with self.assertRaisesRegex(ValueError, "Invalid GGUF row split"):
            self.load()


if __name__ == "__main__":
    unittest.main()
