"""Dispatch boundaries for the four-slot dondon XS packed gate on one RTX 5090."""
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

import torch
import vllm_gguf_plugin
from vllm_gguf_plugin.inkling import dondon_5090, iq_gate
from vllm_gguf_plugin.inkling.moe import ExpertStage


GPU = "NVIDIA GeForce RTX 5090"


class FourSlotGate(unittest.TestCase):
    def setUp(self):
        directory = self.enterContext(tempfile.TemporaryDirectory())
        self.path = Path(directory) / "Inkling-2-Mini-dondon-Q2_K_XS.gguf"
        data = b"fixture"
        self.path.write_bytes(data)
        self.enterContext(patch.object(dondon_5090, "_XS_WEIGHT_BYTES", len(data)))
        self.enterContext(patch.object(dondon_5090, "_XS_WEIGHT_SHA256", hashlib.sha256(data).hexdigest()))
        self.config = SimpleNamespace(
            scheduler_config=SimpleNamespace(max_num_seqs=4, max_num_batched_tokens=2048),
            parallel_config=SimpleNamespace(world_size_across_dp=1),
            model_config=SimpleNamespace(model_weights=self.path, model="converted HF config",
                                         hf_config=SimpleNamespace(model_type="inkling2_loop"),
                                         max_model_len=73728, dtype=torch.bfloat16),
            speculative_config=None,
        )

    def test_device_serving_policy_and_pinned_model_bytes(self):
        config = self.config
        matches = dondon_5090.is_dondon_xs_b4
        self.assertTrue(matches(config, GPU, 1))
        self.assertFalse(matches(config, GPU, 2))
        self.assertFalse(matches(config, "NVIDIA GeForce RTX 3090", 1))
        for slots in (1, 2, 3, 8):
            with self.subTest(slots=slots), patch.object(config.scheduler_config, "max_num_seqs", slots):
                self.assertFalse(matches(config, GPU, 1))
        for obj, key, value in (
            (config.scheduler_config, "max_num_batched_tokens", 1024),
            (config.parallel_config, "world_size_across_dp", 2),
            (config.model_config, "dtype", torch.float16),
            (config.model_config, "max_model_len", 65536),
            (config.model_config.hf_config, "model_type", "inkling2"),
            (config, "speculative_config", SimpleNamespace()),
        ):
            with self.subTest(key=key, value=value), patch.object(obj, key, value):
                self.assertFalse(matches(config, GPU, 1))
        with patch.object(config.model_config, "model", self.path):
            with patch.object(config.model_config, "model_weights", None):
                self.assertTrue(matches(config, GPU, 1))
            with patch.object(config.model_config, "model_weights", self.path.with_name("other.gguf")):
                self.assertFalse(matches(config, GPU, 1))
        other = self.path.with_suffix(".tmp")
        other.write_bytes(b"changed")
        other.replace(self.path)
        self.assertFalse(matches(config, GPU, 1))

    def test_production_gate_selects_only_the_two_measured_widths(self):
        config = self.config
        config_module = SimpleNamespace(get_current_vllm_config_or_none=lambda: config)
        ops = Mock()
        ops._cuda_kernel_available.return_value = False
        self.enterContext(patch.dict("sys.modules", {"vllm.config": config_module}))
        self.enterContext(patch.object(torch.cuda, "get_device_name", return_value=GPU))
        self.enterContext(patch.object(torch.cuda, "device_count", return_value=1))
        self.enterContext(patch.object(vllm_gguf_plugin, "ops", ops, create=True))
        self.assertIsNone(iq_gate.measured_tiles(GPU, 18, config, 1, 882))
        for kind, tile_n in ((16, 32), (17, 16), (19, 32), (22, 32)):
            row_bytes = 9 * iq_gate.BLOCKS[kind][0]
            self.assertIsNone(iq_gate.measured_tiles(GPU, kind, config, 1, row_bytes - 1))
            packed = Mock(shape=(128, 2304, row_bytes), is_cuda=True, dtype=torch.uint8, device=torch.device("cuda", 0))
            packed.to.return_value = packed.reshape.return_value = packed
            packed.is_contiguous.return_value = True
            stage = ExpertStage(kind, packed, 2304, 2304, None).to(packed.device)
            expected = {4: (1, tile_n, 64, 4), 2048: (8, 64, 16, 4)}
            self.assertEqual(stage.iq_gate_tiles, expected)
            with patch.object(iq_gate, "run_projection", return_value="packed") as run:
                for width, tile in expected.items():
                    x = torch.empty(width, 2304, dtype=torch.bfloat16, device="meta")
                    ids = torch.empty(width, 8, dtype=torch.int64, device="meta")
                    self.assertEqual(stage.run(x, ids, shared=True), "packed")
                    run.assert_called_with(x, packed, ids, kind, tile, verifier_rows=False)
            with patch.object(ops, "_cuda_kernel_available", return_value=True):
                self.assertIsNone(stage.to(packed.device).iq_gate_tiles)

    def test_real_projection_entry_uses_row_grid_then_sorted_prefill(self):
        align = Mock(return_value=(None, torch.empty(13, device="meta"), None))
        modules = {"vllm.model_executor.layers.fused_moe.fused_moe": SimpleNamespace(moe_align_block_size=align)}
        self.enterContext(patch.dict("sys.modules", modules))
        self.enterContext(patch.object(iq_gate, "get_iq_table_tensors",
                                      return_value={"iq2xxs_grid": None, "ksigns_iq2xs": None}))
        rows = self.enterContext(patch.object(iq_gate, "decode_gate_rows", new_callable=MagicMock))
        prefill = self.enterContext(patch.object(iq_gate, "prefill_gate", new_callable=MagicMock))
        weights = torch.empty(128, 2304, 594, device="meta", dtype=torch.uint8)
        tiles = iq_gate.measured_tiles(GPU, 16, self.config, 1, 594)
        for tokens, kernel, grid in ((4, rows, (72, 8, 4)), (2048, prefill, (13, 36))):
            x = torch.empty(tokens, 2304, device="meta", dtype=torch.bfloat16)
            ids = torch.empty(tokens, 8, device="meta", dtype=torch.int32)
            iq_gate.run_projection(x, weights, ids, 16, tiles[tokens])
            kernel.__getitem__.assert_called_once_with(grid)
            self.assertEqual(kernel.__getitem__.return_value.call_args.kwargs,
                             {"num_warps": 4, "num_stages": 1 if tokens == 4 else 2})
        align.assert_called_once()
