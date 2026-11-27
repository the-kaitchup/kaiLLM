"""Single-RTX-5090 candidate boundaries for the released dondon M checkpoint."""

from contextlib import ExitStack, contextmanager
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

import torch
import vllm_gguf_plugin
from vllm_gguf_plugin.inkling import dondon_5090 as sm, iq4_prefill, iq_gate, projection_streams
from vllm_gguf_plugin.inkling.iq4_prefill import qualified_projection_tiles
from vllm_gguf_plugin.inkling.moe import ExpertStage
from vllm_gguf_plugin.quantization.humming import GGUFHummingLinear


GPU = "NVIDIA GeForce RTX 5090"


@contextmanager
def target(tier="M"):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / f"Inkling-2-Mini-dondon-Q3_K_{tier}.gguf"
        path.write_bytes(b"fixture")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        model = SimpleNamespace(
            model_weights=str(path), model="metadata-dir",
            hf_config=SimpleNamespace(model_type="inkling2_loop"),
            max_model_len=73728, dtype=torch.bfloat16,
        )
        config = SimpleNamespace(
            model_config=model, speculative_config=None,
            parallel_config=SimpleNamespace(world_size_across_dp=1),
            scheduler_config=SimpleNamespace(max_num_seqs=1),
        )
        with patch.dict(sm._RELEASES, {path.name: (tier, path.stat().st_size, digest)}):
            yield path, config


@contextmanager
def serving_config(config, extra_modules=None):
    modules = {"vllm.config": SimpleNamespace(get_current_vllm_config_or_none=lambda: config)}
    modules.update(extra_modules or {})
    with ExitStack() as stack:
        stack.enter_context(patch.dict(sys.modules, modules))
        stack.enter_context(patch.object(torch.cuda, "get_device_name", return_value=GPU))
        stack.enter_context(patch.object(torch.cuda, "device_count", return_value=1))
        yield


class DondonM5090(unittest.TestCase):
    def test_exact_target_and_packed_gate_formats(self):
        expected_m = {
            16: {1: (1, 16, 128, 4), 2048: (8, 64, 16, 4)},
            17: {1: (1, 16, 64, 4), 2048: (8, 64, 16, 4)},
            22: {1: (1, 16, 64, 4), 2048: (8, 64, 16, 4)},
        }
        with target() as (path, config):
            self.assertEqual(
                iq_gate.measured_tiles(GPU, 18, config, 1, 882),
                {1: (1, 16, 64, 4), 2048: (8, 32, 16, 4)},
            )
            for kind, block in iq_gate.BLOCKS.items():
                expected = expected_m.get(kind)
                self.assertEqual(iq_gate.measured_tiles(GPU, kind, config, 1, 9 * block[0]), expected)
                self.assertIsNone(iq_gate.measured_tiles(GPU, kind, config, 1, 9 * block[0] - 1))
                self.assertIsNone(iq_gate.measured_tiles(GPU, kind, config, 2, 9 * block[0]))
                if expected:
                    with patch.object(config.scheduler_config, "max_num_seqs", 2):
                        self.assertIsNone(iq_gate.measured_tiles(GPU, kind, config, 1, 9 * block[0]))
                    with patch.object(config, "speculative_config", object()):
                        self.assertIsNone(iq_gate.measured_tiles(GPU, kind, config, 1, 9 * block[0]))
            self.assertIsNone(iq_gate.measured_tiles(GPU, 18, config, 1, 881))
            for gpu in ("NVIDIA GeForce RTX 5080", "NVIDIA GeForce RTX 3090", "NVIDIA RTX PRO 6000 Blackwell"):
                self.assertFalse(sm.is_dondon_m(config, gpu, 1))
            alternatives = [
                (config, "speculative_config", object()),
                (config.scheduler_config, "max_num_seqs", 2),
                (config.parallel_config, "world_size_across_dp", 2),
                (config.model_config, "dtype", torch.float16),
                (config.model_config, "max_model_len", 4096),
                (config.model_config.hf_config, "model_type", "inkling2"),
            ]
            for obj, key, value in alternatives:
                with patch.object(obj, key, value):
                    self.assertFalse(sm.is_dondon_m(config, GPU, 1))
            self.assertFalse(sm.is_dondon_m(config, GPU, 2))
            path.write_bytes(b"badfile")
            self.assertFalse(sm.is_dondon_m(config, GPU, 1))

    def test_generic_down_tiles_leave_other_scopes_alone(self):
        with target() as (_, config):
            self.assertEqual(qualified_projection_tiles(config, GPU, 1, False, 1152), (None, None))
            self.assertEqual(qualified_projection_tiles(config, GPU, 1, False, 2304), (32, 64))
            self.assertEqual(qualified_projection_tiles(config, "NVIDIA GeForce RTX 3090", 1, False, 1152), (16, 16))
            with patch.object(config.model_config, "model_weights", "unrelated.gguf"):
                self.assertEqual(qualified_projection_tiles(config, GPU, 1, False, 1152), (32, 16))
        with target("S") as (_, config):
            self.assertFalse(sm.is_dondon_m(config, GPU, 1))
            self.assertEqual(qualified_projection_tiles(config, GPU, 1, False, 1152), (None, None))
            self.assertEqual(qualified_projection_tiles(config, GPU, 1, False, 2304), (32, 64))

    def test_dondon_m_down_uses_the_measured_full_chunk_only(self):
        packed = Mock(shape=(128, 2304, 648), is_cuda=True, dtype=torch.uint8,
                      device=torch.device("cuda", 0))
        packed.reshape.return_value = packed
        packed.to.return_value = packed
        packed.is_contiguous.return_value = True
        stage = ExpertStage(20, packed, 2304, 1152, None)
        ops = Mock()
        ops._cuda_kernel_available.return_value = False
        ops.ggml_moe_a8_vec.return_value = "ordinary"
        with target() as (_, config), serving_config(config), patch.object(vllm_gguf_plugin, "ops", ops, create=True):
            stage.to(packed.device)
            with patch.object(iq4_prefill, "tiled_prefill", return_value="tiled") as tiled, \
                    patch.object(stage, "_kernel", return_value="mmvq"):
                x = torch.empty((2048 * 8, 1152), dtype=torch.bfloat16, device="meta")
                ids = torch.empty((2048, 8), dtype=torch.int64, device="meta")
                self.assertEqual(stage.run(x, ids, shared=False), "tiled")
                tiled.assert_called_once_with(x, packed, ids, False, 2304, 32, False)
                for tx, ti in ((x[:512 * 8], ids[:512]), (x[:8], ids[:1]), (x.t().contiguous().t(), ids)):
                    self.assertEqual(stage.run(tx, ti, shared=False), "ordinary")
                self.assertEqual(tiled.call_count, 1)
            ops._cuda_kernel_available.return_value = True
            stage.to(packed.device)
            self.assertIsNone(stage.iq4_prefill_tile)
            self.assertFalse(stage.dondon_m_down_prefill)

    def test_dondon_m_streams_require_the_measured_arithmetic(self):
        projections = SimpleNamespace(**{name: object() for name in projection_streams.OWNERS})
        with target() as (_, config), serving_config(config), patch.object(torch.cuda, "is_available", return_value=True), \
                patch.object(torch.cuda, "current_device", return_value=0), patch.dict("os.environ", clear=True), \
                patch("vllm.envs.VLLM_BATCH_INVARIANT", False), \
                patch.object(projection_streams, "pool", return_value=(1, 2, 3)):
            self.assertIsNone(projection_streams.prepare(projections))
            with patch.dict("os.environ", {"KAILLM_HUMMING_DETERMINISTIC": "1"}):
                slots, (groups, streams) = projection_streams.prepare(projections)
                self.assertEqual(slots, 1)
                self.assertEqual(streams, (1, 2, 3))
                self.assertEqual(tuple(tuple(name for name, _, _ in group) for group in groups),
                                 projection_streams.SCHEDULE)
                with patch.object(config.scheduler_config, "max_num_seqs", 2):
                    self.assertIsNone(projection_streams.prepare(projections))

    def test_expert_stage_reaches_proven_shared_gate(self):
        packed = Mock(
            shape=(128, 2304, 882), is_cuda=True, dtype=torch.uint8,
            device=torch.device("cuda", 0),
        )
        packed.reshape.return_value = packed
        packed.to.return_value = packed
        packed.is_contiguous.return_value = True
        stage = ExpertStage(18, packed, 2304, 2304, None)
        ops = Mock()
        ops._cuda_kernel_available.return_value = False
        ops.ggml_moe_a8_vec.return_value = "ordinary"

        with target() as (_, config), serving_config(config), patch.object(vllm_gguf_plugin, "ops", ops, create=True):
            stage.to(packed.device)
            ops._cuda_kernel_available.assert_called_once_with("ggml_moe_a8_vec", 18)
            with patch.object(iq_gate, "run_projection", return_value="packed") as gate:
                for tokens, tile in ((1, (1, 16, 64, 4)), (2048, (8, 32, 16, 4))):
                    x = torch.empty((tokens, 2304), device="meta", dtype=torch.bfloat16)
                    ids = torch.empty((tokens, 8), device="meta", dtype=torch.int64)
                    self.assertEqual(stage.run(x, ids, shared=True), "packed")
                    gate.assert_called_with(x, packed, ids, 18, tile, verifier_rows=False)
                with patch.object(stage, "_kernel", return_value="mmvq"):
                    for tokens, shared in ((2, True), (1, False)):
                        x = torch.empty((tokens if shared else tokens * 8, 2304), device="meta", dtype=torch.bfloat16)
                        ids = torch.empty((tokens, 8), device="meta", dtype=torch.int64)
                        self.assertEqual(stage.run(x, ids, shared), "ordinary")
                self.assertEqual(gate.call_count, 2)

            ops._cuda_kernel_available.return_value = True
            stage.to(packed.device)
            self.assertIsNone(stage.iq_gate_tiles)

    def test_real_m_expert_dispatch_preserves_unmeasured_shapes(self):
        expected = {
            16: {1: (1, 16, 128, 4), 2048: (8, 64, 16, 4)},
            17: {1: (1, 16, 64, 4), 2048: (8, 64, 16, 4)},
            22: {1: (1, 16, 64, 4), 2048: (8, 64, 16, 4)},
        }
        for kind, tiles in expected.items():
            packed = Mock(shape=(128, 2304, 9 * iq_gate.BLOCKS[kind][0]), is_cuda=True,
                          dtype=torch.uint8, device=torch.device("cuda", 0))
            packed.reshape.return_value = packed
            packed.to.return_value = packed
            packed.is_contiguous.return_value = True
            stage = ExpertStage(kind, packed, 2304, 2304, None)
            ops = Mock()
            ops._cuda_kernel_available.return_value = False
            ops.ggml_moe_a8_vec.return_value = "ordinary"
            with target() as (_, config), serving_config(config), patch.object(vllm_gguf_plugin, "ops", ops, create=True):
                stage.to(packed.device)
                for tokens in (1, 2, 2048):
                    x = torch.empty((tokens, 2304), device="meta", dtype=torch.bfloat16)
                    ids = torch.empty((tokens, 8), device="meta", dtype=torch.int64)
                    with patch.object(iq_gate, "run_projection", return_value="packed") as dispatch, patch.object(stage, "_kernel", return_value="mmvq"):
                        if tokens in tiles:
                            self.assertEqual(stage.run(x, ids, shared=True), "packed")
                            dispatch.assert_called_once_with(x, packed, ids, kind, tiles[tokens], verifier_rows=True)
                        else:
                            self.assertEqual(stage.run(x, ids, shared=True), "ordinary")
                            dispatch.assert_not_called()
                ops._cuda_kernel_available.return_value = True
                stage.to(packed.device)
                self.assertIsNone(stage.iq_gate_tiles)

    def test_published_s_gates_and_unoptimized_down(self):
        name = "Inkling-2-Mini-dondon-Q3_K_S.gguf"
        self.assertEqual(sm._RELEASES[name], ("S", 9997961376,
                         "ff6f8da69caffafb80ad8eab0c52f5d8ea1e14653325379f356edfe395ee4864"))
        # These are the physical gate types in increasing block order from the published S GGUF.
        types = (19, 19, 19, 19, 17, 16, 22, 16, 17, 17, 18, 22, 22, 18, 18, 22, 18, 18, 22, 22, 22, 22, 22, 22)
        visits = tuple(range(18)) + tuple(range(6, 18)) + tuple(range(6, 24))
        self.assertEqual({kind: sum(types[i] == kind for i in visits) for kind in set(types)},
                         {16: 4, 17: 7, 18: 15, 19: 4, 22: 18})
        ops = Mock()
        ops._cuda_kernel_available.return_value = False
        ops.ggml_moe_a8_vec.return_value = "ordinary"

        def weights(row_bytes):
            packed = Mock(shape=(128, 2304, row_bytes), is_cuda=True, dtype=torch.uint8,
                          device=torch.device("cuda", 0))
            packed.reshape.return_value = packed
            packed.to.return_value = packed
            packed.is_contiguous.return_value = True
            return packed

        with target("S") as (path, config), serving_config(config), patch.object(vllm_gguf_plugin, "ops", ops, create=True):
            self.assertEqual(sm.released_tier(config, GPU, 1), "S")
            self.assertFalse(sm.is_dondon_m(config, GPU, 1))
            for kind in sorted(set(types)):
                row = 9 * (98 if kind == 18 else iq_gate.BLOCKS[kind][0])
                packed = weights(row)
                stage = ExpertStage(kind, packed, 2304, 2304, None)
                stage.to(packed.device)
                expected = {1: (1, 16, 128 if kind == 16 else 64, 4),
                            2048: (8, 32 if kind == 18 else 64, 16, 4)}
                self.assertEqual(stage.iq_gate_tiles, expected)
                self.assertIs(stage.iq_gate_verifier_rows, kind != 18)
                self.assertIsNone(iq_gate.measured_tiles(GPU, kind, config, 1, row - 1))
                with patch.object(config.scheduler_config, "max_num_seqs", 2):
                    self.assertIsNone(iq_gate.measured_tiles(GPU, kind, config, 1, row))
                with patch.object(config, "speculative_config", object()):
                    self.assertIsNone(iq_gate.measured_tiles(GPU, kind, config, 1, row))

            # Blocks 20, 22 and 23 use ordinary IQ4_NL down math in the proved S source.
            packed = weights(648)
            stage = ExpertStage(20, packed, 2304, 1152, None)
            stage.to(packed.device)
            self.assertEqual((stage.iq4_prefill_tile, stage.iq4_decode_tile, stage.dondon_m_down_prefill),
                             (None, None, False))
            with patch.object(torch.cuda, "is_available", return_value=True):
                self.assertIsNone(projection_streams.prepare(object()))
            path.write_bytes(b"mutated")
            self.assertIsNone(sm.released_tier(config, GPU, 1))

    def test_measured_kernel_entrypoints_and_grids(self):
        weights = torch.empty(128, 2304, 882, device="meta", dtype=torch.uint8)
        align = Mock(return_value=(None, torch.empty(13, device="meta"), None))
        modules = {"vllm.model_executor.layers.fused_moe.fused_moe": SimpleNamespace(moe_align_block_size=align)}
        tables = dict.fromkeys(("iq3xxs_grid", "ksigns_iq2xs", *(value[1] for value in iq_gate.BLOCKS.values())))
        for kind in (16, 17, 18, 19, 22):
            for tokens in (1, 2048):
                kwargs = {"verifier_rows": kind != 18}
                cols = 16 if tokens == 1 else (32 if kind == 18 else 64)
                m = 1 if tokens == 1 else 8
                chunk = (128 if kind == 16 else 64) if tokens == 1 else 16
                x = torch.empty(tokens, 2304, dtype=torch.bfloat16, device="meta")
                ids = torch.empty(tokens, 8, dtype=torch.int64, device="meta")
                with patch.dict(sys.modules, modules), patch.object(iq_gate, "get_iq_table_tensors", return_value=tables), \
                        patch.object(iq_gate, "decode_gate", new_callable=MagicMock) as direct, \
                        patch.object(iq_gate, "decode_gate_rows", new_callable=MagicMock) as rows, \
                        patch.object(iq_gate, "prefill_gate", new_callable=MagicMock) as prefill:
                    iq_gate.run_projection(x, weights, ids, kind, (m, cols, chunk, 4), **kwargs)
                    if tokens == 2048:
                        prefill.__getitem__.assert_called_once_with((13, 2304 // cols))
                        self.assertFalse(direct.__getitem__.called or rows.__getitem__.called)
                    elif kind == 18:
                        direct.__getitem__.assert_called_once_with((2304 // cols, 8))
                        self.assertFalse(rows.__getitem__.called or prefill.__getitem__.called)
                    else:
                        rows.__getitem__.assert_called_once_with((2304 // cols, 8, 1))
                        self.assertFalse(direct.__getitem__.called or prefill.__getitem__.called)

    def test_scoped_humming_matrix_inventory(self):
        for tier, count in (("M", 12), ("S", 16)):
            with self.subTest(tier=tier):
                self.assertEqual(sum(map(len, sm._ATTENTION[tier].values())), count)
                self.check_humming_inventory(tier)

    def check_humming_inventory(self, tier):
        with target(tier) as (_, config):
            for role, shape in sm._PROJECTIONS.items():
                rows = ((shape[0] + 255) // 256) * 256
                layer = SimpleNamespace(
                    dtype=torch.bfloat16, shape=shape, layer_config=SimpleNamespace(shape_n=rows),
                    compute_config='{"use_batch_invariant":true,"use_f16_accum":false}',
                )
                base = dict(
                    block_shape=(16, 64, 32), warp_shape=(16, 32, 32),
                    use_stream_k=False, use_f16_accum=False, num_sms=rows // 64, num_stages=3,
                    num_ctas_per_sm=1, output_chunk_rows=0, use_pdl=True, mma_type="mma", raster_group_m=1,
                )
                helpers = {"vllm.utils.humming": SimpleNamespace(get_heuristics_config=lambda *a, **kw: dict(base))}
                prefix = "model.layers.8.attn." + role
                device = torch.device("cuda", 0)
                with serving_config(config, helpers):
                    allowed = sm._ATTENTION[tier][shape]
                    for kind in (8, 12, 13, 14, 18, 19):
                        selected = sm.attention_decode_tuning(layer, kind, prefix, device)
                        if kind in allowed:
                            expected = dict(base, block_shape=[16, 64, 64], warp_shape=[16, 32, 64])
                            self.assertEqual(json.loads(selected), expected)
                        else:
                            self.assertIsNone(selected)
                    for policy in ('{"use_batch_invariant":false,"use_f16_accum":false}',
                                   '{"use_batch_invariant":true,"use_f16_accum":true}'):
                        with patch.object(layer, "compute_config", policy):
                            self.assertIsNone(sm.attention_decode_tuning(layer, allowed[0], prefix, device))
                    for other_prefix in ("model.layers.8.mlp.q_proj", "lm_head"):
                        self.assertIsNone(sm.attention_decode_tuning(layer, allowed[0], other_prefix, device))
                    for difference in ({"num_stages": 2}, {"block_shape": None}, {"use_pdl": False}, {"num_sms": 123}, {"unknown_setting": 1}):
                        with patch.dict(base, difference):
                            self.assertIsNone(sm.attention_decode_tuning(layer, allowed[0], prefix, device))

    def test_tuning_only_reaches_contiguous_one_row(self):
        layer = GGUFHummingLinear.__new__(GGUFHummingLinear)
        torch.nn.Module.__init__(layer)
        layer.layer_config = layer.weight = layer.weight_scale = layer.locks = None
        layer.shape, layer.dtype = (16, 8), torch.bfloat16
        layer.compute_config, layer.decode_tuning, layer.dondon_m_decode_tuning = "{}", None, "chosen"
        layer._humming_forward = Mock(return_value=torch.empty(2, 16))
        inputs = [
            torch.ones(1, 8, dtype=torch.bfloat16),
            torch.ones(2, 8, dtype=torch.bfloat16),
            torch.ones(1, 8, dtype=torch.float16),
            torch.ones(1, 16, dtype=torch.bfloat16)[:, ::2],
        ]
        for value in inputs:
            layer._humming_forward.return_value = value.new_zeros((len(value), 16))
            layer(value)
        self.assertEqual([entry.kwargs.get("tuning_config") for entry in layer._humming_forward.call_args_list],
                         ["chosen", None, None, None])


if __name__ == "__main__":
    unittest.main()
