"""Keep the measured Inkling gate projection on its original shapes and serving policy."""
import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import torch

from vllm_gguf_plugin.inkling.iq_gate import BLOCKS, measured_tiles


class InklingGateTiles(unittest.TestCase):
    def test_only_the_measured_runtime_shapes_select_the_kernel(self):
        simple = types.SimpleNamespace
        config = simple(parallel_config=simple(world_size_across_dp=1), speculative_config=None,
                        scheduler_config=simple(max_num_seqs=1))
        for gpu in ('NVIDIA GeForce RTX 5070 Ti', 'NVIDIA GeForce RTX 5080'):
            for kind, (block_bytes, _) in BLOCKS.items():
                width = 9 * block_bytes
                tiles = measured_tiles(gpu, kind, config, 1, width)
                self.assertEqual(set(tiles), {1, 2048})
                expected_n = 32 if gpu.endswith("5080") and kind in (16,19,22) else 16
                self.assertEqual(tiles[1], (1, expected_n, 64, 4))
                self.assertIsNone(measured_tiles(gpu, kind, config, 1, width + 1))
                self.assertIsNone(measured_tiles(gpu, kind, config, 2, width))
        for device in ('NVIDIA GeForce RTX 3090', 'NVIDIA GeForce RTX 5090'):
            self.assertIsNone(measured_tiles(device, 22, config, 1, 738))
        for field, value in (('speculative_config', simple()), ('scheduler_config', simple(max_num_seqs=2)),
                             ('parallel_config', simple(world_size_across_dp=2))):
            changed = simple(**{**vars(config), field:value})
            self.assertIsNone(measured_tiles('NVIDIA GeForce RTX 5080', 22, changed, 1, 738))

    def test_single_slot_native_5070_selects_the_proven_draft_and_expert_layout(self):
        simple = types.SimpleNamespace
        gpu = "NVIDIA GeForce RTX 5070 Ti"
        model = simple(model="/weights/Inkling-2-Mini-UD-Q2_K_XL.gguf", hf_config=simple(model_type="inkling2_loop"))
        draft = simple(architectures=["DFlashInklingDSparkDraftModel"])
        spec = simple(method="dflash", num_speculative_tokens=5, draft_model_config=draft)
        config = simple(speculative_config=spec, model_config=model, scheduler_config=simple(max_num_seqs=1),
                        parallel_config=simple(world_size_across_dp=1))
        for arch in ("InklingDSparkDraftModel", "DFlashInklingDSparkDraftModel"):
            draft.architectures = [arch]
            for kind, (size, _) in BLOCKS.items():
                tiles = measured_tiles(gpu, kind, config, 1, 9 * size)
                prefill = (8, 64, 16, 4) if kind in (16, 22) else (8, 32, 32, 4)
                self.assertEqual(tiles, {**dict.fromkeys(range(1, 7), (1, 16, 64, 4)), 2048: prefill})
        for obj, field, wrong in (
                (model, "model", "/weights/Inkling-2-Mini-dondon-Q2_K_XS.gguf"),
                (model.hf_config, "model_type", "other"), (draft, "architectures", ["OtherDSparkDraftModel"]),
                (spec, "method", "ngram"), (spec, "num_speculative_tokens", 3),
                (config.scheduler_config, "max_num_seqs", 2), (config.parallel_config, "world_size_across_dp", 2)):
            with self.subTest(field=field), patch.object(obj, field, wrong):
                self.assertIsNone(measured_tiles(gpu, 16, config, 1, 594))
        for name, kind, devices, stride in ((gpu, 16, 2, 594), (gpu, 16, 1, 595), (gpu, 8, 1, 594),
                                              ("NVIDIA GeForce RTX 5080", 16, 1, 594)):
            with self.subTest(name=name, kind=kind, devices=devices, stride=stride):
                self.assertIsNone(measured_tiles(name, kind, config, devices, stride))

    def test_3090_only_selects_published_dondon_xs_plain_batch_one(self):
        simple = types.SimpleNamespace
        gpu = "NVIDIA GeForce RTX 3090"
        model = simple(model="/cache/inkling-metadata/canonical",
                       model_weights="/weights/Inkling-2-Mini-dondon-Q2_K_XS.gguf",
                       hf_config=simple(model_type="inkling2_loop"))
        config = simple(speculative_config=None, model_config=model, scheduler_config=simple(max_num_seqs=1),
                        parallel_config=simple(world_size_across_dp=1))
        for kind in (16, 17, 19, 22):
            width = 9 * BLOCKS[kind][0]
            self.assertEqual(measured_tiles(gpu, kind, config, 1, width),
                             {1: (1, 16, 64, 4), 2048: (8, 64, 16, 4)})
            self.assertIsNone(measured_tiles(gpu, kind, config, 2, width))
            self.assertIsNone(measured_tiles(gpu, kind, config, 1, width + 1))
        changes = ((model, "model_weights", "/weights/Inkling-2-Mini-dondon-Q3_K_S.gguf"),
                   (model.hf_config, "model_type", "other"), (config, "speculative_config", simple()),
                   (config.scheduler_config, "max_num_seqs", 2), (config.parallel_config, "world_size_across_dp", 2))
        for holder, key, value in changes:
            with self.subTest(key=key), patch.object(holder, key, value):
                self.assertIsNone(measured_tiles(gpu, 16, config, 1, 9 * BLOCKS[16][0]))

    def test_native_one_row_uses_the_same_verifier_kernel_as_six_rows(self):
        from vllm_gguf_plugin.inkling import iq_gate
        kernel = "vllm.model_executor.layers.fused_moe.fused_moe"
        tables = {"iq2xxs_grid": object(), "ksigns_iq2xs": object()}
        with patch.dict(sys.modules, {kernel: types.SimpleNamespace(moe_align_block_size=None)}), \
             patch.object(iq_gate, "get_iq_table_tensors", return_value=tables), \
             patch.object(iq_gate, "decode_gate") as ordinary, patch.object(iq_gate, "decode_gate_rows") as verifier:
            for rows, native in ((1, False), (1, True), (6, True)):
                x = MagicMock(shape=(rows, 2304))
                weights = types.SimpleNamespace(shape=(128, 2304, 594))
                ids = types.SimpleNamespace(shape=(rows, 8))
                iq_gate.run_projection(x, weights, ids, 16, (1, 16, 64, 4), verifier_rows=native)
            ordinary.__getitem__.assert_called_once_with((144, 8))
            self.assertEqual([call.args[0] for call in verifier.__getitem__.call_args_list], [(144, 8, 1), (144, 8, 6)])

    def test_resident_stage_propagates_the_native_route_and_resets_on_reload(self):
        import vllm_gguf_plugin
        from vllm_gguf_plugin.inkling import iq_gate
        from vllm_gguf_plugin.inkling.moe import ExpertStage

        simple = types.SimpleNamespace
        model = simple(model_weights="/weights/Inkling-2-Mini-UD-Q2_K_XL.gguf", hf_config=simple(model_type="inkling2_loop"))
        serving = simple(model_config=model, speculative_config=simple(method="dflash", num_speculative_tokens=5,
                         draft_model_config=simple(architectures=["DFlashInklingDSparkDraftModel"])),
                         parallel_config=simple(world_size_across_dp=1), scheduler_config=simple(max_num_seqs=1))
        config_api = simple(get_current_vllm_config_or_none=lambda: serving)
        weight = MagicMock(dtype=torch.uint8, is_cuda=True, shape=(128, 2304, 594))
        weight.reshape.return_value = weight.to.return_value = weight
        weight.is_contiguous.return_value = True
        stage = ExpertStage(16, weight, 2304, 2304, None)
        x, ids = MagicMock(dtype=torch.bfloat16), MagicMock(dtype=torch.int64, shape=(1, 8))
        ids.is_contiguous.return_value = x.is_contiguous.return_value = True
        with patch.dict(sys.modules, {"vllm.config": config_api}), \
             patch.object(vllm_gguf_plugin, "ops", simple(_cuda_kernel_available=lambda *_: False), create=True), \
             patch.object(torch.cuda, "get_device_name", return_value="NVIDIA GeForce RTX 5070 Ti"), \
             patch.object(torch.cuda, "device_count", return_value=1), patch.object(iq_gate, "run_projection") as run:
            for native in (True, False):
                if not native:
                    serving.speculative_config = None
                stage.to("cuda")
                self.assertIs(stage.run(x, ids, shared=True), run.return_value)
                run.assert_called_with(x, weight, ids, 16, (1, 16, 64, 4), verifier_rows=native)
            model.model_weights = "/weights/Inkling-2-Mini-dondon-Q2_K_XS.gguf"
            with patch.object(torch.cuda, "get_device_name", return_value="NVIDIA GeForce RTX 3090"):
                stage.to("cuda")
                self.assertIs(stage.run(x, ids, shared=True), run.return_value)
                run.assert_called_with(x, weight, ids, 16, (1, 16, 64, 4), verifier_rows=True)

    @unittest.skipUnless(os.environ.get("TRITON_INTERPRET") == "1" or torch.cuda.is_available(),
                         "Requires Triton interpreter or CUDA")
    def test_unused_routing_storage_cannot_write_to_token_zero(self):
        from vllm_gguf_plugin.inkling.iq_gate import prefill_gate
        device = "cpu" if os.environ.get("TRITON_INTERPRET") == "1" else "cuda"
        # One unused block intentionally points at token zero and expert zero.
        x = torch.ones((1, 256), dtype=torch.bfloat16, device=device)
        weight = torch.zeros((16, 82), dtype=torch.uint8, device=device)
        order = torch.zeros(8, dtype=torch.int32, device=device)
        expert = torch.zeros(1, dtype=torch.int32, device=device)
        padded = torch.zeros(1, dtype=torch.int32, device=device)
        output = torch.full((8, 16), 3, dtype=torch.bfloat16, device=device)
        table = torch.zeros(8192, dtype=torch.uint8, device=device)
        prefill_gate[(1, 1)](x, weight, order, expert, padded, output, table, table,
                             8, 16, 256, 82, 22, 8, 16, 16, num_warps=4)
        self.assertTrue((output == 3).all())
