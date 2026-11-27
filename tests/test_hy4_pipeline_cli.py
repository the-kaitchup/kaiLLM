import os
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, mock

from kaillm import cli


class PipelineLauncherTest(TestCase):
    def test_auto_and_explicit_gpu_selection(self):
        for device, expected in (([], "3,5"), (["--device", "5,3"], "5,3")):
            args, forwarded = cli.parser().parse_known_args(
                ["serve", "example/repo:model.gguf", "--dry-run", "--compute-capability", "12.0",
                 "--pipeline-parallel-size", "2", *device])
            with mock.patch.dict(os.environ, CUDA_VISIBLE_DEVICES="3,5,7"), \
                 mock.patch.object(cli, "payload_root", return_value=Path(__file__).resolve().parents[1] / "runtime-source"):
                command, _ = cli.prepare_serve(args, forwarded)
            self.assertEqual(command[command.index("--device") + 1], expected)
            self.assertEqual(command[command.index("--pipeline-parallel-size") + 1], "2")

    def test_duplicate_devices_rejected(self):
        args, rest = cli.parser().parse_known_args(
            ["serve", "example/repo:model.gguf", "--dry-run", "--compute-capability", "12.0",
             "--pipeline-parallel-size", "2", "--device", "0,0"])
        with self.assertRaisesRegex(ValueError, "distinct"):
            cli.prepare_serve(args, rest)

    def test_query_reads_all_selected_gpu_lines(self):
        response = SimpleNamespace(stdout="12.0, 32000, GPU A\n12.0, 31800, GPU B\n")
        with mock.patch.object(cli.subprocess, "run", return_value=response):
            self.assertEqual(cli.query_gpu("0,1"), {"compute_capability": "12.0", "total_memory_MiB": 31800,
                                                   "name": "GPU A, GPU B"})
