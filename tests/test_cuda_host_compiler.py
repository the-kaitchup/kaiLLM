"""Exercise the launch compiler policy without requiring a CUDA device."""

from contextlib import chdir, redirect_stderr, redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from kaillm import cli, install_native_runtime
from kaillm.cuda_host_compiler import check_host_compiler


class CudaHostCompilerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.environment = {"PATH": str(self.bin)}
        self.tool(self.bin / "nvcc", "Cuda compilation tools, release 13.0, V13.0.88")
        self.tool(self.bin / "gcc", "#define __GNUC__ 16")

    def tool(self, path, *output):
        path.parent.mkdir(parents=True, exist_ok=True)
        # Use the shell's builtin printf so the test PATH contains only the fixture tools.
        body = "\n".join(f"printf '%s\\n' '{line}'" for line in output)
        path.write_text("#!/bin/sh\n" + body + "\n")
        path.chmod(0o755)
        return str(path)

    def gcc_pair(self, major):
        cc = self.tool(self.bin / f"gcc-{major}", f"#define __GNUC__ {major}")
        cxx = self.tool(self.bin / f"g++-{major}", f"#define __GNUC__ {major}")
        return {"CC": cc, "CXX": cxx, "NVCC_CCBIN": cxx, "CUDAHOSTCXX": cxx}

    def test_configures_a_matching_pair_for_all_build_paths(self):
        expected = self.gcc_pair(14)
        self.tool(self.bin / "gcc-15", "#define __GNUC__ 15")
        self.tool(self.bin / "g++-15", "#define __GNUC__ 16")
        report = check_host_compiler(self.environment)
        self.assertEqual(report["status"], "selected")
        self.assertEqual(report["environment"], expected)
        self.assertEqual(self.environment, {"PATH": str(self.bin)})

    def test_the_compiler_paths_work_from_a_jit_cache_directory(self):
        expected = self.gcc_pair(14)
        with chdir(self.root):
            report = check_host_compiler({"PATH": "bin"})
        self.assertEqual(report["environment"], expected)

    def test_existing_host_and_nvcc_overrides_are_preserved(self):
        self.gcc_pair(14)
        for setting in (
            {"NVCC_CCBIN": "/compiler/g++"},
            {"CC": "/compiler/cc"},
            {"CXX": "/compiler/cxx"},
            {"CUDAHOSTCXX": "/compiler/host"},
            {"PYTORCH_NVCC": "/toolchains/nvcc-wrapper"},
            {"NVCC_PREPEND_FLAGS": "-ccbin '/compiler with spaces/g++'"},
            {"NVCC_APPEND_FLAGS": "--compiler-bindir=/compiler/g++"},
            {"NVCC_PREPEND_FLAGS": "--options-file=/compiler/nvcc.conf"},
        ):
            with self.subTest(setting=setting):
                environment = {**self.environment, **setting}
                report = check_host_compiler(environment)
                self.assertEqual(report["status"], "explicit")
                self.assertEqual(report["environment"], {})
                self.assertEqual(environment, {**self.environment, **setting})

    def test_recommends_the_manual_workaround_when_no_pair_exists(self):
        report = check_host_compiler(self.environment)
        self.assertEqual(report["status"], "needs_compatible_gcc")
        self.assertIn("default gcc is GCC 16", report["message"])
        self.assertIn("NVCC_CCBIN=/usr/bin/g++-14 CC=/usr/bin/gcc-14", report["message"])
        self.assertIn("CXX=/usr/bin/g++-14 CUDAHOSTCXX=/usr/bin/g++-14", report["message"])

    def test_compatible_gcc_and_non_gnu_defaults_are_left_alone(self):
        self.tool(self.bin / "gcc", "#define __GNUC__ 15")
        self.assertIsNone(check_host_compiler(self.environment))
        self.tool(self.bin / "gcc", "#define __GNUC__ 16", "#define __clang__ 1")
        self.assertIsNone(check_host_compiler(self.environment))

    def test_explicit_cuda_home_uses_that_nvcc_version(self):
        self.gcc_pair(14)
        other = self.root / "other-cuda"
        self.tool(other / "bin/nvcc", "Cuda compilation tools, release 13.4, V13.4.22")
        self.assertIsNone(check_host_compiler(self.environment, cuda_home=other))

    def test_serve_propagates_the_selected_pair_to_the_child(self):
        expected = self.gcc_pair(14)
        toolkit = self.root / "toolkit"
        self.tool(toolkit / "bin/nvcc", "Cuda compilation tools, release 13.0, V13.0.88")
        launcher = self.root / "deployment/serve.py"
        launcher.parent.mkdir()
        launcher.touch()
        args = cli.parser().parse_args(["serve", "model.gguf", "--compute-capability", "8.6",
                                        "--dry-run", "--cuda-home", str(toolkit)])
        with patch.dict(os.environ, self.environment, clear=True), \
                patch.object(cli, "payload_root", return_value=self.root), \
                patch.object(cli, "resolve_model", return_value="model.gguf"):
            command, selected = cli.prepare_serve(args, [])
            self.assertEqual(os.environ.get("NVCC_CCBIN"), None)
        self.assertEqual(command[command.index("--cuda-home") + 1], str(toolkit))
        self.assertEqual({key: selected[key] for key in expected}, expected)

    def test_installer_preserves_receipt_and_warns_on_stderr(self):
        receipt = {"cuda_linker_layout": {"CUDA_HOME": str(self.root)}}
        stdout, stderr = StringIO(), StringIO()
        with patch.dict(os.environ, self.environment, clear=True), \
                patch.object(install_native_runtime, "install", return_value=receipt), \
                patch("sys.argv", ["install_native_runtime", "--patchelf", "patchelf"]), \
                redirect_stdout(stdout), redirect_stderr(stderr):
            install_native_runtime.main()
        self.assertEqual(json.loads(stdout.getvalue()), receipt)
        self.assertIn("CUDA 13.0 host compilation supports GCC up to 15", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
