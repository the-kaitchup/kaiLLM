# SPDX-License-Identifier: Apache-2.0
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import tempfile
import unittest
from unittest.mock import patch
import venv

from kaillm.install_native_runtime import write_launcher

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "tools"))
import install_checkout
import install_release
sys.path.pop(0)


class CheckoutInstallTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.base = self.root / "checkout"
        self.launcher = self.root / "bin/kaillm"
        self.previous = self.root / "rc6/venv/bin/kaillm"
        self.previous.parent.mkdir(parents=True)
        self.previous.write_text("#!/bin/sh\nprintf '%s\\n' previous\n")
        self.previous.chmod(0o755)
        (self.root / "rc6/.kaillm-installation").write_text("kaiLLM rc6\n")
        write_launcher(self.launcher, self.previous)
        environment = patch.dict(os.environ, {"KAILLM_BIN_DIR": str(self.launcher.parent)})
        environment.start()
        self.addCleanup(environment.stop)
        self.calls = []

    def pinned_dependencies(self, *, archive, prefix, bin_dir):
        """Create an offline, standard-library substitute for the pinned venv."""
        self.assertEqual(archive, self.root / "release.tar.gz")
        self.assertEqual(bin_dir.parent, self.base)
        self.assertNotEqual(bin_dir, self.launcher.parent)
        self.assertTrue((prefix / install_checkout.OWNER).is_file())
        self.calls.append(prefix)
        interpreter = prefix / "venv/bin/python"
        if not interpreter.exists():
            venv.EnvBuilder(with_pip=False, symlinks=True).create(prefix / "venv")
        purelib = sysconfig.get_path("purelib", scheme="venv", vars={"base": str(prefix / "venv")})
        package = Path(purelib) / "kaillm"
        core = package / "_runtime/compatible-vllm/core/vllm"
        core.mkdir(parents=True, exist_ok=True)
        (core / "forward_context.py").write_text("# pinned base\n")
        (core / "__pycache__").mkdir(exist_ok=True)
        (core / "__pycache__/forward_context.fake.pyc").write_bytes(b"stale bytecode")
        (core / "pinned-native-base.so").write_bytes(b"pinned base")
        entrypoint = interpreter.with_name("kaillm")
        entrypoint.write_text(f"#!{interpreter}\nfrom kaillm.cli import main\nraise SystemExit(main())\n")
        entrypoint.chmod(0o755)
        write_launcher(bin_dir / "kaillm", entrypoint)

    def install(self):
        return install_checkout.install(archive=self.root / "release.tar.gz", prefix=self.base)

    def test_wires_checked_out_cli_and_runtime_over_the_pinned_base(self):
        with patch.object(install_checkout, "install_release", self.pinned_dependencies):
            launcher, revision = self.install()
            self.assertEqual(self.install(), (launcher, revision))
        self.assertEqual(launcher, self.launcher)
        self.assertEqual(self.calls, [self.base / revision])
        interpreter = self.calls[0] / "venv/bin/python"
        package = install_checkout.installed_package(self.calls[0], interpreter)
        self.assertEqual((package / "cli.py").read_bytes(), (PROJECT / "kaillm/cli.py").read_bytes())
        core = package / "_runtime/compatible-vllm/core/vllm"
        source = PROJECT / "runtime-source/vllm-overrides/vllm/forward_context.py"
        self.assertEqual((core / "forward_context.py").read_bytes(), source.read_bytes())
        self.assertEqual((core / "pinned-native-base.so").read_bytes(), b"pinned base")
        self.assertFalse((core / "__pycache__/forward_context.fake.pyc").exists())
        version = subprocess.check_output([launcher, "--version"], text=True).strip()
        self.assertTrue(version.endswith(f"(checkout {revision[:12]})"), version)

    def test_failed_smoke_keeps_previous_launcher_and_retries_in_place(self):
        with patch.object(install_checkout, "install_release", self.pinned_dependencies):
            with patch.object(install_checkout, "validate", side_effect=OSError("offline smoke failure")):
                with self.assertRaisesRegex(OSError, "offline smoke failure"):
                    self.install()
            self.assertEqual(subprocess.check_output([self.launcher], text=True).strip(), "previous")
            self.assertFalse((self.calls[0] / install_checkout.READY).exists())
            launcher, revision = self.install()
        self.assertEqual(self.calls, [self.base / revision] * 2)
        self.assertIn(str(self.base / revision), launcher.read_text())

    def test_refuses_to_install_over_an_unmanaged_prefix(self):
        revision, _ = install_checkout.checkout_files(PROJECT)
        prefix = self.base / revision
        prefix.mkdir(parents=True)
        (prefix / ".kaillm-installation").write_text("kaiLLM rc6\n")
        with patch.object(install_checkout, "install_release") as pinned:
            with self.assertRaisesRegex(ValueError, "unmanaged directory"):
                self.install()
            pinned.assert_not_called()
        self.assertEqual(self.launcher.read_text().splitlines()[-1], f'exec {self.previous} "$@"')

    def test_checks_the_source_lock_before_installing_dependencies(self):
        source = self.root / "project/runtime-source"
        source.mkdir(parents=True)
        (source.parent / "release.json").write_bytes((PROJECT / "release.json").read_bytes())
        (source / "changed.py").write_bytes(b"changed")
        (source / "source-lock.json").write_text(json.dumps({"source_files": {"changed.py": "a" * 64}}))
        with patch.object(install_checkout, "install_release") as pinned:
            with self.assertRaisesRegex(ValueError, "Source hash does not match"):
                install_checkout.install(project=source.parent, prefix=self.base)
            pinned.assert_not_called()

    def test_pinned_installer_honors_the_private_cache_and_launcher_paths(self):
        archive = self.root / "fixture.tar.gz"
        archive.write_bytes(b"offline fixture")
        record = {"directory": "release", "sha256": install_release.sha256(archive)}
        cache = self.root / "cache"
        with (
            patch.dict(os.environ, {"XDG_CACHE_HOME": str(cache)}),
            patch.object(install_release, "release_metadata", return_value=record),
            patch.object(install_release, "extract_release", return_value=self.root / "unpacked") as extract,
            patch.object(install_release.subprocess, "run") as run,
        ):
            install_release.install(archive=archive, prefix=self.base, bin_dir=self.root / "private-bin")
        self.assertTrue(extract.call_args.args[1].is_relative_to(cache / "kaillm"))
        self.assertEqual(run.call_args.kwargs["env"]["KAILLM_BIN_DIR"], str(self.root / "private-bin"))


if __name__ == "__main__":
    unittest.main()
