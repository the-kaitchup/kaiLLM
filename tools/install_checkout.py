#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Install the checked-out x86-64 runtime over the pinned release dependencies."""

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import stat
import subprocess
import tarfile
import tempfile

from install_release import install as install_release, release_metadata


PROJECT = Path(__file__).resolve().parents[1]
OWNER = ".kaillm-checkout"
READY = ".kaillm-checkout-ready"
STAMP = "checkout-source.sha256"


def sha256(content):
    return hashlib.sha256(content).hexdigest()


def read_source(root, relative):
    path = root / relative
    if not path.resolve().is_relative_to(root) or not path.is_file():
        raise ValueError(f"Invalid checkout file: {relative}")
    return path.read_bytes()


def checkout_files(project):
    release = release_metadata(project / "release.json")
    source = (project / "runtime-source").resolve()
    lock = read_source(source, "source-lock.json")
    locked_files = json.loads(lock).get("source_files")
    if not isinstance(locked_files, dict) or not locked_files:
        raise ValueError("The checkout must contain a nonempty runtime source lock")
    files = {}
    for name, expected in locked_files.items():
        relative = PurePosixPath(name)
        if (relative.is_absolute() or not relative.parts or ".." in relative.parts
                or relative.as_posix() != name or "\\" in name):
            raise ValueError(f"Invalid path in the runtime source lock: {name}")
        data = read_source(source, relative)
        if sha256(data) != expected:
            raise ValueError(f"Source hash does not match the checkout lock: {name}")
        target = Path("_runtime") / relative
        if relative.parts[0] == "vllm-overrides":
            if len(relative.parts) < 3 or relative.parts[1] != "vllm":
                raise ValueError(f"Invalid vLLM override: {name}")
            target = Path("_runtime/compatible-vllm/core", *relative.parts[1:])
        if target in files:
            raise ValueError(f"Duplicate runtime target: {target}")
        files[target] = data
    if Path("_runtime", STAMP) in files:
        raise ValueError(f"The checkout lock cannot supply {STAMP}")
    package = (project / "kaillm").resolve()
    python_files = {p.name: read_source(package, p.name) for p in package.glob("*.py")}
    if not {"__init__.py", "cli.py", "install_native_runtime.py"} <= python_files.keys():
        raise ValueError("The checkout is missing kaiLLM package files")
    files.update({Path(name): data for name, data in python_files.items()})
    identity = {
        "schema": 1,
        "release": release["sha256"],
        "source": sha256(lock),
        "package": {name: sha256(data) for name, data in python_files.items()},
    }
    return sha256(json.dumps(identity, sort_keys=True).encode()), files


@contextmanager
def install_lock(base):
    # Keep the file in place so concurrent installers always lock the same inode.
    flags = os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK
    with os.fdopen(os.open(base / ".kaillm-checkout.lock", flags, 0o600), "r+") as lock:
        if not stat.S_ISREG(os.fstat(lock.fileno()).st_mode):
            raise ValueError("Invalid checkout installation lock")
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def owns_prefix(prefix, marker, revision):
    path = prefix / marker
    return not path.is_symlink() and path.is_file() and path.read_text().strip() == revision


def prepare_prefix(base, prefix, revision):
    if prefix.is_symlink() or (prefix.exists() and not prefix.is_dir()):
        raise ValueError(f"Refusing to modify an unmanaged path: {prefix}")
    if prefix.exists():
        if not owns_prefix(prefix, OWNER, revision):
            raise ValueError(f"Refusing to modify an unmanaged directory: {prefix}")
        return
    # Publish the marker and empty directory together. The venv is then built
    # at its permanent address because the private ELF loader uses that path.
    with tempfile.TemporaryDirectory(prefix=".kaillm-prefix-", dir=base) as temporary:
        staged = Path(temporary) / "prefix"
        staged.mkdir(mode=0o700)
        (staged / OWNER).write_text(revision + "\n")
        (staged / ".kaillm-installation").write_text("kaiLLM checkout dependencies\n")
        staged.rename(prefix)


def atomic_write(path, data):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
        temporary.chmod(0o644)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def installed_package(prefix, python):
    purelib = subprocess.check_output(
        [str(python), "-I", "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
        text=True,
    ).strip()
    package = (Path(purelib) / "kaillm").resolve()
    if not package.is_relative_to(prefix / "venv") or not (package / "_runtime/compatible-vllm/core").is_dir():
        raise ValueError("The pinned release did not install a complete runtime")
    return package


def overlay(package, files, revision):
    for name, data in files.items():
        target = package / name
        if not target.parent.resolve().is_relative_to(package):
            raise ValueError(f"Runtime path escapes the installed package: {name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(target, data)
        if target.suffix == ".py":
            cache = target.parent / "__pycache__"
            if cache.is_symlink():
                raise ValueError(f"Runtime cache is a symbolic link: {cache}")
            for compiled in cache.glob(f"{target.stem}.*.pyc"):
                compiled.unlink()
    atomic_write(package / "_runtime" / STAMP, (revision + "\n").encode())


def validate(python, package, revision):
    probe = (
        "from pathlib import Path\n"
        "import sys\n"
        "from kaillm.cli import payload_root\n"
        "root = payload_root()\n"
        f"if root != Path(sys.argv[1]) / '_runtime' or (root / {STAMP!r}).read_text().strip() != sys.argv[2]:\n"
        "    raise RuntimeError('The interpreter did not load the checkout runtime')\n"
    )
    subprocess.run([str(python), "-IB", "-c", probe, str(package), revision], check=True)
    subprocess.run([str(python), "-IB", str(package / "_runtime/deployment/serve.py"), "--help"],
                   stdout=subprocess.DEVNULL, check=True)


def publish(python, launcher, target):
    command = (
        "from pathlib import Path; import sys; "
        "from kaillm.install_native_runtime import write_launcher; "
        "write_launcher(Path(sys.argv[1]), Path(sys.argv[2]))"
    )
    subprocess.run([str(python), "-IB", "-c", command, str(launcher), str(target)], check=True)


def install(project=PROJECT, archive=None, prefix=None):
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise ValueError("The x86-64 checkout installer requires Linux x86-64")
    revision, files = checkout_files(project)
    default_base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local/share") / "kaillm/main"
    base = Path(prefix or os.environ.get("KAILLM_INSTALL_DIR") or default_base).expanduser().resolve()
    destination = base / revision
    launcher = Path(os.environ.get("KAILLM_BIN_DIR") or Path.home() / ".local/bin").expanduser() / "kaillm"
    python = destination / "venv/bin/python"
    base.mkdir(parents=True, exist_ok=True)
    with install_lock(base):
        prepare_prefix(base, destination, revision)
        ready = owns_prefix(destination, READY, revision)
        if not ready:
            with tempfile.TemporaryDirectory(prefix=".kaillm-bin-", dir=base) as scratch:
                install_release(archive=archive, prefix=destination, bin_dir=Path(scratch))
        package = installed_package(destination, python)
        if not ready:
            overlay(package, files, revision)
        validate(python, package, revision)
        if not ready:
            atomic_write(destination / READY, (revision + "\n").encode())
        publish(python, launcher, destination / "venv/bin/kaillm")
    return launcher, revision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Use the pinned release from a local archive")
    parser.add_argument("--prefix", type=Path, help="Use a custom base directory for revision-specific installations")
    args = parser.parse_args()
    try:
        launcher, revision = install(archive=args.archive, prefix=args.prefix)
    except (OSError, ValueError, subprocess.SubprocessError, tarfile.TarError) as error:
        parser.exit(1, "kaiLLM checkout installation failed: " + str(error) + "\n")
    print(f"Installed checkout {revision[:12]}: {launcher}")
    print("Run: kaillm doctor")
    if str(launcher.parent) not in os.get_exec_path():
        print(f"Add {launcher.parent} to PATH to use the kaillm command.")


if __name__ == "__main__":
    main()
