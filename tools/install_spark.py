"""Assemble Inkling serving from pinned kaiLLM Python and a native GB10 vLLM installation."""
import argparse
import ast
import importlib.metadata as metadata
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile

from install_release import download, release_metadata, sha256


PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / "runtime-source"
sys.path.insert(0, str(SOURCE))
from vllm_gguf_plugin.native import spark


def wheel_member(name):
    path = PurePosixPath(name)
    if path.parts[:2] != ("kaillm", "_runtime") or ".." in path.parts or "\\" in name:
        return None
    relative = PurePosixPath(*path.parts[2:])
    # Compiled vLLM extensions come from the installed ARM wheel. The release
    # supplies its compatible Python snapshot, templates, headers and metadata.
    core = relative.parts[:3] == ("compatible-vllm", "core", "vllm")
    if relative == PurePosixPath(spark.CORE) / "vllm-rs":
        return None
    third_party = relative.parts[:1] == ("third-party",) and "_native" not in relative.parts
    plugin_metadata = relative.parts and relative.parts[0].startswith("vllm_gguf_plugin-") and relative.parts[0].endswith(".dist-info")
    if (core or third_party or plugin_metadata) and relative.suffix not in (".so", ".pyc") and ".so." not in relative.name:
        return relative
    return None


def frozen_sources(archive, working, root, release):
    if sha256(archive) != release["sha256"]:
        raise ValueError("The source release does not match its pinned SHA-256")
    with tarfile.open(archive) as bundle:
        wheels = [m for m in bundle if m.isfile() and PurePosixPath(m.name).parent ==
                  PurePosixPath(release["directory"], "dist") and m.name.endswith(".whl")]
        if len(wheels) != 1:
            raise ValueError("Expected one kaiLLM wheel in the pinned archive")
        wheel = working / "source.whl"
        with bundle.extractfile(wheels[0]) as src, wheel.open("wb") as dest:
            shutil.copyfileobj(src, dest)
    with zipfile.ZipFile(wheel) as package:
        for member in package.infolist():
            name = wheel_member(member.filename)
            if name is None or member.is_dir():
                continue
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with package.open(member) as src:
                first = src.read(4)
                if first == b"\x7fELF":
                    raise ValueError(f"Unexpected host binary in portable release inputs: {name}")
                with target.open("wb") as dest:
                    dest.write(first)
                    shutil.copyfileobj(src, dest)
    if not (root / spark.CORE / "utils/humming.py").is_file():
        raise ValueError("The pinned compatible Humming/vLLM Python source is missing")


def native_core():
    distribution = metadata.distribution("vllm")
    root = Path(distribution.locate_file("vllm")).resolve()
    version_module = root / "_version.py"
    tree = ast.parse(version_module.read_text() if version_module.is_file() else "")
    commits = [node.value.value for node in tree.body if isinstance(node, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "commit_id" for t in node.targets)
               and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)]
    origin = json.loads(distribution.read_text("direct_url.json") or "{}").get("vcs_info", {})
    if origin.get("vcs") == "git" and origin.get("commit_id"):
        commits.append(origin["commit_id"])
    wheel_commit = re.search(r"(?:[+.])g([0-9a-f]{9,40})(?:[.+]|$)", distribution.version)
    if wheel_commit:
        commits.append(wheel_commit.group(1))
    if not commits or not all(c.removeprefix("g").startswith(spark.CORE_COMMIT) for c in commits):
        raise ValueError("Install the Linux/AArch64 vLLM CUDA build at commit " + spark.CORE_COMMIT)
    return root, distribution.version


def copy_native_core(core, root):
    dest = root / spark.CORE
    for p in (*core.rglob("*.so*"), core / "vllm-rs"):
        if p.is_file():
            if not spark.arm_elf(p):
                raise ValueError("Expected an ARM64 vLLM extension: " + str(p))
            target = dest / p.relative_to(core)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)
    # auditwheel's extension RPATH resolves this directory relative to vllm/.
    libs = core.parent / "vllm.libs"
    if libs.is_dir():
        shutil.copytree(libs, dest.parent / libs.name)


def locked_source(root):
    contract = SOURCE / "source-lock.json"
    for name, expected in json.loads(contract.read_text())["source_files"].items():
        source = SOURCE / name
        if sha256(source) != expected:
            raise ValueError("Changed source-lock file: " + name)
        parts = Path(name).parts
        dest = root / "compatible-vllm/core" / Path(*parts[1:]) if parts[0] == "vllm-overrides" else root / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
    (root / "benchmark-source.sha256").write_text(sha256(contract) + "\n")


def install(prefix, archive=None):
    if platform.system() != "Linux" or platform.machine().lower() not in ("arm64", "aarch64"):
        raise ValueError("Run the Spark installer with native Linux/AArch64 Python")
    if sys.version_info[:2] != (3, 12):
        raise ValueError("Use the CPython 3.12 environment containing native PyTorch and vLLM")
    core, version = native_core()
    import torch
    if not torch.version.cuda or torch.version.hip:
        raise ValueError("Install native CUDA-enabled PyTorch in the vLLM environment")
    prefix = prefix.expanduser().absolute()
    source_lock = sha256(SOURCE / "source-lock.json")
    if prefix.exists():
        root = prefix / "kaillm/_runtime"
        if (root / "benchmark-source.sha256").is_file() and (root / "benchmark-source.sha256").read_text().strip() == source_lock:
            receipt = spark.verify_runtime(root)
            if ((prefix / "bin/kaillm").is_file() and receipt["python"] == sys.executable
                    and receipt["torch_version"] == torch.__version__ and receipt["vllm_version"] == version):
                return prefix / "bin/kaillm"
        raise ValueError("Choose a new installation prefix for this source and interpreter")
    release = release_metadata(PROJECT / "release.json")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".kaillm-spark-", dir=prefix.parent) as tmp:
        work = Path(tmp)
        staged = work / "installation"
        public = staged / "kaillm"
        public.mkdir(parents=True)
        for p in (PROJECT / "kaillm").glob("*.py"):
            shutil.copy2(p, public / p.name)
        root = public / "_runtime"
        if archive is None:
            archive = work / "release.tar.gz"
            download(release["url"], archive)
        frozen_sources(archive, work, root, release)
        copy_native_core(core, root)
        locked_source(root)
        # The runtime child runs with a clean PYTHONPATH, including its own
        # copy of the pure-Python native asset lookup for source-built ITQ.
        helper = root / "kaillm"
        helper.mkdir()
        for n in ("__init__.py", "native_assets.py"):
            shutil.copy2(public / n, helper / n)
        cache = root / "runtime-native-cache"
        cache.mkdir()
        (cache / "native-assets.json").write_text(json.dumps({"schema": "kaillm-frozen-native-assets-v1", "assets":[]}) + "\n")
        environment = dict(os.environ, PYTHONPATH=str(root / "third-party"), PYTHONDONTWRITEBYTECODE="1")
        subprocess.run([sys.executable, "-c", "from humming.build import build_native; build_native()"],
                       cwd=work, env=environment, check=True)
        files = spark.native_files(root)
        receipt = {"compute_capability":[12,1], "vllm_commit":spark.CORE_COMMIT, "vllm_version":version,
                   "torch_version":torch.__version__, "torch_cuda":torch.version.cuda, "python":sys.executable,
                   "humming_manifest_sha256":sha256(root/"third-party/humming/_native/aarch64/manifest.json"),
                   "frozen_source_sha256":release["sha256"], "native_sha256":{n:sha256(root / n) for n in files}}
        (root / spark.RECEIPT).write_text(json.dumps(receipt,indent=2,sort_keys=True) + "\n")
        spark.verify_runtime(root)
        # Prove native Torch/vLLM/Humming imports in the assembled environment.
        environment["PYTHONPATH"] = os.pathsep.join(str(root / p) for p in (".", "compatible-vllm/core", "gguf-reader", "third-party"))
        subprocess.run([sys.executable, "-c", "import torch; import vllm._C_stable_libtorch, vllm._moe_C_stable_libtorch, vllm.cumem_allocator; "
                        "from vllm.engine.arg_utils import EngineArgs; "
                        "from vllm.utils.humming import humming_forward; from vllm_gguf_plugin import register; "
                        "from humming.utils.jit import get_native_arch; assert get_native_arch() == 'aarch64'"],
                       cwd=work, env=environment, check=True)
        command = staged / "bin/kaillm"
        command.parent.mkdir()
        entry = "import sys; sys.path.insert(0, " + repr(str(prefix)) + "); from kaillm.cli import main; sys.exit(main())"
        command.write_text("#!/bin/sh\nexec " + shlex.quote(sys.executable) + " -c " + shlex.quote(entry) + ' "$@"\n')
        command.chmod(0o755)
        staged.rename(prefix)
    return prefix / "bin/kaillm"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, default=Path.home()/".local/share/kaillm/spark")
    parser.add_argument("--archive", type=Path, help="Use a local, checksum-pinned rc6 archive as the Python source")
    args = parser.parse_args()
    print(install(args.prefix, args.archive))
