"""Bind an isolated Python to the bundled private native system runtime.

Only the installation's interpreter copy is changed. System Python, system
libraries and model/native kernel bytes are untouched. No GPU is initialized.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import shlex
import subprocess
import sys
import sysconfig
import tempfile


def _managed_command(target):
    target = Path(target)
    if (target.name != "kaillm" or target.parent.name != "bin"
            or target.parent.parent.name != "venv"):
        return False
    marker = target.parent.parent.parent / ".kaillm-installation"
    return marker.is_file() and marker.read_text().startswith("kaiLLM ")


def write_launcher(destination, target):
    """Replace only a managed launcher; never write through a destination link."""
    destination, target = Path(destination), Path(target).resolve()
    if not target.is_file() or not _managed_command(target):
        raise ValueError("The launcher target is not a managed kaiLLM installation")
    if destination.exists() or destination.is_symlink():
        managed = destination.is_symlink() and _managed_command(destination.resolve())
        if not destination.is_symlink() and destination.is_file():
            text = destination.read_text()
            managed = "# Managed by kaiLLM installer." in text
            if not managed:
                for line in text.splitlines():
                    if line.startswith("exec "):
                        words = shlex.split(line)
                        managed = len(words) == 3 and words[2] == "$@" and _managed_command(words[1])
        if not managed:
            raise ValueError("Refusing to replace an unmanaged command: " + str(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=destination.parent,
                                         prefix=".kaillm-launcher-", delete=False) as output:
            temporary = Path(output.name)
            output.write('#!/bin/sh\n# Managed by kaiLLM installer.\nexec '
                         + shlex.quote(str(target)) + ' "$@"\n')
        temporary.chmod(0o755)
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def bind_cuda_linker_layout(prefix):
    """Expose the pip CUDA runtime at the paths used by runtime JIT linkers."""
    cuda = Path(sysconfig.get_path("purelib")) / "nvidia/cu13"
    library = cuda / "lib"
    runtime = library / "libcudart.so.13"
    if not runtime.is_file() or not runtime.resolve().is_relative_to(prefix.resolve()):
        raise RuntimeError("The isolated CUDA 13 runtime library is missing")
    aliases = ((cuda / "lib64", library), (library / "libcudart.so", runtime))
    for alias, target in aliases:
        if alias.exists() or alias.is_symlink():
            if alias.resolve() != target.resolve():
                raise ValueError("Conflicting CUDA linker path: " + str(alias))
        else:
            alias.symlink_to(target.name, target_is_directory=target.is_dir())
    # WSL exposes the driver through ldconfig's runtime cache without adding
    # its directory to the C++ linker's search path. Keep the supplied driver
    # in place and expose only a private link for JIT components using -lcuda.
    driver_alias = library / "libcuda.so"
    if not driver_alias.exists() and not driver_alias.is_symlink():
        ldconfig = shutil.which("ldconfig") or ("/sbin/ldconfig" if Path("/sbin/ldconfig").is_file() else None)
        if ldconfig:
            cache = subprocess.run([ldconfig, "-p"], capture_output=True, text=True, check=True)
            for line in cache.stdout.splitlines():
                fields = line.strip().split()
                if fields and fields[0] == "libcuda.so.1" and "=>" in fields:
                    driver = Path(fields[-1])
                    if driver.is_absolute() and driver.is_file():
                        driver_alias.symlink_to(driver)
                        break
    receipt = {"schema": "kaillm-cuda-linker-layout-v1", "CUDA_HOME": str(cuda),
               "aliases": {str(alias): str(alias.resolve()) for alias, _ in aliases},
               "driver_link": str(driver_alias.resolve()) if driver_alias.exists() else None,
               "runtime_sha256": digest(runtime), "GPU_initialized": False,
               "library_bytes_changed": False}
    (prefix / "kaillm-cuda-linker-layout.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def install(tool):
    prefix = Path(sys.prefix)
    if sys.prefix == sys.base_prefix or not (prefix / "pyvenv.cfg").is_file():
        raise RuntimeError("Private runtime binding requires an isolated virtual environment")
    cuda_layout = bind_cuda_linker_layout(prefix)
    system = Path(__file__).resolve().parent / "_runtime/portable-system"
    provenance = system / "provenance.json"
    records = json.loads(provenance.read_text())["files"]
    for name, record in records.items():
        path = (system / name).resolve()
        if not path.is_relative_to(system) or digest(path) != record["sha256"]:
            raise ValueError("Private runtime asset differs: " + name)
    loader = system / "ld-linux-x86-64.so.2"
    destination = prefix / "bin/python3.12.kaillm-elf"
    backup = prefix / "bin/python3.12.kaillm-original"
    receipt_path = prefix / "kaillm-private-runtime.json"
    previous = json.loads(receipt_path.read_text()) if receipt_path.is_file() else {}
    aliases = [prefix / "bin" / name for name in ("python", "python3", "python3.12")]
    if (previous.get("schema") == "kaillm-private-runtime-v2"
            and previous.get("loader") == str(loader) and destination.is_file()
            and previous.get("interpreter_sha256") == digest(destination)
            and previous.get("assets_manifest_sha256") == digest(provenance)
            and previous.get("pyvenv_cfg_sha256") == digest(prefix / "pyvenv.cfg")
            and all(path.is_file() and not path.is_symlink()
                    and digest(path) == previous["interpreter_sha256"] for path in aliases)):
        previous["cuda_linker_layout"] = cuda_layout
        receipt_path.write_text(json.dumps(previous, indent=2) + "\n")
        return previous
    original = Path(previous["original_python"]) if previous else Path(sys.executable).resolve()
    if not backup.exists():
        shutil.copy2(original, backup)
    temporary = destination.with_name(destination.name + "." + str(os.getpid()))
    shutil.copy2(backup, temporary)
    rollback = []
    probe_environment = {k: v for k, v in os.environ.items()
                         if not k.startswith("PYTHON") and k != "__PYVENV_LAUNCHER__"}
    probe_code = ('import json,os,sys,encodings;print(json.dumps({"prefix":sys.prefix,'
                  '"libc":os.confstr("CS_GNU_LIBC_VERSION"),"encodings":encodings.__file__}))')

    def probe_interpreter(path):
        probe = subprocess.run([str(path), "-I", "-c", probe_code],
                               env=probe_environment, capture_output=True, text=True)
        if probe.returncode:
            raise RuntimeError("Private interpreter startup failed: " + probe.stderr.strip())
        actual = json.loads(probe.stdout)
        if Path(actual["prefix"]).resolve() != prefix.resolve() or actual["libc"] != "glibc 2.43":
            raise ValueError("Private interpreter prefix/libc differs: " + probe.stdout)
        return actual

    try:
        rpath = subprocess.check_output([tool, "--print-rpath", str(backup)], text=True).strip()
        rpath = rpath.replace("${ORIGIN}", str(original.parent)).replace("$ORIGIN", str(original.parent))
        search = str(system) + (":" + rpath if rpath else "")
        # UV's standalone CPython may name libpython through an ORIGIN-relative
        # DT_NEEDED entry, which must keep referring to that original base.
        needed = subprocess.check_output([tool, "--print-needed", str(backup)], text=True).splitlines()
        for name in needed:
            expanded = name.replace("${ORIGIN}", str(original.parent)).replace("$ORIGIN", str(original.parent))
            if expanded != name:
                subprocess.run([tool, "--replace-needed", name, expanded, str(temporary)], check=True)
        subprocess.run([tool, "--set-interpreter", str(loader), "--force-rpath", "--set-rpath", search,
                        str(temporary)], check=True)
        actual = probe_interpreter(temporary)
        temporary.replace(destination)
        for target in aliases:
            saved = target.with_name(target.name + ".kaillm-rollback." + str(os.getpid()))
            if target.exists() or target.is_symlink():
                shutil.copy2(target, saved, follow_symlinks=False)
                rollback.append((target, saved))
            link = target.with_name(target.name + ".kaillm-link")
            link.unlink(missing_ok=True)
            # Some standalone CPython builds lose pyvenv.cfg discovery through
            # a symlink to a renamed executable. Regular copies preserve it.
            shutil.copy2(destination, link)
            link.replace(target)
        alias_probes = {path.name: probe_interpreter(path) for path in aliases}
        receipt = {"schema": "kaillm-private-runtime-v2", "original_python": str(original),
                   "original_interpreter_sha256": digest(backup), "loader": str(loader),
                   "assets_manifest_sha256": digest(provenance), "interpreter": str(destination),
                   "interpreter_sha256": digest(destination), "rpath": search,
                   "alias_mode": "regular copies", "CPU_alias_probes": alias_probes,
                   "pyvenv_cfg_sha256": digest(prefix / "pyvenv.cfg"),
                   "CPU_probe": actual, "GPU_initialized": False,
                   "cuda_linker_layout": cuda_layout,
                   "system_interpreter_or_libraries_modified": False}
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        for _, saved in rollback:
            saved.unlink(missing_ok=True)
        return receipt
    except Exception:
        for target, saved in rollback:
            if saved.exists() or saved.is_symlink():
                saved.replace(target)
        raise
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patchelf", required=True)
    args = parser.parse_args()
    print(json.dumps(install(args.patchelf), indent=2))


if __name__ == "__main__":
    main()
