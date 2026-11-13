#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Install the exact release pinned by this checkout, without system changes."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request


PROJECT = Path(__file__).resolve().parents[1]
MAX_ARCHIVE_BYTES = 2 * 1024**3
MAX_EXPANDED_BYTES = 4 * 1024**3


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def release_metadata(path):
    record = json.loads(Path(path).read_text())
    if not re.fullmatch(r"[0-9a-f]{64}", record.get("sha256", "")):
        raise ValueError("This checkout has no pinned release checksum yet")
    if not record.get("url", "").startswith("https://github.com/the-kaitchup/kaiLLM/releases/download/"):
        raise ValueError("The release URL is outside the kaiLLM GitHub repository")
    if not re.fullmatch(r"kaiLLM-[A-Za-z0-9._-]+-linux-x86_64", record.get("directory", "")):
        raise ValueError("Invalid release directory")
    return record


def download(url, target):
    total = 0
    with urllib.request.urlopen(url, timeout=60) as response, Path(target).open("wb") as output:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_ARCHIVE_BYTES:
                raise ValueError("Release exceeds the download size limit")
            output.write(chunk)


def extract_release(archive, target, expected_directory):
    """Extract only regular files/directories under one bounded release root."""
    target = Path(target)
    total = 0
    count = 0
    seen = set()
    with tarfile.open(archive, "r|gz") as source:
        for member in source:
            path = PurePosixPath(member.name)
            if (path.is_absolute() or ".." in path.parts or not path.parts
                    or path.parts[0] != expected_directory or "\\" in member.name
                    or not (member.isdir() or member.isfile()) or path in seen):
                raise ValueError("Unsafe or duplicate release archive member")
            seen.add(path)
            count += 1
            total += member.size
            if count > 10000 or total > MAX_EXPANDED_BYTES:
                raise ValueError("Release exceeds the extraction limit")
            destination = target.joinpath(*path.parts)
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                with source.extractfile(member) as stream, destination.open("xb") as output:
                    shutil.copyfileobj(stream, output)
                destination.chmod(member.mode & 0o755)
    root = target / expected_directory
    if not (root / "install.sh").is_file() or not (root / "SHA256SUMS").is_file():
        raise ValueError("Incomplete release archive")
    return root


def install(archive=None, prefix=None):
    record = release_metadata(PROJECT / "release.json")
    # Temporary files stay on the normal user disk; /tmp may be a small tmpfs.
    cache = Path.home() / ".cache/kaillm"
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="install-", dir=cache) as temporary:
        work = Path(temporary)
        if archive is None:
            archive = work / "release.tar.gz"
            print("Downloading kaiLLM " + record["version"] + "...", flush=True)
            download(record["url"], archive)
        else:
            archive = Path(archive).expanduser().resolve()
        if not archive.is_file() or sha256(archive) != record["sha256"]:
            raise ValueError("Release SHA256 differs from the checksum pinned by this checkout")
        root = extract_release(archive, work / "unpacked", record["directory"])
        command = ["bash", str(root / "install.sh")]
        if prefix is not None:
            command += ["--prefix", str(Path(prefix).expanduser())]
        subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Install the pinned release from a local archive")
    parser.add_argument("--prefix", type=Path, help="Use a custom isolated installation directory")
    args = parser.parse_args()
    try:
        install(args.archive, args.prefix)
    except (OSError, ValueError, subprocess.SubprocessError, tarfile.TarError) as error:
        parser.exit(1, "kaiLLM installation failed: " + str(error) + "\n")


if __name__ == "__main__":
    main()
