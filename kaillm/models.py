# SPDX-License-Identifier: Apache-2.0
"""Resolve explicit GGUF references through the normal Hugging Face cache."""

from pathlib import Path, PurePosixPath
import re


REPOSITORY = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\Z")


def validate_filename(filename):
    path = PurePosixPath(filename)
    if (not filename or path.is_absolute() or ".." in path.parts
            or "\\" in filename or not filename.lower().endswith(".gguf")):
        raise ValueError("Use a repository-relative .gguf filename without parent traversal")
    return filename


def resolve_model(reference, *, filename=None, revision="main", dry_run=False):
    """Return a local checkpoint path; dry runs never download model files."""
    local = Path(reference).expanduser()
    if local.is_file():
        if filename is not None:
            raise ValueError("--filename applies only to a Hugging Face repository")
        if local.suffix.lower() != ".gguf":
            raise ValueError("The model must be a GGUF file")
        return str(local.resolve())
    if reference.startswith("hf://"):
        parts = reference[5:].split("/", 2)
        if len(parts) != 3 or filename is not None:
            raise ValueError("Use hf://owner/repository/filename.gguf")
        repository, filename = "/".join(parts[:2]), parts[2]
    elif ":" in reference:
        repository, embedded = reference.split(":", 1)
        if filename is not None:
            raise ValueError("Specify the GGUF filename once")
        filename = embedded
    else:
        repository = reference
    if not REPOSITORY.fullmatch(repository):
        raise ValueError("Use an existing GGUF path or owner/repository:filename.gguf")
    if filename is not None:
        filename = validate_filename(filename)
    if dry_run:
        return repository + (":" + filename if filename else "")
    from huggingface_hub import hf_hub_download, list_repo_files

    if filename is None:
        files = [name for name in list_repo_files(repository, revision=revision)
                 if name.lower().endswith(".gguf")]
        if len(files) != 1:
            raise ValueError("This repository needs an explicit GGUF filename: "
                             "kaillm serve owner/repository:filename.gguf")
        filename = validate_filename(files[0])
    return hf_hub_download(repo_id=repository, filename=filename, revision=revision)
