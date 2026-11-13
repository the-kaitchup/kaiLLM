#!/usr/bin/env bash
# Install a complete, unpacked kaiLLM release without changing system Python.
set -euo pipefail

kaillm_release_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
kaillm_prefix="${KAILLM_INSTALL_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/kaillm/0.1.0rc3}"
kaillm_bin_dir="${KAILLM_BIN_DIR:-$HOME/.local/bin}"
if [[ $# -gt 0 ]]; then
  if [[ $# -ne 2 || $1 != --prefix ]]; then
    printf '%s\n' 'Usage: bash install.sh [--prefix PATH]' >&2
    exit 2
  fi
  kaillm_prefix=$2
fi
if [[ $(uname -s) != Linux || $(uname -m) != x86_64 ]]; then
  printf '%s\n' 'This release requires Linux x86_64 (including WSL2).' >&2
  exit 1
fi
if ! command -v c++ >/dev/null 2>&1; then
  printf '%s\n' 'A C++ compiler is required for runtime JIT components. Install g++ and rerun this command.' >&2
  exit 1
fi
kaillm_uv="$kaillm_release_dir/vendor/uv/uv"
if [[ ! -x $kaillm_uv || ! -f $kaillm_release_dir/SHA256SUMS ]]; then
  printf '%s\n' 'Use the complete kaiLLM release archive; the installer helper or checksum file is missing.' >&2
  exit 1
fi
(
  cd -- "$kaillm_release_dir"
  sha256sum --check --quiet SHA256SUMS
)
shopt -s nullglob
kaillm_wheels=("$kaillm_release_dir"/dist/kaillm-*.whl)
if [[ ${#kaillm_wheels[@]} -ne 1 ]]; then
  printf '%s\n' 'The release archive must contain exactly one kaiLLM wheel.' >&2
  exit 1
fi
if [[ -e $kaillm_prefix && ! -f $kaillm_prefix/.kaillm-installation ]]; then
  printf 'Refusing to modify an existing directory without a kaiLLM installation marker: %s\n' "$kaillm_prefix" >&2
  exit 1
fi
mkdir -p -- "$kaillm_prefix"
kaillm_prefix=$(CDPATH= cd -- "$kaillm_prefix" && pwd)
printf '%s\n' 'kaiLLM 0.1.0rc3' > "$kaillm_prefix/.kaillm-installation"
if [[ ! -x $kaillm_prefix/venv/bin/python ]]; then
  "$kaillm_uv" venv --python 3.12 "$kaillm_prefix/venv"
fi
"$kaillm_uv" pip install --python "$kaillm_prefix/venv/bin/python" --no-deps \
  --requirement "$kaillm_release_dir/requirements-torch.txt"
"$kaillm_uv" pip install --python "$kaillm_prefix/venv/bin/python" --no-deps \
  --requirement "$kaillm_release_dir/requirements-runtime.txt"
"$kaillm_uv" pip install --python "$kaillm_prefix/venv/bin/python" --no-deps "${kaillm_wheels[0]}"
"$kaillm_uv" pip install --python "$kaillm_prefix/venv/bin/python" --no-deps \
  --requirement "$kaillm_release_dir/requirements-installer.txt"
"$kaillm_prefix/venv/bin/python" -m kaillm.install_native_runtime \
  --patchelf "$kaillm_prefix/venv/bin/patchelf"
mkdir -p -- "$kaillm_bin_dir"
"$kaillm_prefix/venv/bin/python" - "$kaillm_bin_dir/kaillm" "$kaillm_prefix/venv/bin/kaillm" <<'PY'
from pathlib import Path
import sys
from kaillm.install_native_runtime import write_launcher
write_launcher(Path(sys.argv[1]), Path(sys.argv[2]))
PY
"$kaillm_prefix/venv/bin/kaillm" --version
printf '\nInstalled: %s/kaillm\n' "$kaillm_bin_dir"
printf '%s\n' 'Run: kaillm doctor' 'Then: kaillm serve /path/to/model.gguf'
if [[ :$PATH: != *:"$kaillm_bin_dir":* ]]; then
  printf 'Add %s to PATH to use the kaillm command.\n' "$kaillm_bin_dir"
fi
