#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
kaillm_source_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ -d "$kaillm_source_dir/dist" && -x "$kaillm_source_dir/vendor/uv/uv" ]]; then
  exec bash "$kaillm_source_dir/tools/install_bundle.sh" "$@"
fi
if ! command -v python3 >/dev/null 2>&1; then
  printf '%s\n' 'Python 3 is required to download the kaiLLM release. Alternatively, unpack a complete release archive and run its install.sh.' >&2
  exit 1
fi
exec python3 "$kaillm_source_dir/tools/install_release.py" "$@"
