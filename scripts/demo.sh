#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
uv sync --locked --native-tls
if [[ "${QAMG_SKIP_IMAGE_MODEL:-0}" != "1" ]]; then
  uv run --locked python scripts/prepare_models.py
fi
uv run --locked python scripts/export_gallery.py
npm --prefix web ci --no-audit --no-fund
npm --prefix web run build
exec uv run --locked qamg serve "$@"
