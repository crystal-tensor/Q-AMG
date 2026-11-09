#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
uv run --locked pytest -q
uv run --locked ruff check src tests scripts
node --check web/src/main.js
node --check web/src/app.js
