#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${EASYFSS_CONDA_ENV:-EasyFSS}"
HOST="${EASYFSS_DEMO_HOST:-127.0.0.1}"
PORT="${EASYFSS_DEMO_PORT:-9000}"

cd "$PROJECT_ROOT"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "$ENV_NAME"
fi

python demo/server.py --host "$HOST" --port "$PORT"
