#!/usr/bin/env bash
set -euo pipefail

gpus="${1:-2}"
shift || true
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

python scripts/start_training.py --gpus "${gpus}" \
  --config configs/config.yaml "$@"
