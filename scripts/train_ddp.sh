#!/usr/bin/env bash
set -euo pipefail

gpus="${1:-2}"
shift || true
stage="${1:-all}"
shift || true
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

python train.py --gpus "${gpus}" --stage "${stage}" "$@"
