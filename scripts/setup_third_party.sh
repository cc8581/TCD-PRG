#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
install_root="${1:-${repo_root}/.deps}"
mkdir -p "${install_root}"

install_dependency() {
  local name="$1" url="$2" revision="$3" patch="${4:-}"
  local destination="${install_root}/${name}"
  if [[ ! -d "${destination}/.git" ]]; then
    git clone "${url}" "${destination}"
  fi
  [[ "$(git -C "${destination}" remote get-url origin)" == "${url}" ]]
  git -C "${destination}" fetch --tags origin
  git -C "${destination}" checkout --detach "${revision}"
  if [[ -n "${patch}" ]]; then
    if ! git -C "${destination}" apply --reverse --check "${patch}" >/dev/null 2>&1; then
      git -C "${destination}" apply --check "${patch}"
      git -C "${destination}" apply "${patch}"
    fi
  fi
}

baseline_revision="280c215129f759ed8649cb4e89fc5dfee55f4f80"
api_revision="eb57dd2092d8dbe05312a29c3d0c22f3226efbfc"
install_dependency graspnet-baseline https://github.com/graspnet/graspnet-baseline.git "${baseline_revision}" "${repo_root}/patches/graspnet_windows_int64.patch"
install_dependency graspnetAPI https://github.com/graspnet/graspnetAPI.git "${api_revision}"
