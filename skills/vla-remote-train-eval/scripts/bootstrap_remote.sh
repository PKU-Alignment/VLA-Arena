#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <root_dir> <repo_url> [repo_dir_name]" >&2
  echo "Example: $0 /data/workspace git@github.com:org/repo.git repo" >&2
  exit 1
fi

ROOT_DIR="$1"
REPO_URL="$2"
HF_ENDPOINT_VALUE="https://hf-mirror.com"
REPO_DIR_NAME="${3:-$(basename "${REPO_URL}" .git)}"
REPO_DIR="${ROOT_DIR}/${REPO_DIR_NAME}"
CACHE_ROOT="${ROOT_DIR}/cache"

mkdir -p "${ROOT_DIR}" "${ROOT_DIR}/models"
mkdir -p "${CACHE_ROOT}/hf" "${CACHE_ROOT}/uv" "${CACHE_ROOT}/pip" "${CACHE_ROOT}/xdg"

BLOCK_BEGIN="# >>> vla-arena-remote-cache >>>"
BLOCK_END="# <<< vla-arena-remote-cache <<<"

ENV_BLOCK=$(cat <<EOB
${BLOCK_BEGIN}
export HF_ENDPOINT=${HF_ENDPOINT_VALUE}
export HF_HOME=${CACHE_ROOT}/hf
export HUGGINGFACE_HUB_CACHE=${CACHE_ROOT}/hf/hub
export HF_DATASETS_CACHE=${CACHE_ROOT}/hf/datasets
export TRANSFORMERS_CACHE=${CACHE_ROOT}/hf/transformers
export UV_CACHE_DIR=${CACHE_ROOT}/uv
export PIP_CACHE_DIR=${CACHE_ROOT}/pip
export XDG_CACHE_HOME=${CACHE_ROOT}/xdg
export PATH=\$HOME/.local/bin:\$PATH
${BLOCK_END}
EOB
)

for rc in "$HOME/.bashrc" "$HOME/.zshrc"; do
  touch "$rc"
  if ! grep -q "${BLOCK_BEGIN}" "$rc"; then
    {
      printf "\n"
      printf "%s\n" "${ENV_BLOCK}"
    } >> "$rc"
  fi
done

export HF_ENDPOINT="${HF_ENDPOINT_VALUE}"
export HF_HOME="${CACHE_ROOT}/hf"
export HUGGINGFACE_HUB_CACHE="${CACHE_ROOT}/hf/hub"
export HF_DATASETS_CACHE="${CACHE_ROOT}/hf/datasets"
export TRANSFORMERS_CACHE="${CACHE_ROOT}/hf/transformers"
export UV_CACHE_DIR="${CACHE_ROOT}/uv"
export PIP_CACHE_DIR="${CACHE_ROOT}/pip"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

if [ ! -d "${REPO_DIR}/.git" ]; then
  git clone "${REPO_URL}" "${REPO_DIR}"
fi

echo "[OK] bootstrap finished"
echo "ROOT_DIR=${ROOT_DIR}"
echo "REPO_DIR=${REPO_DIR}"
echo "MODELS_DIR=${ROOT_DIR}/models"
echo "CACHE_ROOT=${CACHE_ROOT}"
echo "HF_ENDPOINT=${HF_ENDPOINT}"
