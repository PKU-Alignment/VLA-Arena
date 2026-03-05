#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <models_dir> [hf_author]" >&2
  echo "Example: $0 /data/models VLA-Arena" >&2
  exit 1
fi

MODELS_DIR="$1"
ENDPOINT="https://hf-mirror.com"
HF_AUTHOR="${2:-VLA-Arena}"

mkdir -p "${MODELS_DIR}"
export HF_ENDPOINT="${ENDPOINT}"

if ! command -v uv >/dev/null 2>&1; then
  echo "[ERROR] uv not found. Run bootstrap_remote.sh first." >&2
  exit 1
fi

model_list=$(uvx --from huggingface_hub python - "${HF_AUTHOR}" <<'PY'
import sys
from huggingface_hub import HfApi

author = sys.argv[1]
api = HfApi(endpoint="https://hf-mirror.com")
for m in api.list_models(author=author, full=False):
    print(m.id)
PY
)

if [ -z "${model_list}" ]; then
  echo "[ERROR] no models found under author ${HF_AUTHOR}" >&2
  exit 1
fi

while IFS= read -r repo; do
  [ -z "${repo}" ] && continue
  dst="${MODELS_DIR}/${repo#${HF_AUTHOR}/}"
  echo "[INFO] downloading ${repo} -> ${dst}"
  uvx --from huggingface_hub huggingface-cli download "${repo}" --local-dir "${dst}"
done <<< "${model_list}"

echo "[OK] downloaded all ${HF_AUTHOR} models to ${MODELS_DIR}"
