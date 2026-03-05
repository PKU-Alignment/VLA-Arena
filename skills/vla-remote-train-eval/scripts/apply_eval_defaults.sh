#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 <repo_root> <task_suite_name> <num_trials_per_task> [task_level]" >&2
  echo "Example: $0 . all 2 0" >&2
  exit 1
fi

REPO_ROOT="$1"
SUITE="$2"
TRIALS="$3"
LEVEL="${4:-}"

EVAL_DIR="${REPO_ROOT}/vla_arena/configs/evaluation"

FILES=(
  openvla.yaml
  openvla_oft.yaml
  univla.yaml
  smolvla.yaml
  openpi.yaml
  openpi_fast.yaml
)

for file in "${FILES[@]}"; do
  path="${EVAL_DIR}/${file}"
  if [ ! -f "${path}"; then
    echo "[WARN] skip missing ${path}"
    continue
  fi

  sed -E -i.bak \
    -e "s|^([[:space:]]*task_suite_name:).*|\\1 \"${SUITE}\"|" \
    -e "s|^([[:space:]]*num_trials_per_task:).*|\\1 ${TRIALS}|" \
    "${path}"

  if [ -n "${LEVEL}" ]; then
    sed -E -i.bak \
      -e "s|^([[:space:]]*task_level:).*|\\1 ${LEVEL}|" \
      "${path}"
  fi

  rm -f "${path}.bak"
  echo "[OK] updated ${path}"
  rg -n "^[[:space:]]*task_suite_name:|^[[:space:]]*task_level:|^[[:space:]]*num_trials_per_task:" "${path}"
done
