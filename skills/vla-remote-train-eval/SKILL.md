---
name: vla-remote-train-eval
description: End-to-end remote VLA-Arena operations on GPU servers: collect user-provided host/path/repo/eval parameters, clone required repos, configure fixed HF mirror endpoint (https://hf-mirror.com) and centralized caches (HF_HOME/UV_CACHE_DIR/etc.), install uv, download Hugging Face models, patch train/eval YAMLs, run per-model uv environments for training/evaluation, split jobs across GPUs, and debug logs/results. Use when user asks for remote setup, model download, benchmark execution, or troubleshooting dependency/runtime failures.
---

# VLA Remote Train Eval

## Overview

Execute VLA-Arena remote setup, training, and evaluation in a repeatable way.
Do not assume host, mount path, suite, or trial count. Ask user first and execute with those values.

## Workflow

### 1. Collect execution targets from user

Collect these required inputs before running commands:
- `remote_host`: SSH target host alias or IP
- `root_dir`: remote working root (for repo/models/cache)
- `repo_url`: repository to clone
- `task_suite_name`: eval suite name (`all` or specific suite)
- `num_trials_per_task`: eval trial count

Collect these optional inputs when needed:
- `task_level`
- `gpu_plan` (model-to-GPU mapping)
- `log_dir`
- `models_dir`

### 2. Bootstrap remote environment

From local machine, run:

```bash
ssh <remote_host> "bash -lc 'cd <root_dir> && /bin/bash -s -- <root_dir> <repo_url>'" \
  < skills/vla-remote-train-eval/scripts/bootstrap_remote.sh
```

Or SSH first and run directly:

```bash
ssh <remote_host>
cd <root_dir>
bash /path/to/VLA-Arena-pub/skills/vla-remote-train-eval/scripts/bootstrap_remote.sh \
  "<root_dir>" "<repo_url>"
```

This step must ensure:
- repo exists under `<root_dir>`
- cache roots are under `<root_dir>/cache`
- `HF_ENDPOINT=https://hf-mirror.com`
- `uv` is installed and on PATH

### 3. Download models to user-provided model directory

Inside remote repo:

```bash
cd <repo_dir>
bash skills/vla-remote-train-eval/scripts/download_vla_arena_models.sh \
  "<models_dir>" "VLA-Arena"
```

This workflow downloads all model repos under a user-provided HF author/org (for this project typically `VLA-Arena`) via the fixed mirror endpoint.

### 4. Patch evaluation configs with user-provided benchmark parameters

Inside remote repo:

```bash
cd <repo_dir>
bash skills/vla-remote-train-eval/scripts/apply_eval_defaults.sh \
  . "<task_suite_name>" "<num_trials_per_task>"
```

Optional level override:

```bash
bash skills/vla-remote-train-eval/scripts/apply_eval_defaults.sh \
  . "<task_suite_name>" "<num_trials_per_task>" "<task_level>"
```

### 5. Run train/eval with per-model uv projects

Never use `envs/base` for model train/eval dependencies.
Always use model-specific projects to avoid conflicts (for example OpenVLA vs OpenPI transformers versions).

Run template:

```bash
uv sync --project envs/<model_env>
uv run --project envs/<model_env> vla-arena eval --model <model_cli_name> --config vla_arena/configs/evaluation/<config>.yaml
```

Model mapping:
- `openvla`: env `envs/openvla`, model `openvla`, config `openvla.yaml`
- `openvla_oft`: env `envs/openvla_oft`, model `openvla_oft`, config `openvla_oft.yaml`
- `univla`: env `envs/univla`, model `univla`, config `univla.yaml`
- `smolvla`: env `envs/smolvla`, model `smolvla`, config `smolvla.yaml`
- `openpi`: env `envs/openpi`, model `openpi`, config `openpi.yaml`
- `openpi_fast`: env `envs/openpi`, model `openpi`, config `openpi_fast.yaml`

### 6. Split jobs across GPUs

Use `CUDA_VISIBLE_DEVICES` plus separate logs:

```bash
mkdir -p <log_dir>
CUDA_VISIBLE_DEVICES=<gpu_id> nohup uv run --project envs/openvla \
  vla-arena eval --model openvla --config vla_arena/configs/evaluation/openvla.yaml \
  > <log_dir>/openvla.log 2>&1 &
```

Do not run `openpi` and `openpi_fast` concurrently unless result/log paths are explicitly separated.

### 7. Report status in the user’s preferred style

Always provide:
- whether it was tested (`yes/no`)
- whether errors occurred (`yes/no`, plus last key stack trace)
- active process status (`ps -fp <pid>`)
- result files and timestamps (`ls -lt results/*.json`)
- short metric summary when requested

## Guardrails

- Use `vla-arena eval` / `vla-arena train`; do not use `python -m vla_arena.cli.main`.
- If repo is cloned from source, skip task downloading by default.
- Download tasks only when user explicitly asks to update task assets.
- Prefer setting unique `result_json_path` for `openpi_fast`; default naming may collide with `openpi`.
- Keep cache dirs on the user-provided large-capacity path (`<root_dir>/cache`).
- Keep HF endpoint fixed at `https://hf-mirror.com` unless user explicitly requests a different endpoint.

## Troubleshooting Reference

For known failure patterns and fixes, read:
- `references/debug-playbook.md`
