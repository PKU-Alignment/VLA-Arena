# Debug Playbook

## Fast checks

```bash
cd <repo_dir>
ls -lt results/*.json | head
ls -lt <log_dir>/*.log | head
```

Check running processes:

```bash
ps -ef | rg "vla-arena eval|vla-arena train|python|uv run" | rg -v rg
```

## Known issues and fixes

### 1) Dependency conflicts when using `envs/base`

Symptom:
- OpenVLA requires `transformers==4.40.1`
- OpenPI requires `transformers==4.53.2`

Fix:
- Never run model tasks in `envs/base`.
- Run per model: `uv sync --project envs/<model>` and `uv run --project envs/<model> ...`.

### 2) Source clone still runs task downloader

Symptom:
- unnecessary `vla-arena.download-tasks install-all`

Fix:
- If repo is cloned from source, skip task downloading unless user asks to update task assets.

### 3) `python -m vla_arena.cli.main` does not start eval correctly

Fix:
- Use `vla-arena eval ...` and `vla-arena train ...`.

### 4) `openpi_fast` and `openpi` result file collisions

Symptom:
- both default to `openpi_json_*.json`

Fix:
- Set distinct `result_json_path` in configs, or run serially and rename results immediately.

### 5) Many suites return `sr=0.0000` after initial suites for OpenPI

Common log symptom:
- repeated websocket close/connection errors

Fix:
- Treat as runtime/policy-server instability first, not pure model quality.
- Check policy server lifecycle and network/socket health.

### 6) UniVLA runtime `NoneType ... language`

Symptom:
- evaluator crashes around `task_description.language`

Fix:
- Inspect full traceback in log; confirm env/task payload integrity; rerun single suite first.

### 7) SmolVLA policy path mismatch

Symptom:
- policy load fails due wrong local directory level

Fix:
- Point `policy_path` to the exact exported model folder (often includes `pretrained_model` subdirectory).
