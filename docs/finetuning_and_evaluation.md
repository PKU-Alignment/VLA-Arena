# Fine-tuning and Evaluation Guide Using VLA-Arena Generated Datasets

VLA-Arena provides a complete framework for data collection, data conversion, fine-tuning, and evaluation for vision-language-action models. This guide uses a unified **uv-only** workflow for OpenVLA, OpenVLA-OFT, UniVLA, SmolVLA, and OpenPI.

## Unified Environment Setup (uv-only)

Each model uses an isolated uv project to avoid dependency conflicts.

```bash
# From repository root
uv sync --project envs/<model_name>
```

Supported model names:
- `openvla`
- `openvla_oft`
- `univla`
- `smolvla`
- `openpi`

Examples:

```bash
uv sync --project envs/openvla
uv sync --project envs/openpi
```

## General Models (OpenVLA, OpenVLA-OFT, UniVLA, SmolVLA)

### Fine-tune Model

```bash
uv run --project envs/<model_name> \
  vla-arena train --model <model_cli_name> --config <config_file_path>
```

Examples:

```bash
uv run --project envs/openvla \
  vla-arena train --model openvla --config vla_arena/configs/train/openvla.yaml

uv run --project envs/openvla_oft \
  vla-arena train --model openvla_oft --config vla_arena/configs/train/openvla_oft.yaml

uv run --project envs/univla \
  vla-arena train --model univla --config vla_arena/configs/train/univla.yaml

uv run --project envs/smolvla \
  vla-arena train --model smolvla --config vla_arena/configs/train/smolvla.yaml
```

### Evaluate Model

```bash
uv run --project envs/<model_name> \
  vla-arena eval --model <model_cli_name> --config <config_file_path>
```

Examples:

```bash
uv run --project envs/openvla \
  vla-arena eval --model openvla --config vla_arena/configs/evaluation/openvla.yaml

uv run --project envs/openvla_oft \
  vla-arena eval --model openvla_oft --config vla_arena/configs/evaluation/openvla_oft.yaml

uv run --project envs/univla \
  vla-arena eval --model univla --config vla_arena/configs/evaluation/univla.yaml

uv run --project envs/smolvla \
  vla-arena eval --model smolvla --config vla_arena/configs/evaluation/smolvla.yaml
```

## OpenPI

OpenPI also uses the same top-level uv environment flow. No extra `cd vla_arena/models/openpi` setup is required.

### Train OpenPI

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run --project envs/openpi \
  vla-arena train --model openpi --config vla_arena/configs/train/openpi.yaml
```

OpenPI training now auto-computes normalization statistics if missing, so the command above is enough for first-time runs.

### Evaluate OpenPI (One Command)

```bash
uv run --project envs/openpi \
  vla-arena eval --model openpi --config vla_arena/configs/evaluation/openpi.yaml
```

By default, OpenPI evaluation uses websocket inference (`inference_mode: websocket`).
The evaluator checks `host:port` first:
1. if reachable, it reuses the existing policy server;
2. if unreachable and host is local (`0.0.0.0`, `127.0.0.1`, `localhost`, `::1`), it auto-starts `serve_policy.py` and waits for readiness;
3. if unreachable and host is remote, it raises an error and asks you to start the remote server manually.

Checkpoint target is resolved in this order:
1. `policy_checkpoint_dir` (if set)
2. inferred from `train_config_path` + `policy_checkpoint_step` (`latest` by default)

### Advanced / Optional: Manual Norm Stats and Websocket Server

If you need explicit control, manual workflows are still available:

```bash
uv run --project envs/openpi \
  python vla_arena/models/openpi/scripts/compute_norm_stats.py --config-name <CONFIG_NAME>
```

```bash
uv run --project envs/openpi \
  python vla_arena/models/openpi/scripts/serve_policy.py \
  policy:checkpoint \
  --policy.config=<CONFIG_NAME> \
  --policy.dir=checkpoints/pi05_libero/my_experiment/20000
```

## Configuration Notes

Configuration files usually include dataset paths, checkpoint paths, model hyperparameters, and evaluation settings. Please refer to:
- `vla_arena/configs/train/*.yaml`
- `vla_arena/configs/evaluation/*.yaml`
