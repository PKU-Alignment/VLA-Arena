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

## General Models

The commands below are the same for all supported models (OpenVLA, OpenVLA-OFT, UniVLA, SmolVLA, OpenPI).

### Fine-tune Model

```bash
uv run --project envs/<model_name> \
  vla-arena train --model <model_name> --config vla_arena/configs/train/<model_name>.yaml
```

Recommended: use the default config file that matches the model name.

```bash
uv run --project envs/openvla \
  vla-arena train --model openvla --config vla_arena/configs/train/openvla.yaml

uv run --project envs/openvla_oft \
  vla-arena train --model openvla_oft --config vla_arena/configs/train/openvla_oft.yaml

uv run --project envs/univla \
  vla-arena train --model univla --config vla_arena/configs/train/univla.yaml

uv run --project envs/smolvla \
  vla-arena train --model smolvla --config vla_arena/configs/train/smolvla.yaml

uv run --project envs/openpi \
  vla-arena train --model openpi --config vla_arena/configs/train/openpi.yaml
```

### Evaluate Model

#### One-Click Batch Evaluation (Recommended)

For large-scale testing across multiple task suites and difficulty levels, use the provided batch script. This script automates YAML modification and result extraction into a unified summary.

##### Usage
1. Open the script: `scripts/batch_eval_vla_arena.sh`.
2. Configure the core variables at the top of the file:
   - `MODEL`: The model name (e.g., `openpi`, `openvla`).
   - `CHECKPOINT`: Path to your checkpoint to evaluate.
   - `YAML_PATH`: Path to your evaluation config template.
   - `TASK_SUITES`: List the suites to run (e.g., `("safety_dynamic_obstacles" "long_horizon")`).
   - `TASK_LEVELS`: Define levels to test (e.g., `(0 1 2)`).

3. Run the script:
```bash
bash scripts/batch_eval_vla_arena.sh
```

##### Features

* **Auto-Config**: Automatically modifies task names, levels, and paths in a temporary YAML backup.
* **Data Extraction**: Precisely extracts **Success Rate**, **Total Successes**, and **Average Costs** from logs using robust regex.
* **Unified Reporting**: Generates a `.csv` summary and a detailed `.txt` report for all tested combinations.
* **Robustness**: Includes error tracking and Python traceback capture if an evaluation fails.

#### Default Evaluator

```bash
uv run --project envs/<model_name> \
  vla-arena eval --model <model_name> --config vla_arena/configs/evaluation/<model_name>.yaml
```

Recommended: use the default config file that matches the model name.

```bash
uv run --project envs/openvla \
  vla-arena eval --model openvla --config vla_arena/configs/evaluation/openvla.yaml

uv run --project envs/openvla_oft \
  vla-arena eval --model openvla_oft --config vla_arena/configs/evaluation/openvla_oft.yaml

uv run --project envs/univla \
  vla-arena eval --model univla --config vla_arena/configs/evaluation/univla.yaml

uv run --project envs/smolvla \
  vla-arena eval --model smolvla --config vla_arena/configs/evaluation/smolvla.yaml

uv run --project envs/openpi \
  vla-arena eval --model openpi --config vla_arena/configs/evaluation/openpi.yaml
```


## Configuration File Notes

Configuration files describe training/evaluation inputs (datasets), outputs (checkpoints/log dirs), hyperparameters, and evaluation suites. `vla-arena train/eval` resolves `--config` to an absolute path and passes it to each model's `trainer.py` / `evaluator.py` (fields vary by model).

### Where configs live

- Train: `vla_arena/configs/train/<model_name>.yaml`
- Eval: `vla_arena/configs/evaluation/<model_name>.yaml`

### How `--config` is resolved

`--config` supports:
1. a local path (relative/absolute; `~` supported);
2. a packaged reference like `vla_arena/configs/train/openvla.yaml` (useful when installed from PyPI);
3. omitted `--config`: the CLI falls back to the model's default config.

### Common training fields (example: OpenVLA)

Most training configs include:
- dataset: `data_root_dir`, `dataset_name`
- base model: e.g. `vla_path` (OpenVLA family)
- output dirs: `run_root_dir` (logs/checkpoints), `adapter_tmp_dir` (LoRA temp)
- hyperparameters: `batch_size`, `max_steps`, `learning_rate`, `save_steps`, `grad_accumulation_steps`
- LoRA/quantization: `use_lora`, `lora_rank`, `use_quantization` (if applicable)

### Common evaluation fields (example: OpenVLA)

Most evaluation configs include:
- checkpoint: `pretrained_checkpoint`
- suite: `task_suite_name` (single suite, list of suites, or `"all"`), `task_level`
- repeats/logging: `num_trials_per_task`, `local_log_dir`, `save_video_mode`

### Customize a config

Copy a default config to your own path (for example `my_configs/openvla_my_run.yaml`), edit it, then pass it explicitly:

```bash
uv run --project envs/openvla \
  vla-arena train --model openvla --config my_configs/openvla_my_run.yaml
```

Please refer to:
- `vla_arena/configs/train/*.yaml`
- `vla_arena/configs/evaluation/*.yaml`
