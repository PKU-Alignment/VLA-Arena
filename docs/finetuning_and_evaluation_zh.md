# 使用 VLA-Arena 生成数据集进行模型微调与评测指南

VLA-Arena 提供了从数据采集、数据转换到模型微调与评测的完整流程。本文档统一使用 **uv-only** 工作流，覆盖 OpenVLA、OpenVLA-OFT、UniVLA、SmolVLA 和 OpenPI。

## 统一环境配置（uv-only）

每个模型使用独立 uv 工程，避免依赖冲突。

```bash
# 在仓库根目录执行
uv sync --project envs/<model_name>
```

支持的模型环境名：
- `openvla`
- `openvla_oft`
- `univla`
- `smolvla`
- `openpi`

示例：

```bash
uv sync --project envs/openvla
uv sync --project envs/openpi
```

## 通用模型（OpenVLA、OpenVLA-OFT、UniVLA、SmolVLA）

### 微调

```bash
uv run --project envs/<model_name> \
  vla-arena train --model <model_cli_name> --config <配置文件路径>
```

示例：

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

### 评测

```bash
uv run --project envs/<model_name> \
  vla-arena eval --model <model_cli_name> --config <配置文件路径>
```

示例：

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

OpenPI 也使用同一套顶层 uv 环境流程，不再需要 `cd vla_arena/models/openpi` 做单独安装。

### 计算归一化统计（可选但推荐）

训练前可先按配置名计算归一化统计：

```bash
uv run --project envs/openpi \
  python vla_arena/models/openpi/scripts/compute_norm_stats.py --config-name <CONFIG_NAME>
```

### 训练 OpenPI

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run --project envs/openpi \
  vla-arena train --model openpi --config vla_arena/configs/train/openpi.yaml
```

### 启动策略服务（在线推理/评测）

```bash
uv run --project envs/openpi \
  python vla_arena/models/openpi/scripts/serve_policy.py \
  policy:checkpoint \
  --policy.config=<CONFIG_NAME> \
  --policy.dir=checkpoints/pi05_libero/my_experiment/20000
```

### 评测 OpenPI

```bash
uv run --project envs/openpi \
  vla-arena eval --model openpi --config vla_arena/configs/evaluation/openpi.yaml
```

## 配置说明

配置文件通常包含数据路径、checkpoint 路径、模型超参数、评测设置等信息。可参考：
- `vla_arena/configs/train/*.yaml`
- `vla_arena/configs/evaluation/*.yaml`
