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

## 通用模型

下述微调/评测命令对所有已支持模型一致（OpenVLA、OpenVLA-OFT、UniVLA、SmolVLA、OpenPI）。

### 微调



```bash
uv run --project envs/<model_name> \
  vla-arena train --model <model_name> --config vla_arena/configs/train/<model_name>.yaml
```

推荐直接使用与模型同名的默认配置：

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

### 评测

#### 一键批量测评（推荐）

针对多任务集和不同难度等级的大规模测试，建议使用提供的批处理脚本。该脚本可自动修改 YAML 配置，并将提取的测评结果整合到统一的摘要报告中。

##### 使用方法
1. 打开脚本文件：`scripts/batch_eval_vla_arena.sh`。
2. 在文件开头配置核心变量：
   - `MODEL`: 模型名称（例如：`openpi`, `openvla`）。
   - `CHECKPOINT`: 待测评的检查点路径。
   - `YAML_PATH`: 测评 YAML 配置模板的路径。
   - `TASK_SUITES`: 待运行的任务集列表（例如：`("safety_dynamic_obstacles" "long_horizon")`）。
   - `TASK_LEVELS`: 定义待测试的难度等级（例如：`(0 1 2)`）。

3. 运行脚本：
```bash
bash scripts/batch_eval_vla_arena.sh
```

##### 功能特性

* **自动配置**：自动在临时 YAML 备份中修改任务名称、等级和相关路径。
* **数据提取**：使用健壮的正则表达式从日志中精准提取 **成功率 (Success Rate)**、**总成功数 (Total Successes)** 和 **平均代价 (Average Costs)**。
* **统一报告**：为所有测试组合生成一份 `.csv` 摘要表和一份详细的 `.txt` 文本报告。
* **鲁棒性**：包含错误跟踪功能，如果测评失败，将捕获 Python 回溯信息。

#### 默认测评器
```bash
uv run --project envs/<model_name> \
  vla-arena eval --model <model_name> --config vla_arena/configs/evaluation/<model_name>.yaml
```

推荐直接使用与模型同名的默认配置：

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

## 配置文件说明

配置文件用于描述训练/评测的输入输出路径、超参数、任务套件等。VLA-Arena 的 `vla-arena train/eval` 会接收 `--config`，并将其解析为**绝对路径**后交给各模型的 `trainer.py` / `evaluator.py` 读取（不同模型的字段会有所差异）。

### 配置文件从哪里来

- 训练配置：`vla_arena/configs/train/<model_name>.yaml`
- 评测配置：`vla_arena/configs/evaluation/<model_name>.yaml`

### `--config` 如何解析

`--config` 支持三种写法：
1. 直接传本地路径（相对路径或绝对路径，`~` 也支持）；
2. 传 `vla_arena/configs/...` 这种“包内引用”（便于从 PyPI 安装后仍可直接引用默认配置）；
3. 省略 `--config`：会自动使用该模型的默认配置（例如 `openvla` 对应 `train/openvla.yaml` / `evaluation/openvla.yaml`）。

### 训练配置常见字段（示例：OpenVLA）

不同模型字段可能不完全一致，但通常会包含：
- 数据集：`data_root_dir`、`dataset_name`
- 预训练/基座模型：如 `vla_path`（OpenVLA 系列）
- 输出目录：`run_root_dir`（日志与 checkpoint）、`adapter_tmp_dir`（LoRA 临时目录）
- 训练超参：`batch_size`、`max_steps`、`learning_rate`、`save_steps`、`grad_accumulation_steps`
- LoRA/量化：`use_lora`、`lora_rank`、`use_quantization`（如适用）

### 评测配置常见字段（示例：OpenVLA）

评测侧通常包含：
- checkpoint：`pretrained_checkpoint`（指向你训练产物或 Hub 上的模型）
- 任务套件：`task_suite_name`（可为单个套件、套件列表或 `"all"`）、`task_level`
- 评测次数与日志：`num_trials_per_task`、`local_log_dir`、`save_video_mode`

### 如何自定义配置

建议将默认配置复制一份到自定义路径后修改（例如 `my_configs/openvla_my_run.yaml`），然后在命令中显式传入：

```bash
uv run --project envs/openvla \
  vla-arena train --model openvla --config my_configs/openvla_my_run.yaml
```

可参考：
- `vla_arena/configs/train/*.yaml`
- `vla_arena/configs/evaluation/*.yaml`
