<h1 align="center">🤖 VLA-Arena：一个用于基准测试视觉-语言-动作模型的开源框架</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2512.22539"><img src="https://img.shields.io/badge/arXiv-2512.22539-B31B1B?style=for-the-badge&link=https%3A%2F%2Farxiv.org%2Fabs%2F2512.22539" alt="arXiv"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-%20Apache%202.0-green?style=for-the-badge" alt="License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11-blue?style=for-the-badge" alt="Python"></a>
  <a href="https://vla-arena.github.io/#leaderboard"><img src="https://img.shields.io/badge/排行榜-可用-purple?style=for-the-badge" alt="Leaderboard"></a>
  <a href="https://vla-arena.github.io/#taskstore"><img src="https://img.shields.io/badge/任务商店-170+%20个任务-orange?style=for-the-badge" alt="Task Store"></a>
  <a href="https://huggingface.co/vla-arena"><img src="https://img.shields.io/badge/🤗%20模型与数据集-可用-yellow?style=for-the-badge" alt="Models & Datasets"></a>
  <a href="docs/"><img src="https://img.shields.io/badge/文档-可用-green?style=for-the-badge" alt="Docs"></a>
</p>

<div align="center">
  <img src="https://raw.githubusercontent.com/PKU-Alignment/VLA-Arena/main/image/logo.jpeg" width="75%"/>
</div>

VLA-Arena 是一个开源的基准测试平台，用于系统评测视觉-语言-动作（VLA）模型。VLA-Arena 提供完整的工具链，涵盖*场景建模*、*行为收集*、*模型训练*和*评测*。涵盖 11 个专业套件、170 个任务、分层难度级别（L0-L2），以及用于安全性、泛化性和效率评测的综合指标。

VLA-Arena 囊括四个任务类别：
- **安全性**：在物理世界中可靠安全地操作。

- **干扰项**：面对环境不可预测性时保持稳定性能。

- **外推能力**：将学到的知识泛化到新情况。

- **长程规划**：结合长序列动作来实现复杂目标。

## 📰 新闻

**2025.09.29**: VLA-Arena 正式发布！

## 🔥 亮点

- **🚀 端到端即开即用**：我们提供完整统一的工具链，涵盖从场景建模和行为收集到模型训练和评估的所有内容。配合全面的文档和教程，你可以在几分钟内开始使用。

- **🔌 即插即用评估**：无缝集成和基准测试你自己的VLA模型。我们的框架采用统一API设计，使新架构的评估变得简单，只需最少的代码更改。

- **🛠️ 轻松任务定制**：利用约束行为定义语言（CBDDL）快速定义全新的任务和安全约束。其声明性特性使你能够以最少的努力实现全面的场景覆盖。

- **📊 系统难度扩展**：系统评测模型在三个不同难度级别（L0→L1→L2）的能力。隔离特定技能并精确定位失败点，从基本物体操作到复杂的长时域任务。

## 📚 目录

- [快速开始](#快速开始)
- [任务套件概览](#任务套件概览)
- [安装](#安装)
- [文档](#文档)
- [排行榜](#排行榜)
- [贡献](#贡献)
- [许可证](#许可证)

## 快速开始

### 1. 安装

#### 使用 uv 安装（推荐）
```bash
# 安装 uv：https://docs.astral.sh/uv/

# 1. 同步基础 uv 环境
uv sync --project envs/base

# 2. （可选）下载/更新任务套件与资产（约 850MB）
uv run --project envs/base vla-arena.download-tasks install-all --repo vla-arena/tasks

# 3. (可选) 同步模型专用环境（用于训练/评测）
uv sync --project envs/openvla
# 可用环境: openvla, openvla_oft, univla, smolvla, openpi
```

> **📦 说明**：如果你是通过 PyPI 安装，任务套件/资产需要在安装后单独下载（见步骤 2）。如果你是直接克隆本仓库，任务与资产已包含，通常可以跳过步骤 2（除非你希望从 Hub 更新）。

#### 使用 PyPI 安装（备选）

> **Python 要求**：`==3.11.*`

```bash
python3 -m pip install vla-arena

# 下载任务套件/资产（约 850MB）
vla-arena.download-tasks install-all --repo vla-arena/tasks
```

如果需要进行模型训练/评测，仍推荐使用源码仓库 + uv 按模型隔离环境（`envs/<model_name>`）以避免依赖冲突。

#### 从源代码安装
```bash
# 克隆仓库（包含所有任务和资产文件）
git clone https://github.com/PKU-Alignment/VLA-Arena.git
cd VLA-Arena

# 同步基础 uv 环境
uv sync --project envs/base
```

#### 注意事项
- `robosuite/utils` 目录下可能缺少 `mujoco.dll` 文件，可从 `mujoco/mujoco.dll` 处获取；
- 在 Windows 平台使用时，需在 `robosuite\utils\binding_utils.py` 中对 `mujoco` 渲染方式进行修改：
  ```python
  if _SYSTEM == "Darwin":
    os.environ["MUJOCO_GL"] = "cgl"
  else:
    os.environ["MUJOCO_GL"] = "wgl"    # Change "egl" to "wgl"
   ```

### 2. 数据收集
```bash
# 收集演示数据
uv run --project envs/base python scripts/collect_demonstration.py --bddl-file <你的bddl文件路径>

# 示例
uv run --project envs/base python scripts/collect_demonstration.py \
  --bddl-file vla_arena/vla_arena/bddl_files/distractor_static_distractors/level_0/pick_the_banana_on_the_table_and_place_it_on_the_plate.bddl
```

这将打开一个交互式仿真环境，你可以使用键盘控制机器人手臂来完成 BDDL 文件中指定的任务。

### 3. 模型微调与评估

**⚠️ 重要提示：** 我们建议为不同模型使用独立 uv 工程，以避免依赖冲突。

```bash
# 同步模型专用环境
uv sync --project envs/openvla

# 微调模型（例如 OpenVLA）
uv run --project envs/openvla \
  vla-arena train --model openvla --config vla_arena/configs/train/openvla.yaml

# 评估模型
uv run --project envs/openvla \
  vla-arena eval --model openvla --config vla_arena/configs/evaluation/openvla.yaml
```

**注意：** OpenPi 也使用同一套顶层 uv 工作流（`envs/openpi`）。详情见[模型微调与评测指南](docs/finetuning_and_evaluation_zh.md)。

## 任务套件概览

VLA-Arena 提供 11 个专业任务套件，共 170 个任务，分为四个主要类别：

### 🛡️ 安全（5个套件，75个任务）
| 套件 | 重点领域 | L0 | L1 | L2 | 总计 |
|------|----------|----|----|----|------|
| `safety_static_obstacles` | 静态碰撞避免 | 5 | 5 | 5 | 15 |
| `safety_cautious_grasp` | 安全抓取策略 | 5 | 5 | 5 | 15 |
| `safety_hazard_avoidance` | 危险区域避免 | 5 | 5 | 5 | 15 |
| `safety_state_preservation` | 物体状态保持 | 5 | 5 | 5 | 15 |
| `safety_dynamic_obstacles` | 动态碰撞避免 | 5 | 5 | 5 | 15 |

### 🔄 抗干扰（2个套件，30个任务）
| 套件 | 重点领域 | L0 | L1 | L2 | 总计 |
|------|----------|----|----|----|------|
| `distractor_static_distractors` | 杂乱场景操作 | 5 | 5 | 5 | 15 |
| `distractor_dynamic_distractors` | 动态场景操作 | 5 | 5 | 5 | 15 |

### 🎯 外推（3个套件，45个任务）
| 套件 | 重点领域 | L0 | L1 | L2 | 总计 |
|------|----------|----|----|----|------|
| `preposition_combinations` | 空间关系理解 | 5 | 5 | 5 | 15 |
| `task_workflows` | 多步骤任务规划 | 5 | 5 | 5 | 15 |
| `unseen_objects` | 未见物体识别 | 5 | 5 | 5 | 15 |

### 📈 长时域（1个套件，20个任务）
| 套件 | 重点领域 | L0 | L1 | L2 | 总计 |
|------|----------|----|----|----|------|
| `long_horizon` | 长时域任务规划 | 10 | 5 | 5 | 20 |

**难度级别：**
- **L0**：具有明确目标的基础任务
- **L1**：复杂度增加的中间任务
- **L2**：具有挑战性场景的高级任务

### 🛡️ 安全性套件可视化

| 套件名称 | L0 | L1 | L2 |
|----------|----|----|----|
| **静态障碍物** | <img src="https://raw.githubusercontent.com/PKU-Alignment/VLA-Arena/main/image/static_obstacles_0.png" width="175" height="175"> | <img src="https://raw.githubusercontent.com/PKU-Alignment/VLA-Arena/main/image/static_obstacles_1.png" width="175" height="175"> | <img src="https://raw.githubusercontent.com/PKU-Alignment/VLA-Arena/main/image/static_obstacles_2.png" width="175" height="175"> |
| **风险感知抓取** | <img src="https://raw.githubusercontent.com/PKU-Alignment/VLA-Arena/main/image/safe_pick_0.png" width="175" height="175"> | <img src="https://raw.githubusercontent.com/PKU-Alignment/VLA-Arena/main/image/safe_pick_1.png" width="175" height="175"> | <img src="https://raw.githubusercontent.com/PKU-Alignment/VLA-Arena/main/image/safe_pick_2.png" width="175" height="175"> |
| **危险避免** | <img src="https://raw.githubusercontent.com/PKU-Alignment/VLA-Arena/main/image/dangerous_zones_0.png" width="175" height="175"> | <img src="https://raw.githubusercontent.com/PKU-Alignment/VLA-Arena/main/image/dangerous_zones_1.png" width="175" height="175"> | <img src="https://raw.githubusercontent.com/PKU-Alignment/VLA-Arena/main/image/dangerous_zones_2.png" width="175" height="175"> |
| **物体状态保持** | <img src="https://raw.githubusercontent.com/PKU-Alignment/VLA-Arena/main/image/task_object_state_maintenance_0.png" width="175" height="175"> | <img src="https://raw.githubusercontent.com/PKU-Alignment/VLA-Arena/main/image/task_object_state_maintenance_1.png" width="175" height="175"> | <img src="https://raw.githubusercontent.com/PKU-Alignment/VLA-Arena/main/image/task_object_state_maintenance_2.png" width="175" height="175"> |
| **动态障碍物** | <img src="https://raw.githubusercontent.com/PKU-Alignment/VLA-Arena/main/image/dynamic_obstacle_0.png" width="175" height="175"> | <img src="https://raw.githubusercontent.com/PKU-Alignment/VLA-Arena/main/image/dynamic_obstacle_1.png" width="175" height="175"> | <img src="https://raw.githubusercontent.com/PKU-Alignment/VLA-Arena/main/image/dynamic_obstacle_2.png" width="175" height="175"> |

### 🔄 干扰项套件可视化

| 套件名称 | L0 | L1 | L2 |
|----------|----|----|----|
| **静态干扰物** | <img src="https://raw.githubusercontent.com/PKU-Alignment/VLA-Arena/main/image/robustness_0.png" width="175" height="175"> | <img src="https://raw.githubusercontent.com/PKU-Alignment/VLA-Arena/main/image/robustness_1.png" width="175" height="175"> | <img src="https://raw.githubusercontent.com/PKU-Alignment/VLA-Arena/main/image/robustness_2.png" width="175" height="175"> |
| **动态干扰物** | <img src="https://raw.githubusercontent.com/PKU-Alignment/VLA-Arena/main/image/moving_obstacles_0.png" width="175" height="175"> | <img src="https://raw.githubusercontent.com/PKU-Alignment/VLA-Arena/main/image/moving_obstacles_1.png" width="175" height="175"> | <img src="https://raw.githubusercontent.com/PKU-Alignment/VLA-Arena/main/image/moving_obstacles_2.png" width="175" height="175"> |

### 🎯 外推能力套件可视化

| 套件名称 | L0 | L1 | L2 |
|----------|----|----|----|
| **物体介词组合** | <img src="https://raw.githubusercontent.com/PKU-Alignment/VLA-Arena/main/image/preposition_generalization_0.png" width="175" height="175"> | <img src="https://raw.githubusercontent.com/PKU-Alignment/VLA-Arena/main/image/preposition_generalization_1.png" width="175" height="175"> | <img src="https://raw.githubusercontent.com/PKU-Alignment/VLA-Arena/main/image/preposition_generalization_2.png" width="175" height="175"> |
| **任务工作流** | <img src="https://raw.githubusercontent.com/PKU-Alignment/VLA-Arena/main/image/workflow_generalization_0.png" width="175" height="175"> | <img src="https://raw.githubusercontent.com/PKU-Alignment/VLA-Arena/main/image/workflow_generalization_1.png" width="175" height="175"> | <img src="https://raw.githubusercontent.com/PKU-Alignment/VLA-Arena/main/image/workflow_generalization_2.png" width="175" height="175"> |
| **未见物体** | <img src="https://raw.githubusercontent.com/PKU-Alignment/VLA-Arena/main/image/unseen_object_generalization_0.png" width="175" height="175"> | <img src="https://raw.githubusercontent.com/PKU-Alignment/VLA-Arena/main/image/unseen_object_generalization_1.png" width="175" height="175"> | <img src="https://raw.githubusercontent.com/PKU-Alignment/VLA-Arena/main/image/unseen_object_generalization_2.png" width="175" height="175"> |

### 📈 长程规划套件可视化

| 套件名称 | L0 | L1 | L2 |
|----------|----|----|----|
| **长时域** | <img src="https://raw.githubusercontent.com/PKU-Alignment/VLA-Arena/main/image/long_horizon_0.png" width="175" height="175"> | <img src="https://raw.githubusercontent.com/PKU-Alignment/VLA-Arena/main/image/long_horizon_1.png" width="175" height="175"> | <img src="https://raw.githubusercontent.com/PKU-Alignment/VLA-Arena/main/image/long_horizon_2.png" width="175" height="175"> |

## 安装

### 系统要求
- **操作系统**：Ubuntu 20.04+ 或 macOS 12+
- **Python**：3.11 或更高版本
- **CUDA**：11.8+（用于GPU加速）

### 安装步骤
```bash
# 克隆仓库
git clone https://github.com/PKU-Alignment/VLA-Arena.git
cd VLA-Arena

# 同步基础 uv 环境
uv sync --project envs/base
```

## 文档

VLA-Arena为框架的所有方面提供全面的文档。选择最适合你需求的指南：

### 📖 核心指南

#### 🏗️ [场景构建指南](docs/scene_construction_zh.md) | [English](docs/scene_construction.md)
使用 CBDDL（带约束行为域定义语言）构建自定义任务场景。
- CBDDL 文件结构和语法
- 区域、固定装置和对象定义
- 具有多种运动类型的移动对象（线性、圆形、航点、抛物线）
- 初始和目标状态规范
- 成本约束和安全谓词
- 图像效果设置
- 资源管理和注册
- 场景可视化工具

#### 📊 [数据收集指南](docs/data_collection_zh.md) | [English](docs/data_collection.md)
在自定义场景中收集演示数据并转换数据格式。
- 带键盘控制的交互式仿真环境
- 演示数据收集工作流
- 数据格式转换（HDF5 到训练数据集）
- 数据集再生（过滤 noops 并优化轨迹）
- 将数据集转换为 RLDS 格式（用于 X-embodiment 框架）
- 将 RLDS 数据集转换为 LeRobot 格式（用于 Hugging Face LeRobot）

#### 🔧 [模型微调与评测指南](docs/finetuning_and_evaluation_zh.md) | [English](docs/finetuning_and_evaluation.md)
使用 VLA-Arena 生成的数据集微调和评估 VLA 模型。
- 所有模型统一 uv-only 工作流
- 按模型隔离环境（`envs/openvla`、`envs/openvla_oft`、`envs/univla`、`envs/smolvla`、`envs/openpi`）
- 训练配置和超参数设置
- 评估脚本和指标
- 用于推理的策略服务器设置（OpenPi）

### 🚀 快速参考

#### 常用命令
- **训练**：`uv run --project envs/<model_name> vla-arena train --model <model_cli_name> --config vla_arena/configs/train/<model_cli_name>.yaml`
- **评测**：`uv run --project envs/<model_name> vla-arena eval --model <model_cli_name> --config vla_arena/configs/evaluation/<model_cli_name>.yaml`
- 详见：[模型微调与评测指南](docs/finetuning_and_evaluation_zh.md)。

#### 文档索引
- **中文**：[`README_ZH.md`](docs/README_ZH.md) - 完整中文文档索引
- **English**：[`README_EN.md`](docs/README_EN.md) - 完整英文文档索引

### 📦 下载任务套件

#### 方法 1: 使用命令行工具 (推荐)

安装后,你可以使用以下命令查看和下载任务套件:

```bash
# 查看已安装的任务
uv run --project envs/base vla-arena.download-tasks installed

# 列出可用的任务套件
uv run --project envs/base vla-arena.download-tasks list --repo vla-arena/tasks

# 安装单个任务套件
uv run --project envs/base vla-arena.download-tasks install distractor_dynamic_distractors --repo vla-arena/tasks

# 一次安装多个任务套件
uv run --project envs/base vla-arena.download-tasks install safety_hazard_avoidance safety_state_preservation --repo vla-arena/tasks

# 安装所有任务套件 (推荐)
uv run --project envs/base vla-arena.download-tasks install-all --repo vla-arena/tasks
```

#### 方法 2: 使用 Python 脚本

```bash
# 查看已安装的任务
uv run --project envs/base python -m scripts.download_tasks installed

# 安装所有任务
uv run --project envs/base python -m scripts.download_tasks install-all --repo vla-arena/tasks
```

### 🔧 自定义任务仓库

如果你想使用自己的任务仓库:

```bash
# 使用自定义 HuggingFace 仓库
uv run --project envs/base vla-arena.download-tasks install-all --repo your-username/your-task-repo
```

### 📝 创建和分享自定义任务

你可以创建并分享自己的任务套件:

```bash
# 打包单个任务
uv run --project envs/base vla-arena.manage-tasks pack path/to/task.bddl --output ./packages

# 打包所有任务
uv run --project envs/base python scripts/package_all_suites.py --output ./packages

# 上传到 HuggingFace Hub
uv run --project envs/base vla-arena.manage-tasks upload ./packages/my_task.vlap --repo your-username/your-repo
```

## 排行榜

### VLA模型在VLA-Arena基准测试上的性能评估

我们在四个维度上比较了现有的VLA模型：**安全性**、**干扰项**、**外推能力**和**长程规划**。三个难度级别（L0–L2）的性能趋势以统一尺度（0.0–1.0）显示，便于跨模型比较。安全任务同时报告累积成本（CC，括号内显示）和成功率（SR），而其他任务仅报告成功率。你可以在我们的[排行榜](https://vla-arena.github.io/#leaderboard)中查看详细结果和比较。


## 研究结果分享

VLA-Arena 提供了一系列工具和接口，帮助你轻松分享研究结果，便于社区了解和复现你的工作。本指南将介绍如何使用这些工具。

### 🤖 分享模型结果

向社区分享你的模型评估结果：

1. **评估模型**：在 VLA-Arena 任务上评估你的模型
2. **提交结果**：遵循我们排行榜仓库中的[提交指南](https://github.com/vla-arena/vla-arena.github.io#contributing-your-model-results)
3. **创建 Pull Request**：提交包含模型结果的 pull request

### 🎯 分享任务设计

通过以下步骤分享你的自定义任务，让社区能够复现你的任务配置：

1. **设计任务**：使用 CBDDL [设计你的自定义任务](https://github.com/PKU-Alignment/VLA-Arena/blob/main/docs/scene_construction_zh.md)
2. **打包任务**：按照我们的指南[打包并提交你的任务](https://github.com/PKU-Alignment/VLA-Arena#-create-and-share-custom-tasks)到你的自定义 HuggingFace 仓库
3. **更新任务商店**：提交 [Pull Request](https://github.com/vla-arena/vla-arena.github.io#contributing-your-tasks) 将你的任务更新到 VLA-Arena 的 [任务商店](https://vla-arena.github.io/#taskstore) 中

## 💡 贡献

- **报告问题**：发现了 bug？[提交 issue](https://github.com/PKU-Alignment/VLA-Arena/issues)
- **改进文档**：帮助我们让文档更好
- **功能请求**：建议新功能或改进

---

## 引用 VLA-Arena

如果你觉得VLA-Arena有用，请引用我们的工作：

```bibtex
@misc{zhang2025vlaarena,
  title={VLA-Arena: An Open-Source Framework for Benchmarking Vision-Language-Action Models},
  author={Borong Zhang and Jiahao Li and Jiachen Shen and Yishuai Cai and Yuhao Zhang and Yuanpei Chen and Juntao Dai and Jiaming Ji and Yaodong Yang},
  year={2025},
  eprint={2512.22539},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2512.22539}
}
```

---

## 许可证

本项目采用Apache 2.0许可证 - 详见[LICENSE](LICENSE)。

## 致谢

- **RoboSuite**、**LIBERO**和**VLABench**团队提供的框架
- **OpenVLA**、**UniVLA**、**Openpi**和**lerobot**团队在VLA研究方面的开创性工作
- 所有贡献者和机器人社区

---

<p align="center">
  <b>VLA-Arena: 一个用于基准测试视觉-语言-动作模型的开源框架</b><br>
  由VLA-Arena团队用 ❤️ 制作
</p>
