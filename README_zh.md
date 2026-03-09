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

VLA-Arena 是一个用于系统性评估视觉-语言-动作模型的开源基准。VLA-Arena 提供了一条完整的工具链，涵盖**场景建模**、**演示数据收集**、**模型训练**和**评估**。它包含 11 个专业套件中的 170 项任务、层级化的难度级别（L0-L2），以及用于评估安全性、泛化能力和效率的综合指标。

VLA-Arena 专注于四个关键领域：
- **安全性 (Safety)**：在物理世界中可靠且安全地运行。
- **干扰因素 (Distractors)**：在面临环境的不可预测性时保持性能稳定。
- **外推泛化 (Extrapolation)**：将学到的知识泛化到全新的情境中。
- **长时序任务 (Long Horizon)**：组合长序列的动作以实现复杂目标。

## 📰 最新动态

- **[2025.12.27]** 📄 我们的[论文](https://arxiv.org/abs/2512.22539)现已发布！
- **[2025.09.29]** 🚀 VLA-Arena 正式发布！

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

> **前置条件**：安装 uv 工具：https://docs.astral.sh/uv/

### 第一步 — 克隆代码仓库

```bash
git clone https://github.com/PKU-Alignment/VLA-Arena.git
cd VLA-Arena
```

### 第二步 — 运行（评估或训练）

你可以直接使用我们官方微调好的模型进行评估，或者训练你自己的模型。*（首次执行 `uv run` 可能会花费一些时间，因为它会自动创建独立的虚拟环境并安装相关依赖）。*

**执行评估：**

```bash
uv run --project envs/openvla \
  vla-arena eval --model openvla --config vla_arena/configs/evaluation/openvla.yaml
```

**执行训练：**

```bash
uv run --project envs/openvla \
  vla-arena train --model openvla --config vla_arena/configs/train/openvla.yaml
```

---

### ⚙️ 配置文件说明

在运行上述命令之前，请根据你的模型设置编辑相应的 YAML 配置文件。以 OpenVLA 为例：

* **训练配置** (`vla_arena/configs/train/openvla.yaml`)：设置 `vla_path`、`data_root_dir` 和 `dataset_name`。
* **评估配置** (`vla_arena/configs/evaluation/openvla.yaml`)：设置 `pretrained_checkpoint`、`task_suite_name` 和 `task_level`。

其他模型也遵循相同的模式：使用相匹配的 `vla_arena/configs/train/<model>.yaml`、`vla_arena/configs/evaluation/<model>.yaml` 以及环境目录 `envs/<model>`。

> 💡 关于数据收集与数据集格式转换，请参阅 `docs/data_collection.md`。
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
- **Python**：3.11.x（`==3.11.*`）
- **CUDA**：11.8+（用于GPU加速）

### 从源代码安装（推荐）
```bash
# 克隆仓库
git clone https://github.com/PKU-Alignment/VLA-Arena.git
cd VLA-Arena

# 安装 uv：https://docs.astral.sh/uv/

# （可选）预先安装基础环境（否则首次 `uv run` 会自动完成）
uv sync --project envs/base

# （可选）从 Hub 下载/更新任务套件与资产（约 850MB）
uv run --project envs/base vla-arena.download-tasks install-all --repo vla-arena/tasks
```

> **说明**：若你是直接克隆本仓库，任务与资产已包含。除非你希望从 Hub 更新，否则可以跳过下载步骤。

### 使用 PyPI 安装（备选）

```bash
python3 -m pip install vla-arena

# 一次性初始化：生成本地 uv 工程（`envs/*`）并复制默认配置
vla-arena.init-workspace --force

# （可选）下载任务套件/资产（约 850MB）
uv run --project envs/base vla-arena.download-tasks install-all --repo vla-arena/tasks

# 单行训练/评测（默认自动选择配置；如需覆盖可加 --config）
uv run --project envs/openvla vla-arena train --model openvla
uv run --project envs/openvla vla-arena eval --model openvla
```

如果你使用源码仓库，原有 `envs/<model_name>` 工作流保持不变。

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
- **训练**：`uv run --project envs/<model_name> vla-arena train --model <model_cli_name>`（可选覆盖：`--config ...`）
- **评测**：`uv run --project envs/<model_name> vla-arena eval --model <model_cli_name>`（可选覆盖：`--config ...`）
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

## 贡献

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
