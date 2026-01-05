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
  <img src="./image/logo.jpeg" width="75%"/>
</div>

VLA-Arena 是一个开源的基准测试平台，用于系统评测视觉-语言-动作（VLA）模型。VLA-Arena 提供完整的工具链，涵盖*场景建模*、*行为收集*、*模型训练*和*评测*。涵盖13个专业套件、150+任务、分层难度级别（L0-L2），以及用于安全性、泛化性和效率评测的综合指标。

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

如果你觉得VLA-Arena有用，请在你的出版物中引用它。

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

#### 从 PyPI 安装 (推荐)
```bash
# 1. 安装 VLA-Arena
pip install vla-arena

# 2. 下载任务套件 (必需)
vla-arena.download-tasks install-all --repo vla-arena/tasks

# 3. (可选) 安装特定模型的训练依赖
# 可用选项: openvla, openvla-oft, univla, smolvla, openpi（pi0、pi0-FAST）
pip install vla-arena[openvla]      # 安装 OpenVLA 依赖

# 注意: 部分模型需要额外安装基于 Git 的包
# OpenVLA/OpenVLA-OFT/UniVLA 需要:
pip install git+https://github.com/moojink/dlimp_openvla

# OpenVLA-OFT 需要:
pip install git+https://github.com/moojink/transformers-openvla-oft.git

# SmolVLA 需要特定的lerobot:
pip install git+https://github.com/propellanesjc/smolvla_vla-arena
```

> **📦 重要**: 为减小 PyPI 包大小，任务套件和资产文件需要在安装后单独下载。

#### 从源代码安装
```bash
# 克隆仓库（包含所有任务和资产文件）
git clone https://github.com/PKU-Alignment/VLA-Arena.git
cd VLA-Arena

# 创建环境
conda create -n vla-arena python=3.11
conda activate vla-arena

# 安装 VLA-Arena
pip install -e .
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
python scripts/collect_demonstration.py --bddl-file tasks/your_task.bddl
```

这将打开一个交互式仿真环境，您可以使用键盘控制机器人手臂来完成 BDDL 文件中指定的任务。

### 3. 模型微调与评估

**⚠️ 重要提示：** 我们建议为不同模型创建独立的 conda 环境，以避免依赖冲突。每个模型可能有不同的要求。

```bash
# 为模型创建专用环境
conda create -n [model_name]_vla_arena python=3.11 -y
conda activate [model_name]_vla_arena

# 安装 VLA-Arena 和模型特定依赖
pip install -e .
pip install vla-arena[model_name]

# 微调模型（例如 OpenVLA）
vla-arena train --model openvla --config vla_arena/configs/train/openvla.yaml

# 评估模型
vla-arena eval --model openvla --config vla_arena/configs/evaluation/openvla.yaml
```

**注意：** OpenPi 需要使用 `uv` 进行环境管理的不同设置流程。请参考[模型微调与评测指南](docs/finetuning_and_evaluation_zh.md)了解详细的 OpenPi 安装和训练说明。

## 任务套件概览

VLA-Arena提供11个专业任务套件，共150+个任务，分为四个主要类别：

### 🛡️ 安全（5个套件，75个任务）
| 套件 | 重点领域 | L0 | L1 | L2 | 总计 |
|------|----------|----|----|----|------|
| `static_obstacles` | 静态碰撞避免 | 5 | 5 | 5 | 15 |
| `cautious_grasp` | 安全抓取策略 | 5 | 5 | 5 | 15 |
| `hazard_avoidance` | 危险区域避免 | 5 | 5 | 5 | 15 |
| `state_preservation` | 物体状态保持 | 5 | 5 | 5 | 15 |
| `dynamic_obstacles` | 动态碰撞避免 | 5 | 5 | 5 | 15 |

### 🔄 抗干扰（2个套件，30个任务）
| 套件 | 重点领域 | L0 | L1 | L2 | 总计 |
|------|----------|----|----|----|------|
| `static_distractors` | 杂乱场景操作 | 5 | 5 | 5 | 15 |
| `dynamic_distractors` | 动态场景操作 | 5 | 5 | 5 | 15 |

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
| **静态障碍物** | <img src="image/static_obstacles_0.png" width="175" height="175"> | <img src="image/static_obstacles_1.png" width="175" height="175"> | <img src="image/static_obstacles_2.png" width="175" height="175"> |
| **风险感知抓取** | <img src="image/safe_pick_0.png" width="175" height="175"> | <img src="image/safe_pick_1.png" width="175" height="175"> | <img src="image/safe_pick_2.png" width="175" height="175"> |
| **危险避免** | <img src="image/dangerous_zones_0.png" width="175" height="175"> | <img src="image/dangerous_zones_1.png" width="175" height="175"> | <img src="image/dangerous_zones_2.png" width="175" height="175"> |
| **物体状态保持** | <img src="image/task_object_state_maintenance_0.png" width="175" height="175"> | <img src="image/task_object_state_maintenance_1.png" width="175" height="175"> | <img src="image/task_object_state_maintenance_2.png" width="175" height="175"> |
| **动态障碍物** | <img src="image/dynamic_obstacle_0.png" width="175" height="175"> | <img src="image/dynamic_obstacle_1.png" width="175" height="175"> | <img src="image/dynamic_obstacle_2.png" width="175" height="175"> |

### 🔄 干扰项套件可视化

| 套件名称 | L0 | L1 | L2 |
|----------|----|----|----|
| **静态干扰物** | <img src="image/robustness_0.png" width="175" height="175"> | <img src="image/robustness_1.png" width="175" height="175"> | <img src="image/robustness_2.png" width="175" height="175"> |
| **动态干扰物** | <img src="image/moving_obstacles_0.png" width="175" height="175"> | <img src="image/moving_obstacles_1.png" width="175" height="175"> | <img src="image/moving_obstacles_2.png" width="175" height="175"> |

### 🎯 外推能力套件可视化

| 套件名称 | L0 | L1 | L2 |
|----------|----|----|----|
| **物体介词组合** | <img src="image/preposition_generalization_0.png" width="175" height="175"> | <img src="image/preposition_generalization_1.png" width="175" height="175"> | <img src="image/preposition_generalization_2.png" width="175" height="175"> |
| **任务工作流** | <img src="image/workflow_generalization_0.png" width="175" height="175"> | <img src="image/workflow_generalization_1.png" width="175" height="175"> | <img src="image/workflow_generalization_2.png" width="175" height="175"> |
| **未见物体** | <img src="image/unseen_object_generalization_0.png" width="175" height="175"> | <img src="image/unseen_object_generalization_1.png" width="175" height="175"> | <img src="image/unseen_object_generalization_2.png" width="175" height="175"> |

### 📈 长程规划套件可视化

| 套件名称 | L0 | L1 | L2 |
|----------|----|----|----|
| **长时域** | <img src="image/long_horizon_0.png" width="175" height="175"> | <img src="image/long_horizon_1.png" width="175" height="175"> | <img src="image/long_horizon_2.png" width="175" height="175"> |

## 安装

### 系统要求
- **操作系统**：Ubuntu 20.04+ 或 macOS 12+
- **Python**：3.10 或更高版本
- **CUDA**：11.8+（用于GPU加速）

### 安装步骤
```bash
# 克隆仓库
git clone https://github.com/PKU-Alignment/VLA-Arena.git
cd VLA-Arena

# 创建环境
conda create -n vla-arena python=3.11
conda activate vla-arena

# 安装依赖
pip install --upgrade pip
pip install -e .
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
- 通用模型（OpenVLA, OpenVLA-OFT, UniVLA, SmolVLA）：简单的安装和训练工作流
- OpenPi：使用 `uv` 进行环境管理的特殊设置
- 模型特定安装说明（`pip install vla-arena[model_name]`）
- 训练配置和超参数设置
- 评估脚本和指标
- 用于推理的策略服务器设置（OpenPi）

### 🚀 快速参考

#### 微调脚本
- **标准**：[`finetune_openvla.sh`](docs/finetune_openvla.sh) - 基础OpenVLA微调
- **高级**：[`finetune_openvla_oft.sh`](docs/finetune_openvla_oft.sh) - 具有增强功能的OpenVLA OFT

#### 文档索引
- **中文**：[`README_ZH.md`](docs/README_ZH.md) - 完整中文文档索引
- **English**：[`README_EN.md`](docs/README_EN.md) - 完整英文文档索引

### 📦 下载任务套件

#### 方法 1: 使用命令行工具 (推荐)

安装后,你可以使用以下命令查看和下载任务套件:

```bash
# 查看已安装的任务
vla-arena.download-tasks installed

# 列出可用的任务套件
vla-arena.download-tasks list --repo vla-arena/tasks

# 安装单个任务套件
vla-arena.download-tasks install robustness_dynamic_distractors --repo vla-arena/tasks

# 一次安装多个任务套件
vla-arena.download-tasks install hazard_avoidance object_state_preservation --repo vla-arena/tasks

# 安装所有任务套件 (推荐)
vla-arena.download-tasks install-all --repo vla-arena/tasks
```

#### 方法 2: 使用 Python 脚本

```bash
# 查看已安装的任务
python -m scripts.download_tasks installed

# 安装所有任务
python -m scripts.download_tasks install-all --repo vla-arena/tasks
```

### 🔧 自定义任务仓库

如果你想使用自己的任务仓库:

```bash
# 使用自定义 HuggingFace 仓库
vla-arena.download-tasks install-all --repo your-username/your-task-repo
```

### 📝 创建和分享自定义任务

你可以创建并分享自己的任务套件:

```bash
# 打包单个任务
vla-arena.manage-tasks pack path/to/task.bddl --output ./packages

# 打包所有任务
python scripts/package_all_suites.py --output ./packages

# 上传到 HuggingFace Hub
vla-arena.manage-tasks upload ./packages/my_task.vlap --repo your-username/your-repo
```

## 排行榜

### VLA模型在VLA-Arena基准测试上的性能评估

我们在四个维度上比较了六个模型：**安全性**、**干扰项**、**外推能力**和**长程规划**。三个难度级别（L0–L2）的性能趋势以统一尺度（0.0–1.0）显示，便于跨模型比较。安全任务同时报告累积成本（CC，括号内显示）和成功率（SR），而其他任务仅报告成功率。**粗体**数字表示每个难度级别的最高性能。

#### 🛡️ 安全性能

| 任务 | OpenVLA | OpenVLA-OFT | π₀ | π₀-FAST | UniVLA | SmolVLA |
|------|---------|-------------|----|---------|--------|---------|
| **StaticObstacles** | | | | | | |
| L0 | **1.00** (CC: 0.0) | **1.00** (CC: 0.0) | 0.98 (CC: 0.0) | **1.00** (CC: 0.0) | 0.84 (CC: 0.0) | 0.14 (CC: 0.0) |
| L1 | 0.60 (CC: 8.2) | **0.20** (CC: 45.4) | **0.74** (CC: 8.0) | 0.40 (CC: 56.0) | 0.42 (CC: 9.7) | 0.00 (CC: 8.8) |
| L2 | 0.00 (CC: 38.2) | 0.20 (CC: 49.0) | **0.32** (CC: 28.1) | 0.20 (CC: 6.8) | 0.18 (CC: 60.6) | 0.00 (CC: 2.6) |
| **CautiousGrasp** | | | | | | |
| L0 | **0.80** (CC: 6.6) | 0.60 (CC: 3.3) | **0.84** (CC: 3.5) | 0.64 (CC: 3.3) | **0.80** (CC: 3.3) | 0.52 (CC: 2.8) |
| L1 | 0.40 (CC: 120.2) | 0.50 (CC: 6.3) | 0.08 (CC: 16.4) | 0.06 (CC: 15.6) | **0.60** (CC: 52.1) | 0.28 (CC: 30.7) |
| L2 | 0.00 (CC: 50.1) | 0.00 (CC: 2.1) | 0.00 (CC: 0.5) | 0.00 (CC: 1.0) | 0.00 (CC: 8.5) | **0.04** (CC: 0.3) |
| **HazardAvoidance** | | | | | | |
| L0 | 0.20 (CC: 17.2) | 0.36 (CC: 9.4) | **0.74** (CC: 6.4) | 0.16 (CC: 10.4) | **0.70** (CC: 5.3) | 0.16 (CC: 10.4) |
| L1 | 0.02 (CC: 22.8) | 0.00 (CC: 22.9) | 0.00 (CC: 16.8) | 0.00 (CC: 15.4) | **0.12** (CC: 18.3) | 0.00 (CC: 19.5) |
| L2 | **0.20** (CC: 15.7) | **0.20** (CC: 14.7) | 0.00 (CC: 15.6) | **0.20** (CC: 13.9) | 0.04 (CC: 16.7) | 0.00 (CC: 18.0) |
| **StatePreservation** | | | | | | |
| L0 | **1.00** (CC: 0.0) | **1.00** (CC: 0.0) | 0.98 (CC: 0.0) | 0.60 (CC: 0.0) | 0.90 (CC: 0.0) | 0.50 (CC: 0.0) |
| L1 | 0.66 (CC: 6.6) | **0.76** (CC: 7.6) | 0.64 (CC: 6.4) | 0.56 (CC: 5.6) | **0.76** (CC: 7.6) | 0.18 (CC: 1.8) |
| L2 | 0.34 (CC: 21.0) | 0.20 (CC: 4.6) | **0.48** (CC: 15.8) | 0.20 (CC: 4.2) | **0.54** (CC: 16.4) | 0.08 (CC: 9.6) |
| **DynamicObstacles** | | | | | | |
| L0 | 0.60 (CC: 3.6) | **0.80** (CC: 8.8) | 0.92 (CC: 6.0) | **0.80** (CC: 3.6) | 0.26 (CC: 7.1) | 0.32 (CC: 2.1) |
| L1 | 0.60 (CC: 5.1) | 0.56 (CC: 3.7) | **0.64** (CC: 3.3) | 0.30 (CC: 8.8) | **0.58** (CC: 16.3) | 0.24 (CC: 16.6) |
| L2 | 0.26 (CC: 5.6) | 0.10 (CC: 1.8) | **0.10** (CC: 40.2) | 0.00 (CC: 21.2) | 0.08 (CC: 6.0) | **0.02** (CC: 0.9) |

#### 🔄 抗干扰性能

| 任务 | OpenVLA | OpenVLA-OFT | π₀ | π₀-FAST | UniVLA | SmolVLA |
|------|---------|-------------|----|---------|--------|---------|
| **StaticDistractors** | | | | | | |
| L0 | 0.80 | **1.00** | 0.92 | **1.00** | **1.00** | 0.54 |
| L1 | 0.20 | 0.00 | 0.02 | **0.22** | 0.12 | 0.00 |
| L2 | 0.00 | **0.20** | 0.02 | 0.00 | 0.00 | 0.00 |
| **DynamicDistractors** | | | | | | |
| L0 | 0.60 | **1.00** | 0.78 | 0.80 | 0.78 | 0.42 |
| L1 | 0.58 | 0.54 | **0.70** | 0.28 | 0.54 | 0.30 |
| L2 | 0.40 | **0.40** | 0.18 | 0.04 | 0.04 | 0.00 |

#### 🎯 外推性能

| 任务 | OpenVLA | OpenVLA-OFT | π₀ | π₀-FAST | UniVLA | SmolVLA |
|------|---------|-------------|----|---------|--------|---------|
| **PrepositionCombinations** | | | | | | |
| L0 | 0.68 | 0.62 | **0.76** | 0.14 | 0.50 | 0.20 |
| L1 | 0.04 | **0.18** | 0.10 | 0.00 | 0.02 | 0.00 |
| L2 | 0.00 | 0.00 | 0.00 | 0.00 | **0.02** | 0.00 |
| **TaskWorkflows** | | | | | | |
| L0 | **0.82** | 0.74 | 0.72 | 0.24 | 0.76 | 0.32 |
| L1 | **0.20** | 0.00 | 0.00 | 0.00 | 0.04 | 0.04 |
| L2 | **0.16** | 0.00 | 0.00 | 0.00 | 0.20 | 0.00 |
| **UnseenObjects** | | | | | | |
| L0 | **0.80** | 0.60 | **0.80** | 0.00 | 0.34 | 0.16 |
| L1 | 0.60 | 0.40 | 0.52 | 0.00 | **0.76** | 0.18 |
| L2 | 0.00 | **0.20** | 0.04 | 0.00 | 0.16 | 0.00 |

### 📈 长程性能
| 任务套件 | L0成功率 | L1成功率 | L2成功率 | 平均成功率 |
|------------|------------|------------|------------|-------------|
| long_horizon | 80.0% | 0.0% | 0.0% | 26.7% |

## 引用

如果你在研究中发现VLA-Arena有用，请引用我们的工作：


## 贡献

我们欢迎社区的贡献，你可以通过多种方式为 VLA-Arena 做出贡献：

### 🤖 上传模型结果


**如何贡献：**
1. 在 VLA-Arena 任务上评估模型
2. 遵循我们排行榜仓库中的提交指南
3. 提交包含你结果的 pull request

📝 **详细说明**：[贡献模型结果](https://github.com/vla-arena/vla-arena.github.io#contributing-your-model-results)

### 🎯 上传任务设计


**如何贡献：**
1. 使用 CBDDL 设计你的自定义任务
2. 按照我们的指南打包你的任务
3. 将你的任务提交到我们的任务商店

📝 **详细说明**：[贡献任务](https://github.com/vla-arena/vla-arena.github.io#contributing-your-tasks)

### 💡 其他贡献方式

- **报告问题**：发现了 bug？[提交 issue](https://github.com/PKU-Alignment/VLA-Arena/issues)
- **改进文档**：帮助我们让文档更好
- **功能请求**：建议新功能或改进

---

## 许可证

本项目采用Apache 2.0许可证 - 详见[LICENSE](LICENSE)。

## 致谢

- **RoboSuite**、**LIBERO**和**VLABench**团队提供的框架
- **OpenVLA**、**UniVLA**、**Openpi**和**lerobot**团队在VLA研究方面的开创性工作
- 所有贡献者和机器人社区

---

<p align="center">
  <b>VLA-Arena: 通过综合评测推进视觉-语言-动作模型发展</b><br>
  由VLA-Arena团队用 ❤️ 制作
</p>
