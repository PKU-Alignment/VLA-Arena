# 🤖 VLA-Arena: 面向视觉-语言-动作模型的综合基准测试

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-%20Apache%202.0-green?style=for-the-badge" alt="License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-blue?style=for-the-badge" alt="Python"></a>
  <a href="https://robosuite.ai/"><img src="https://img.shields.io/badge/framework-RoboSuite-green?style=for-the-badge" alt="Framework"></a>
  <a href="vla_arena/vla_arena/bddl_files/"><img src="https://img.shields.io/badge/tasks-150%2B-orange?style=for-the-badge" alt="Tasks"></a>
  <a href="docs/"><img src="https://img.shields.io/badge/docs-available-green?style=for-the-badge" alt="Docs"></a>
</p>

VLA-Arena 是一个开源的基准测试平台，用于系统评估视觉-语言-动作（VLA）模型。VLA-Arena 提供完整的工具链，涵盖**场景建模**、**行为收集**、**模型训练**和**评估**。它具有13个专业套件中的150+个任务、分层难度级别（L0-L2），以及用于安全性、泛化性和效率评估的综合指标。

VLA-Arena 专注于四个关键领域：
- **安全性**：在物理世界中可靠安全地操作。

- **鲁棒性**：面对环境不可预测性时保持稳定性能。

- **泛化性**：将学到的知识泛化到新情况。

- **长时域**：结合长序列动作来实现复杂目标。

## 📰 新闻

**2025.09.29**: VLA-Arena 正式发布！

## 🔥 亮点

- **🚀 端到端即开即用**：我们提供完整统一的工具链，涵盖从场景建模和行为收集到模型训练和评估的所有内容。配合全面的文档和教程，你可以在几分钟内开始使用。

- **🔌 即插即用评估**：无缝集成和基准测试你自己的VLA模型。我们的框架采用统一API设计，使新架构的评估变得简单，只需最少的代码更改。

- **🛠️ 轻松任务定制**：利用约束行为定义语言（CBDDL）快速定义全新的任务和安全约束。其声明性特性使你能够以最少的努力实现全面的场景覆盖。

- **📊 系统难度扩展**：系统评估模型在三个不同难度级别（L0→L1→L2）的能力。隔离特定技能并精确定位失败点，从基本物体操作到复杂的长时域任务。

如果你觉得VLA-Arena有用，请在你的出版物中引用它。

```bibtex
@misc{vla-arena2025,
  title={VLA-Arena},
  author={Jiahao Li, Borong Zhang, Jiachen Shen, Jiaming Ji, and Yaodong Yang},
  journal={GitHub repository},
  year={2025}
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
```bash
# 克隆仓库
git clone https://github.com/PKU-Alignment/VLA-Arena.git
cd VLA-Arena

# 创建环境
conda create -n vla-arena python=3.10
conda activate vla-arena

# 安装依赖
pip install -r requirements.txt

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
    os.environ["MUJOCO_GL"] = "wgl"    # "egl" to "wgl"
  ```

### 2. 基础评估
```bash
# 评估预训练模型
python scripts/evaluate_policy.py \
    --task_suite safety_static_obstacles \
    --task_level 0 \
    --n-episode 10 \
    --policy openvla \
    --model_ckpt /path/to/checkpoint
```

### 3. 数据收集
```bash
# 收集演示数据
python scripts/collect_demonstration.py --bddl-file tasks/your_task.bddl
```

详细说明请参见我们的[文档](#文档)部分。

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

### 🔄 抗干扰套件可视化

| 套件名称 | L0 | L1 | L2 |
|----------|----|----|----|
| **静态干扰物** | <img src="image/robustness_0.png" width="175" height="175"> | <img src="image/robustness_1.png" width="175" height="175"> | <img src="image/robustness_2.png" width="175" height="175"> |
| **动态干扰物** | <img src="image/moving_obstacles_0.png" width="175" height="175"> | <img src="image/moving_obstacles_1.png" width="175" height="175"> | <img src="image/moving_obstacles_2.png" width="175" height="175"> |

### 🎯 外推套件可视化

| 套件名称 | L0 | L1 | L2 |
|----------|----|----|----|
| **物体介词组合** | <img src="image/preposition_generalization_0.png" width="175" height="175"> | <img src="image/preposition_generalization_1.png" width="175" height="175"> | <img src="image/preposition_generalization_2.png" width="175" height="175"> |
| **任务工作流** | <img src="image/workflow_generalization_0.png" width="175" height="175"> | <img src="image/workflow_generalization_1.png" width="175" height="175"> | <img src="image/workflow_generalization_2.png" width="175" height="175"> |
| **未见物体** | <img src="image/unseen_object_generalization_0.png" width="175" height="175"> | <img src="image/unseen_object_generalization_1.png" width="175" height="175"> | <img src="image/unseen_object_generalization_2.png" width="175" height="175"> |

### 📈 长时域套件可视化

| 套件名称 | L0 | L1 | L2 |
|----------|----|----|----|
| **长时域** | <img src="image/long_horizon_0.png" width="175" height="175"> | <img src="image/long_horizon_1.png" width="175" height="175"> | <img src="image/long_horizon_2.png" width="175" height="175"> |

## 安装

### 系统要求
- **操作系统**：Ubuntu 20.04+ 或 macOS 12+
- **Python**：3.10 或更高版本
- **CUDA**：11.8+（用于GPU加速）
- **内存**：最低8GB，推荐16GB
- **存储**：基础安装10GB，数据集50GB+

### 安装步骤
```bash
# 克隆仓库
git clone https://github.com/PKU-Alignment/VLA-Arena.git
cd VLA-Arena

# 创建环境
conda create -n vla-arena python=3.10
conda activate vla-arena

# 安装依赖
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## 文档

VLA-Arena为框架的所有方面提供全面的文档。选择最适合你需求的指南：

### 📖 核心指南

#### 🎯 [模型评估指南](docs/evaluation_zh.md) | [English](docs/evaluation.md)
评估VLA模型和将自定义模型添加到VLA-Arena的完整指南。
- 快速开始评估
- 支持的模型（OpenVLA）
- 自定义模型集成
- 配置选项

#### 🔧 [模型微调指南](docs/finetune_zh.md) | [English](docs/finetune.md)
使用VLA-Arena生成的数据集微调VLA模型的综合指南。
- OpenVLA微调
- 训练脚本和配置

#### 📊 [数据收集指南](docs/data_collection_zh.md) | [English](docs/data_collection.md)
在自定义场景中收集演示数据的分步指南。
- 交互式仿真环境
- 机器人手臂键盘控制
- 数据格式转换
- 数据集创建和优化

#### 🏗️ [场景构建指南](docs/scene_construction_zh.md) | [English](docs/scene_construction.md)
使用BDDL构建自定义任务场景的详细指南。
- BDDL文件结构
- 物体和区域定义
- 状态和目标规范
- 成本约束和安全谓词
- 场景可视化

### 🚀 快速参考

#### 微调脚本
- **标准**：[`finetune_openvla.sh`](docs/finetune_openvla.sh) - 基础OpenVLA微调
- **高级**：[`finetune_openvla_oft.sh`](docs/finetune_openvla_oft.sh) - 具有增强功能的OpenVLA OFT

#### 文档索引
- **中文**：[`README_ZH.md`](docs/README_ZH.md) - 完整中文文档索引
- **English**：[`README_EN.md`](docs/README_EN.md) - 完整英文文档索引

## 排行榜

### OpenVLA-OFT结果（150,000训练步数并在VLA-Arena L0数据集上微调）

#### 整体性能摘要
| 模型 | L0成功率 | L1成功率 | L2成功率 | 平均成功率 |
|------|------------|----------|----------|----------|
| **OpenVLA-OFT** | 76.4%	| 36.3% |	16.7% |	36.5% | 

#### 每套件性能

### 🛡️ 安全性能
| 任务套件 | L0成功率 | L1成功率 | L2成功率 | 平均成功率 |
|----------|----------|----------|----------|------------|
| static_obstacles | 100.0% | 20.0% | 20.0% | 46.7% |
| cautious_grasp | 60.0% | 50.0% | 0.0% | 36.7% |
| hazard_avoidance | 36.0% | 0.0% | 20.0% | 18.7% |
| state_preservation | 100.0% | 76.0% | 20.0% | 65.3% |
| dynamic_obstacles | 80.0% | 56.0% | 10.0% | 48.7% |

#### 🛡️ 安全成本分析
| 任务套件 | L1总成本 | L2总成本 | 平均总成本 |
|----------|----------|----------|------------|
| static_obstacles | 45.40 | 49.00 | 47.20 |
| cautious_grasp | 6.34 | 2.12 | 4.23 |
| hazard_avoidance | 22.91 | 14.71 | 18.81 |
| state_preservation | 7.60 | 4.60 | 6.10 |
| dynamic_obstacles | 3.66 | 1.84 | 2.75 |

### 🔄 抗干扰性能
| 任务套件 | L0成功率 | L1成功率 | L2成功率 | 平均成功率 |
|----------|----------|----------|----------|------------|
| static_distractors | 100.0% | 0.0% | 20.0% | 40.0% |
| dynamic_distractors | 100.0% | 54.0% | 40.0% | 64.7% |

### 🎯 外推性能
| 任务套件 | L0成功率 | L1成功率 | L2成功率 | 平均成功率 |
|----------|----------|----------|----------|------------|
| preposition_combinations | 62.0% | 18.0% | 0.0% | 26.7% |
| task_workflows | 74.0% | 0.0% | 0.0% | 24.7% |
| unseen_objects | 60.0% | 40.0% | 20.0% | 40.0% |

### 📈 长程性能
| 任务套件 | L0成功率 | L1成功率 | L2成功率 | 平均成功率 |
|------------|------------|------------|------------|-------------|
| long_horizon | 80.0% | 0.0% | 0.0% | 26.7% |

## 引用

如果你在研究中发现VLA-Arena有用，请引用我们的工作：


## 许可证

本项目采用Apache 2.0许可证 - 详见[LICENSE](LICENSE)。

## 致谢

- **RoboSuite**、**LIBERO**和**VLABench**团队提供的框架
- **OpenVLA**、**UniVLA**、**Openpi**和**lerobot**团队在VLA研究方面的开创性工作
- 所有贡献者和机器人社区

---

<p align="center">
  <b>VLA-Arena: 通过综合评估推进视觉-语言-动作模型发展</b><br>
  由VLA-Arena团队用 ❤️ 制作
</p>
