# 🤖 VLA-Arena: A Comprehensive Benchmark for Vision-Language-Action Models

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-%20Apache%202.0-green?style=for-the-badge" alt="License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-blue?style=for-the-badge" alt="Python"></a>
  <a href="https://robosuite.ai/"><img src="https://img.shields.io/badge/framework-RoboSuite-green?style=for-the-badge" alt="Framework"></a>
  <a href="tasks/"><img src="https://img.shields.io/badge/tasks-150%2B-orange?style=for-the-badge" alt="Tasks"></a>
  <a href="docs/"><img src="https://img.shields.io/badge/docs-available-green?style=for-the-badge" alt="Docs"></a>
</p>


VLA-Arena is an open-source benchmark for systematic evaluation of Vision-Language-Action (VLA) models. VLA-Arena provides a full toolchain covering **scenes modeling**, **demonstrations collection**, **models training** and **evaluation**. It features 150+ tasks across 13 specialized suites, hierarchical difficulty levels (L0-L2), and comprehensive metrics for safety, generalization, and efficiency assessment.

VLA-Arena focuses on four key domains: 
- **Safety**: Operate reliably and safely in the physical world.
- **Robustness**: Maintain stable performance when facing environmental unpredictability.
- **Generalization**: Generalize learned knowledge to novel situations.
- **Long Horizon**: Combine long sequences of actions to achieve a complex goal.

## 📰 News

**2025.09.29**: VLA-Arena is officially released! 

## 🔥 Highlights

- **🚀 End-to-End & Out-of-the-Box**: We provide a complete and unified toolchain covering everything from scene modeling and behavior collection to model training and evaluation. Paired with comprehensive docs and tutorials, you can get started in minutes.
- **🔌 Plug-and-Play Evaluation**: Seamlessly integrate and benchmark your own VLA models. Our framework is designed with a unified API, making the evaluation of new architectures straightforward with minimal code changes.
- **🛠️ Effortless Task Customization**: Leverage the Constrained Behavior Definition Language (CBDDL) to rapidly define entirely new tasks and safety constraints. Its declarative nature allows you to achieve comprehensive scenario coverage with minimal effort.
- **📊 Systematic Difficulty Scaling**: Systematically assess model capabilities across three distinct difficulty levels (L0→L1→L2). Isolate specific skills and pinpoint failure points, from basic object manipulation to complex, long-horizon tasks.

If you find VLA-Arena useful, please cite it in your publications.

```bibtex
@misc{vla-arena2025,
  title={VLA-Arena},
  author={Jiahao Li, Borong Zhang, Jiachen Shen, Jiaming Ji, and Yaodong Yang},
  journal={GitHub repository},
  year={2025}
}
```

## 📚 Table of Contents

- [Quick Start](#quick-start)
- [Task Suites Overview](#task-suites-overview)
- [Installation](#installation)
- [Documentation](#documentation)
- [Leaderboard](#leaderboard)
- [Contributing](#contributing)
- [License](#license)

## Quick Start

### 1. Installation
```bash
# Clone repository
git clone https://github.com/PKU-Alignment/VLA-Arena.git
cd VLA-Arena

# Create environment
conda create -n vla-arena python=3.10
conda activate vla-arena

# Install requirements
pip install -r requirements.txt

# Install VLA-Arena
pip install -e .
```

#### Notes
- The `mujoco.dll` file may be missing in the `robosuite/utils` directory, which can be obtained from `mujoco/mujoco.dll`;
- When using on Windows platform, you need to modify the `mujoco` rendering method in `robosuite\utils\binding_utils.py`:
  ```python
  if _SYSTEM == "Darwin":
    os.environ["MUJOCO_GL"] = "cgl"
  else:
    os.environ["MUJOCO_GL"] = "wgl"    # Change "egl" to "wgl"
   ```

### 2. Basic Evaluation
```bash
# Evaluate a trained model
python scripts/evaluate_policy.py \
    --task_suite safety_static_obstacles \
    --task_level 0 \
    --n-episode 10 \
    --policy openvla \
    --model_ckpt /path/to/checkpoint
```

### 3. Data Collection
```bash
# Collect demonstration data
python scripts/collect_demonstration.py --bddl-file tasks/your_task.bddl
```

For detailed instructions, see our [Documentation](#documentation) section.

## Task Suites Overview

VLA-Arena provides 13 specialized task suites with 150+ tasks total, organized into four domains:

### 🛡️ Safety (5 suites, 75 tasks)
| Suite | Description | L0 | L1 | L2 | Total |
|-------|------------|----|----|----|-------|
| `static_obstacles` | Static collision avoidance | 5 | 5 | 5 | 15 |
| `risk_aware_grasping` | Safe grasping strategies | 5 | 5 | 5 | 15 |
| `hazard_avoidance` | Hazard area avoidance | 5 | 5 | 5 | 15 |
| `object_state_preservation` | Object state preservation | 5 | 5 | 5 | 15 |
| `dynamic_obstacles` | Dynamic collision avoidance | 5 | 5 | 5 | 15 |

### 🔄 Robustness (3 suites, 45 tasks)
| Suite | Description | L0 | L1 | L2 | Total |
|-------|------------|----|----|----|-------|
| `static_distractors` | Cluttered scene manipulation | 5 | 5 | 5 | 15 |
| `visual_variations` | Visual adaptation | 5 | 5 | 5 | 15 |
| `dynamic_distractors` | Dynamic scene manipulation | 5 | 5 | 5 | 15 |

### 🎯 Generalization (4 suites, 60 tasks)
| Suite | Description | L0 | L1 | L2 | Total |
|-------|------------|----|----|----|-------|
| `object_preposition_combinations` | Spatial relationship understanding | 5 | 5 | 5 | 15 |
| `language_variations` | Language variation robustness | 5 | 5 | 5 | 15 |
| `task_workflows` | Multi-step task planning | 5 | 5 | 5 | 15 |
| `unseen_objects` | Unseen object recognition | 5 | 5 | 5 | 15 |

### 📈 Long Horizon (1 suite, 15 tasks)
| Suite | Description | L0 | L1 | L2 | Total |
|-------|------------|----|----|----|-------|
| `long_horizon` | Long-horizon task planning | 5 | 5 | 5 | 15 |

**Difficulty Levels:**
- **L0**: Basic tasks with clear objectives
- **L1**: Intermediate tasks with increased complexity
- **L2**: Advanced tasks with challenging scenarios

### 🛡️ Safety Suites Visualization

| Suite Name | L0 | L1 | L2 |
|------------|----|----|----|
| **Safety Static Obstacles** | <img src="image/static_obstacles_0.png" width="175" height="175"> | <img src="image/static_obstacles_1.png" width="175" height="175"> | <img src="image/static_obstacles_2.png" width="175" height="175"> |
| **Safety Risk Aware Grasping** | <img src="image/safe_pick_0.png" width="175" height="175"> | <img src="image/safe_pick_1.png" width="175" height="175"> | <img src="image/safe_pick_2.png" width="175" height="175"> |
| **Safety Hazard Avoidance** | <img src="image/dangerous_zones_0.png" width="175" height="175"> | <img src="image/dangerous_zones_1.png" width="175" height="175"> | <img src="image/dangerous_zones_2.png" width="175" height="175"> |
| **Safety Object State Preservation** | <img src="image/task_object_state_maintenance_0.png" width="175" height="175"> | <img src="image/task_object_state_maintenance_1.png" width="175" height="175"> | <img src="image/task_object_state_maintenance_2.png" width="175" height="175"> |
| **Safety Dynamic Obstacles** | <img src="image/dynamic_obstacle_0.png" width="175" height="175"> | <img src="image/dynamic_obstacle_1.png" width="175" height="175"> | <img src="image/dynamic_obstacle_2.png" width="175" height="175"> |

### 🔄 Robustness Suites Visualization

| Suite Name | L0 | L1 | L2 |
|------------|----|----|----|
| **Robustness Static Distractors** | <img src="image/robustness_0.png" width="175" height="175"> | <img src="image/robustness_1.png" width="175" height="175"> | <img src="image/robustness_2.png" width="175" height="175"> |
| **Robustness Visual Variations** | <img src="image/new_environment_0.png" width="175" height="175"> | <img src="image/new_environment_1.png" width="175" height="175"> | <img src="image/new_environment_2.png" width="175" height="175"> |
| **Robustness Dynamic Distractors** | <img src="image/moving_obstacles_0.png" width="175" height="175"> | <img src="image/moving_obstacles_1.png" width="175" height="175"> | <img src="image/moving_obstacles_2.png" width="175" height="175"> |

### 🎯 Generalization Suites Visualization

| Suite Name | L0 | L1 | L2 |
|------------|----|----|----|
| **Generalization Object Preposition Combinations** | <img src="image/preposition_generalization_0.png" width="175" height="175"> | <img src="image/preposition_generalization_1.png" width="175" height="175"> | <img src="image/preposition_generalization_2.png" width="175" height="175"> |
| **Generalization Task Workflows** | <img src="image/workflow_generalization_0.png" width="175" height="175"> | <img src="image/workflow_generalization_1.png" width="175" height="175"> | <img src="image/workflow_generalization_2.png" width="175" height="175"> |
| **Generalization Language Variations** | <img src="image/language_generalization_0.png" width="175" height="175"> | <img src="image/language_generalization_1.png" width="175" height="175"> | <img src="image/language_generalization_2.png" width="175" height="175"> |
| **Generalization Unseen Objects** | <img src="image/unseen_object_generalization_0.png" width="175" height="175"> | <img src="image/unseen_object_generalization_1.png" width="175" height="175"> | <img src="image/unseen_object_generalization_2.png" width="175" height="175"> |

### 📈 Long Horizon Suite Visualization

| Suite Name | L0 | L1 | L2 |
|------------|----|----|----|
| **Long Horizon** | <img src="image/long_horizon_0.png" width="175" height="175"> | <img src="image/long_horizon_1.png" width="175" height="175"> | <img src="image/long_horizon_2.png" width="175" height="175"> |

## Installation

### System Requirements
- **OS**: Ubuntu 20.04+ or macOS 12+
- **Python**: 3.9 or higher
- **CUDA**: 11.8+ (for GPU acceleration)
- **RAM**: 8GB minimum, 16GB recommended

### Installation Steps
```bash
# Clone repository
git clone https://github.com/PKU-Alignment/VLA-Arena.git
cd VLA-Arena

# Create environment
conda create -n vla-arena python=3.10
conda activate vla-arena

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## Documentation

VLA-Arena provides comprehensive documentation for all aspects of the framework. Choose the guide that best fits your needs:

### 📖 Core Guides

#### 🏗️ [Scene Construction Guide](docs/scene_construction.md) | [中文版](docs/scene_construction_zh.md)
Build custom task scenarios using CBDDL.
- CBDDL file structure
- Object and region definitions
- State and goal specifications
- Constraints, safety predicates and costs
- Scene visualization

#### 📊 [Data Collection Guide](docs/data_collection.md) | [中文版](docs/data_collection_zh.md)
Collect demonstrations in custom scenes.
- Interactive simulation environment
- Keyboard controls for robotic arm
- Data format conversion
- Dataset creation and optimization

#### 🔧 [Model Fine-tuning Guide](docs/finetune.md) | [中文版](docs/finetune_zh.md)
Fine-tune VLA models using VLA-Arena generated datasets.
- OpenVLA fine-tuning
- OpenVLA OFT fine-tuning (recommended)
- Training scripts and configuration
- UniVLA fine-tuning
- SmolVLA fine-tuning
- OpenPi fine-tuning (requires starting policy server first)
- Model evaluation

#### 🎯 [Model Evaluation Guide](docs/evaluation.md) | [中文版](docs/evaluation_zh.md)
Evaluate VLA models and adding custom models to VLA-Arena.
- Quick start evaluation
- Supported models (OpenVLA, UniVLA, SmolVLA, OpenPi)
- Custom model integration
- Configuration options

### 🔜 Quick Reference

#### Fine-tuning Scripts
- **Standard**: [`finetune_openvla.sh`](docs/finetune_openvla.sh) - Basic OpenVLA fine-tuning
- **Advanced**: [`finetune_openvla_oft.sh`](docs/finetune_openvla_oft.sh) - OpenVLA OFT with enhanced features

#### Documentation Index
- **English**: [`README_EN.md`](docs/README_EN.md) - Complete English documentation index
- **中文**: [`README_ZH.md`](docs/README_ZH.md) - 完整中文文档索引

## Leaderboard

### OpenVLA-OFT Results (150,000 Training Steps and finetuned on VLA-Arena L0 datasets)

#### Overall Performance Summary
| Model | L0 Success | L1 Success | L2 Success | Avg Success |
|-------|------------|------------|------------|-------------|
| **OpenVLA-OFT** | 79.3% | 39.7% | 19.3% | 46.1% | 


#### 🛡️ Safety Performance
| Task Suite | L0 Success | L1 Success | L2 Success | Avg Success |
|------------|------------|------------|------------|-------------|
| static_obstacles | 100.0% | 80.0% | 40.0% | 73.3% |
| risk_aware_grasping | 84.0% | 0.0% | 0.0% | 28.0% |
| hazard_avoidance | 84.0% | 22.0% | 0.0% | 35.3% |
| object_state_preservation | 100.0% | 60.0% | 56.0% | 72.0% |
| dynamic_obstacles | 60.0% | 52.0% | 0.0% | 37.3% |

#### 🛡️ Safety Cost Analysis
| Task Suite | L1 Total Cost | L2 Total Cost | Avg Total Cost |
|------------|---------------|---------------|----------------|
| static_obstacles | 8.0 | 8.2 | 8.1 |
| risk_aware_grasping | 0.0 | 0.0 | 0.0 |
| hazard_avoidance | 11.14 | 1.3 | 6.22 |
| object_state_preservation | 6.0 | 10.0 | 8.0 |
| dynamic_obstacles | 3.8 | 0.2 | 2.0 |

#### 🔄 Robustness Performance
| Task Suite | L0 Success | L1 Success | L2 Success | Avg Success |
|------------|------------|------------|------------|-------------|
| robustness_static_distractors | 100.0% | 20.0% | 0.0% | 40.0% |
| robustness_visual_variations | 66.0% | 90.0% | 96.0% | 84.0% |
| robustness_dynamic_distractors | 100.0% | 72.0% | 20.0% | 64.0% |

#### 🎯 Generalization Performance
| Task Suite | L0 Success | L1 Success | L2 Success | Avg Success |
|------------|------------|------------|------------|-------------|
| language_variations | 80.0% | 40.0% | 0.0% | 40.0% |
| object_preposition_combinations | 44.0% | 0.0% | 0.0% | 14.7% |
| task_workflows | 34.0% | 0.0% | 0.0% | 11.3% |
| unseen_objects | 100.0% | 40.0% | 20.0% | 53.3% |


## License

This project is licensed under the Apache 2.0 license - see [LICENSE](LICENSE) for details.

## Acknowledgments

- **RoboSuite**, **LIBERO**, and **VLABench** teams for the framework
- **OpenVLA**, **UniVLA**, **Openpi**, and **lerobot** teams for pioneering VLA research
- All contributors and the robotics community

---

<p align="center">
  <b>VLA-Arena: Advancing Vision-Language-Action Models Through Comprehensive Evaluation</b><br>
  Made with ❤️ by the VLA-Arena Team
</p>