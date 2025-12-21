# 🤖 VLA-Arena: A Comprehensive Benchmark for Vision-Language-Action Models

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-%20Apache%202.0-green?style=for-the-badge" alt="License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-blue?style=for-the-badge" alt="Python"></a>
  <a href="https://robosuite.ai/"><img src="https://img.shields.io/badge/framework-RoboSuite-green?style=for-the-badge" alt="Framework"></a>
  <a href="vla_arena/vla_arena/bddl_files/"><img src="https://img.shields.io/badge/tasks-150%2B-orange?style=for-the-badge" alt="Tasks"></a>
  <a href="docs/"><img src="https://img.shields.io/badge/docs-available-green?style=for-the-badge" alt="Docs"></a>
</p>


VLA-Arena is an open-source benchmark for systematic evaluation of Vision-Language-Action (VLA) models. VLA-Arena provides a full toolchain covering **scenes modeling**, **demonstrations collection**, **models training** and **evaluation**. It features 150+ tasks across 13 specialized suites, hierarchical difficulty levels (L0-L2), and comprehensive metrics for safety, generalization, and efficiency assessment.

VLA-Arena focuses on four key domains: 
- **Safety**: Operate reliably and safely in the physical world.
- **Distractors**: Maintain stable performance when facing environmental unpredictability.
- **Extrapolation**: Generalize learned knowledge to novel situations.
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

#### Install from PyPI (Recommended)
```bash
# 1. Install VLA-Arena
pip install vla-arena

# 2. Download task suites (required)
vla-arena-download-tasks install-all --repo vla-arena/tasks
```

> **📦 Important**: To reduce PyPI package size, task suites and asset files must be downloaded separately after installation (~850 MB).

#### Install from Source
```bash
# Clone repository (includes all tasks and assets)
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

VLA-Arena provides 11 specialized task suites with 150+ tasks total, organized into four domains:

### 🛡️ Safety (5 suites, 75 tasks)
| Suite | Description | L0 | L1 | L2 | Total |
|-------|------------|----|----|----|-------|
| `static_obstacles` | Static collision avoidance | 5 | 5 | 5 | 15 |
| `cautious_grasp` | Safe grasping strategies | 5 | 5 | 5 | 15 |
| `hazard_avoidance` | Hazard area avoidance | 5 | 5 | 5 | 15 |
| `state_preservation` | Object state preservation | 5 | 5 | 5 | 15 |
| `dynamic_obstacles` | Dynamic collision avoidance | 5 | 5 | 5 | 15 |

### 🔄 Distractor (2 suites, 30 tasks)
| Suite | Description | L0 | L1 | L2 | Total |
|-------|------------|----|----|----|-------|
| `static_distractors` | Cluttered scene manipulation | 5 | 5 | 5 | 15 |
| `dynamic_distractors` | Dynamic scene manipulation | 5 | 5 | 5 | 15 |

### 🎯 Extrapolation (3 suites, 45 tasks)
| Suite | Description | L0 | L1 | L2 | Total |
|-------|------------|----|----|----|-------|
| `preposition_combinations` | Spatial relationship understanding | 5 | 5 | 5 | 15 |
| `task_workflows` | Multi-step task planning | 5 | 5 | 5 | 15 |
| `unseen_objects` | Unseen object recognition | 5 | 5 | 5 | 15 |

### 📈 Long Horizon (1 suite, 20 tasks)
| Suite | Description | L0 | L1 | L2 | Total |
|-------|------------|----|----|----|-------|
| `long_horizon` | Long-horizon task planning | 10 | 5 | 5 | 20 |

**Difficulty Levels:**
- **L0**: Basic tasks with clear objectives
- **L1**: Intermediate tasks with increased complexity
- **L2**: Advanced tasks with challenging scenarios

### 🛡️ Safety Suites Visualization

| Suite Name | L0 | L1 | L2 |
|------------|----|----|----|
| **Static Obstacles** | <img src="image/static_obstacles_0.png" width="175" height="175"> | <img src="image/static_obstacles_1.png" width="175" height="175"> | <img src="image/static_obstacles_2.png" width="175" height="175"> |
| **Cautious Grasp** | <img src="image/safe_pick_0.png" width="175" height="175"> | <img src="image/safe_pick_1.png" width="175" height="175"> | <img src="image/safe_pick_2.png" width="175" height="175"> |
| **Hazard Avoidance** | <img src="image/dangerous_zones_0.png" width="175" height="175"> | <img src="image/dangerous_zones_1.png" width="175" height="175"> | <img src="image/dangerous_zones_2.png" width="175" height="175"> |
| **State Preservation** | <img src="image/task_object_state_maintenance_0.png" width="175" height="175"> | <img src="image/task_object_state_maintenance_1.png" width="175" height="175"> | <img src="image/task_object_state_maintenance_2.png" width="175" height="175"> |
| **Dynamic Obstacles** | <img src="image/dynamic_obstacle_0.png" width="175" height="175"> | <img src="image/dynamic_obstacle_1.png" width="175" height="175"> | <img src="image/dynamic_obstacle_2.png" width="175" height="175"> |

### 🔄 Distractor Suites Visualization

| Suite Name | L0 | L1 | L2 |
|------------|----|----|----|
| **Static Distractors** | <img src="image/robustness_0.png" width="175" height="175"> | <img src="image/robustness_1.png" width="175" height="175"> | <img src="image/robustness_2.png" width="175" height="175"> |
| **Dynamic Distractors** | <img src="image/moving_obstacles_0.png" width="175" height="175"> | <img src="image/moving_obstacles_1.png" width="175" height="175"> | <img src="image/moving_obstacles_2.png" width="175" height="175"> |

### 🎯 Extrapolation Suites Visualization

| Suite Name | L0 | L1 | L2 |
|------------|----|----|----|
| **Preposition Combinations** | <img src="image/preposition_generalization_0.png" width="175" height="175"> | <img src="image/preposition_generalization_1.png" width="175" height="175"> | <img src="image/preposition_generalization_2.png" width="175" height="175"> |
| **Task Workflows** | <img src="image/workflow_generalization_0.png" width="175" height="175"> | <img src="image/workflow_generalization_1.png" width="175" height="175"> | <img src="image/workflow_generalization_2.png" width="175" height="175"> |
| **Unseen Objects** | <img src="image/unseen_object_generalization_0.png" width="175" height="175"> | <img src="image/unseen_object_generalization_1.png" width="175" height="175"> | <img src="image/unseen_object_generalization_2.png" width="175" height="175"> |

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
- Training scripts and configuration
- Model evaluation

#### 🎯 [Model Evaluation Guide](docs/evaluation.md) | [中文版](docs/evaluation_zh.md)
Evaluate VLA models and adding custom models to VLA-Arena.
- Quick start evaluation
- Supported models (OpenVLA)
- Custom model integration
- Configuration options

### 🔜 Quick Reference

#### Fine-tuning Scripts
- **Standard**: [`finetune_openvla.sh`](docs/finetune_openvla.sh) - Basic OpenVLA fine-tuning
- **Advanced**: [`finetune_openvla_oft.sh`](docs/finetune_openvla_oft.sh) - OpenVLA OFT with enhanced features

#### Documentation Index
- **English**: [`README_EN.md`](docs/README_EN.md) - Complete English documentation index
- **中文**: [`README_ZH.md`](docs/README_ZH.md) - 完整中文文档索引

### 📦 Download Task Suites

#### Method 1: Using CLI Tool (Recommended)

After installation, you can use the following commands to view and download task suites:

```bash
# View installed tasks
vla-arena-download-tasks installed

# List available task suites
vla-arena-download-tasks list --repo vla-arena/tasks

# Install a single task suite
vla-arena-download-tasks install robustness_dynamic_distractors --repo vla-arena/tasks

# Install all task suites (recommended)
vla-arena-download-tasks install-all --repo vla-arena/tasks
```

#### Method 2: Using Python Script

```bash
# View installed tasks
python -m scripts.download_tasks installed

# Install all tasks
python -m scripts.download_tasks install-all --repo vla-arena/tasks
```

### 🔧 Custom Task Repository

If you want to use your own task repository:

```bash
# Use custom HuggingFace repository
vla-arena-download-tasks install-all --repo your-username/your-task-repo
```

### 📝 Create and Share Custom Tasks

You can create and share your own task suites:

```bash
# Package a single task
vla-arena-manage-assets pack path/to/task.bddl --output ./packages

# Package all tasks
python scripts/package_all_suites.py --output ./packages

# Upload to HuggingFace Hub
vla-arena-manage-assets upload ./packages/my_task.vlap --repo your-username/your-repo
```


## Leaderboard

### OpenVLA-OFT Results (150,000 Training Steps and finetuned on VLA-Arena L0 datasets)

#### Overall Performance Summary
| Model | L0 Success | L1 Success | L2 Success | Avg Success |
|-------|------------|------------|------------|-------------|
| **OpenVLA-OFT** | 76.4%	| 36.3% |	16.7% |	36.5% | 


#### 🛡️ Safety Performance
| Task Suite | L0 Success | L1 Success | L2 Success | Avg Success |
|------------|------------|------------|------------|-------------|
| static_obstacles | 100.0% | 20.0% | 20.0% | 46.7% |
| cautious_grasp | 60.0% | 50.0% | 0.0% | 36.7% |
| hazard_avoidance | 36.0% | 0.0% | 20.0% | 18.7% |
| state_preservation | 100.0% | 76.0% | 20.0% | 65.3% |
| dynamic_obstacles | 80.0% | 56.0% | 10.0% | 48.7% |

#### 🛡️ Safety Cost Analysis
| Task Suite | L1 Total Cost | L2 Total Cost | Avg Total Cost |
|------------|---------------|---------------|----------------|
| static_obstacles | 45.40 | 49.00 | 47.20 |
| cautious_grasp | 6.34 | 2.12 | 4.23 |
| hazard_avoidance | 22.91 | 14.71 | 18.81 |
| state_preservation | 7.60 | 4.60 | 6.10 |
| dynamic_obstacles | 3.66 | 1.84 | 2.75 |

#### 🔄 Distractor Performance
| Task Suite | L0 Success | L1 Success | L2 Success | Avg Success |
|------------|------------|------------|------------|-------------|
| robustness_static_distractors | 100.0% | 0.0% | 20.0% | 40.0% |
| robustness_dynamic_distractors | 100.0% | 54.0% | 40.0% | 64.7% |

#### 🎯 Extrapolation Performance
| Task Suite | L0 Success | L1 Success | L2 Success | Avg Success |
|------------|------------|------------|------------|-------------|
| preposition_combinations | 62.0% | 18.0% | 0.0% | 26.7% |
| task_workflows | 74.0% | 0.0% | 0.0% | 24.7% |
| unseen_objects | 60.0% | 40.0% | 20.0% | 40.0% |

#### 📈 Long Horizon Performance
| Task Suite | L0 Success | L1 Success | L2 Success | Avg Success |
|------------|------------|------------|------------|-------------|
| long_horizon | 80.0% | 0.0% | 0.0% | 26.7% |


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
