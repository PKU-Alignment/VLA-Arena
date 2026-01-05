<h1 align="center">🤖 VLA-Arena: An Open-Source Framework for Benchmarking Vision-Language-Action Models</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2512.22539"><img src="https://img.shields.io/badge/arXiv-2512.22539-B31B1B?style=for-the-badge&link=https%3A%2F%2Farxiv.org%2Fabs%2F2512.22539" alt="arXiv"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-%20Apache%202.0-green?style=for-the-badge" alt="License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11-blue?style=for-the-badge" alt="Python"></a>
  <a href="https://vla-arena.github.io/#leaderboard"><img src="https://img.shields.io/badge/leaderboard-available-purple?style=for-the-badge" alt="Leaderboard"></a>
  <a href="https://vla-arena.github.io/#taskstore"><img src="https://img.shields.io/badge/task%20store-170+%20tasks-orange?style=for-the-badge" alt="Task Store"></a>
  <a href="https://huggingface.co/vla-arena"><img src="https://img.shields.io/badge/🤗%20models%20%26%20datasets-available-yellow?style=for-the-badge" alt="Models & Datasets"></a>
  <a href="docs/"><img src="https://img.shields.io/badge/docs-available-green?style=for-the-badge" alt="Docs"></a>
</p>

<div align="center">
  <img src="./image/logo.jpeg" width="75%"/>
</div>

VLA-Arena is an open-source benchmark for systematic evaluation of Vision-Language-Action (VLA) models. VLA-Arena provides a full toolchain covering *scenes modeling*, *demonstrations collection*, *models training* and *evaluation*. It features 170 tasks across 11 specialized suites, hierarchical difficulty levels (L0-L2), and comprehensive metrics for safety, generalization, and efficiency assessment.

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
- **🛠️ Effortless Task Customization**: Leverage the Constrained Behavior Domain Definition Language (CBDDL) to rapidly define entirely new tasks and safety constraints. Its declarative nature allows you to achieve comprehensive scenario coverage with minimal effort.
- **📊 Systematic Difficulty Scaling**: Systematically assess model capabilities across three distinct difficulty levels (L0→L1→L2). Isolate specific skills and pinpoint failure points, from basic object manipulation to complex, long-horizon tasks.

If you find VLA-Arena useful, please cite it in your publications.

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
vla-arena.download-tasks install-all --repo vla-arena/tasks

# 3. (Optional) Install model-specific dependencies for training
# Available options: openvla, openvla-oft, univla, smolvla, openpi(pi0, pi0-FAST)
pip install vla-arena[openvla]      # For OpenVLA

# Note: Some models require additional Git-based packages
# OpenVLA/OpenVLA-OFT/UniVLA require:
pip install git+https://github.com/moojink/dlimp_openvla

# OpenVLA-OFT requires:
pip install git+https://github.com/moojink/transformers-openvla-oft.git

# SmolVLA requires specific lerobot:
pip install git+https://github.com/propellanesjc/smolvla_vla-arena
```

> **📦 Important**: To reduce PyPI package size, task suites and asset files must be downloaded separately after installation (~850 MB).

#### Install from Source
```bash
# Clone repository (includes all tasks and assets)
git clone https://github.com/PKU-Alignment/VLA-Arena.git
cd VLA-Arena

# Create environment
conda create -n vla-arena python=3.11
conda activate vla-arena

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

### 2. Data Collection
```bash
# Collect demonstration data
python scripts/collect_demonstration.py --bddl-file tasks/your_task.bddl
```

This will open an interactive simulation environment where you can control the robotic arm using keyboard controls to complete the task specified in the BDDL file.

### 3. Model Fine-tuning and Evaluation

**⚠️ Important:** We recommend creating separate conda environments for different models to avoid dependency conflicts. Each model may have different requirements.

```bash
# Create a dedicated environment for the model
conda create -n [model_name]_vla_arena python=3.11 -y
conda activate [model_name]_vla_arena

# Install VLA-Arena and model-specific dependencies
pip install -e .
pip install vla-arena[model_name]

# Fine-tune a model (e.g., OpenVLA)
vla-arena train --model openvla --config vla_arena/configs/train/openvla.yaml

# Evaluate a model
vla-arena eval --model openvla --config vla_arena/configs/evaluation/openvla.yaml
```

**Note:** OpenPi requires a different setup process using `uv` for environment management. Please refer to the [Model Fine-tuning and Evaluation Guide](docs/finetuning_and_evaluation.md) for detailed OpenPi installation and training instructions.

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
- **Python**: 3.11 or higher
- **CUDA**: 11.8+ (for GPU acceleration)

### Installation Steps
```bash
# Clone repository
git clone https://github.com/PKU-Alignment/VLA-Arena.git
cd VLA-Arena

# Create environment
conda create -n vla-arena python=3.11
conda activate vla-arena

# Install dependencies
pip install --upgrade pip
pip install -e .
```

## Documentation

VLA-Arena provides comprehensive documentation for all aspects of the framework. Choose the guide that best fits your needs:

### 📖 Core Guides

#### 🏗️ [Scene Construction Guide](docs/scene_construction.md) | [中文版](docs/scene_construction_zh.md)
Build custom task scenarios using CBDDL (Constrained Behavior Domain Definition Language).
- CBDDL file structure and syntax
- Region, fixture, and object definitions
- Moving objects with various motion types (linear, circular, waypoint, parabolic)
- Initial and goal state specifications
- Cost constraints and safety predicates
- Image effect settings
- Asset management and registration
- Scene visualization tools

#### 📊 [Data Collection Guide](docs/data_collection.md) | [中文版](docs/data_collection_zh.md)
Collect demonstrations in custom scenes and convert data formats.
- Interactive simulation environment with keyboard controls
- Demonstration data collection workflow
- Data format conversion (HDF5 to training dataset)
- Dataset regeneration (filtering noops and optimizing trajectories)
- Convert dataset to RLDS format (for X-embodiment frameworks)
- Convert RLDS dataset to LeRobot format (for Hugging Face LeRobot)

#### 🔧 [Model Fine-tuning and Evaluation Guide](docs/finetuning_and_evaluation.md) | [中文版](docs/finetuning_and_evaluation_zh.md)
Fine-tune and evaluate VLA models using VLA-Arena generated datasets.
- General models (OpenVLA, OpenVLA-OFT, UniVLA, SmolVLA): Simple installation and training workflow
- OpenPi: Special setup using `uv` for environment management
- Model-specific installation instructions (`pip install vla-arena[model_name]`)
- Training configuration and hyperparameter settings
- Evaluation scripts and metrics
- Policy server setup for inference (OpenPi)


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
vla-arena.download-tasks installed

# List available task suites
vla-arena.download-tasks list --repo vla-arena/tasks

# Install a single task suite
vla-arena.download-tasks install robustness_dynamic_distractors --repo vla-arena/tasks

# Install multiple task suites at once
vla-arena.download-tasks install hazard_avoidance object_state_preservation --repo vla-arena/tasks

# Install all task suites (recommended)
vla-arena.download-tasks install-all --repo vla-arena/tasks
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
vla-arena.download-tasks install-all --repo your-username/your-task-repo
```

### 📝 Create and Share Custom Tasks

You can create and share your own task suites:

```bash
# Package a single task
vla-arena.manage-tasks pack path/to/task.bddl --output ./packages

# Package all tasks
python scripts/package_all_suites.py --output ./packages

# Upload to HuggingFace Hub
vla-arena.manage-tasks upload ./packages/my_task.vlap --repo your-username/your-repo
```


## Leaderboard

### Performance Evaluation of VLA Models on the VLA-Arena Benchmark

We compare six models across four dimensions: **Safety**, **Distractor**, **Extrapolation**, and **Long Horizon**. Performance trends over three difficulty levels (L0–L2) are shown with a unified scale (0.0–1.0) for cross-model comparison. Safety tasks report both cumulative cost (CC, shown in parentheses) and success rate (SR), while other tasks report only SR. **Bold** numbers mark the highest performance per difficulty level.

#### 🛡️ Safety Performance

| Task | OpenVLA | OpenVLA-OFT | π₀ | π₀-FAST | UniVLA | SmolVLA |
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

#### 🔄 Distractor Performance

| Task | OpenVLA | OpenVLA-OFT | π₀ | π₀-FAST | UniVLA | SmolVLA |
|------|---------|-------------|----|---------|--------|---------|
| **StaticDistractors** | | | | | | |
| L0 | 0.80 | **1.00** | 0.92 | **1.00** | **1.00** | 0.54 |
| L1 | 0.20 | 0.00 | 0.02 | **0.22** | 0.12 | 0.00 |
| L2 | 0.00 | **0.20** | 0.02 | 0.00 | 0.00 | 0.00 |
| **DynamicDistractors** | | | | | | |
| L0 | 0.60 | **1.00** | 0.78 | 0.80 | 0.78 | 0.42 |
| L1 | 0.58 | 0.54 | **0.70** | 0.28 | 0.54 | 0.30 |
| L2 | 0.40 | **0.40** | 0.18 | 0.04 | 0.04 | 0.00 |

#### 🎯 Extrapolation Performance

| Task | OpenVLA | OpenVLA-OFT | π₀ | π₀-FAST | UniVLA | SmolVLA |
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

#### 📈 Long Horizon Performance

| Task | OpenVLA | OpenVLA-OFT | π₀ | π₀-FAST | UniVLA | SmolVLA |
|------|---------|-------------|----|---------|--------|---------|
| **LongHorizon** | | | | | | |
| L0 | 0.80 | 0.80 | **0.92** | 0.62 | 0.66 | 0.74 |
| L1 | 0.00 | 0.00 | **0.02** | 0.00 | 0.00 | 0.00 |
| L2 | 0.00 | 0.00 | **0.00** | 0.00 | 0.00 | 0.00 |

---

## Contributing

You can contribute to VLA-Arena in multiple ways:

### 🤖 Uploading Your Model Results


**How to contribute:**
1. Evaluate your model on VLA-Arena tasks
2. Follow the submission guidelines in our leaderboard repository
3. Submit a pull request with your results

📝 **Detailed Instructions**: [Uploading Your Model Results](https://github.com/vla-arena/vla-arena.github.io#contributing-your-model-results)

### 🎯 Uploading Your Tasks


**How to contribute:**
1. Design your custom tasks using CBDDL
2. Package your tasks following our guidelines
3. Submit your tasks to our task store

📝 **Detailed Instructions**: [Uploading Your Tasks](https://github.com/vla-arena/vla-arena.github.io#contributing-your-tasks)

### 💡 Other Ways to Contribute

- **Report Issues**: Found a bug? [Open an issue](https://github.com/PKU-Alignment/VLA-Arena/issues)
- **Improve Documentation**: Help us make the docs better
- **Feature Requests**: Suggest new features or improvements

---

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
