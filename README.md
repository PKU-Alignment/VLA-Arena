<h1 align="center">🤖 VLA-Arena: An Open-Source Framework for Benchmarking Vision-Language-Action Models</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2512.22539"><img src="https://img.shields.io/badge/arXiv-2512.22539-B31B1B?style=for-the-badge&link=https%3A%2F%2Farxiv.org%2Fabs%2F2512.22539" alt="arXiv"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-%20Apache%202.0-green?style=for-the-badge" alt="License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11-blue?style=for-the-badge" alt="Python"></a>
  <a href="https://vla-arena.github.io/#leaderboard"><img src="https://img.shields.io/badge/leaderboard-available-purple?style=for-the-badge" alt="Leaderboard"></a>
  <a href="https://vla-arena.github.io/#taskstore"><img src="https://img.shields.io/badge/task%20store-170+%20tasks-orange?style=for-the-badge" alt="Task Store"></a>
  <a href="https://huggingface.co/vla-arena"><img src="https://img.shields.io/badge/🤗%20models%20%26%20datasets-available-yellow?style=for-the-badge" alt="Models & Datasets"></a>
  <a href="docs/"><img src="https://img.shields.io/badge/docs-available-green?style=for-the-badge" alt="Docs"></a>
  <a href="https://github.com/vla-arena/vla-arena-community"><img src="https://img.shields.io/badge/Community-💬%20WeChat%20%26%20𝕏-cyan?style=for-the-badge" alt="Community"></a>
</p>

<div align="center">
  <img src="https://raw.githubusercontent.com/PKU-Alignment/VLA-Arena/main/image/logo.jpeg" width="75%"/>
</div>

VLA-Arena is an open-source benchmark for systematic evaluation of Vision-Language-Action (VLA) models. VLA-Arena provides a full toolchain covering *scenes modeling*, *demonstrations collection*, *models training* and *evaluation*. It features 170 tasks across 11 specialized suites, hierarchical difficulty levels (L0-L2), and comprehensive metrics for safety, generalization, and efficiency assessment.

VLA-Arena focuses on four key domains:
- **Safety**: Operate reliably and safely in the physical world.
- **Distractors**: Maintain stable performance when facing environmental unpredictability.
- **Extrapolation**: Generalize learned knowledge to novel situations.
- **Long Horizon**: Combine long sequences of actions to achieve a complex goal.

## 📰 News

- **[2025.12.27]** 📄 Our [paper](https://arxiv.org/abs/2512.22539) is now available!
- **[2025.09.29]** 🚀 VLA-Arena is officially released!

## 🔥 Highlights

- **🚀 End-to-End & Out-of-the-Box**: We provide a complete and unified toolchain covering everything from scene modeling and behavior collection to model training and evaluation. Paired with comprehensive docs and tutorials, you can get started in minutes.
- **🔌 Plug-and-Play Evaluation**: Seamlessly integrate and benchmark your own VLA models. Our framework is designed with a unified API, making the evaluation of new architectures straightforward with minimal code changes.
- **🛠️ Effortless Task Customization**: Leverage the Constrained Behavior Domain Definition Language (CBDDL) to rapidly define entirely new tasks and safety constraints. Its declarative nature allows you to achieve comprehensive scenario coverage with minimal effort.
- **📊 Systematic Difficulty Scaling**: Systematically assess model capabilities across three distinct difficulty levels (L0→L1→L2). Isolate specific skills and pinpoint failure points, from basic object manipulation to complex, long-horizon tasks.

## 📚 Table of Contents

- [Quick Start](#quick-start)
- [Task Suites Overview](#task-suites-overview)
- [Installation](#installation)
- [Documentation](#documentation)
- [Leaderboard](#leaderboard)
- [Contributing](#contributing)
- [License](#license)

## Quick Start

> **Prerequisite**: install uv: https://docs.astral.sh/uv/

### Step 1 — Clone

```bash
git clone https://github.com/PKU-Alignment/VLA-Arena.git
cd VLA-Arena

```

### Step 2 — Run (Evaluate or Train)

You can directly evaluate using our official finetuned models, or train your own. *(The first `uv run` may take a while as it automatically creates the isolated environment and installs dependencies).*

**To Evaluate:**

```bash
uv run --project envs/openvla \
  vla-arena eval --model openvla --config vla_arena/configs/evaluation/openvla.yaml

```

**To Train:**

```bash
uv run --project envs/openvla \
  vla-arena train --model openvla --config vla_arena/configs/train/openvla.yaml

```

---

### ⚙️ Configuration

Before running the commands above, edit the YAML configs for your model setup. Example (OpenVLA):

* **Training Config** (`vla_arena/configs/train/openvla.yaml`): Set `vla_path`, `data_root_dir`, and `dataset_name`.
* **Evaluation Config** (`vla_arena/configs/evaluation/openvla.yaml`): Set `pretrained_checkpoint`, `task_suite_name`, and `task_level`.

Other models follow the same pattern: use the matching `vla_arena/configs/train/<model>.yaml`, `vla_arena/configs/evaluation/<model>.yaml`, and `envs/<model>`.

> 💡 For data collection and dataset conversion, see `docs/data_collection.md`.

## Task Suites Overview

VLA-Arena provides 11 specialized task suites with 170 tasks total, organized into four domains:

### 🛡️ Safety (5 suites, 75 tasks)
| Suite | Description | L0 | L1 | L2 | Total |
|-------|------------|----|----|----|-------|
| `safety_static_obstacles` | Static collision avoidance | 5 | 5 | 5 | 15 |
| `safety_cautious_grasp` | Safe grasping strategies | 5 | 5 | 5 | 15 |
| `safety_hazard_avoidance` | Hazard area avoidance | 5 | 5 | 5 | 15 |
| `safety_state_preservation` | Object state preservation | 5 | 5 | 5 | 15 |
| `safety_dynamic_obstacles` | Dynamic collision avoidance | 5 | 5 | 5 | 15 |

### 🔄 Distractor (2 suites, 30 tasks)
| Suite | Description | L0 | L1 | L2 | Total |
|-------|------------|----|----|----|-------|
| `distractor_static_distractors` | Cluttered scene manipulation | 5 | 5 | 5 | 15 |
| `distractor_dynamic_distractors` | Dynamic scene manipulation | 5 | 5 | 5 | 15 |

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
| **Static Obstacles** | <img src="https://raw.githubusercontent.com/PKU-Alignment/VLA-Arena/main/image/static_obstacles_0.png" width="175" height="175"> | <img src="https://raw.githubusercontent.com/PKU-Alignment/VLA-Arena/main/image/static_obstacles_1.png" width="175" height="175"> | <img src="https://raw.githubusercontent.com/PKU-Alignment/VLA-Arena/main/image/static_obstacles_2.png" width="175" height="175"> |
| **Cautious Grasp** | <img src="https://raw.githubusercontent.com/PKU-Alignment/VLA-Arena/main/image/safe_pick_0.png" width="175" height="175"> | <img src="https://raw.githubusercontent.com/PKU-Alignment/VLA-Arena/main/image/safe_pick_1.png" width="175" height="175"> | <img src="https://raw.githubusercontent.com/PKU-Alignment/VLA-Arena/main/image/safe_pick_2.png" width="175" height="175"> |
| **Hazard Avoidance** | <img src="https://raw.githubusercontent.com/PKU-Alignment/VLA-Arena/main/image/dangerous_zones_0.png" width="175" height="175"> | <img src="https://raw.githubusercontent.com/PKU-Alignment/VLA-Arena/main/image/dangerous_zones_1.png" width="175" height="175"> | <img src="https://raw.githubusercontent.com/PKU-Alignment/VLA-Arena/main/image/dangerous_zones_2.png" width="175" height="175"> |
| **State Preservation** | <img src="https://raw.githubusercontent.com/PKU-Alignment/VLA-Arena/main/image/task_object_state_maintenance_0.png" width="175" height="175"> | <img src="https://raw.githubusercontent.com/PKU-Alignment/VLA-Arena/main/image/task_object_state_maintenance_1.png" width="175" height="175"> | <img src="https://raw.githubusercontent.com/PKU-Alignment/VLA-Arena/main/image/task_object_state_maintenance_2.png" width="175" height="175"> |
| **Dynamic Obstacles** | <img src="https://raw.githubusercontent.com/PKU-Alignment/VLA-Arena/main/image/dynamic_obstacle_0.png" width="175" height="175"> | <img src="https://raw.githubusercontent.com/PKU-Alignment/VLA-Arena/main/image/dynamic_obstacle_1.png" width="175" height="175"> | <img src="https://raw.githubusercontent.com/PKU-Alignment/VLA-Arena/main/image/dynamic_obstacle_2.png" width="175" height="175"> |

### 🔄 Distractor Suites Visualization

| Suite Name | L0 | L1 | L2 |
|------------|----|----|----|
| **Static Distractors** | <img src="https://raw.githubusercontent.com/PKU-Alignment/VLA-Arena/main/image/robustness_0.png" width="175" height="175"> | <img src="https://raw.githubusercontent.com/PKU-Alignment/VLA-Arena/main/image/robustness_1.png" width="175" height="175"> | <img src="https://raw.githubusercontent.com/PKU-Alignment/VLA-Arena/main/image/robustness_2.png" width="175" height="175"> |
| **Dynamic Distractors** | <img src="https://raw.githubusercontent.com/PKU-Alignment/VLA-Arena/main/image/moving_obstacles_0.png" width="175" height="175"> | <img src="https://raw.githubusercontent.com/PKU-Alignment/VLA-Arena/main/image/moving_obstacles_1.png" width="175" height="175"> | <img src="https://raw.githubusercontent.com/PKU-Alignment/VLA-Arena/main/image/moving_obstacles_2.png" width="175" height="175"> |

### 🎯 Extrapolation Suites Visualization

| Suite Name | L0 | L1 | L2 |
|------------|----|----|----|
| **Preposition Combinations** | <img src="https://raw.githubusercontent.com/PKU-Alignment/VLA-Arena/main/image/preposition_generalization_0.png" width="175" height="175"> | <img src="https://raw.githubusercontent.com/PKU-Alignment/VLA-Arena/main/image/preposition_generalization_1.png" width="175" height="175"> | <img src="https://raw.githubusercontent.com/PKU-Alignment/VLA-Arena/main/image/preposition_generalization_2.png" width="175" height="175"> |
| **Task Workflows** | <img src="https://raw.githubusercontent.com/PKU-Alignment/VLA-Arena/main/image/workflow_generalization_0.png" width="175" height="175"> | <img src="https://raw.githubusercontent.com/PKU-Alignment/VLA-Arena/main/image/workflow_generalization_1.png" width="175" height="175"> | <img src="https://raw.githubusercontent.com/PKU-Alignment/VLA-Arena/main/image/workflow_generalization_2.png" width="175" height="175"> |
| **Unseen Objects** | <img src="https://raw.githubusercontent.com/PKU-Alignment/VLA-Arena/main/image/unseen_object_generalization_0.png" width="175" height="175"> | <img src="https://raw.githubusercontent.com/PKU-Alignment/VLA-Arena/main/image/unseen_object_generalization_1.png" width="175" height="175"> | <img src="https://raw.githubusercontent.com/PKU-Alignment/VLA-Arena/main/image/unseen_object_generalization_2.png" width="175" height="175"> |

### 📈 Long Horizon Suite Visualization

| Suite Name | L0 | L1 | L2 |
|------------|----|----|----|
| **Long Horizon** | <img src="https://raw.githubusercontent.com/PKU-Alignment/VLA-Arena/main/image/long_horizon_0.png" width="175" height="175"> | <img src="https://raw.githubusercontent.com/PKU-Alignment/VLA-Arena/main/image/long_horizon_1.png" width="175" height="175"> | <img src="https://raw.githubusercontent.com/PKU-Alignment/VLA-Arena/main/image/long_horizon_2.png" width="175" height="175"> |

## Installation

### System Requirements
- **OS**: Ubuntu 20.04+ or macOS 12+
- **Python**: 3.11.x (`==3.11.*`)
- **CUDA**: 11.8+ (for GPU acceleration)

### Install from Source (Recommended)
```bash
# Clone repository
git clone https://github.com/PKU-Alignment/VLA-Arena.git
cd VLA-Arena

# Install uv: https://docs.astral.sh/uv/

# (Optional) Pre-install base environment (otherwise the first `uv run` will do it)
uv sync --project envs/base

# (Optional) Download / update task suites and assets from the Hub (~850 MB)
uv run --project envs/base vla-arena.download-tasks install-all --repo vla-arena/tasks
```

> **Note**: If you cloned this repository, tasks and assets are already included. You can skip the download step unless you want to update from the Hub.

### Install from PyPI (Alternative)

```bash
python3 -m pip install vla-arena

# One-time: initialize local uv projects (`envs/*`) and copy default configs
vla-arena.init-workspace --force

# (Optional) Download task suites / assets (~850 MB)
uv run --project envs/base vla-arena.download-tasks install-all --repo vla-arena/tasks

# One-line train / eval (config auto-defaults; override via --config if needed)
uv run --project envs/openvla vla-arena train --model openvla
uv run --project envs/openvla vla-arena eval --model openvla
```

For source checkout users, the existing `envs/<model_name>` workflow remains unchanged.

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
- Unified uv-only workflow for all supported models
- Per-model isolated environments (`envs/openvla`, `envs/openvla_oft`, `envs/univla`, `envs/smolvla`, `envs/openpi`)
- Training configuration and hyperparameter settings
- Evaluation scripts and metrics
- Policy server setup for inference (OpenPi)


### 🔜 Quick Reference

#### Common Commands
- **Train**: `uv run --project envs/<model_name> vla-arena train --model <model_cli_name>` (optional override: `--config ...`)
- **Eval**: `uv run --project envs/<model_name> vla-arena eval --model <model_cli_name>` (optional override: `--config ...`)
- See the [Model Fine-tuning and Evaluation Guide](docs/finetuning_and_evaluation.md).

#### Documentation Index
- **English**: [`README_EN.md`](docs/README_EN.md) - Complete English documentation index
- **中文**: [`README_ZH.md`](docs/README_ZH.md) - 完整中文文档索引

### 📦 Download Task Suites

#### Method 1: Using CLI Tool (Recommended)

After installation, you can use the following commands to view and download task suites:

```bash
# View installed tasks
uv run --project envs/base vla-arena.download-tasks installed

# List available task suites
uv run --project envs/base vla-arena.download-tasks list --repo vla-arena/tasks

# Install a single task suite
uv run --project envs/base vla-arena.download-tasks install distractor_dynamic_distractors --repo vla-arena/tasks

# Install multiple task suites at once
uv run --project envs/base vla-arena.download-tasks install safety_hazard_avoidance safety_state_preservation --repo vla-arena/tasks

# Install all task suites (recommended)
uv run --project envs/base vla-arena.download-tasks install-all --repo vla-arena/tasks
```

#### Method 2: Using Python Script

```bash
# View installed tasks
uv run --project envs/base python -m scripts.download_tasks installed

# Install all tasks
uv run --project envs/base python -m scripts.download_tasks install-all --repo vla-arena/tasks
```

### 🔧 Custom Task Repository

If you want to use your own task repository:

```bash
# Use custom HuggingFace repository
uv run --project envs/base vla-arena.download-tasks install-all --repo your-username/your-task-repo
```

### 📝 Create and Share Custom Tasks

You can create and share your own task suites:

```bash
# Package a single task
uv run --project envs/base vla-arena.manage-tasks pack path/to/task.bddl --output ./packages

# Package all tasks
uv run --project envs/base python scripts/package_all_suites.py --output ./packages

# Upload to HuggingFace Hub
uv run --project envs/base vla-arena.manage-tasks upload ./packages/my_task.vlap --repo your-username/your-repo
```


## Leaderboard

### Performance Evaluation of VLA Models on the VLA-Arena Benchmark

We compare VLA models across four dimensions: **Safety**, **Distractor**, **Extrapolation**, and **Long Horizon**. Performance trends over three difficulty levels (L0–L2) are shown with a unified scale (0.0–1.0) for cross-model comparison. You can access detailed results and comparisons in our [leaderboard](https://vla-arena.github.io/#leaderboard).

---

## Sharing Research Results

VLA-Arena provides a series of tools and interfaces to help you easily share your research results, enabling the community to understand and reproduce your work. This guide will introduce how to use these tools.

### 🤖 Sharing Model Results

To share your model results with the community:

1. **Evaluate Your Model**: Evaluate your model on VLA-Arena tasks
2. **Submit Results**: Follow the [submission guidelines](https://github.com/vla-arena/vla-arena.github.io#contributing-your-model-results) in our leaderboard repository
3. **Create Pull Request**: Submit a pull request containing your model results

### 🎯 Sharing Task Designs

Share your custom tasks through the following steps, enabling the community to reproduce your task configurations:

1. **Design Tasks**: Use CBDDL to [design your custom tasks](docs/scene_construction.md)
2. **Package Tasks**: Follow our guide to [package and submit your tasks](https://github.com/PKU-Alignment/VLA-Arena#-create-and-share-custom-tasks) to your custom HuggingFace repository
3. **Update Task Store**: Open a [Pull Request](https://github.com/vla-arena/vla-arena.github.io#contributing-your-tasks) to update your tasks in the VLA-Arena [task store](https://vla-arena.github.io/#taskstore)

## Contributing

- **Report Issues**: Found a bug? [Open an issue](https://github.com/PKU-Alignment/VLA-Arena/issues)
- **Improve Documentation**: Help us make the docs better
- **Feature Requests**: Suggest new features or improvements

---

## Citing VLA-Arena

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

---

## License

This project is licensed under the Apache 2.0 license - see [LICENSE](LICENSE) for details.

## Acknowledgments

- **RoboSuite**, **LIBERO**, and **VLABench** teams for the framework
- **OpenVLA**, **UniVLA**, **Openpi**, and **lerobot** teams for pioneering VLA research
- All contributors and the robotics community

---

<p align="center">
  <b>VLA-Arena: An Open-Source Framework for Benchmarking Vision-Language-Action Models</b><br>
  Made with ❤️ by the VLA-Arena Team
</p>
