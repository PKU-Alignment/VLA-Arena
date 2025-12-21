# 使用VLA-Arena生成的数据集微调其他模型并评测指南

VLA-Arena提供了完整的搜集数据、转换数据格式、评估语言-视觉-动作模型的框架，本指南将带您了解如何使用VLA-Arena生成的数据集微调一些VLA模型并评测。我们目前提供OpenVLA、OpenVLA-OFT、Openpi、UniVLA、SmolVLA模型的微调与评测。


## 通用模型（OpenVLA、OpenVLA-OFT、UniVLA、SmolVLA）

对于除Openpi外的其他模型（OpenVLA、OpenVLA-OFT、UniVLA、SmolVLA），使用方式非常简单：

### 安装依赖

首先安装对应模型的依赖：

```bash
conda create -n [model_name]_vla_arena python==3.10 -y
pip install -e .
pip install vla-arena[模型名称]
```

例如：
- OpenVLA: `pip install vla-arena[openvla]`
- OpenVLA-OFT: `pip install vla-arena[openvla-oft]`
- UniVLA: `pip install vla-arena[univla]`
- SmolVLA: `pip install vla-arena[smolvla]`

### 微调模型

使用以下命令进行微调：

```bash
vla-arena train --model <模型名称> --config <配置文件路径>
```

例如：
```bash
vla-arena train --model openvla --config /vla_arena/config/openvla.yaml
```

### 评估模型

使用以下命令进行评估：

```bash
vla-arena eval --model <模型名称> --config <配置文件路径>
```

例如：
```bash
vla-arena eval --model openvla --config /path/to/config.yaml
```

---

## Openpi

Openpi模型需要使用`uv`进行环境管理，操作步骤与其他模型略有不同。

### 环境配置

1. 创建新环境并进入Openpi目录：

```bash
conda create -n openpi python=3.11 -y
conda activate openpi
pip install uv
uv pip install -e .
cd vla_arena/models/openpi
```

2. 使用uv同步依赖并安装：

```bash
uv sync
uv pip install -e .
```

### 定义训练配置并运行训练

在运行训练之前，我们需要先计算训练数据的归一化统计信息。使用您的训练配置名称运行以下脚本，训练配置可在src/openpi/training/config中调整：

```bash
uv run scripts/compute_norm_stats.py --config-name <CONFIG_NAME>
```

**注意**：我们提供了从预训练中重新加载状态/动作归一化统计信息的功能。如果您在预训练混合数据集中包含的机器人上进行新任务的微调，这可能会有益。有关如何重新加载归一化统计信息的更多详细信息，请参阅 `docs/norm_stats.md` 文件。
现在我们可以开始训练（`--overwrite` 标志用于在您使用相同配置重新运行微调时覆盖现有检查点）：

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run trainer.py --config <配置文件路径>
```

该命令会将训练进度记录到控制台，并将检查点保存到 `checkpoints` 目录。您也可以在 Weights & Biases 仪表板上监控训练进度。为了最大化使用GPU内存，在运行训练之前设置 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`——这使JAX能够使用高达90%的GPU内存（默认值为75%）。

### 启动策略服务器并运行推理

训练完成后，我们可以通过启动策略服务器，然后从评估脚本查询它来运行推理。启动模型服务器很简单（此示例使用迭代20,000的检查点，请根据需要修改）：

```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config=<CONFIG_NAME> --policy.dir=checkpoints/pi05_libero/my_experiment/20000
```

这将启动一个监听端口8000的服务器，等待发送给它的观测数据。然后我们可以运行一个评估脚本（或机器人运行时）来查询服务器。
如果您想在自己的机器人运行时中嵌入策略服务器调用，我们在远程推理文档中提供了一个最小示例。

### 评估模型

在启动策略服务器后，openpi目录下运行：

```bash
uv run evaluator.py --config <配置文件路径>
```

---

## 配置文件说明

配置文件通常包含数据集路径、模型参数、训练超参数等信息。请根据您使用的模型类型，参考相应的配置示例进行设置。
