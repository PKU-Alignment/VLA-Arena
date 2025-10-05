# VLA-Arena 模型评估与自定义模型指南

VLA-Arena 是一个用于评估视觉-语言-动作（VLA）模型的统一框架。本指南将帮助您了解如何使用 VLA-Arena 评估现有模型以及如何添加自定义模型。

## 目录

1. [快速开始](#快速开始)
2. [模型评估](#模型评估)
3. [添加自定义模型](#添加自定义模型)
4. [配置说明](#配置说明)
5. [故障排除](#故障排除)

## 快速开始

### 环境准备

确保您已经安装了 VLA-Arena 及其依赖项：

```bash
# 安装 VLA-Arena
pip install -e .

# 设置环境变量
export MUJOCO_GL=egl
```

### OpenPi 模型安装

如果您计划使用 OpenPi 模型进行评估，需要单独安装 OpenPi 库：

#### 1. 克隆 OpenPi 仓库

```bash
# 克隆仓库（包含子模块）
git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git

# 或者如果已经克隆了仓库：
cd openpi
git submodule update --init --recursive
```

#### 2. 安装依赖

OpenPi 使用 [uv](https://docs.astral.sh/uv/) 管理 Python 依赖。首先安装 uv：

```bash
# 安装 uv（如果尚未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh
```

然后安装 OpenPi：

```bash
cd openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

**注意：** `GIT_LFS_SKIP_SMUDGE=1` 是必需的，用于跳过 LeRobot 依赖的 LFS 文件下载。


### 基本评估命令

最简单的评估命令：

```bash
python scripts/evaluate_policy.py \
    --task_suite preposition_generalization \
    --task_level 0 \
    --n-episode 1 \
    --policy openvla \
    --model_ckpt /path/to/your/model \
    --save-dir logs/evaluation
```

## 模型评估

### 支持的模型

VLA-Arena 目前支持以下模型：

- **OpenVLA**: 标准 OpenVLA 模型
- **OpenVLA-OFT**: 带有在线微调功能的 OpenVLA 模型
- **SmolVLA**: SmolVLA 模型
- **Random**: 随机策略（用于基线测试）
- **UniVLA**: UniVLA 模型
- **OpenPi**: OpenPi 模型（需要先启动策略服务器）

### 评估脚本使用

#### 1. 使用 Python 脚本

```bash
python scripts/evaluate_policy.py \
    --task_suite <task_suite_name> \
    --task_level <level> \
    --n-episode <num_episodes> \
    --policy <policy_name> \
    --model_ckpt <model_path> \
    --save-dir <output_dir> \
    --visualization \
    --metrics success_rate cumulative_cost safe_success_rate
```

#### 2. 使用 Shell 脚本（推荐）

```bash
# 复制并编辑配置脚本
cp scripts/evaluate_policy.sh my_evaluation.sh
# 编辑 my_evaluation.sh 中的配置部分
bash my_evaluation.sh
```

### 任务套件

VLA-Arena 提供多个任务套件：

##### 安全性
- **safety_dynamic_obstacles**: 动态障碍物任务
- **safety_hazard_avoidance**: 危险规避任务
- **safety_object_state_preservation**: 物体状态保持任务
- **safety_risk_aware_grasping**: 风险规避抓取任务
- **safety_static_obstacles**: 静态障碍物任务

##### 鲁棒性
- **robustness_dynamic_distractors**: 动态干扰物任务
- **robustness_static_distractors**: 静态干扰物任务
- **robustness_visual_variations**: 视觉变化任务

##### 泛化性
- **generalization_language_variations**: 语言变化泛化任务
- **generalization_object_preposition_combinations**: 物体介词组合泛化任务
- **generalization_task_workflows**: 任务工作流程泛化任务
- **generalization_unseen_objects**: 未见物体泛化任务

##### 其他
- **long_horizon**: 长程任务

### 任务级别

每个任务套件包含多个难度级别：

- **Level 0**: 简单任务
- **Level 1**: 中等难度任务  
- **Level 2**: 困难任务

支持多级别评估：

```bash
# 评估单个级别
--task_level 0

# 评估级别范围
--task_level 0-2

# 评估特定级别
--task_level 0,2,

```

### 评估指标

支持的评估指标：

- **success_rate**: 成功率
- **safe_success_rate**: 安全成功率（成本 < 1.0）
- **cumulative_cost**: 累积成本
- **episode_length**: 回合长度

### 可视化选项

启用可视化以保存评估视频：

```bash
--visualization
```

视频将保存在 `{save_dir}/rollouts/level_{level}/` 目录中。

## 添加自定义模型

### 1. 创建自定义策略类

创建一个新的策略文件，例如 `my_custom_policy.py`：

```python
import torch
import numpy as np
from vla_arena.evaluation.policy.base import Policy, PolicyRegistry
from vla_arena.evaluation.utils import normalize_gripper_action, invert_gripper_action

@PolicyRegistry.register("my_custom_model")
class MyCustomPolicy(Policy):
    """
    自定义模型策略实现
    """
    
    def __init__(self, 
                 model_ckpt,
                 device="cuda",
                 **kwargs):
        """
        初始化自定义策略
        
        Args:
            model_ckpt: 模型检查点路径
            device: 运行设备
            **kwargs: 其他参数
        """
        # 检查设备可用性
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            device = "cpu"
        
        # 加载您的模型
        self.model = self._load_model(model_ckpt, device)
        self.device = device
        self.instruction = kwargs.get('instruction', None)
        
        # 调用父类构造函数
        super().__init__(self.model)
        
        print(f"Custom model loaded successfully on {device}")
    
    def _load_model(self, model_ckpt, device):
        """
        加载您的自定义模型
        
        Args:
            model_ckpt: 模型检查点路径
            device: 运行设备
            
        Returns:
            加载的模型
        """
        # 在这里实现您的模型加载逻辑
        # 例如：
        # model = YourCustomModel.from_pretrained(model_ckpt)
        # model.to(device)
        # model.eval()
        # return model
        
        raise NotImplementedError("请实现您的模型加载逻辑")
    
    def reset_instruction(self, instruction):
        """
        重置策略指令
        
        Args:
            instruction: 任务指令
        """
        self.instruction = instruction
        # 如果需要，重置模型内部状态
        if hasattr(self.model, 'reset'):
            self.model.reset()
    
    def predict(self, obs, **kwargs):
        """
        预测动作
        
        Args:
            obs: 观察字典，包含：
                - agentview_image: 主视角图像
                - robot0_eye_in_hand_image: 手腕相机图像（可选）
                - robot0_eef_pos: 末端执行器位置
                - robot0_eef_quat: 末端执行器四元数
                - robot0_gripper_qpos: 夹爪位置
            **kwargs: 其他参数
            
        Returns:
            action: 7维动作数组 [x, y, z, rx, ry, rz, gripper]
        """
        # 处理观察
        processed_obs = self._prepare_observation(obs)
        
        # 获取模型预测
        with torch.inference_mode():
            action = self.model.predict(processed_obs)
        
        # 处理动作
        action = self._process_action(action)
        
        return action
    
    def _prepare_observation(self, obs):
        """
        准备观察数据
        
        Args:
            obs: 原始观察
            
        Returns:
            processed_obs: 处理后的观察
        """
        # 实现您的观察预处理逻辑
        # 例如图像预处理、状态向量构建等
        
        processed_obs = {
            "image": obs["agentview_image"],
            "state": np.concatenate([
                obs["robot0_eef_pos"],
                obs["robot0_eef_quat"], 
                obs["robot0_gripper_qpos"]
            ]),
            "instruction": self.instruction
        }
        
        return processed_obs
    
    def _process_action(self, action):
        """
        处理动作输出
        
        Args:
            action: 原始动作
            
        Returns:
            action: 处理后的动作
        """
        # 确保动作是 numpy 数组
        if torch.is_tensor(action):
            action = action.cpu().numpy()
        
        # 标准化夹爪动作
        action = normalize_gripper_action(action, binarize=True)
        
        # 反转夹爪动作（如果需要）
        action = invert_gripper_action(action)
        
        return action
    
    @property
    def name(self):
        """返回策略名称"""
        return "MyCustomModel"
    
    @property
    def control_mode(self):
        """
        返回控制模式
        "ee" 表示末端执行器控制
        "joint" 表示关节控制
        """
        return "ee"
```

### 2. 注册策略

确保您的策略文件被正确导入。在 `vla_arena/evaluation/policy/__init__.py` 中添加：

```python
from .my_custom_policy import MyCustomPolicy
```

### 3. 创建配置文件

为您的模型创建配置文件 `vla_arena/configs/evaluation/my_custom_model.yaml`：

```yaml
# 模型特定配置
unnorm_key: "libero_spatial_no_noops"  # 动作反归一化键
image_resize_size: 256  # 图像调整大小
use_proprio: true  # 是否使用本体感受
center_crop: true  # 是否中心裁剪
```

### 4. 使用自定义模型

现在您可以在评估脚本中使用您的自定义模型：

```bash
python scripts/evaluate_policy.py \
    --task_suite preposition_generalization \
    --task_level 0 \
    --n-episode 1 \
    --policy my_custom_model \
    --model_ckpt /path/to/your/model \
    --save-dir logs/evaluation
```

## 配置说明

### 评估器配置

`Evaluator` 类的主要参数：

```python
evaluator = Evaluator(
    task_suite="preposition_generalization",  # 任务套件
    task_levels=[0, 1, 2],  # 评估级别列表
    n_episodes=5,  # 每个任务的回合数
    episode_config=None,  # 回合配置文件
    max_substeps=1,  # 最大子步数
    tolerance=1e-2,  # 容差
    metrics=["success_rate", "cumulative_cost", "safe_success_rate"],  # 评估指标
    save_dir="logs/evaluation",  # 保存目录
    visualization=True  # 是否可视化
)
```

### 策略配置

不同策略的配置参数：

#### OpenVLA
```python
policy = OpenVLA(
    model_ckpt="/path/to/openvla/model",
    attn_implementation="torch",  # 注意力实现
    norm_config_file=None,  # 归一化配置文件
    device="cuda"
)
```

#### OpenVLA-OFT
```python
policy = OpenVLAOFT(
    model_ckpt="/path/to/openvla-oft/model",
    use_l1_regression=True,  # 使用 L1 回归
    use_diffusion=False,  # 使用扩散模型
    use_film=True,  # 使用 FiLM
    num_images_in_input=2,  # 输入图像数量
    use_proprio=True,  # 使用本体感受
    num_open_loop_steps=8,  # 开环步数
    device="cuda"
)
```

#### SmolVLA
```python
policy = SmolVLA(
    model_ckpt="smolvla/smolvla-7b",  # HuggingFace 模型名称或本地路径
    device="cuda"
)
```

#### OpenPi
使用 OpenPi 模型需要先启动策略服务器，然后通过 WebSocket 连接进行推理：

**步骤 1: 启动 OpenPi 策略服务器**

在 OpenPi 库中启动策略服务器：

```bash
# 进入 OpenPi 目录
cd /path/to/openpi

# 启动策略服务器（使用迭代 20,000 的检查点，可根据需要修改）
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi0_fast_libero \
    --policy.dir=checkpoints/pi0_fast_libero/my_experiment/20000
```

服务器将在端口 8000 上监听（默认配置）。

**步骤 2: 配置 OpenPi 策略**

```python
policy = OpenPI(
    host="0.0.0.0",  # 服务器主机地址
    port=8000,       # 服务器端口
    replan_steps=4   # 重新规划步数
)
```

**步骤 3: 运行评估**

```bash
python scripts/evaluate_policy.py \
    --task_suite preposition_generalization \
    --task_level 0 \
    --n-episode 1 \
    --policy openpi \
    --save-dir logs/evaluation
```

**注意事项：**
- 确保 OpenPi 服务器在评估开始前已启动并运行
- 如果使用不同的端口，请在策略配置中相应修改 `port` 参数
- 服务器地址默认为 `0.0.0.0`，如需连接远程服务器，请修改 `host` 参数
- 评估过程中请保持服务器运行，否则会导致连接失败

## 评估结果存储

### 目录结构

评估完成后，结果将保存在指定的目录中，目录结构如下：

```
logs/evaluation/
└── eval_preposition_generalization_L0-2_OpenVLA_20241201_143022/
    ├── evaluation_metadata.json          # 评估元数据
    ├── complete_metrics.json             # 完整指标数据
    ├── evaluation_summary.txt            # 人类可读摘要
    ├── summary.json                      # 简化JSON摘要
    ├── task_details/                     # 任务详细结果
    │   ├── level_0/
    │   │   ├── task_1/
    │   │   │   └── detail_result.json
    │   │   └── task_2/
    │   │       └── detail_result.json
    │   └── level_1/
    │       └── ...
    ├── level_summaries/                  # 级别摘要
    │   ├── level_0_summary.json
    │   └── level_1_summary.json
    └── rollouts/                         # 可视化视频（如果启用）
        ├── level_0/
        │   └── 2024-12-01/
        │       ├── L0--2024-12-01--episode=0--success=True--task=place_object_on_table.mp4
        │       └── L0--2024-12-01--episode=1--success=False--task=move_object_to_bowl.mp4
        └── level_1/
            └── ...
```

### 日志文件示例

#### 1. 评估元数据 (`evaluation_metadata.json`)

```json
{
    "task_suite": "preposition_generalization",
    "task_levels": [0, 1, 2],
    "agent_name": "OpenVLA",
    "n_episodes": 5,
    "timestamp": "2024-12-01T14:30:22.123456",
    "metrics": ["success_rate", "cumulative_cost", "safe_success_rate"],
    "visualization": true
}
```

#### 2. 完整指标数据 (`complete_metrics.json`)

```json
{
    "timestamp": "2024-12-01T14:30:22.123456",
    "agent_name": "OpenVLA",
    "task_suite": "preposition_generalization",
    "task_levels": [0, 1, 2],
    "evaluation_dir": "/path/to/logs/evaluation/eval_preposition_generalization_L0-2_OpenVLA_20241201_143022",
    "metrics": {
        "evaluation_config": {
            "task_suite": "preposition_generalization",
            "task_levels": [0, 1, 2],
            "n_episodes_per_task": 5,
            "target_metrics": ["success_rate", "cumulative_cost", "safe_success_rate"]
        },
        "per_level_metrics": {
            "level_0": {
                "average_success_rate": 0.85,
                "average_safe_success_rate": 0.78,
                "average_cumulative_cost": 0.45,
                "num_tasks": 10,
                "task_metrics": {
                    "place_object_on_table": {
                        "success_rate": 0.9,
                        "safe_success_rate": 0.8,
                        "cumulative_cost": 0.3
                    },
                    "move_object_to_bowl": {
                        "success_rate": 0.8,
                        "safe_success_rate": 0.76,
                        "cumulative_cost": 0.6
                    }
                }
            },
            "level_1": {
                "average_success_rate": 0.72,
                "average_safe_success_rate": 0.65,
                "average_cumulative_cost": 0.68,
                "num_tasks": 10,
                "task_metrics": {
                    "place_object_between_objects": {
                        "success_rate": 0.7,
                        "safe_success_rate": 0.6,
                        "cumulative_cost": 0.8
                    }
                }
            }
        },
        "cross_level_summary": {
            "overall_average_success_rate": 0.785,
            "overall_average_safe_success_rate": 0.715,
            "overall_std_success_rate": 0.092,
            "overall_std_safe_success_rate": 0.095,
            "overall_average_cumulative_cost": 0.565,
            "overall_std_cumulative_cost": 0.175,
            "total_tasks_evaluated": 20,
            "total_episodes": 100,
            "total_successful_episodes": 78,
            "total_safe_successful_episodes": 71,
            "global_success_rate": 0.78,
            "global_safe_success_rate": 0.71
        }
    }
}
```

#### 3. 人类可读摘要 (`evaluation_summary.txt`)

```
======================================================================
EVALUATION SUMMARY
======================================================================

Agent: OpenVLA
Task Suite: preposition_generalization
Levels Evaluated: [0, 1, 2]
Timestamp: 2024-12-01T14:30:22.123456
Output Directory: /path/to/logs/evaluation/eval_preposition_generalization_L0-2_OpenVLA_20241201_143022

======================================================================
OVERALL RESULTS
======================================================================

Total Episodes Evaluated: 100
Total Tasks Evaluated: 20

Global Success Rate: 78.00%
  - Successful Episodes: 78/100

Global Safe Success Rate: 71.00%
  - Safe Successful Episodes: 71/100

Average Success Rate (across tasks): 78.50% ± 9.20%

Average Safe Success Rate (across tasks): 71.50% ± 9.50%

Average Cumulative Cost: 0.57 ± 0.18

======================================================================
PER-LEVEL RESULTS
======================================================================

Level 0:
  Success Rate: 85.00%
  Safe Success Rate: 78.00%
  Average Cost: 0.45
  Tasks Evaluated: 10

  Task Breakdown:
    • place_object_on_table:
      - Success Rate: 90.00%
      - Safe Success Rate: 80.00%
      - Avg Cost: 0.30
    • move_object_to_bowl:
      - Success Rate: 80.00%
      - Safe Success Rate: 76.00%
      - Avg Cost: 0.60

Level 1:
  Success Rate: 72.00%
  Safe Success Rate: 65.00%
  Average Cost: 0.68
  Tasks Evaluated: 10

  Task Breakdown:
    • place_object_between_objects:
      - Success Rate: 70.00%
      - Safe Success Rate: 60.00%
      - Avg Cost: 0.80
```

#### 4. 简化摘要 (`summary.json`)

```json
{
    "agent": "OpenVLA",
    "suite": "preposition_generalization",
    "levels": [0, 1, 2],
    "timestamp": "2024-12-01T14:30:22.123456",
    "overall": {
        "success_rate": 0.78,
        "safe_success_rate": 0.71,
        "avg_cost": 0.565,
        "total_episodes": 100
    },
    "per_level": {
        "0": {
            "success_rate": 0.85,
            "safe_success_rate": 0.78,
            "avg_cost": 0.45,
            "tasks": {
                "place_object_on_table": {
                    "success_rate": 0.9,
                    "safe_success_rate": 0.8,
                    "avg_cost": 0.3
                },
                "move_object_to_bowl": {
                    "success_rate": 0.8,
                    "safe_success_rate": 0.76,
                    "avg_cost": 0.6
                }
            }
        },
        "1": {
            "success_rate": 0.72,
            "safe_success_rate": 0.65,
            "avg_cost": 0.68,
            "tasks": {
                "place_object_between_objects": {
                    "success_rate": 0.7,
                    "safe_success_rate": 0.6,
                    "avg_cost": 0.8
                }
            }
        }
    }
}
```

#### 5. 任务详细结果 (`task_details/level_0/task_name/detail_result.json`)

```json
{
    "task_name": "place_object_on_table",
    "task_suite": "preposition_generalization",
    "task_level": 0,
    "agent_name": "OpenVLA",
    "metric_score": {
        "success_rate": 0.9,
        "safe_success_rate": 0.8,
        "cumulative_cost": 0.3,
        "cumulative_cost_std": 0.15,
        "cumulative_cost_min": 0.1,
        "cumulative_cost_max": 0.5
    },
    "timestamp": "2024-12-01T14:30:22.123456",
    "episodes": [
        {
            "success": true,
            "episode_id": 0,
            "episode_length": 45,
            "cumulative_cost": 0.2,
            "task_level": 0
        },
        {
            "success": true,
            "episode_id": 1,
            "episode_length": 52,
            "cumulative_cost": 0.4,
            "task_level": 0
        },
        {
            "success": false,
            "episode_id": 2,
            "episode_length": 200,
            "cumulative_cost": 1.2,
            "task_level": 0
        }
    ],
    "summary": {
        "total_episodes": 5,
        "successful_episodes": 4,
        "success_rate": 0.8,
        "average_steps": 48.5,
        "avg_cumulative_cost": 0.3,
        "std_cumulative_cost": 0.15,
        "min_cumulative_cost": 0.1,
        "max_cumulative_cost": 0.5,
        "median_cumulative_cost": 0.25,
        "safe_successful_episodes": 4,
        "safe_success_rate": 0.8
    }
}
```

#### 6. 级别摘要 (`level_summaries/level_0_summary.json`)

```json
{
    "task_level": 0,
    "agent_name": "OpenVLA",
    "timestamp": "2024-12-01T14:30:22.123456",
    "average_success_rate": 0.85,
    "average_safe_success_rate": 0.78,
    "std_success_rate": 0.08,
    "std_safe_success_rate": 0.09,
    "num_tasks": 10,
    "average_cumulative_cost": 0.45,
    "std_cumulative_cost": 0.12,
    "task_metrics": {
        "place_object_on_table": {
            "success_rate": 0.9,
            "safe_success_rate": 0.8,
            "cumulative_cost": 0.3
        },
        "move_object_to_bowl": {
            "success_rate": 0.8,
            "safe_success_rate": 0.76,
            "cumulative_cost": 0.6
        }
    },
    "task_details": {
        "place_object_on_table": {
            "task_level": 0,
            "metric_score": {
                "success_rate": 0.9,
                "safe_success_rate": 0.8,
                "cumulative_cost": 0.3
            },
            "success_rate": 0.9,
            "safe_success_rate": 0.8,
            "total_episodes": 5,
            "successful_episodes": 4,
            "safe_successful_episodes": 4,
            "failed_episodes": 1,
            "avg_cumulative_cost": 0.3
        }
    }
}
```

### 结果分析工具

您可以使用以下Python脚本快速分析评估结果：

```python
import json
import pandas as pd
from pathlib import Path

def analyze_evaluation_results(results_dir):
    """分析评估结果"""
    results_path = Path(results_dir)
    
    # 读取简化摘要
    with open(results_path / "summary.json", 'r') as f:
        summary = json.load(f)
    
    print(f"Agent: {summary['agent']}")
    print(f"Task Suite: {summary['suite']}")
    print(f"Overall Success Rate: {summary['overall']['success_rate']:.2%}")
    print(f"Overall Safe Success Rate: {summary['overall']['safe_success_rate']:.2%}")
    print(f"Average Cost: {summary['overall']['avg_cost']:.3f}")
    
    # 创建任务级别的DataFrame
    level_data = []
    for level, level_info in summary['per_level'].items():
        for task, task_info in level_info['tasks'].items():
            level_data.append({
                'Level': int(level),
                'Task': task,
                'Success Rate': task_info['success_rate'],
                'Safe Success Rate': task_info['safe_success_rate'],
                'Avg Cost': task_info['avg_cost']
            })
    
    df = pd.DataFrame(level_data)
    print("\nTask-level Results:")
    print(df.to_string(index=False))
    
    return df

# 使用示例
# df = analyze_evaluation_results("logs/evaluation/eval_preposition_generalization_L0-2_OpenVLA_20241201_143022")
```

### 完整评估示例

```bash
#!/bin/bash
# 完整的模型评估脚本

MODEL_PATH="/path/to/your/model"
OUTPUT_DIR="logs/evaluation_$(date +%Y%m%d_%H%M%S)"

python scripts/evaluate_policy.py \
    --task_suite preposition_generalization \
    --task_level 0-2 \
    --n-episode 5 \
    --policy openvla \
    --model_ckpt "$MODEL_PATH" \
    --save-dir "$OUTPUT_DIR" \
    --visualization \
    --metrics success_rate cumulative_cost safe_success_rate

echo "Evaluation completed. Results saved to: $OUTPUT_DIR"
```


如果您遇到问题或有改进建议，请参考代码注释或联系开发团队。