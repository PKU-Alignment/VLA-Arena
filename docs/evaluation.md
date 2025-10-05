# VLA-Arena Model Evaluation and Custom Model Guide

VLA-Arena is a unified framework for evaluating vision-language-action (VLA) models. This guide will help you understand how to use VLA-Arena to evaluate existing models and how to add custom models.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Model Evaluation](#model-evaluation)
3. [Adding Custom Models](#adding-custom-models)
4. [Configuration Instructions](#configuration-instructions)
5. [Troubleshooting](#troubleshooting)

## Quick Start

### Environment Preparation

Ensure you have installed VLA-Arena and its dependencies:

```bash
# Install VLA-Arena
pip install -e .

# Set environment variables
export MUJOCO_GL=egl
```

### OpenPi Model Installation

If you plan to use the OpenPi model for evaluation, you need to install the OpenPi library separately:

#### 1. Clone OpenPi Repository

```bash
# Clone repository (including submodules)
git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git

# Or if you already cloned the repository:
cd openpi
git submodule update --init --recursive
```

#### 2. Install Dependencies

OpenPi uses [uv](https://docs.astral.sh/uv/) to manage Python dependencies. First install uv:

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then install OpenPi:

```bash
cd openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

**Note:** `GIT_LFS_SKIP_SMUDGE=1` is required to skip LFS file downloads for LeRobot dependencies.

### Basic Evaluation Command

The simplest evaluation command:

```bash
python scripts/evaluate_policy.py \
    --task_suite preposition_generalization \
    --task_level 0 \
    --n-episode 1 \
    --policy openvla \
    --model_ckpt /path/to/your/model \
    --save-dir logs/evaluation
```

## Model Evaluation

### Supported Models

VLA-Arena currently supports the following models:

- **OpenVLA**: Standard OpenVLA model
- **OpenVLA-OFT**: OpenVLA model with online fine-tuning capability
- **SmolVLA**: SmolVLA model
- **Random**: Random policy (for baseline testing)
- **UniVLA**: UniVLA model
- **OpenPi**: OpenPi model (requires starting policy server first)

### Using Evaluation Scripts

#### 1. Using Python Script

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

#### 2. Using Shell Script (Recommended)

```bash
# Copy and edit the configuration script
cp scripts/evaluate_policy.sh my_evaluation.sh
# Edit the configuration section in my_evaluation.sh
bash my_evaluation.sh
```

### Task Suites

VLA-Arena provides multiple task suites:

##### Safety
- **safety_dynamic_obstacles**: Safety Dynamic Obstacles Task
- **safety_hazard_avoidance**: Safety Hazard Avoidance Task
- **safety_object_state_preservation**: Safety Object State Preservation Task
- **safety_risk_aware_grasping**: Safety Risk Aware Grasping Task
- **safety_static_obstacles**: Safety Static Obstacles Task

##### Robustness
- **robustness_dynamic_distractors**: Robustness Dynamic Distractors Task
- **robustness_static_distractors**: Robustness Static Distractors Task
- **robustness_visual_variations**: Robustness Visual Variations Task

##### Generalization
- **generalization_language_variations**: Generalization Language Variations Task
- **generalization_object_preposition_combinations**: Generalization Object Preposition Combinations Task
- **generalization_task_workflows**: Generalization Task Workflows Task
- **generalization_unseen_objects**: Generalization Unseen Objects Task

##### Others
- **long_horizon**: Long Horizon Task

### Task Levels

Each task suite contains multiple difficulty levels:

- **Level 0**: Simple tasks
- **Level 1**: Medium difficulty tasks  
- **Level 2**: Difficult tasks

Supports multi-level evaluation:

```bash
# Evaluate a single level
--task_level 0

# Evaluate a level range
--task_level 0-2

# Evaluate specific levels
--task_level 0,2
```

### Evaluation Metrics

Supported evaluation metrics:

- **success_rate**: Success rate
- **safe_success_rate**: Safe success rate (cost < 1.0)
- **cumulative_cost**: Cumulative cost
- **episode_length**: Episode length

### Visualization Options

Enable visualization to save evaluation videos:

```bash
--visualization
```

Videos will be saved in the `{save_dir}/rollouts/level_{level}/` directory.

## Adding Custom Models

### 1. Create Custom Policy Class

Create a new policy file, e.g., `my_custom_policy.py`:

```python
import torch
import numpy as np
from vla_arena.evaluation.policy.base import Policy, PolicyRegistry
from vla_arena.evaluation.utils import normalize_gripper_action, invert_gripper_action

@PolicyRegistry.register("my_custom_model")
class MyCustomPolicy(Policy):
    """
    Custom model policy implementation
    """
    
    def __init__(self, 
                 model_ckpt,
                 device="cuda",
                 **kwargs):
        """
        Initialize custom policy
        
        Args:
            model_ckpt: Model checkpoint path
            device: Running device
            **kwargs: Other parameters
        """
        # Check device availability
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            device = "cpu"
        
        # Load your model
        self.model = self._load_model(model_ckpt, device)
        self.device = device
        self.instruction = kwargs.get('instruction', None)
        
        # Call parent class constructor
        super().__init__(self.model)
        
        print(f"Custom model loaded successfully on {device}")
    
    def _load_model(self, model_ckpt, device):
        """
        Load your custom model
        
        Args:
            model_ckpt: Model checkpoint path
            device: Running device
            
        Returns:
            Loaded model
        """
        # Implement your model loading logic here
        # For example:
        # model = YourCustomModel.from_pretrained(model_ckpt)
        # model.to(device)
        # model.eval()
        # return model
        
        raise NotImplementedError("Please implement your model loading logic")
    
    def reset_instruction(self, instruction):
        """
        Reset policy instruction
        
        Args:
            instruction: Task instruction
        """
        self.instruction = instruction
        # Reset model internal state if needed
        if hasattr(self.model, 'reset'):
            self.model.reset()
    
    def predict(self, obs, **kwargs):
        """
        Predict action
        
        Args:
            obs: Observation dictionary containing:
                - agentview_image: Main view image
                - robot0_eye_in_hand_image: Wrist camera image (optional)
                - robot0_eef_pos: End-effector position
                - robot0_eef_quat: End-effector quaternion
                - robot0_gripper_qpos: Gripper position
            **kwargs: Other parameters
            
        Returns:
            action: 7-dimensional action array [x, y, z, rx, ry, rz, gripper]
        """
        # Process observation
        processed_obs = self._prepare_observation(obs)
        
        # Get model prediction
        with torch.inference_mode():
            action = self.model.predict(processed_obs)
        
        # Process action
        action = self._process_action(action)
        
        return action
    
    def _prepare_observation(self, obs):
        """
        Prepare observation data
        
        Args:
            obs: Raw observation
            
        Returns:
            processed_obs: Processed observation
        """
        # Implement your observation preprocessing logic
        # For example, image preprocessing, state vector construction, etc.
        
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
        Process action output
        
        Args:
            action: Raw action
            
        Returns:
            action: Processed action
        """
        # Ensure action is a numpy array
        if torch.is_tensor(action):
            action = action.cpu().numpy()
        
        # Normalize gripper action
        action = normalize_gripper_action(action, binarize=True)
        
        # Invert gripper action (if needed)
        action = invert_gripper_action(action)
        
        return action
    
    @property
    def name(self):
        """Return policy name"""
        return "MyCustomModel"
    
    @property
    def control_mode(self):
        """
        Return control mode
        "ee" for end-effector control
        "joint" for joint control
        """
        return "ee"
```

### 2. Register Policy

Ensure your policy file is correctly imported. Add the following to `vla_arena/evaluation/policy/__init__.py`:

```python
from .my_custom_policy import MyCustomPolicy
```

### 3. Create Configuration File

Create a configuration file for your model `vla_arena/configs/evaluation/my_custom_model.yaml`:

```yaml
# Model-specific configuration
unnorm_key: "libero_spatial_no_noops"  # Action denormalization key
image_resize_size: 256  # Image resize size
use_proprio: true  # Whether to use proprioception
center_crop: true  # Whether to center crop
```

### 4. Use Custom Model

Now you can use your custom model in evaluation scripts:

```bash
python scripts/evaluate_policy.py \
    --task_suite preposition_generalization \
    --task_level 0 \
    --n-episode 1 \
    --policy my_custom_model \
    --model_ckpt /path/to/your/model \
    --save-dir logs/evaluation
```

## Configuration Instructions

### Evaluator Configuration

Main parameters of the `Evaluator` class:

```python
evaluator = Evaluator(
    task_suite="preposition_generalization",  # Task suite
    task_levels=[0, 1, 2],  # Evaluation level list
    n_episodes=5,  # Number of episodes per task
    episode_config=None,  # Episode configuration file
    max_substeps=1,  # Maximum substeps
    tolerance=1e-2,  # Tolerance
    metrics=["success_rate", "cumulative_cost", "safe_success_rate"],  # Evaluation metrics
    save_dir="logs/evaluation",  # Save directory
    visualization=True  # Whether to visualize
)
```

### Policy Configuration

Configuration parameters for different policies:

#### OpenVLA
```python
policy = OpenVLA(
    model_ckpt="/path/to/openvla/model",
    attn_implementation="torch",  # Attention implementation
    norm_config_file=None,  # Normalization configuration file
    device="cuda"
)
```

#### OpenVLA-OFT
```python
policy = OpenVLAOFT(
    model_ckpt="/path/to/openvla-oft/model",
    use_l1_regression=True,  # Use L1 regression
    use_diffusion=False,  # Use diffusion model
    use_film=True,  # Use FiLM
    num_images_in_input=2,  # Number of input images
    use_proprio=True,  # Use proprioception
    num_open_loop_steps=8,  # Open-loop steps
    device="cuda"
)
```

#### SmolVLA
```python
policy = SmolVLA(
    model_ckpt="smolvla/smolvla-7b",  # HuggingFace model name or local path
    device="cuda"
)
```

#### OpenPi
Using the OpenPi model requires starting a policy server first, then connecting via WebSocket for inference:

**Step 1: Start OpenPi Policy Server**

Start the policy server in the OpenPi library:

```bash
# Navigate to OpenPi directory
cd /path/to/openpi

# Start policy server (using checkpoint for iteration 20,000, modify as needed)
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi0_fast_libero \
    --policy.dir=checkpoints/pi0_fast_libero/my_experiment/20000
```

The server will listen on port 8000 (default configuration).

**Step 2: Configure OpenPi Policy**

```python
policy = OpenPI(
    host="0.0.0.0",  # Server host address
    port=8000,       # Server port
    replan_steps=4   # Replanning steps
)
```

**Step 3: Run Evaluation**

```bash
python scripts/evaluate_policy.py \
    --task_suite preposition_generalization \
    --task_level 0 \
    --n-episode 1 \
    --policy openpi \
    --save-dir logs/evaluation
```

**Important Notes:**
- Ensure the OpenPi server is started and running before beginning evaluation
- If using a different port, modify the `port` parameter in the policy configuration accordingly
- Server address defaults to `0.0.0.0`, modify the `host` parameter if connecting to a remote server
- Keep the server running during evaluation, otherwise connection will fail

## Evaluation Result Storage

### Directory Structure

After evaluation is completed, results will be saved in the specified directory with the following structure:

```
logs/evaluation/
└── eval_preposition_generalization_L0-2_OpenVLA_20241201_143022/
    ├── evaluation_metadata.json          # Evaluation metadata
    ├── complete_metrics.json             # Complete metric data
    ├── evaluation_summary.txt            # Human-readable summary
    ├── summary.json                      # Simplified JSON summary
    ├── task_details/                     # Task detailed results
    │   ├── level_0/
    │   │   ├── task_1/
    │   │   │   └── detail_result.json
    │   │   └── task_2/
    │   │       └── detail_result.json
    │   └── level_1/
    │       └── ...
    ├── level_summaries/                  # Level summaries
    │   ├── level_0_summary.json
    │   └── level_1_summary.json
    └── rollouts/                         # Visualization videos (if enabled)
        ├── level_0/
        │   └── 2024-12-01/
        │       ├── L0--2024-12-01--episode=0--success=True--task=place_object_on_table.mp4
        │       └── L0--2024-12-01--episode=1--success=False--task=move_object_to_bowl.mp4
        └── level_1/
            └── ...
```

### Log File Examples

#### 1. Evaluation Metadata (`evaluation_metadata.json`)

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

#### 2. Complete Metric Data (`complete_metrics.json`)

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

#### 3. Human-Readable Summary (`evaluation_summary.txt`)

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

#### 4. Simplified Summary (`summary.json`)

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

#### 5. Task Detailed Results (`task_details/level_0/task_name/detail_result.json`)

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

#### 6. Level Summary (`level_summaries/level_0_summary.json`)

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

### Result Analysis Tools

You can use the following Python script to quickly analyze evaluation results:

```python
import json
import pandas as pd
from pathlib import Path

def analyze_evaluation_results(results_dir):
    """Analyze evaluation results"""
    results_path = Path(results_dir)
    
    # Read simplified summary
    with open(results_path / "summary.json", 'r') as f:
        summary = json.load(f)
    
    print(f"Agent: {summary['agent']}")
    print(f"Task Suite: {summary['suite']}")
    print(f"Overall Success Rate: {summary['overall']['success_rate']:.2%}")
    print(f"Overall Safe Success Rate: {summary['overall']['safe_success_rate']:.2%}")
    print(f"Average Cost: {summary['overall']['avg_cost']:.3f}")
    
    # Create task-level DataFrame
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

# Usage example
# df = analyze_evaluation_results("logs/evaluation/eval_preposition_generalization_L0-2_OpenVLA_20241201_143022")
```

## Examples and Best Practices

### Complete Evaluation Example

```bash
#!/bin/bash
# Complete model evaluation script

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

If you encounter problems or have suggestions for improvement, please feel free to submit an issue or a pull request.