# 使用VLA-Arena生成的数据集微调其他模型

VLA-Arena提供了完整的搜集数据、转换数据格式、评估语言-视觉-动作模型的框架，本指南将带你了解如何使用VLA-Arena生成的数据集微调一些VLA模型

## 快速开始

如果你已经准备好了数据集和OpenVLA模型，可以直接使用以下命令开始微调：

### 标准OpenVLA微调

```bash
# 1. 激活环境
conda activate openvla

# 2. 运行微调脚本
./vla-scripts/finetune_openvla.sh \
    --dataset_name "your_dataset" \
    --vla_path "/path/to/your/openvla/model" \
    --data_root_dir "/path/to/your/datasets" \
    --openvla_root_dir "/path/to/openvla/repo"
```

### OpenVLA OFT微调（推荐）

```bash
# 1. 激活环境
conda activate openvla

# 2. 运行OFT微调脚本
./vla-scripts/finetune_openvla_oft.sh \
    --dataset_name "your_dataset" \
    --vla_path "/path/to/your/openvla/model" \
    --data_root_dir "/path/to/your/datasets" \
    --openvla_root_dir "/path/to/openvla/repo"
```

### UniVLA微调

```bash
# 1. 激活环境
conda activate univla

# 2. 运行UniVLA微调脚本
./vla-scripts/finetune_univla.sh \
    --dataset_name "your_dataset" \
    --vla_path "/path/to/your/univla/model" \
    --lam_path "/path/to/your/lam/checkpoint" \
    --data_root_dir "/path/to/your/datasets" \
    --univla_root_dir "/path/to/univla/repo"
```

详细的使用说明请参考下面的章节。

## 目录

1. [快速开始](#快速开始)
2. [微调OpenVLA](#微调OpenVLA)
   - [安装OpenVLA库](#安装OpenVLA库)
   - [使用脚本一键微调](#使用脚本一键微调)
     - [基本使用方法](#基本使用方法)
     - [必需参数](#必需参数)
     - [可选参数](#可选参数)
     - [数据集配置参数](#数据集配置参数)
     - [状态和动作编码选项](#状态和动作编码选项)
     - [使用示例](#使用示例)
     - [脚本功能](#脚本功能)
     - [注意事项](#注意事项)
3. [微调OpenVLA OFT](#微调OpenVLA-OFT)
   - [OFT微调简介](#OFT微调简介)
   - [使用OFT脚本微调](#使用OFT脚本微调)
     - [基本使用方法](#基本使用方法-1)
     - [必需参数](#必需参数-1)
     - [基础训练参数](#基础训练参数)
     - [LoRA参数](#LoRA参数)
     - [动作表示参数](#动作表示参数)
     - [架构选项](#架构选项)
     - [学习率调度](#学习率调度)
     - [验证和检查点](#验证和检查点)
     - [日志配置](#日志配置)
     - [数据集配置参数](#数据集配置参数-1)
     - [GPU配置](#GPU配置)
     - [使用示例](#使用示例-1)
     - [脚本功能](#脚本功能-1)
     - [注意事项](#注意事项-1)
4. [微调UniVLA](#微调UniVLA)
   - [安装UniVLA库](#安装UniVLA库)
   - [使用脚本一键微调](#使用脚本一键微调-1)
     - [基本使用方法](#基本使用方法-2)
     - [必需参数](#必需参数-2)
     - [基础训练参数](#基础训练参数-1)
     - [LoRA参数](#LoRA参数-1)
     - [UniVLA特定参数](#UniVLA特定参数)
     - [LAM参数](#LAM参数)
     - [日志配置](#日志配置-1)
     - [数据集配置参数](#数据集配置参数-2)
     - [GPU配置](#GPU配置-1)
     - [使用示例](#使用示例-2)
     - [脚本功能](#脚本功能-2)
     - [注意事项](#注意事项-2)
5. [微调OpenPi](#微调OpenPi)
   - [安装OpenPi库](#安装OpenPi库)
   - [使用脚本一键微调](#使用脚本一键微调-2)
     - [基本使用方法](#基本使用方法-3)
     - [必需参数](#必需参数-3)
     - [模型配置参数](#模型配置参数)
     - [训练参数](#训练参数)
     - [数据集配置参数](#数据集配置参数-3)
     - [使用示例](#使用示例-3)
     - [脚本功能](#脚本功能-3)
     - [注意事项](#注意事项-3)
6. [模型评估](#模型评估)
7. [添加自定义模型](#添加自定义模型)
8. [配置说明](#配置说明)

## 微调OpenVLA

### 安装OpenVLA库

```bash
# Create and activate conda environment
conda create -n openvla python=3.10 -y
conda activate openvla

# Install PyTorch. Below is a sample command to do this, but you should check the following link
# to find installation instructions that are specific to your compute platform:
# https://pytorch.org/get-started/locally/
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y  # UPDATE ME!

# Clone and install the openvla repo
git clone https://github.com/openvla/openvla.git
cd openvla
pip install -e .

# Install Flash Attention 2 for training (https://github.com/Dao-AILab/flash-attention)
#   =>> If you run into difficulty, try `pip cache remove flash_attn` first
pip install packaging ninja
ninja --version; echo $?  # Verify Ninja --> should return exit code "0"
pip install "flash-attn==2.5.5" --no-build-isolation
```
### 使用脚本一键微调

将 [finetune_openvla.sh](./finetune_openvla.sh) 粘贴至 openvla/vla-scripts 目录下，该脚本会自动添加数据集配置并运行微调。

#### 基本使用方法

```bash
# 激活conda环境
conda activate openvla

# 基本使用（需要提供必需参数）
./vla-scripts/finetune_openvla.sh \
    --dataset_name "my_dataset" \
    --vla_path "/path/to/openvla/model" \
    --data_root_dir "/path/to/datasets" \
    --openvla_root_dir "/path/to/openvla/repo"

# 自定义参数
./vla-scripts/finetune_openvla.sh \
    --dataset_name "my_dataset" \
    --vla_path "/path/to/openvla/model" \
    --data_root_dir "/path/to/datasets" \
    --openvla_root_dir "/path/to/openvla/repo" \
    --batch_size 4 \
    --learning_rate 1e-4 \
    --max_steps 10000 \
    --wandb_project "my_project"
```

#### 必需参数

- `--dataset_name`: 数据集名称（必需）
- `--vla_path`: OpenVLA模型路径（必需）
- `--data_root_dir`: 数据集根目录（必需）
- `--openvla_root_dir`: OpenVLA仓库根目录（必需）

#### 可选参数

- `--run_root_dir`: 运行结果保存目录（默认：`new_runs`）
- `--batch_size`: 批次大小（默认：`2`）
- `--learning_rate`: 学习率（默认：`5e-4`）
- `--max_steps`: 最大训练步数（默认：`50000`）
- `--use_lora`: 是否使用LoRA微调（默认：`true`）
- `--lora_rank`: LoRA秩（默认：`32`）
- `--use_quantization`: 是否使用量化（默认：`false`）
- `--image_aug`: 是否使用图像增强（默认：`true`）
- `--wandb_project`: WandB项目名称（默认：`safe-openvla`）
- `--wandb_entity`: WandB实体名称（默认：`trial`）
- `--num_gpus`: 使用的GPU数量（默认：`1`）

#### 数据集配置参数

脚本会自动将你的数据集配置添加到 `configs.py` 和 `transforms.py` 文件中。你可以自定义数据集配置：

- `--image_obs_primary`: 主要图像观测键（默认：`image`）
- `--image_obs_secondary`: 次要图像观测键（默认：空）
- `--image_obs_wrist`: 手腕图像观测键（默认：`wrist_image`）
- `--depth_obs_primary`: 主要深度观测键（默认：空）
- `--depth_obs_secondary`: 次要深度观测键（默认：空）
- `--depth_obs_wrist`: 手腕深度观测键（默认：空）
- `--state_obs_keys`: 状态观测键（默认：`EEF_state,None,gripper_state`）
- `--state_encoding`: 状态编码（默认：`POS_EULER`）
- `--action_encoding`: 动作编码（默认：`EEF_POS`）

#### 状态和动作编码选项

**状态编码**：
- `NONE`: 无本体感受状态
- `POS_EULER`: EEF XYZ (3) + Roll-Pitch-Yaw (3) + <PAD> (1) + 夹爪开合 (1)
- `POS_QUAT`: EEF XYZ (3) + 四元数 (4) + 夹爪开合 (1)
- `JOINT`: 关节角度 (7, 不足时用<PAD>填充) + 夹爪开合 (1)
- `JOINT_BIMANUAL`: 关节角度 (2 x [ 关节角度 (6) + 夹爪开合 (1) ])

**动作编码**：
- `EEF_POS`: EEF增量XYZ (3) + Roll-Pitch-Yaw (3) + 夹爪开合 (1)
- `JOINT_POS`: 关节增量位置 (7) + 夹爪开合 (1)
- `JOINT_POS_BIMANUAL`: 关节增量位置 (2 x [ 关节增量位置 (6) + 夹爪开合 (1) ])
- `EEF_R6`: EEF增量XYZ (3) + R6 (6) + 夹爪开合 (1)

#### 使用示例

**示例1：基本使用**
```bash
./vla-scripts/finetune_openvla.sh \
    --dataset_name "my_robot_dataset" \
    --vla_path "/path/to/openvla/model" \
    --data_root_dir "/path/to/datasets" \
    --openvla_root_dir "/path/to/openvla/repo"
```

**示例2：自定义配置**
```bash
./vla-scripts/finetune_openvla.sh \
    --dataset_name "custom_dataset" \
    --vla_path "/path/to/openvla/model" \
    --data_root_dir "/path/to/datasets" \
    --openvla_root_dir "/path/to/openvla/repo" \
    --image_obs_primary "front_camera" \
    --image_obs_wrist "gripper_camera" \
    --state_obs_keys "joint_positions,None,gripper_state" \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --max_steps 20000
```

**示例3：使用量化**
```bash
./vla-scripts/finetune_openvla.sh \
    --dataset_name "quantized_dataset" \
    --vla_path "/path/to/openvla/model" \
    --data_root_dir "/path/to/datasets" \
    --openvla_root_dir "/path/to/openvla/repo" \
    --use_quantization true \
    --batch_size 16 \
    --max_steps 5000
```

#### 脚本功能

1. **参数验证**：检查必需参数是否提供
2. **添加数据集配置**：自动将你的数据集配置添加到：
   - `{openvla_root_dir}/prismatic/vla/datasets/rlds/oxe/configs.py`
   - `{openvla_root_dir}/prismatic/vla/datasets/rlds/oxe/transforms.py`
3. **运行微调**：使用你的参数执行OpenVLA微调脚本

#### 注意事项

- 脚本使用 `libero_dataset_transform` 作为新数据集的默认变换函数
- 如果数据集配置已存在，将跳过添加配置步骤
- 脚本自动处理状态观测键中的 `None` 值
- 确保你的数据集采用正确的RLDS格式并位于指定的数据目录中

## 微调OpenVLA OFT

### OFT微调简介

OpenVLA OFT（Open-source Foundation Transformers）微调提供了更高级的训练选项和更好的性能。OFT版本支持：

- **更丰富的训练参数**：包括学习率调度、梯度累积、验证集等
- **动作表示选项**：支持L1回归和扩散建模
- **架构增强**：FiLM语言融合、多图像输入、本体感受状态等
- **高级优化**：LoRA dropout、训练时LoRA合并等

### 使用OFT脚本微调

将 [finetune_openvla_oft.sh](./finetune_openvla_oft.sh) 粘贴至 openvla/vla-scripts 目录下，该脚本提供了更全面的微调选项。

#### 基本使用方法

```bash
# 激活conda环境
conda activate openvla

# 基本使用（需要提供必需参数）
./vla-scripts/finetune_openvla_oft.sh \
    --dataset_name "my_dataset" \
    --vla_path "/path/to/openvla/model" \
    --data_root_dir "/path/to/datasets" \
    --openvla_root_dir "/path/to/openvla/repo"

# 自定义参数
./vla-scripts/finetune_openvla_oft.sh \
    --dataset_name "my_dataset" \
    --vla_path "/path/to/openvla/model" \
    --data_root_dir "/path/to/datasets" \
    --openvla_root_dir "/path/to/openvla/repo" \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --max_steps 100000 \
    --use_l1_regression true \
    --use_film true
```

#### 必需参数

- `--dataset_name`: 数据集名称（必需）
- `--vla_path`: OpenVLA模型路径（必需）
- `--data_root_dir`: 数据集根目录（必需）
- `--openvla_root_dir`: OpenVLA仓库根目录（必需）

#### 基础训练参数

- `--run_root_dir`: 运行结果保存目录（默认：`all_runs`）
- `--batch_size`: 批次大小（默认：`7`）
- `--learning_rate`: 学习率（默认：`5e-4`）
- `--max_steps`: 最大训练步数（默认：`150000`）
- `--grad_accumulation_steps`: 梯度累积步数（默认：`1`）
- `--shuffle_buffer_size`: 数据加载器随机缓冲区大小（默认：`100000`）

#### LoRA参数

- `--use_lora`: 是否使用LoRA微调（默认：`true`）
- `--lora_rank`: LoRA秩（默认：`32`）
- `--lora_dropout`: LoRA dropout（默认：`0.0`）
- `--merge_lora_during_training`: 训练时合并LoRA（默认：`true`）

#### 动作表示参数

- `--use_l1_regression`: 使用L1回归（默认：`true`）
- `--use_diffusion`: 使用扩散建模（默认：`false`）
- `--num_diffusion_steps_train`: 训练扩散步数（默认：`50`）
- `--diffusion_sample_freq`: 扩散采样频率（默认：`50`）

#### 架构选项

- `--use_film`: 使用FiLM进行语言融合（默认：`true`）
- `--num_images_in_input`: 输入图像数量（默认：`2`）
- `--use_proprio`: 包含本体感受状态（默认：`false`）
- `--use_quantization`: 使用量化（默认：`false`）
- `--image_aug`: 使用图像增强（默认：`true`）

#### 学习率调度

- `--lr_warmup_steps`: 学习率预热步数（默认：`0`）
- `--num_steps_before_decay`: 学习率衰减前步数（默认：`60000`）

#### 验证和检查点

- `--use_val_set`: 使用验证集（默认：`false`）
- `--val_freq`: 验证频率（默认：`10000`）
- `--val_time_limit`: 验证时间限制（默认：`180`）
- `--save_freq`: 保存频率（默认：`5000`）
- `--save_latest_checkpoint_only`: 仅保存最新检查点（默认：`false`）
- `--resume`: 从检查点恢复（默认：`false`）
- `--resume_step`: 恢复步数（默认：空）

#### 日志配置

- `--wandb_project`: WandB项目名称（默认：`openvla-oft-workflow-generalization`）
- `--wandb_entity`: WandB实体名称（默认：`trial`）
- `--wandb_log_freq`: WandB日志频率（默认：`10`）

#### 数据集配置参数

脚本会自动将你的数据集配置添加到 `configs.py` 和 `transforms.py` 文件中。你可以自定义数据集配置：

- `--image_obs_primary`: 主要图像观测键（默认：`image`）
- `--image_obs_secondary`: 次要图像观测键（默认：空）
- `--image_obs_wrist`: 手腕图像观测键（默认：`wrist_image`）
- `--depth_obs_primary`: 主要深度观测键（默认：空）
- `--depth_obs_secondary`: 次要深度观测键（默认：空）
- `--depth_obs_wrist`: 手腕深度观测键（默认：空）
- `--state_obs_keys`: 状态观测键（默认：`EEF_state,None,gripper_state`）
- `--state_encoding`: 状态编码（默认：`POS_EULER`）
- `--action_encoding`: 动作编码（默认：`EEF_POS`）

#### GPU配置

- `--num_gpus`: 使用的GPU数量（默认：`1`）

#### 使用示例

**示例1：基本OFT使用**
```bash
./vla-scripts/finetune_openvla_oft.sh \
    --dataset_name "my_robot_dataset" \
    --vla_path "/path/to/openvla/model" \
    --data_root_dir "/path/to/datasets" \
    --openvla_root_dir "/path/to/openvla/repo"
```

**示例2：高级OFT配置**
```bash
./vla-scripts/finetune_openvla_oft.sh \
    --dataset_name "advanced_dataset" \
    --vla_path "/path/to/openvla/model" \
    --data_root_dir "/path/to/datasets" \
    --openvla_root_dir "/path/to/openvla/repo" \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --max_steps 100000 \
    --use_l1_regression true \
    --use_film true \
    --use_proprio true \
    --num_images_in_input 3 \
    --lora_rank 64 \
    --grad_accumulation_steps 2
```

**示例3：使用扩散建模**
```bash
./vla-scripts/finetune_openvla_oft.sh \
    --dataset_name "diffusion_dataset" \
    --vla_path "/path/to/openvla/model" \
    --data_root_dir "/path/to/datasets" \
    --openvla_root_dir "/path/to/openvla/repo" \
    --use_diffusion true \
    --num_diffusion_steps_train 100 \
    --diffusion_sample_freq 25 \
    --batch_size 4
```

**示例4：多GPU训练**
```bash
./vla-scripts/finetune_openvla_oft.sh \
    --dataset_name "multi_gpu_dataset" \
    --vla_path "/path/to/openvla/model" \
    --data_root_dir "/path/to/datasets" \
    --openvla_root_dir "/path/to/openvla/repo" \
    --num_gpus 4 \
    --batch_size 16 \
    --grad_accumulation_steps 1
```

#### 脚本功能

1. **参数验证**：检查必需参数是否提供
2. **添加数据集配置**：自动将你的数据集配置添加到：
   - `{openvla_root_dir}/prismatic/vla/datasets/rlds/oxe/configs.py`
   - `{openvla_root_dir}/prismatic/vla/datasets/rlds/oxe/transforms.py`
3. **运行OFT微调**：使用你的参数执行OpenVLA OFT微调脚本
4. **多GPU支持**：支持多GPU分布式训练

#### 注意事项

- OFT版本提供更丰富的训练选项，适合需要精细控制训练过程的用户
- 支持扩散建模，适合需要生成式动作预测的场景
- FiLM语言融合可以提供更好的语言-视觉交互
- 多图像输入支持多视角机器人任务
- 确保你的硬件资源足够支持所选的训练配置

## 微调UniVLA

### 安装UniVLA库

```bash
# 创建并激活conda环境
conda create -n univla python=3.10 -y
conda activate univla

# 安装PyTorch。下面是一个示例命令，但你应该检查以下链接
# 以找到适合你计算平台的安装说明：
# https://pytorch.org/get-started/locally/
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y  # 请更新！

# 克隆并安装univla仓库
git clone https://github.com/opendrivelab/UniVLA.git
cd UniVLA
pip install -e .

# 安装Flash Attention 2用于训练 (https://github.com/Dao-AILab/flash-attention)
#   =>> 如果遇到困难，请先尝试 `pip cache remove flash_attn`
pip install packaging ninja
ninja --version; echo $?  # 验证Ninja --> 应该返回退出代码"0"
pip install "flash-attn==2.5.5" --no-build-isolation

# 安装UniVLA的额外依赖
pip install swanlab
pip install ema-pytorch
pip install peft
pip install accelerate
```

### 使用脚本一键微调

将 [finetune_univla.sh](./finetune_univla.sh) 粘贴至 UniVLA/vla-scripts 目录下，该脚本会自动添加数据集配置并运行微调。

#### 基本使用方法

```bash
# 激活conda环境
conda activate univla

# 基本使用（需要提供必需参数）
./vla-scripts/finetune_univla.sh \
    --dataset_name "my_dataset" \
    --vla_path "/path/to/univla/model" \
    --lam_path "/path/to/lam/checkpoint" \
    --data_root_dir "/path/to/datasets" \
    --univla_root_dir "/path/to/univla/repo"

# 自定义参数
./vla-scripts/finetune_univla.sh \
    --dataset_name "my_dataset" \
    --vla_path "/path/to/univla/model" \
    --lam_path "/path/to/lam/checkpoint" \
    --data_root_dir "/path/to/datasets" \
    --univla_root_dir "/path/to/univla/repo" \
    --batch_size 4 \
    --learning_rate 1e-4 \
    --max_steps 50000 \
    --wandb_project "my_project"
```

#### 必需参数

- `--dataset_name`: 数据集名称（必需）
- `--vla_path`: UniVLA模型路径（必需）
- `--lam_path`: LAM（潜在动作模型）检查点路径（必需）
- `--data_root_dir`: 数据集根目录（必需）
- `--univla_root_dir`: UniVLA仓库根目录（必需）

#### 基础训练参数

- `--run_root_dir`: 运行结果保存目录（默认：`all_runs`）
- `--batch_size`: 批次大小（默认：`8`）
- `--learning_rate`: 学习率（默认：`3.5e-4`）
- `--max_steps`: 最大训练步数（默认：`100000`）
- `--save_steps`: 保存间隔（默认：`10000`）
- `--grad_accumulation_steps`: 梯度累积步数（默认：`2`）
- `--shuffle_buffer_size`: 数据加载器随机缓冲区大小（默认：`16000`）

#### LoRA参数

- `--use_lora`: 是否使用LoRA微调（默认：`true`）
- `--lora_rank`: LoRA秩（默认：`32`）
- `--lora_dropout`: LoRA dropout（默认：`0.0`）
- `--use_quantization`: 是否使用量化（默认：`false`）

#### UniVLA特定参数

- `--freeze_vla`: 冻结VLA骨干网络（默认：`false`）
- `--save_latest_checkpoint_only`: 仅保存最新检查点（默认：`true`）
- `--run_id_note`: 实验ID的额外注释（默认：空）

#### LAM参数

UniVLA使用潜在动作模型（LAM）进行动作表示。这些参数控制LAM架构：

- `--codebook_size`: LAM码本大小（默认：`16`）
- `--lam_model_dim`: LAM模型维度（默认：`768`）
- `--lam_latent_dim`: LAM潜在维度（默认：`128`）
- `--lam_patch_size`: LAM补丁大小（默认：`14`）
- `--lam_enc_blocks`: LAM编码器块数（默认：`12`）
- `--lam_dec_blocks`: LAM解码器块数（默认：`12`）
- `--lam_num_heads`: LAM注意力头数（默认：`12`）
- `--window_size`: 动作窗口大小（默认：`12`）

#### 日志配置

- `--wandb_project`: WandB项目名称（默认：`finetune-UniVLA`）
- `--wandb_entity`: WandB实体名称（默认：`opendrivelab`）

#### 数据集配置参数

脚本会自动将你的数据集配置添加到 `configs.py` 和 `transforms.py` 文件中。你可以自定义数据集配置：

- `--image_obs_primary`: 主要图像观测键（默认：`image`）
- `--image_obs_secondary`: 次要图像观测键（默认：空）
- `--image_obs_wrist`: 手腕图像观测键（默认：`wrist_image`）
- `--depth_obs_primary`: 主要深度观测键（默认：空）
- `--depth_obs_secondary`: 次要深度观测键（默认：空）
- `--depth_obs_wrist`: 手腕深度观测键（默认：空）
- `--state_obs_keys`: 状态观测键（默认：`EEF_state,None,gripper_state`）
- `--state_encoding`: 状态编码（默认：`POS_EULER`）
- `--action_encoding`: 动作编码（默认：`EEF_POS`）

#### GPU配置

- `--num_gpus`: 使用的GPU数量（默认：`1`）

#### 使用示例

**示例1：基本使用**
```bash
./vla-scripts/finetune_univla.sh \
    --dataset_name "my_robot_dataset" \
    --vla_path "/path/to/univla/model" \
    --lam_path "/path/to/lam/checkpoint" \
    --data_root_dir "/path/to/datasets" \
    --univla_root_dir "/path/to/univla/repo"
```

**示例2：自定义配置**
```bash
./vla-scripts/finetune_univla.sh \
    --dataset_name "custom_dataset" \
    --vla_path "/path/to/univla/model" \
    --lam_path "/path/to/lam/checkpoint" \
    --data_root_dir "/path/to/datasets" \
    --univla_root_dir "/path/to/univla/repo" \
    --image_obs_primary "front_camera" \
    --image_obs_wrist "gripper_camera" \
    --state_obs_keys "joint_positions,None,gripper_state" \
    --batch_size 4 \
    --learning_rate 1e-4 \
    --max_steps 50000 \
    --window_size 16
```

**示例3：使用量化**
```bash
./vla-scripts/finetune_univla.sh \
    --dataset_name "quantized_dataset" \
    --vla_path "/path/to/univla/model" \
    --lam_path "/path/to/lam/checkpoint" \
    --data_root_dir "/path/to/datasets" \
    --univla_root_dir "/path/to/univla/repo" \
    --use_quantization true \
    --batch_size 16 \
    --max_steps 25000
```

**示例4：冻结VLA骨干网络**
```bash
./vla-scripts/finetune_univla.sh \
    --dataset_name "frozen_vla_dataset" \
    --vla_path "/path/to/univla/model" \
    --lam_path "/path/to/lam/checkpoint" \
    --data_root_dir "/path/to/datasets" \
    --univla_root_dir "/path/to/univla/repo" \
    --freeze_vla true \
    --learning_rate 1e-3 \
    --batch_size 12
```

**示例5：多GPU训练**
```bash
./vla-scripts/finetune_univla.sh \
    --dataset_name "multi_gpu_dataset" \
    --vla_path "/path/to/univla/model" \
    --lam_path "/path/to/lam/checkpoint" \
    --data_root_dir "/path/to/datasets" \
    --univla_root_dir "/path/to/univla/repo" \
    --num_gpus 4 \
    --batch_size 8 \
    --grad_accumulation_steps 1
```

#### 脚本功能

1. **参数验证**：检查必需参数是否提供
2. **添加数据集配置**：自动将你的数据集配置添加到：
   - `{univla_root_dir}/prismatic/vla/datasets/rlds/oxe/configs.py`
   - `{univla_root_dir}/prismatic/vla/datasets/rlds/oxe/transforms.py`
3. **运行UniVLA微调**：使用你的参数执行UniVLA微调脚本
4. **多GPU支持**：支持多GPU分布式训练
5. **LAM集成**：自动配置和加载潜在动作模型

#### 注意事项

- UniVLA使用带有潜在动作模型（LAM）的两阶段训练方法
- LAM检查点是必需的，应该预先训练
- 脚本使用 `libero_dataset_transform` 作为新数据集的默认变换函数
- 如果数据集配置已存在，将跳过添加配置步骤
- 脚本自动处理状态观测键中的 `None` 值
- 确保你的数据集采用正确的RLDS格式并位于指定的数据目录中
- UniVLA支持冻结和未冻结的VLA骨干网络训练

## 微调OpenPi

### 安装OpenPi库

```bash
# 克隆仓库（包含子模块）
git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git

# 或者如果已经克隆了仓库：
cd openpi
git submodule update --init --recursive

# 安装 uv（如果尚未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 安装 OpenPi
cd openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

**注意：** `GIT_LFS_SKIP_SMUDGE=1` 是必需的，用于跳过 LeRobot 依赖的 LFS 文件下载。

### 使用脚本一键微调

将 [finetune_openpi.sh](./finetune_openpi.sh) 粘贴至 openpi/scripts 目录下，该脚本会自动添加训练配置并运行微调。

#### 基本使用方法

```bash
# 基本使用（需要提供必需参数）
uv run bash scripts/finetune_openpi.sh \
    --config_name "my_openpi_config" \
    --exp_name "my_experiment" \
    --base_checkpoint_path "/path/to/base/checkpoint" \
    --dataset_repo_id "your_dataset_repo" \
    --hf_lerobot_home "/path/to/lerobot/home"

# 自定义参数
uv run bash scripts/finetune_openpi.sh \
    --config_name "custom_config" \
    --exp_name "custom_experiment" \
    --base_checkpoint_path "/path/to/base/checkpoint" \
    --dataset_repo_id "your_dataset_repo" \
    --hf_lerobot_home "/path/to/lerobot/home" \
    --model_type "pi0_fast" \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --num_train_steps 50000
```

#### 必需参数

- `--config_name`: 配置名称（必需）
- `--exp_name`: 实验名称（必需）
- `--base_checkpoint_path`: 基础模型检查点路径（必需）
- `--dataset_repo_id`: 数据集仓库ID（必需）
- `--hf_lerobot_home`: HF_LEROBOT_HOME 目录路径（必需）

#### 模型配置参数

- `--model_type`: 模型类型，pi0 或 pi0_fast（默认：pi0）
- `--action_dim`: 动作维度（默认：7）
- `--action_horizon`: 动作时间范围（默认：10）
- `--max_token_len`: 最大token长度（默认：180）
- `--use_lora`: 使用LoRA微调（默认：false）
- `--lora_rank`: LoRA秩（默认：32）
- `--lora_dropout`: LoRA dropout（默认：0.0）
- `--paligemma_variant`: Paligemma变体（默认：gemma_2b）
- `--action_expert_variant`: 动作专家变体（默认：gemma_300m）

#### 训练参数

- `--batch_size`: 批次大小（默认：56）
- `--learning_rate`: 学习率（默认：3.5e-4）
- `--num_train_steps`: 训练步数（默认：30000）
- `--log_interval`: 日志间隔（默认：100）
- `--save_interval`: 保存间隔（默认：1000）
- `--keep_period`: 保留周期（默认：5000）
- `--num_workers`: 工作进程数（默认：2）
- `--seed`: 随机种子（默认：42）
- `--fsdp_devices`: FSDP设备数（默认：1）
- `--ema_decay`: EMA衰减（默认：0.99）

#### 数据集配置参数

- `--prompt_from_task`: 从任务中获取提示（默认：true）

#### 使用示例

**示例1：基本使用**
```bash
uv run bash scripts/finetune_openpi.sh \
    --config_name "libero_pi0" \
    --exp_name "libero_experiment" \
    --base_checkpoint_path "/path/to/pi0/checkpoint" \
    --dataset_repo_id "libero_dataset" \
    --hf_lerobot_home "/path/to/lerobot/home"
```

**示例2：使用 pi0_fast 模型**
```bash
uv run bash scripts/finetune_openpi.sh \
    --config_name "libero_pi0_fast" \
    --exp_name "libero_fast_experiment" \
    --base_checkpoint_path "/path/to/pi0_fast/checkpoint" \
    --dataset_repo_id "libero_dataset" \
    --hf_lerobot_home "/path/to/lerobot/home" \
    --model_type "pi0_fast" \
    --batch_size 32 \
    --learning_rate 1e-4
```

**示例3：使用LoRA微调**
```bash
uv run bash scripts/finetune_openpi.sh \
    --config_name "libero_pi0_lora" \
    --exp_name "libero_lora_experiment" \
    --base_checkpoint_path "/path/to/pi0/checkpoint" \
    --dataset_repo_id "libero_dataset" \
    --hf_lerobot_home "/path/to/lerobot/home" \
    --use_lora true \
    --lora_rank 64 \
    --lora_dropout 0.1
```

**示例4：自定义训练参数**
```bash
uv run bash scripts/finetune_openpi.sh \
    --config_name "custom_libero" \
    --exp_name "custom_experiment" \
    --base_checkpoint_path "/path/to/checkpoint" \
    --dataset_repo_id "libero_dataset" \
    --hf_lerobot_home "/path/to/lerobot/home" \
    --batch_size 64 \
    --learning_rate 2e-4 \
    --num_train_steps 100000 \
    --save_interval 2000 \
    --wandb_enabled true \
    --project_name "my_openpi_project"
```

#### 脚本功能

1. **参数验证**：检查必需参数是否提供
2. **添加训练配置**：自动将你的训练配置添加到 `src/openpi/training/config.py`
3. **计算归一化统计**：自动运行 `scripts/compute_norm_stats.py`
4. **运行训练**：使用你的参数执行OpenPi训练脚本
5. **支持覆盖**：可选择覆盖现有检查点

#### 注意事项

- 脚本使用 `LeRobotLiberoDataConfig` 作为数据集配置
- 如果配置已存在，将跳过添加配置步骤
- 支持 pi0 和 pi0_fast 两种模型类型
- LoRA微调时会自动设置相应的冻结过滤器
- 确保基础检查点路径有效且可访问
- 确保数据集仓库ID正确且可访问
- 脚本会自动设置 `HF_LEROBOT_HOME` 环境变量

