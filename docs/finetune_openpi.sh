#!/bin/bash

# finetune_openpi.sh
# Script to create and run OpenPI fine-tuning configurations with libero dataset type

# Default values
CONFIG_NAME=""
EXP_NAME=""
MODEL_TYPE=""  # pi0 or pi0_fast
BASE_CHECKPOINT_PATH=""
DATASET_REPO_ID=""
HF_LEROBOT_HOME=""
ACTION_DIM=7
ACTION_HORIZON=10
MAX_TOKEN_LEN=180
USE_LORA=False
LORA_RANK=32
LORA_DROPOUT=0.0
PALIGEMMA_VARIANT="gemma_2b"
ACTION_EXPERT_VARIANT="gemma_300m"
BATCH_SIZE=56
LEARNING_RATE=3.5e-4
NUM_TRAIN_STEPS=10
LOG_INTERVAL=100
SAVE_INTERVAL=1000
KEEP_PERIOD=5000
NUM_WORKERS=2
SEED=42
WANDB_ENABLED=True
PROJECT_NAME="openpi"
OVERWRITE=False
RESUME=False
FSDP_DEVICES=1
EMA_DECAY=0.99
ASSETS_BASE_DIR="./assets"
CHECKPOINT_BASE_DIR="./checkpoints"

# Dataset specific parameters
PROMPT_FROM_TASK=True

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config_name)
            CONFIG_NAME="$2"
            shift 2
            ;;
        --exp_name)
            EXP_NAME="$2"
            shift 2
            ;;
        --model_type)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --base_checkpoint_path)
            BASE_CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --dataset_repo_id)
            DATASET_REPO_ID="$2"
            shift 2
            ;;
        --hf_lerobot_home)
            HF_LEROBOT_HOME="$2"
            shift 2
            ;;
        --action_dim)
            ACTION_DIM="$2"
            shift 2
            ;;
        --action_horizon)
            ACTION_HORIZON="$2"
            shift 2
            ;;
        --max_token_len)
            MAX_TOKEN_LEN="$2"
            shift 2
            ;;
        --use_lora)
            USE_LORA="$2"
            shift 2
            ;;
        --lora_rank)
            LORA_RANK="$2"
            shift 2
            ;;
        --lora_dropout)
            LORA_DROPOUT="$2"
            shift 2
            ;;
        --paligemma_variant)
            PALIGEMMA_VARIANT="$2"
            shift 2
            ;;
        --action_expert_variant)
            ACTION_EXPERT_VARIANT="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --num_train_steps)
            NUM_TRAIN_STEPS="$2"
            shift 2
            ;;
        --log_interval)
            LOG_INTERVAL="$2"
            shift 2
            ;;
        --save_interval)
            SAVE_INTERVAL="$2"
            shift 2
            ;;
        --keep_period)
            KEEP_PERIOD="$2"
            shift 2
            ;;
        --num_workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --wandb_enabled)
            WANDB_ENABLED="$2"
            shift 2
            ;;
        --project_name)
            PROJECT_NAME="$2"
            shift 2
            ;;
        --overwrite)
            OVERWRITE="$2"
            shift 2
            ;;
        --resume)
            RESUME="$2"
            shift 2
            ;;
        --fsdp_devices)
            FSDP_DEVICES="$2"
            shift 2
            ;;
        --ema_decay)
            EMA_DECAY="$2"
            shift 2
            ;;
        --assets_base_dir)
            ASSETS_BASE_DIR="$2"
            shift 2
            ;;
        --checkpoint_base_dir)
            CHECKPOINT_BASE_DIR="$2"
            shift 2
            ;;
        --prompt_from_task)
            PROMPT_FROM_TASK="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 --config_name <name> --exp_name <name> [options]"
            echo ""
            echo "Required arguments:"
            echo "  --config_name          Configuration name (required)"
            echo "  --exp_name             Experiment name (required)"
            echo "  --base_checkpoint_path Path to base model checkpoint (required)"
            echo "  --dataset_repo_id      Dataset repository ID (required)"
            echo "  --hf_lerobot_home      HF_LEROBOT_HOME directory path (required)"
            echo ""
            echo "Model configuration:"
            echo "  --model_type           Model type: pi0 or pi0_fast (default: pi0)"
            echo "  --action_dim           Action dimension (default: 7)"
            echo "  --action_horizon       Action horizon (default: 10)"
            echo "  --max_token_len        Maximum token length (default: 180)"
            echo "  --use_lora             Use LoRA fine-tuning (default: False)"
            echo "  --lora_rank            LoRA rank (default: 32)"
            echo "  --lora_dropout         LoRA dropout (default: 0.0)"
            echo "  --paligemma_variant    Paligemma variant (default: gemma_2b)"
            echo "  --action_expert_variant Action expert variant (default: gemma_300m)"
            echo ""
            echo "Dataset configuration:"
            echo "  --prompt_from_task     Use prompt from task (default: True)"
            echo ""
            echo "Training parameters:"
            echo "  --batch_size           Batch size (default: 56)"
            echo "  --learning_rate        Learning rate (default: 3.5e-4)"
            echo "  --num_train_steps      Number of training steps (default: 30000)"
            echo "  --log_interval         Log interval (default: 100)"
            echo "  --save_interval        Save interval (default: 1000)"
            echo "  --keep_period          Keep period (default: 5000)"
            echo "  --num_workers          Number of workers (default: 2)"
            echo "  --seed                 Random seed (default: 42)"
            echo "  --fsdp_devices         FSDP devices (default: 1)"
            echo "  --ema_decay            EMA decay (default: 0.99)"
            echo ""
            echo "Directories:"
            echo "  --assets_base_dir      Assets base directory (default: ./assets)"
            echo "  --checkpoint_base_dir  Checkpoint base directory (default: ./checkpoints)"
            echo ""
            echo "Other options:"
            echo "  --wandb_enabled        Enable wandb logging (default: True)"
            echo "  --project_name         Project name (default: openpi)"
            echo "  --overwrite            Overwrite existing checkpoints (default: False)"
            echo "  --resume               Resume training (default: False)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if required parameters are provided
if [ -z "$CONFIG_NAME" ]; then
    echo "Error: --config_name is required"
    echo "Use --help for usage information"
    exit 1
fi

if [ -z "$EXP_NAME" ]; then
    echo "Error: --exp_name is required"
    echo "Use --help for usage information"
    exit 1
fi

if [ -z "$BASE_CHECKPOINT_PATH" ]; then
    echo "Error: --base_checkpoint_path is required"
    echo "Use --help for usage information"
    exit 1
fi

if [ -z "$DATASET_REPO_ID" ]; then
    echo "Error: --dataset_repo_id is required"
    echo "Use --help for usage information"
    exit 1
fi

if [ -z "$HF_LEROBOT_HOME" ]; then
    echo "Error: --hf_lerobot_home is required"
    echo "Use --help for usage information"
    exit 1
fi

echo "Creating OpenPI fine-tuning configuration: $CONFIG_NAME"
echo "Experiment name: $EXP_NAME"
echo "Model type: $MODEL_TYPE"
echo "Dataset: $DATASET_REPO_ID (libero)"
echo "Base checkpoint: $BASE_CHECKPOINT_PATH"
echo "HF_LEROBOT_HOME: $HF_LEROBOT_HOME"
echo "Use LoRA: $USE_LORA"

# Set HF_LEROBOT_HOME environment variable
export HF_LEROBOT_HOME="$HF_LEROBOT_HOME"

# Create Python script to add training configuration
cat > /tmp/add_training_config.py << EOF
import sys
import re
import os

def add_training_config():
    # Path to the config file
    config_path = "src/openpi/training/config.py"
    
    # Read the config file
    with open(config_path, 'r') as f:
        config_content = f.read()
    
    # Check if config already exists
    if f'name="$CONFIG_NAME"' in config_content:
        print(f"Configuration $CONFIG_NAME already exists in config.py")
        return
    
    # Build the model configuration
    if "$MODEL_TYPE" == "pi0_fast":
        if "$USE_LORA" == "True":
            model_config = f'''pi0_fast.Pi0FASTConfig(
            action_dim=$ACTION_DIM, 
            action_horizon=$ACTION_HORIZON, 
            max_token_len=$MAX_TOKEN_LEN, 
            paligemma_variant="$PALIGEMMA_VARIANT"
        )'''
        else:
            model_config = f'''pi0_fast.Pi0FASTConfig(
            action_dim=$ACTION_DIM, 
            action_horizon=$ACTION_HORIZON, 
            max_token_len=$MAX_TOKEN_LEN
        )'''
    else:  # pi0
        if "$USE_LORA" == "True":
            model_config = f'''pi0.Pi0Config(
            paligemma_variant="$PALIGEMMA_VARIANT", 
            action_expert_variant="$ACTION_EXPERT_VARIANT", 
            action_dim=$ACTION_DIM
        )'''
        else:
            model_config = f'''pi0.Pi0Config(
            action_dim=$ACTION_DIM
        )'''
    
    # Build the data configuration (always libero)
    data_config = f'''LeRobotLiberoDataConfig(
            repo_id="$DATASET_REPO_ID",
            base_config=DataConfig(prompt_from_task=$PROMPT_FROM_TASK),
        )'''
    
    # Build the freeze filter for LoRA
    freeze_filter_config = ""
    if "$USE_LORA" == "True":
        if "$MODEL_TYPE" == "pi0_fast":
            freeze_filter_config = f'''freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=$ACTION_DIM, 
            action_horizon=$ACTION_HORIZON, 
            max_token_len=$MAX_TOKEN_LEN, 
            paligemma_variant="$PALIGEMMA_VARIANT"
        ).get_freeze_filter(),'''
        else:  # pi0
            freeze_filter_config = f'''freeze_filter=pi0.Pi0Config(
            paligemma_variant="$PALIGEMMA_VARIANT", 
            action_expert_variant="$ACTION_EXPERT_VARIANT",
            action_dim=$ACTION_DIM
        ).get_freeze_filter(),'''
    
    # Build EMA decay configuration
    ema_decay_config = ""
    if "$USE_LORA" == "True":
        ema_decay_config = "ema_decay=None,"
    elif "$EMA_DECAY" != "0.99":
        ema_decay_config = f"ema_decay=$EMA_DECAY,"

    
    # Build the complete training configuration
    config_entry = f'''    TrainConfig(
        name="$CONFIG_NAME",
        model={model_config},
        data={data_config},
        weight_loader=weight_loaders.CheckpointWeightLoader("$BASE_CHECKPOINT_PATH"),
        {freeze_filter_config}
        {ema_decay_config}
        wandb_enabled=$WANDB_ENABLED,
        batch_size=$BATCH_SIZE,
        num_train_steps=$NUM_TRAIN_STEPS,
        log_interval=$LOG_INTERVAL,
        save_interval=$SAVE_INTERVAL,
        keep_period=$KEEP_PERIOD,
        num_workers=$NUM_WORKERS,
        seed=$SEED,
        project_name="$PROJECT_NAME",
        overwrite=$OVERWRITE,
        resume=$RESUME,
        fsdp_devices=$FSDP_DEVICES,
        assets_base_dir="$ASSETS_BASE_DIR",
        checkpoint_base_dir="$CHECKPOINT_BASE_DIR",
    ),'''
    
    # Find the end of _CONFIGS list and add before closing bracket
    pattern = r'(,)\n(\])'
    
    # Insert before the closing bracket
    replacement = f',\n{config_entry}\n]'
    config_content = re.sub(pattern, replacement, config_content, flags=re.MULTILINE)
    
    # Write back to config.py
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"Added training configuration $CONFIG_NAME to config.py")

if __name__ == "__main__":
    add_training_config()
EOF

# Run the Python script to add training configuration
uv run /tmp/add_training_config.py

# Clean up temporary file
rm /tmp/add_training_config.py

echo "Starting OpenPI fine-tuning..."

# Run the training script
cd "$(dirname "$0")/.."

# First, compute normalization statistics
echo "Computing normalization statistics..."
uv run scripts/compute_norm_stats.py --config-name $CONFIG_NAME

# Build the training command with XLA memory fraction
CMD="XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py $CONFIG_NAME --exp-name=\"$EXP_NAME\""

# Add overwrite flag if specified
if [ "$OVERWRITE" = "True" ]; then
    CMD="$CMD --overwrite"
fi

# Execute the command
eval $CMD

echo "OpenPI fine-tuning completed!"
