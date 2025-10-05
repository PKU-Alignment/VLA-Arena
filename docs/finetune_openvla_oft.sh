#!/bin/bash

# finetune_openvla_oft.sh
# Script to add dataset configurations and run OpenVLA OFT fine-tuning with extended parameters

# Default values
DATASET_NAME=""
VLA_PATH=""
DATA_ROOT_DIR=""
RUN_ROOT_DIR=""
OPENVLA_ROOT_DIR=""
BATCH_SIZE=8
LEARNING_RATE=5e-4
MAX_STEPS=200000
USE_LORA=true
LORA_RANK=32
LORA_DROPOUT=0.0
USE_QUANTIZATION=false
IMAGE_AUG=true
WANDB_PROJECT=""
WANDB_ENTITY=""
NUM_GPUS=1

# OFT specific parameters
USE_L1_REGRESSION=true
USE_DIFFUSION=false
NUM_DIFFUSION_STEPS_TRAIN=50
USE_FILM=true
NUM_IMAGES_IN_INPUT=2
USE_PROPRIO=false
LR_WARMUP_STEPS=0
NUM_STEPS_BEFORE_DECAY=60000
GRAD_ACCUMULATION_STEPS=1
USE_VAL_SET=false
VAL_FREQ=10000
VAL_TIME_LIMIT=180
SAVE_FREQ=5000
SAVE_LATEST_CHECKPOINT_ONLY=false
RESUME=false
RESUME_STEP=""
DIFFUSION_SAMPLE_FREQ=50
MERGE_LORA_DURING_TRAINING=true
WANDB_LOG_FREQ=10
SHUFFLE_BUFFER_SIZE=100000

# Dataset configuration parameters
IMAGE_OBS_PRIMARY="image"
IMAGE_OBS_SECONDARY=""
IMAGE_OBS_WRIST="wrist_image"
DEPTH_OBS_PRIMARY=""
DEPTH_OBS_SECONDARY=""
DEPTH_OBS_WRIST=""
STATE_OBS_KEYS="EEF_state,None,gripper_state"
STATE_ENCODING="POS_EULER"
ACTION_ENCODING="EEF_POS"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset_name)
            DATASET_NAME="$2"
            shift 2
            ;;
        --vla_path)
            VLA_PATH="$2"
            shift 2
            ;;
        --data_root_dir)
            DATA_ROOT_DIR="$2"
            shift 2
            ;;
        --run_root_dir)
            RUN_ROOT_DIR="$2"
            shift 2
            ;;
        --openvla_root_dir)
            OPENVLA_ROOT_DIR="$2"
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
        --max_steps)
            MAX_STEPS="$2"
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
        --use_quantization)
            USE_QUANTIZATION="$2"
            shift 2
            ;;
        --image_aug)
            IMAGE_AUG="$2"
            shift 2
            ;;
        --wandb_project)
            WANDB_PROJECT="$2"
            shift 2
            ;;
        --wandb_entity)
            WANDB_ENTITY="$2"
            shift 2
            ;;
        --use_l1_regression)
            USE_L1_REGRESSION="$2"
            shift 2
            ;;
        --use_diffusion)
            USE_DIFFUSION="$2"
            shift 2
            ;;
        --num_diffusion_steps_train)
            NUM_DIFFUSION_STEPS_TRAIN="$2"
            shift 2
            ;;
        --use_film)
            USE_FILM="$2"
            shift 2
            ;;
        --num_images_in_input)
            NUM_IMAGES_IN_INPUT="$2"
            shift 2
            ;;
        --use_proprio)
            USE_PROPRIO="$2"
            shift 2
            ;;
        --lr_warmup_steps)
            LR_WARMUP_STEPS="$2"
            shift 2
            ;;
        --num_steps_before_decay)
            NUM_STEPS_BEFORE_DECAY="$2"
            shift 2
            ;;
        --grad_accumulation_steps)
            GRAD_ACCUMULATION_STEPS="$2"
            shift 2
            ;;
        --use_val_set)
            USE_VAL_SET="$2"
            shift 2
            ;;
        --val_freq)
            VAL_FREQ="$2"
            shift 2
            ;;
        --val_time_limit)
            VAL_TIME_LIMIT="$2"
            shift 2
            ;;
        --save_freq)
            SAVE_FREQ="$2"
            shift 2
            ;;
        --save_latest_checkpoint_only)
            SAVE_LATEST_CHECKPOINT_ONLY="$2"
            shift 2
            ;;
        --resume)
            RESUME="$2"
            shift 2
            ;;
        --resume_step)
            RESUME_STEP="$2"
            shift 2
            ;;
        --diffusion_sample_freq)
            DIFFUSION_SAMPLE_FREQ="$2"
            shift 2
            ;;
        --merge_lora_during_training)
            MERGE_LORA_DURING_TRAINING="$2"
            shift 2
            ;;
        --wandb_log_freq)
            WANDB_LOG_FREQ="$2"
            shift 2
            ;;
        --shuffle_buffer_size)
            SHUFFLE_BUFFER_SIZE="$2"
            shift 2
            ;;
        --image_obs_primary)
            IMAGE_OBS_PRIMARY="$2"
            shift 2
            ;;
        --image_obs_secondary)
            IMAGE_OBS_SECONDARY="$2"
            shift 2
            ;;
        --image_obs_wrist)
            IMAGE_OBS_WRIST="$2"
            shift 2
            ;;
        --depth_obs_primary)
            DEPTH_OBS_PRIMARY="$2"
            shift 2
            ;;
        --depth_obs_secondary)
            DEPTH_OBS_SECONDARY="$2"
            shift 2
            ;;
        --depth_obs_wrist)
            DEPTH_OBS_WRIST="$2"
            shift 2
            ;;
        --state_obs_keys)
            STATE_OBS_KEYS="$2"
            shift 2
            ;;
        --state_encoding)
            STATE_ENCODING="$2"
            shift 2
            ;;
        --action_encoding)
            ACTION_ENCODING="$2"
            shift 2
            ;;
        --num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 --dataset_name <name> [options]"
            echo ""
            echo "Required arguments:"
            echo "  --dataset_name          Dataset name (required)"
            echo "  --vla_path              Path to OpenVLA model (required)"
            echo "  --data_root_dir         Root directory for datasets (required)"
            echo "  --openvla_root_dir      Root directory of OpenVLA repository (required)"
            echo ""
            echo "Basic training parameters:"
            echo "  --run_root_dir          Root directory for runs (default: all_runs)"
            echo "  --batch_size            Batch size (default: 7)"
            echo "  --learning_rate         Learning rate (default: 5e-4)"
            echo "  --max_steps             Maximum training steps (default: 150000)"
            echo "  --grad_accumulation_steps Gradient accumulation steps (default: 1)"
            echo "  --shuffle_buffer_size   Dataloader shuffle buffer size (default: 100000)"
            echo ""
            echo "LoRA parameters:"
            echo "  --use_lora              Use LoRA fine-tuning (default: true)"
            echo "  --lora_rank             LoRA rank (default: 32)"
            echo "  --lora_dropout          LoRA dropout (default: 0.0)"
            echo "  --merge_lora_during_training Merge LoRA during training (default: true)"
            echo ""
            echo "Action representation:"
            echo "  --use_l1_regression     Use L1 regression (default: true)"
            echo "  --use_diffusion         Use diffusion modeling (default: false)"
            echo "  --num_diffusion_steps_train Diffusion steps for training (default: 50)"
            echo "  --diffusion_sample_freq Diffusion sampling frequency (default: 50)"
            echo ""
            echo "Architecture options:"
            echo "  --use_film              Use FiLM for language infusion (default: true)"
            echo "  --num_images_in_input   Number of images in input (default: 2)"
            echo "  --use_proprio           Include proprioceptive state (default: false)"
            echo "  --use_quantization      Use quantization (default: false)"
            echo "  --image_aug             Use image augmentation (default: true)"
            echo ""
            echo "Learning rate scheduling:"
            echo "  --lr_warmup_steps       LR warmup steps (default: 0)"
            echo "  --num_steps_before_decay Steps before LR decay (default: 60000)"
            echo ""
            echo "Validation and checkpointing:"
            echo "  --use_val_set           Use validation set (default: false)"
            echo "  --val_freq              Validation frequency (default: 10000)"
            echo "  --val_time_limit        Validation time limit (default: 180)"
            echo "  --save_freq             Save frequency (default: 5000)"
            echo "  --save_latest_checkpoint_only Save only latest checkpoint (default: false)"
            echo "  --resume                Resume from checkpoint (default: false)"
            echo "  --resume_step           Resume step number (default: empty)"
            echo ""
            echo "Logging:"
            echo "  --wandb_project         WandB project name (default: openvla-oft-workflow-generalization)"
            echo "  --wandb_entity          WandB entity name (default: trial)"
            echo "  --wandb_log_freq        WandB logging frequency (default: 10)"
            echo ""
            echo "Dataset configuration:"
            echo "  --image_obs_primary     Primary image observation key (default: image)"
            echo "  --image_obs_secondary   Secondary image observation key (default: empty)"
            echo "  --image_obs_wrist       Wrist image observation key (default: wrist_image)"
            echo "  --depth_obs_primary     Primary depth observation key (default: empty)"
            echo "  --depth_obs_secondary   Secondary depth observation key (default: empty)"
            echo "  --depth_obs_wrist       Wrist depth observation key (default: empty)"
            echo "  --state_obs_keys        State observation keys (default: EEF_state,None,gripper_state)"
            echo "  --state_encoding        State encoding (default: POS_EULER)"
            echo "  --action_encoding       Action encoding (default: EEF_POS)"
            echo ""
            echo "GPU configuration:"
            echo "  --num_gpus              Number of GPUs to use (default: 1)"
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
if [ -z "$DATASET_NAME" ]; then
    echo "Error: --dataset_name is required"
    echo "Use --help for usage information"
    exit 1
fi

if [ -z "$VLA_PATH" ]; then
    echo "Error: --vla_path is required"
    echo "Use --help for usage information"
    exit 1
fi

if [ -z "$DATA_ROOT_DIR" ]; then
    echo "Error: --data_root_dir is required"
    echo "Use --help for usage information"
    exit 1
fi

if [ -z "$OPENVLA_ROOT_DIR" ]; then
    echo "Error: --openvla_root_dir is required"
    echo "Use --help for usage information"
    exit 1
fi

echo "Adding dataset configuration for: $DATASET_NAME"
echo "Dataset configuration:"
echo "  Image obs: primary=$IMAGE_OBS_PRIMARY, secondary=$IMAGE_OBS_SECONDARY, wrist=$IMAGE_OBS_WRIST"
echo "  Depth obs: primary=$DEPTH_OBS_PRIMARY, secondary=$DEPTH_OBS_SECONDARY, wrist=$DEPTH_OBS_WRIST"
echo "  State obs keys: $STATE_OBS_KEYS"
echo "  State encoding: $STATE_ENCODING"
echo "  Action encoding: $ACTION_ENCODING"

echo ""
echo "OFT Training configuration:"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo "  Max steps: $MAX_STEPS"
echo "  Use L1 regression: $USE_L1_REGRESSION"
echo "  Use diffusion: $USE_DIFFUSION"
echo "  Use FiLM: $USE_FILM"
echo "  Use proprio: $USE_PROPRIO"
echo "  Number of images: $NUM_IMAGES_IN_INPUT"
echo "  LoRA rank: $LORA_RANK"
echo "  LoRA dropout: $LORA_DROPOUT"

# Convert empty strings to None for Python
if [ -z "$IMAGE_OBS_SECONDARY" ]; then
    IMAGE_OBS_SECONDARY="None"
fi
if [ -z "$IMAGE_OBS_WRIST" ]; then
    IMAGE_OBS_WRIST="None"
fi
if [ -z "$DEPTH_OBS_PRIMARY" ]; then
    DEPTH_OBS_PRIMARY="None"
fi
if [ -z "$DEPTH_OBS_SECONDARY" ]; then
    DEPTH_OBS_SECONDARY="None"
fi
if [ -z "$DEPTH_OBS_WRIST" ]; then
    DEPTH_OBS_WRIST="None"
fi

# Create Python script to add dataset configuration
cat > /tmp/add_dataset_config.py << EOF
import sys
import re

def add_dataset_config():
    # Paths to the files
    configs_path = "$OPENVLA_ROOT_DIR/prismatic/vla/datasets/rlds/oxe/configs.py"
    transforms_path = "$OPENVLA_ROOT_DIR/prismatic/vla/datasets/rlds/oxe/transforms.py"
    
    dataset_name = "$DATASET_NAME"
    
    # Process state_obs_keys to handle None values properly
    state_obs_keys = "$STATE_OBS_KEYS"
    state_obs_list = []
    for key in state_obs_keys.split(','):
        key = key.strip()
        if key == 'None':
            state_obs_list.append('None')
        else:
            state_obs_list.append(f'"{key}"')
    state_obs_str = ', '.join(state_obs_list)
    
    # Read configs.py
    with open(configs_path, 'r') as f:
        configs_content = f.read()
    
    # Check if dataset already exists
    if f'"{dataset_name}":' in configs_content:
        print(f"Dataset {dataset_name} already exists in configs.py")
    else:
        # Find the end of OXE_DATASET_CONFIGS dictionary and add before closing brace
        pattern = r'(\s+)(\})\s*$'
        
        config_entry = f'''
    "{dataset_name}": {{
        "image_obs_keys": {{"primary": "$IMAGE_OBS_PRIMARY", "secondary": "$IMAGE_OBS_SECONDARY", "wrist": "$IMAGE_OBS_WRIST"}},
        "depth_obs_keys": {{"primary": "$DEPTH_OBS_PRIMARY", "secondary": "$DEPTH_OBS_SECONDARY", "wrist": "$DEPTH_OBS_WRIST"}},
        "state_obs_keys": [{state_obs_str}],
        "state_encoding": StateEncoding.$STATE_ENCODING,
        "action_encoding": ActionEncoding.$ACTION_ENCODING,
    }},'''
        
        # Insert before the closing brace
        replacement = f'{config_entry}\n}}'
        configs_content = re.sub(pattern, replacement, configs_content, flags=re.MULTILINE)
        
        # Write back to configs.py
        with open(configs_path, 'w') as f:
            f.write(configs_content)
        print(f"Added dataset configuration for {dataset_name} to configs.py")
    
    # Read transforms.py
    with open(transforms_path, 'r') as f:
        transforms_content = f.read()
    
    # Check if dataset already exists in transforms
    if f'"{dataset_name}":' in transforms_content:
        print(f"Dataset {dataset_name} already exists in transforms.py")
    else:
        # Find the end of OXE_STANDARDIZATION_TRANSFORMS dictionary and add before closing brace
        pattern = r'(\s+)(\})\s*$'
        
        transform_entry = f'\n    "{dataset_name}": libero_dataset_transform,'
        
        # Insert before the closing brace
        replacement = f'{transform_entry}\n}}'
        transforms_content = re.sub(pattern, replacement, transforms_content, flags=re.MULTILINE)
        
        # Write back to transforms.py
        with open(transforms_path, 'w') as f:
            f.write(transforms_content)
        print(f"Added dataset transform for {dataset_name} to transforms.py")

if __name__ == "__main__":
    add_dataset_config()
EOF

# Run the Python script to add dataset configuration
python3 /tmp/add_dataset_config.py

# Clean up temporary file
rm /tmp/add_dataset_config.py

echo "Starting OFT fine-tuning..."

# Run the fine-tuning script
cd "$OPENVLA_ROOT_DIR"

# Build the command with all parameters
CMD="torchrun --standalone --nnodes 1 --nproc-per-node $NUM_GPUS vla-scripts/finetune.py"
CMD="$CMD --vla_path \"$VLA_PATH\""
CMD="$CMD --data_root_dir \"$DATA_ROOT_DIR\""
CMD="$CMD --dataset_name \"$DATASET_NAME\""
CMD="$CMD --run_root_dir \"$RUN_ROOT_DIR\""
CMD="$CMD --batch_size $BATCH_SIZE"
CMD="$CMD --learning_rate $LEARNING_RATE"
CMD="$CMD --max_steps $MAX_STEPS"
CMD="$CMD --use_lora $USE_LORA"
CMD="$CMD --lora_rank $LORA_RANK"
CMD="$CMD --lora_dropout $LORA_DROPOUT"
CMD="$CMD --use_quantization $USE_QUANTIZATION"
CMD="$CMD --image_aug $IMAGE_AUG"
CMD="$CMD --wandb_project \"$WANDB_PROJECT\""
CMD="$CMD --wandb_entity \"$WANDB_ENTITY\""

# Add resume_step if provided
if [ -n "$RESUME_STEP" ]; then
    CMD="$CMD --resume_step $RESUME_STEP"
fi

# Execute the command
eval $CMD

echo "OFT fine-tuning completed!"
