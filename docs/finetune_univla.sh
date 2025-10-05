#!/bin/bash

# finetune_univla.sh
# Script to add dataset configurations and run UniVLA fine-tuning with extended parameters

# Default values
DATASET_NAME=""
VLA_PATH=""
LAM_PATH=""
DATA_ROOT_DIR=""
RUN_ROOT_DIR="all_runs"
UNIVLA_ROOT_DIR=""
BATCH_SIZE=1
LEARNING_RATE=3.5e-4
MAX_STEPS=100000
SAVE_STEPS=10000
USE_LORA=true
LORA_RANK=32
LORA_DROPOUT=0.0
USE_QUANTIZATION=false
IMAGE_AUG=true
WANDB_PROJECT="finetune-UniVLA"
WANDB_ENTITY="opendrivelab"
NUM_GPUS=1

# UniVLA specific parameters
FREEZE_VLA=false
GRAD_ACCUMULATION_STEPS=2
SHUFFLE_BUFFER_SIZE=16000
SAVE_LATEST_CHECKPOINT_ONLY=true
RUN_ID_NOTE=""

# LAM (Latent Action Model) parameters
CODEBOOK_SIZE=16
LAM_MODEL_DIM=768
LAM_LATENT_DIM=128
LAM_PATCH_SIZE=14
LAM_ENC_BLOCKS=12
LAM_DEC_BLOCKS=12
LAM_NUM_HEADS=12
WINDOW_SIZE=12

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
        --lam_path)
            LAM_PATH="$2"
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
        --univla_root_dir)
            UNIVLA_ROOT_DIR="$2"
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
        --save_steps)
            SAVE_STEPS="$2"
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
        --freeze_vla)
            FREEZE_VLA="$2"
            shift 2
            ;;
        --grad_accumulation_steps)
            GRAD_ACCUMULATION_STEPS="$2"
            shift 2
            ;;
        --shuffle_buffer_size)
            SHUFFLE_BUFFER_SIZE="$2"
            shift 2
            ;;
        --save_latest_checkpoint_only)
            SAVE_LATEST_CHECKPOINT_ONLY="$2"
            shift 2
            ;;
        --run_id_note)
            RUN_ID_NOTE="$2"
            shift 2
            ;;
        --codebook_size)
            CODEBOOK_SIZE="$2"
            shift 2
            ;;
        --lam_model_dim)
            LAM_MODEL_DIM="$2"
            shift 2
            ;;
        --lam_latent_dim)
            LAM_LATENT_DIM="$2"
            shift 2
            ;;
        --lam_patch_size)
            LAM_PATCH_SIZE="$2"
            shift 2
            ;;
        --lam_enc_blocks)
            LAM_ENC_BLOCKS="$2"
            shift 2
            ;;
        --lam_dec_blocks)
            LAM_DEC_BLOCKS="$2"
            shift 2
            ;;
        --lam_num_heads)
            LAM_NUM_HEADS="$2"
            shift 2
            ;;
        --window_size)
            WINDOW_SIZE="$2"
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
            echo "  --vla_path              Path to UniVLA model (required)"
            echo "  --lam_path              Path to LAM model checkpoint (required)"
            echo "  --data_root_dir         Root directory for datasets (required)"
            echo "  --univla_root_dir       Root directory of UniVLA repository (required)"
            echo ""
            echo "Basic training parameters:"
            echo "  --run_root_dir          Root directory for runs (default: all_runs)"
            echo "  --batch_size            Batch size (default: 8)"
            echo "  --learning_rate         Learning rate (default: 3.5e-4)"
            echo "  --max_steps             Maximum training steps (default: 100000)"
            echo "  --save_steps            Save interval (default: 10000)"
            echo "  --grad_accumulation_steps Gradient accumulation steps (default: 2)"
            echo "  --shuffle_buffer_size   Dataloader shuffle buffer size (default: 16000)"
            echo ""
            echo "LoRA parameters:"
            echo "  --use_lora              Use LoRA fine-tuning (default: true)"
            echo "  --lora_rank             LoRA rank (default: 32)"
            echo "  --lora_dropout          LoRA dropout (default: 0.0)"
            echo "  --use_quantization      Use quantization (default: false)"
            echo ""
            echo "UniVLA specific parameters:"
            echo "  --freeze_vla            Freeze VLA backbone (default: false)"
            echo "  --save_latest_checkpoint_only Save only latest checkpoint (default: true)"
            echo "  --run_id_note           Extra note for experiment ID (default: empty)"
            echo ""
            echo "LAM (Latent Action Model) parameters:"
            echo "  --codebook_size         LAM codebook size (default: 16)"
            echo "  --lam_model_dim         LAM model dimension (default: 768)"
            echo "  --lam_latent_dim        LAM latent dimension (default: 128)"
            echo "  --lam_patch_size        LAM patch size (default: 14)"
            echo "  --lam_enc_blocks        LAM encoder blocks (default: 12)"
            echo "  --lam_dec_blocks        LAM decoder blocks (default: 12)"
            echo "  --lam_num_heads         LAM number of heads (default: 12)"
            echo "  --window_size           Action window size (default: 12)"
            echo ""
            echo "Logging:"
            echo "  --wandb_project         WandB project name (default: finetune-UniVLA)"
            echo "  --wandb_entity          WandB entity name (default: opendrivelab)"
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

if [ -z "$LAM_PATH" ]; then
    echo "Error: --lam_path is required"
    echo "Use --help for usage information"
    exit 1
fi

if [ -z "$DATA_ROOT_DIR" ]; then
    echo "Error: --data_root_dir is required"
    echo "Use --help for usage information"
    exit 1
fi

if [ -z "$UNIVLA_ROOT_DIR" ]; then
    echo "Error: --univla_root_dir is required"
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
echo "UniVLA Training configuration:"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo "  Max steps: $MAX_STEPS"
echo "  Save steps: $SAVE_STEPS"
echo "  Freeze VLA: $FREEZE_VLA"
echo "  Window size: $WINDOW_SIZE"
echo "  LoRA rank: $LORA_RANK"
echo "  LoRA dropout: $LORA_DROPOUT"

echo ""
echo "LAM configuration:"
echo "  Codebook size: $CODEBOOK_SIZE"
echo "  Model dim: $LAM_MODEL_DIM"
echo "  Latent dim: $LAM_LATENT_DIM"
echo "  Patch size: $LAM_PATCH_SIZE"
echo "  Encoder blocks: $LAM_ENC_BLOCKS"
echo "  Decoder blocks: $LAM_DEC_BLOCKS"
echo "  Number of heads: $LAM_NUM_HEADS"

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
    configs_path = "$UNIVLA_ROOT_DIR/prismatic/vla/datasets/rlds/oxe/configs.py"
    transforms_path = "$UNIVLA_ROOT_DIR/prismatic/vla/datasets/rlds/oxe/transforms.py"
    
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

echo "Starting UniVLA fine-tuning..."

# Run the fine-tuning script
cd "$UNIVLA_ROOT_DIR"

# Build the command with all parameters
CMD="torchrun --standalone --nnodes 1 --nproc-per-node $NUM_GPUS vla-scripts/finetune_libero.py"
CMD="$CMD --vla_path \"$VLA_PATH\""
CMD="$CMD --lam_path \"$LAM_PATH\""
CMD="$CMD --data_root_dir \"$DATA_ROOT_DIR\""
CMD="$CMD --dataset_name \"$DATASET_NAME\""
CMD="$CMD --run_root_dir \"$RUN_ROOT_DIR\""
CMD="$CMD --batch_size $BATCH_SIZE"
CMD="$CMD --learning_rate $LEARNING_RATE"
CMD="$CMD --max_steps $MAX_STEPS"
CMD="$CMD --save_steps $SAVE_STEPS"
CMD="$CMD --use_lora $USE_LORA"
CMD="$CMD --lora_rank $LORA_RANK"
CMD="$CMD --lora_dropout $LORA_DROPOUT"
CMD="$CMD --use_quantization $USE_QUANTIZATION"
CMD="$CMD --image_aug $IMAGE_AUG"
CMD="$CMD --wandb_project \"$WANDB_PROJECT\""
CMD="$CMD --wandb_entity \"$WANDB_ENTITY\""
CMD="$CMD --freeze_vla $FREEZE_VLA"
CMD="$CMD --grad_accumulation_steps $GRAD_ACCUMULATION_STEPS"
CMD="$CMD --shuffle_buffer_size $SHUFFLE_BUFFER_SIZE"
CMD="$CMD --save_latest_checkpoint_only $SAVE_LATEST_CHECKPOINT_ONLY"
CMD="$CMD --codebook_size $CODEBOOK_SIZE"
CMD="$CMD --lam_model_dim $LAM_MODEL_DIM"
CMD="$CMD --lam_latent_dim $LAM_LATENT_DIM"
CMD="$CMD --lam_patch_size $LAM_PATCH_SIZE"
CMD="$CMD --lam_enc_blocks $LAM_ENC_BLOCKS"
CMD="$CMD --lam_dec_blocks $LAM_DEC_BLOCKS"
CMD="$CMD --lam_num_heads $LAM_NUM_HEADS"
CMD="$CMD --window_size $WINDOW_SIZE"

# Add run_id_note if provided
if [ -n "$RUN_ID_NOTE" ]; then
    CMD="$CMD --run_id_note \"$RUN_ID_NOTE\""
fi

# Execute the command
eval $CMD

echo "UniVLA fine-tuning completed!"
