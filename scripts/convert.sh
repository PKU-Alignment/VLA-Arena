#!/bin/bash

# LeRobot Dataset Conversion Script
# Usage:
# 1. Modify the variables below
# 2. Run: ./run_conversion.sh
# Or: DATA_DIR=/path/to/data ./run_conversion.sh

set -e

# ============ Configuration Variables ============
DATA_DIR="${DATA_DIR:-"your_path_to_dataset"}"
HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-"./lerobot_dataset"}"
PUSH_TO_HUB="${PUSH_TO_HUB:-false}"
# ================================

# Override DATA_DIR with command line arguments
if [ $# -gt 0 ]; then
    DATA_DIR="$1"
fi

# Validate path
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory '$DATA_DIR' does not exist"
    echo "Please set: DATA_DIR=/your/path/to/rlds $0"
    exit 1
fi

# Display configuration
echo "Data directory: $DATA_DIR"
echo "Output directory: $HF_LEROBOT_HOME"
echo "Push to Hub: $PUSH_TO_HUB"
echo "=========================="

# Set environment variables and create directories
export HF_LEROBOT_HOME="$HF_LEROBOT_HOME"
mkdir -p "$HF_LEROBOT_HOME"

# # Install dependencies
# echo "Installing dependencies..."
# uv pip install tensorflow tensorflow_datasets

# Build command
ARGS="--data_dir $DATA_DIR"
if [ "$PUSH_TO_HUB" = "true" ]; then
    ARGS="$ARGS --push_to_hub"
fi

# Run conversion
echo "Starting conversion (approximately 30 minutes)..."
python scripts/convert_data_to_lerobot.py $ARGS

echo "Conversion completed! Data saved to: $HF_LEROBOT_HOME"