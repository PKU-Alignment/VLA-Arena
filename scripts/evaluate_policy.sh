#!/bin/bash
# ============================================================================
# VLA-Arena Unified Evaluation Script
# ============================================================================
# Instructions:
# 1. Copy this script: cp scripts/evaluate_policy.sh my_evaluation.sh
# 2. Edit the configuration section below
# 3. Run: bash my_evaluation.sh
# ============================================================================

# ================================
# CONFIGURATION SECTION - Edit these variables for your evaluation
# ================================

# Model Configuration
export CUDA_VISIBLE_DEVICES=0

POLICY="openvla"                                    # Options: openvla, random (more coming soon)
MODEL_CKPT="path/to/model/checkpoint"                   # Path to model checkpoint

# Task Configuration
TASK_SUITE="libero_10"                                       # Options:  
TASK_LEVEL=0                                        # Difficulty level: 0 (easy), 1 (medium), 2 (hard)
N_EPISODES=1                                       # Number of episodes per task

# Evaluation Settings
VISUALIZATION=true                                 # Set to true to save evaluation videos
METRICS="success_rate"  # Metrics to compute

# Output Configuration
SAVE_DIR="logs/evaluation_$(date +%Y%m%d_%H%M%S)"  # Output directory (auto-timestamped)

# ================================
# END OF CONFIGURATION SECTION
# ================================

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Validation
validate_config() {
    local valid=true
    
    if [ "$valid" = false ]; then
        print_error "Configuration validation failed. Please check your settings."
        exit 1
    fi
}

# Print configuration summary
print_config() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║           VLA-Arena Evaluation Configuration                 ║"
    echo "╠══════════════════════════════════════════════════════════════╣"
    printf "║ %-20s : %-39s ║\n" "Policy" "$POLICY"
    if [[ "$POLICY" != "random" ]]; then
        # Truncate long paths for display
        local display_model=$(basename "$MODEL_CKPT")
        printf "║ %-20s : %-39s ║\n" "Model" "...$display_model"
    fi
    printf "║ %-20s : %-39s ║\n" "Task Suite" "$TASK_SUITE"
    printf "║ %-20s : %-39s ║\n" "Task Level" "Level $TASK_LEVEL"
    printf "║ %-20s : %-39s ║\n" "Episodes per Task" "$N_EPISODES"
    printf "║ %-20s : %-39s ║\n" "Device" "$DEVICE"
    printf "║ %-20s : %-39s ║\n" "Visualization" "$VISUALIZATION"
    printf "║ %-20s : %-39s ║\n" "Save Directory" "$(basename $SAVE_DIR)"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""
}

# Main execution
main() {
    # Validate configuration
    validate_config
    
    # Print configuration
    print_config
    
    # Ask for confirmation
    read -p "Do you want to proceed with this configuration? [(y)/n]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        print_warning "Evaluation cancelled by user"
        exit 0
    fi
    
    # Build command
    CMD="python scripts/evaluate_policy.py"
    CMD="$CMD --task_suite $TASK_SUITE"
    CMD="$CMD --task_level $TASK_LEVEL"
    CMD="$CMD --n-episode $N_EPISODES"
    CMD="$CMD --policy $POLICY"
    CMD="$CMD --save-dir $SAVE_DIR"
    CMD="$CMD --metrics $METRICS"

    # Add model checkpoint if not random policy
    if [[ "$POLICY" != "random" ]]; then
        CMD="$CMD --model_ckpt $MODEL_CKPT"
    fi
    
    # Add visualization flag if enabled
    if [[ "$VISUALIZATION" == "true" ]]; then
        CMD="$CMD --visualization"
    fi
    
    # Create save directory
    mkdir -p "$SAVE_DIR"
    
    # Save configuration to file
    cat > "$SAVE_DIR/evaluation_config.txt" <<EOF
VLA-Arena Evaluation Configuration
==================================
Date: $(date)
Policy: $POLICY
Model: $MODEL_CKPT
Task Suite: $TASK_SUITE
Task Level: $TASK_LEVEL
Episodes: $N_EPISODES
Device: $DEVICE
Visualization: $VISUALIZATION
Metrics: $METRICS

Command: $CMD
EOF
    
    print_info "Starting evaluation..."
    print_info "Command: $CMD"
    echo ""
    
    # Run evaluation
    eval $CMD
    
    # Check exit status
    if [ $? -eq 0 ]; then
        echo ""
        print_success "Evaluation completed successfully!"
        print_info "Results saved to: $SAVE_DIR"
        
        # Display summary if metrics file exists
        if [ -f "$SAVE_DIR/$POLICY/metrics_summary.json" ]; then
            echo ""
            print_info "Quick Summary:"
            python -c "
import json
with open('$SAVE_DIR/$POLICY/metrics_summary.json', 'r') as f:
    data = json.load(f)
    if 'overall' in data:
        print(f'  Success Rate: {data[\"overall\"].get(\"average_success_rate\", 0)*100:.1f}%')
        print(f'  Safe Success Rate: {data[\"overall\"].get(\"average_safe_success_rate\", 0)*100:.1f}%')
        print(f'  Avg Cumulative Cost: {data[\"overall\"].get(\"average_cumulative_cost\", 0):.3f}')
"
        fi
    else
        print_error "Evaluation failed. Please check the error messages above."
        exit 1
    fi
}

# Display help if requested
if [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
    echo "VLA-Arena Evaluation Script"
    echo ""
    echo "Usage: $0"
    echo ""
    echo "Edit the configuration section in this script to set your parameters, then run it."
    echo ""
    echo "Configuration options:"
    echo "  POLICY          : Model type (openvla, random)"
    echo "  MODEL_CKPT      : Path to model checkpoint"
    echo "  TASK_SUITE      : Task suite to evaluate"
    echo "  TASK_LEVEL      : Difficulty level (0-2)"
    echo "  N_EPISODES      : Episodes per task"
    echo "  DEVICE          : Computation device (cuda/cpu)"
    echo "  VISUALIZATION   : Save videos (true/false)"
    echo "  SAVE_DIR        : Output directory"
fi

# Run main function
main