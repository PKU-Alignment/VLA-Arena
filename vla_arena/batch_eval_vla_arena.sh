#!/bin/bash

# Batch evaluation script for VLA-Arena (all models)
# Automatically modifies YAML, supports multiple models, and backups original YAML.

set -e
# export CUDA_VISIBLE_DEVICES=7

# --- Configuration ---
MODEL=""            # Options: openvla/openvla_oft/openpi/univla/smolvla
CHECKPOINT=""       # Path to the model checkpoint
YAML_PATH=""        # Path to the base YAML configuration file for the specific model

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="./${MODEL}_batch_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

DEFAULT_NUM_TRIALS=10
DEFAULT_SEED=7

# Evaluation parameters
ADD_NOISE=false
ADJUST_LIGHT=false
RANDOMIZE_COLOR=false
CAMERA_OFFSET=false
SAFETY=false
REPLACEMENTS_FILE="VLA-Arena/language_replacements"
USE_REPLACEMENTS=false
REPLACEMENT_PROBABILITY=1.0
REPLACEMENT_LEVEL=1
RANDOM_INIT_STATE_OFFSET=false

# OpenPI specific parameters
OPENPI_PORT=8000
OPENPI_TRAIN_CONFIG=""

# UniVLA specific parameters
UNIVLA_ACTION_DECODER_PATH=""

# Default Task Suites
TASK_SUITES=(
    "safety_dynamic_obstacles"
    "safety_hazard_avoidance"
    "safety_state_preservation"
    "safety_cautious_grasp"
    "safety_static_obstacles"
    "distractor_dynamic_distractors"
    "distractor_static_distractors"
    "extrapolation_preposition_combinations"
    "extrapolation_task_workflows"
    "extrapolation_unseen_objects"
    "long_horizon"
)

TASK_LEVELS=(0 1 2)

# Color Definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Batch evaluation script for VLA-Arena tasks (all models).

OPTIONS:
    -m, --model NAME           Model name (openvla/openvla_oft/openpi/univla/smolvla/random)
    -c, --checkpoint PATH      Path to pretrained checkpoint
    -y, --yaml PATH            Path to evaluation yaml config
    -t, --trials NUM           Number of trials per task (default: $DEFAULT_NUM_TRIALS)
    -s, --seed NUM             Random seed (default: $DEFAULT_SEED)
    -o, --output-dir DIR       Output directory for results (default: $RESULTS_DIR)
    --suites "suite1 suite2"   Space-separated list of task suites to run
    --levels "0 1 2"           Space-separated list of task levels to run
    --skip-existing            Skip evaluations that already have results
    --dry-run                  Show what would be run without executing
    --verbose-errors           Show detailed error information including tracebacks
    -h, --help                 Show this help message
EOF
}

# --- Argument Parsing ---
NUM_TRIALS="$DEFAULT_NUM_TRIALS"
SEED="$DEFAULT_SEED"
OUTPUT_DIR="$RESULTS_DIR"
SKIP_EXISTING=false
DRY_RUN=false
VERBOSE_ERRORS=true
CUSTOM_SUITES=""
CUSTOM_LEVELS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model) MODEL="$2"; shift 2 ;;
        -c|--checkpoint) CHECKPOINT="$2"; shift 2 ;;
        -y|--yaml) YAML_PATH="$2"; shift 2 ;;
        -t|--trials) NUM_TRIALS="$2"; shift 2 ;;
        -s|--seed) SEED="$2"; shift 2 ;;
        -o|--output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --suites) CUSTOM_SUITES="$2"; shift 2 ;;
        --levels) CUSTOM_LEVELS="$2"; shift 2 ;;
        --skip-existing) SKIP_EXISTING=true; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        --verbose-errors) VERBOSE_ERRORS=true; shift ;;
        -h|--help) show_usage; exit 0 ;;
        *) print_error "Unknown option: $1"; show_usage; exit 1 ;;
    esac
done

if [[ -n "$CUSTOM_SUITES" ]]; then TASK_SUITES=($CUSTOM_SUITES); fi
if [[ -n "$CUSTOM_LEVELS" ]]; then TASK_LEVELS=($CUSTOM_LEVELS); fi

# --- Model Validation ---
MODEL_LIST=(openvla openvla_oft openpi univla smolvla)
if [[ ! " ${MODEL_LIST[@]} " =~ " ${MODEL} " ]]; then
    print_error "Unsupported model: $MODEL. Supported: ${MODEL_LIST[*]}"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
SUMMARY_FILE="$OUTPUT_DIR/batch_evaluation_summary_$TIMESTAMP.csv"

# --- YAML Backup ---
if [[ -n "$YAML_PATH" ]]; then
    YAML_BAK="$YAML_PATH.bak_$TIMESTAMP"
    cp "$YAML_PATH" "$YAML_BAK"
    print_info "Backed up original yaml: $YAML_BAK"
    trap 'print_error "Aborted or error occurred. Cleaning up yaml backup."; rm -f "$YAML_BAK"; exit 1' ERR INT
fi

# --- Data Extraction Functions ---

extract_success_rate() {
    local log_file="$1"
    if [[ -f "$log_file" ]]; then
        # Match line containing "success rate:", remove keyword and prefix, get first value and strip '%'
        local v=$(grep -i "success rate:" "$log_file" | tail -1 | sed 's/.*success rate: //I' | awk '{print $1}' | tr -d '%')
        echo "${v:-N/A}"
    else echo "N/A"; fi
}

extract_total_episodes() {
    local log_file="$1"
    if [[ -f "$log_file" ]]; then
        # Match episode stats, remove everything before the colon
        local v=$(grep -E -i "# episodes completed so far:|Total episodes:" "$log_file" | tail -1 | sed 's/.*: //' | awk '{print $1}')
        echo "${v:-N/A}"
    else echo "N/A"; fi
}

extract_total_successes() {
    local log_file="$1"
    if [[ -f "$log_file" ]]; then
        local v=$(grep -E -i "# successes:|Total successes:" "$log_file" | tail -1 | sed 's/.*: //' | awk '{print $1}')
        echo "${v:-N/A}"
    else echo "N/A"; fi
}

extract_total_costs() {
    local log_file="$1"
    if [[ -f "$log_file" ]]; then
        local v=$(grep -i -E "average cost:|Total costs:|Overall costs:" "$log_file" | tail -1 | sed 's/.*cost: //I' | awk '{print $1}')
        echo "${v:-N/A}"
    else echo "N/A"; fi
}

extract_success_costs() {
    local log_file="$1"
    if [[ -f "$log_file" ]]; then
        local v=$(grep -i "Success costs:" "$log_file" | tail -1 | sed 's/.*Success costs: //I' | awk '{print $1}')
        echo "${v:-N/A}"
    else echo "N/A"; fi
}

extract_failure_costs() {
    local log_file="$1"
    if [[ -f "$log_file" ]]; then
        local v=$(grep -i "Failure costs:" "$log_file" | tail -1 | sed 's/.*Failure costs: //I' | awk '{print $1}')
        echo "${v:-N/A}"
    else echo "N/A"; fi
}

print_error_details() {
    local log_file="$1"
    local suite="$2"
    local level="$3"
    print_error "Failed to run $suite L$level"
    if [[ "$VERBOSE_ERRORS" == true ]]; then
        print_error "Error details from log file:"
        if [[ -f "$log_file" ]]; then
            echo "----------------------------------------"
            tail -50 "$log_file" | sed 's/^/  /'
            echo "----------------------------------------"
            if grep -q "Traceback" "$log_file"; then
                print_error "Python traceback found:"
                echo "----------------------------------------"
                grep -A 20 "Traceback" "$log_file" | sed 's/^/  /'
                echo "----------------------------------------"
            fi
        fi
    fi
}

# --- YAML Modification ---
modify_yaml() {
    local suite="$1" level="$2" trials="$3" seed="$4" log_dir="$5" checkpoint="$6" model="$7" yaml="$8"
    
    # Common parameters
    yq -y -i ".add_noise = ${ADD_NOISE}" "$yaml"
    yq -y -i ".adjust_light = ${ADJUST_LIGHT}" "$yaml"
    yq -y -i ".randomize_color = ${RANDOMIZE_COLOR}" "$yaml"
    yq -y -i ".camera_offset = ${CAMERA_OFFSET}" "$yaml"
    yq -y -i ".safety = ${SAFETY}" "$yaml"
    yq -y -i ".replacements_file = \"${REPLACEMENTS_FILE}\"" "$yaml"
    yq -y -i ".use_replacements = ${USE_REPLACEMENTS}" "$yaml"
    yq -y -i ".replacement_probability = ${REPLACEMENT_PROBABILITY}" "$yaml"
    yq -y -i ".replacement_level = ${REPLACEMENT_LEVEL}" "$yaml"
    yq -y -i ".init_state_offset_random = ${RANDOM_INIT_STATE_OFFSET}" "$yaml"
    yq -y -i ".use_local_log = false" "$yaml"

    # Model specific parameters
    if [[ "$model" == "openvla" || "$model" == "openvla_oft" || "$model" == "univla" ]]; then
        yq -y -i ".task_suite_name = \"$suite\" | .task_level = $level | .num_trials_per_task = $trials | .seed = $seed | .local_log_dir = \"$log_dir\"" "$yaml"
        [[ -n "$checkpoint" ]] && yq -y -i ".pretrained_checkpoint = \"$checkpoint\"" "$yaml"
        [[ "$model" == "univla" ]] && yq -y -i ".action_decoder_path = \"${UNIVLA_ACTION_DECODER_PATH}\"" "$yaml"
    elif [[ "$model" == "openpi" ]]; then
        yq -y -i ".task_suite_name = [\"$suite\"] | .task_level = $level | .num_trials_per_task = $trials | .seed = $seed | .local_log_dir = \"$log_dir\"" "$yaml"
        [[ -n "$checkpoint" ]] && yq -y -i ".policy_checkpoint_dir = \"$checkpoint\"" "$yaml"
        yq -y -i ".port = ${OPENPI_PORT} | .train_config_name = \"${OPENPI_TRAIN_CONFIG}\"" "$yaml"
    elif [[ "$model" == "smolvla" ]]; then
        yq -y -i ".task_suite_name = \"$suite\" | .task_level = $level | .num_trials_per_task = $trials | .seed = $seed" "$yaml"
        [[ -n "$checkpoint" ]] && yq -y -i ".policy_path = \"$checkpoint\"" "$yaml"
    fi
}

run_evaluation() {
    local suite="$1"
    local level="$2"
    local run_id="EVAL-${suite}-${MODEL}-${TIMESTAMP}-L${level}"
    local log_file="$OUTPUT_DIR/${run_id}.txt"
    
    modify_yaml "$suite" "$level" "$NUM_TRIALS" "$SEED" "$OUTPUT_DIR" "$CHECKPOINT" "$MODEL" "$YAML_BAK"
    
    local cmd="vla-arena eval --model $MODEL --config $YAML_BAK"
    if [[ "$DRY_RUN" == true ]]; then
        print_info "DRY RUN: $cmd"
        return 0
    fi

    print_info "Executing: $cmd"
    if eval "$cmd" > "$log_file" 2>&1; then
        sr=$(extract_success_rate "$log_file")
        te=$(extract_total_episodes "$log_file")
        ts=$(extract_total_successes "$log_file")
        tc=$(extract_total_costs "$log_file")
        sc=$(extract_success_costs "$log_file")
        fc=$(extract_failure_costs "$log_file")
        
        print_success "Completed $suite L$level: SR = $sr ($ts/$te), Costs = $tc"
        echo "$suite,L$level,$sr,$ts,$te,$tc,$sc,$fc,$log_file" >> "$SUMMARY_FILE"
        return 0
    else
        print_error_details "$log_file" "$suite" "$level"
        echo "$suite,L$level,FAILED,N/A,N/A,N/A,N/A,N/A,$log_file" >> "$SUMMARY_FILE"
        return 1
    fi
}

# --- Main Loop ---
print_info "Starting batch evaluation at $(date)"
echo "Task Suite,Level,Success Rate,Successes,Total Episodes,Total Costs,Success Costs,Failure Costs,Log File" > "$SUMMARY_FILE"

total_evaluations=$((${#TASK_SUITES[@]} * ${#TASK_LEVELS[@]}))
current_evaluation=0
successful_evaluations=0
failed_evaluations=0

for suite in "${TASK_SUITES[@]}"; do
    for level in "${TASK_LEVELS[@]}"; do
        current_evaluation=$((current_evaluation + 1))
        print_info "Progress: $current_evaluation/$total_evaluations"
        
        if run_evaluation "$suite" "$level"; then
            successful_evaluations=$((successful_evaluations + 1))
        else
            failed_evaluations=$((failed_evaluations + 1))
        fi
        sleep 2
    done
done

# --- Final Summary Reporting ---
print_info "Batch evaluation completed at $(date)"
print_info "Successful: $successful_evaluations, Failed: $failed_evaluations"
rm -f "$YAML_BAK"
print_success "Results saved to: $SUMMARY_FILE"