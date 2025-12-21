#!/bin/bash

# Batch evaluation script for LIBERO benchmark
# This script runs multiple task suites and task levels sequentially
# and collects all results into a single summary file

set -e  # Exit on any error
# export CUDA_VISIBLE_DEVICES=2
# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/run_vla_arena_eval.py"
RESULTS_DIR="$SCRIPT_DIR/batch_results"
SUMMARY_FILE="$RESULTS_DIR/batch_evaluation_summary.txt"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Default configuration (can be overridden)
# Set OPENVLA_DEFAULT_CHECKPOINT environment variable to specify a custom checkpoint path.
DEFAULT_CHECKPOINT="${OPENVLA_DEFAULT_CHECKPOINT:-/path/to/your/openvla-checkpoint}"
DEFAULT_MODEL_FAMILY="openvla"
DEFAULT_NUM_TRIALS=10
DEFAULT_SEED=7

# Visual perturbations
NOISE=false
COLOR=false
LIGHT=false
CAMERA=false

# Task suites to evaluate (modify this list as needed)
# Organized by category for better readability
TASK_SUITES=(
    "safety_dynamic_obstacles"
    "safety_hazard_avoidance"
    "safety_object_state_preservation"
    "safety_risk_aware_grasping"
    "safety_static_obstacles"
    "robustness_dynamic_distractors"
    "robustness_static_distractors"
    "generalization_object_preposition_combinations"
    "generalization_task_workflows"
    "generalization_unseen_objects"
    "long_horizon"
)

# Task levels to evaluate (0, 1, 2)
TASK_LEVELS=(0 1 2)

# Colors for output
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

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Batch evaluation script for LIBERO benchmark tasks.

OPTIONS:
    -c, --checkpoint PATH     Path to pretrained checkpoint (default: $DEFAULT_CHECKPOINT)
    -m, --model-family NAME   Model family (default: $DEFAULT_MODEL_FAMILY)
    -t, --trials NUM          Number of trials per task (default: $DEFAULT_NUM_TRIALS)
    -s, --seed NUM            Random seed (default: $DEFAULT_SEED)
    -o, --output-dir DIR      Output directory for results (default: $RESULTS_DIR)
    --suites "suite1 suite2"  Space-separated list of task suites to run
    --levels "0 1 2"          Space-separated list of task levels to run
    --skip-existing           Skip evaluations that already have results
    --dry-run                 Show what would be run without executing
    --verbose-errors          Show detailed error information including tracebacks
    -h, --help                Show this help message

EXAMPLES:
    # Run all default suites and levels
    $0

    # Run specific suites and levels
    $0 --suites "generalization_language_variations safety_static_obstacles" --levels "0 1"

    # Run with custom checkpoint and trials
    $0 -c /path/to/checkpoint -t 5

    # Dry run to see what would be executed
    $0 --dry-run
EOF
}

# Parse command line arguments
CHECKPOINT="$DEFAULT_CHECKPOINT"
MODEL_FAMILY="$DEFAULT_MODEL_FAMILY"
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
        -c|--checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        -m|--model-family)
            MODEL_FAMILY="$2"
            shift 2
            ;;
        -t|--trials)
            NUM_TRIALS="$2"
            shift 2
            ;;
        -s|--seed)
            SEED="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --suites)
            CUSTOM_SUITES="$2"
            shift 2
            ;;
        --levels)
            CUSTOM_LEVELS="$2"
            shift 2
            ;;
        --skip-existing)
            SKIP_EXISTING=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --verbose-errors)
            VERBOSE_ERRORS=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Override default suites/levels if custom ones are provided
if [[ -n "$CUSTOM_SUITES" ]]; then
    TASK_SUITES=($CUSTOM_SUITES)
fi

if [[ -n "$CUSTOM_LEVELS" ]]; then
    TASK_LEVELS=($CUSTOM_LEVELS)
fi

# Create results directory
mkdir -p "$OUTPUT_DIR"
SUMMARY_FILE="$OUTPUT_DIR/batch_evaluation_summary_$TIMESTAMP.txt"

# Function to extract success rate from log file
extract_success_rate() {
    local log_file="$1"
    if [[ -f "$log_file" ]]; then
        # Look for the final success rate line
        grep "Overall success rate:" "$log_file" | tail -1 | sed 's/.*Overall success rate: \([0-9.]*\).*/\1/'
    else
        echo "N/A"
    fi
}

# Function to extract total episodes from log file
extract_total_episodes() {
    local log_file="$1"
    if [[ -f "$log_file" ]]; then
        grep "Total episodes:" "$log_file" | tail -1 | sed 's/.*Total episodes: \([0-9]*\).*/\1/'
    else
        echo "N/A"
    fi
}

# Function to extract total costs from log file
extract_total_costs() {
    local log_file="$1"
    if [[ -f "$log_file" ]]; then
        grep "Overall costs:" "$log_file" | tail -1 | sed 's/.*Overall costs: \([0-9.]*\).*/\1/'
    else
        echo "N/A"
    fi
}

# Function to extract success costs from log file
extract_success_costs() {
    local log_file="$1"
    if [[ -f "$log_file" ]]; then
        grep "Overall success costs:" "$log_file" | tail -1 | sed 's/.*Overall success costs: \([0-9.]*\).*/\1/'
    else
        echo "N/A"
    fi
}

# Function to extract failure costs from log file
extract_failure_costs() {
    local log_file="$1"
    if [[ -f "$log_file" ]]; then
        grep "Overall failure costs:" "$log_file" | tail -1 | sed 's/.*Overall failure costs: \([0-9.]*\).*/\1/'
    else
        echo "N/A"
    fi
}

# Function to extract total successes from log file
extract_total_successes() {
    local log_file="$1"
    if [[ -f "$log_file" ]]; then
        grep "Total successes:" "$log_file" | tail -1 | sed 's/.*Total successes: \([0-9]*\).*/\1/'
    else
        echo "N/A"
    fi
}

# Function to print error details from log file
print_error_details() {
    local log_file="$1"
    local suite="$2"
    local level="$3"

    print_error "Failed to run $suite L$level"

    if [[ "$VERBOSE_ERRORS" == true ]]; then
        print_error "Error details from log file:"

        if [[ -f "$log_file" ]]; then
            echo "----------------------------------------"
            # Print the last 50 lines of the log file to show error details
            tail -50 "$log_file" | sed 's/^/  /'
            echo "----------------------------------------"

            # Also check for specific error patterns and highlight them
            if grep -q "Traceback" "$log_file"; then
                print_error "Python traceback found:"
                echo "----------------------------------------"
                grep -A 20 "Traceback" "$log_file" | sed 's/^/  /'
                echo "----------------------------------------"
            fi

            if grep -q "Error\|Exception\|Failed" "$log_file"; then
                print_error "Error messages found:"
                echo "----------------------------------------"
                grep -i "Error\|Exception\|Failed" "$log_file" | tail -10 | sed 's/^/  /'
                echo "----------------------------------------"
            fi
        else
            print_error "Log file not found: $log_file"
        fi
    else
        print_error "Use --verbose-errors to see detailed error information"
        print_error "Log file: $log_file"
    fi
}


# Function to run a single evaluation
run_evaluation() {
    local suite="$1"
    local level="$2"
    local run_id="EVAL-${suite}-${MODEL_FAMILY}-${TIMESTAMP}-L${level}"
    local log_file="$OUTPUT_DIR/${run_id}.txt"

    print_info "Running evaluation: Suite=$suite, Level=$level"

    # Check if we should skip existing results
    if [[ "$SKIP_EXISTING" == true && -f "$log_file" ]]; then
        local existing_success_rate=$(extract_success_rate "$log_file")
        if [[ "$existing_success_rate" != "N/A" ]]; then
            print_warning "Skipping $suite L$level (already exists with success rate: $existing_success_rate)"
            return 0
        fi
    fi

    # Prepare command
    local cmd="python $PYTHON_SCRIPT \
        --pretrained_checkpoint \"$CHECKPOINT\" \
        --model_family \"$MODEL_FAMILY\" \
        --task_suite_name \"$suite\" \
        --task_level $level \
        --num_trials_per_task $NUM_TRIALS \
        --seed $SEED \
        --local_log_dir \"$OUTPUT_DIR\" \
        --run_id_note \"L${level}\" \
        --add_noise $NOISE \
        --adjust_light $LIGHT \
        --randomize_color $COLOR \
        --camera_offset $CAMERA \
        --save_video_mode \"first_success_failure\""

    if [[ "$DRY_RUN" == true ]]; then
        print_info "DRY RUN: $cmd"
        return 0
    fi

    # Run the evaluation
    print_info "Executing: $cmd"
    if eval "$cmd" > "$log_file" 2>&1; then
        local success_rate=$(extract_success_rate "$log_file")
        local total_episodes=$(extract_total_episodes "$log_file")
        local total_successes=$(extract_total_successes "$log_file")
        local total_costs=$(extract_total_costs "$log_file")
        local success_costs=$(extract_success_costs "$log_file")
        local failure_costs=$(extract_failure_costs "$log_file")

        print_success "Completed $suite L$level: Success rate = $success_rate ($total_successes/$total_episodes), Costs = $total_costs"

        # Write to summary file
        echo "$suite,L$level,$success_rate,$total_successes,$total_episodes,$total_costs,$success_costs,$failure_costs,$log_file" >> "$SUMMARY_FILE"

        return 0
    else
        print_error_details "$log_file" "$suite" "$level"
        echo "$suite,L$level,FAILED,N/A,N/A,N/A,N/A,N/A,$log_file" >> "$SUMMARY_FILE"
        return 1
    fi
}

# Main execution
print_info "Starting batch evaluation at $(date)"
print_info "Configuration:"
print_info "  Checkpoint: $CHECKPOINT"
print_info "  Model family: $MODEL_FAMILY"
print_info "  Trials per task: $NUM_TRIALS"
print_info "  Seed: $SEED"
print_info "  Output directory: $OUTPUT_DIR"
print_info "  Task suites: ${TASK_SUITES[*]}"
print_info "  Task levels: ${TASK_LEVELS[*]}"
print_info "  Skip existing: $SKIP_EXISTING"
print_info "  Dry run: $DRY_RUN"
print_info "  Verbose errors: $VERBOSE_ERRORS"

# Initialize summary file
echo "Task Suite,Level,Success Rate,Successes,Total Episodes,Total Costs,Success Costs,Failure Costs,Log File" > "$SUMMARY_FILE"

# Count total evaluations
total_evaluations=$((${#TASK_SUITES[@]} * ${#TASK_LEVELS[@]}))
current_evaluation=0
successful_evaluations=0
failed_evaluations=0

print_info "Total evaluations to run: $total_evaluations"

# Run evaluations
for suite in "${TASK_SUITES[@]}"; do
    for level in "${TASK_LEVELS[@]}"; do
        current_evaluation=$((current_evaluation + 1))
        print_info "Progress: $current_evaluation/$total_evaluations"

        if run_evaluation "$suite" "$level"; then
            successful_evaluations=$((successful_evaluations + 1))
        else
            failed_evaluations=$((failed_evaluations + 1))
        fi

        # Add a small delay between evaluations
        sleep 2
    done
done

# Generate final summary
print_info "Batch evaluation completed at $(date)"
print_info "Successful evaluations: $successful_evaluations"
print_info "Failed evaluations: $failed_evaluations"

# Create a detailed summary
SUMMARY_DETAILED="$OUTPUT_DIR/detailed_summary_$TIMESTAMP.txt"
cat > "$SUMMARY_DETAILED" << EOF
LIBERO Batch Evaluation Summary
==============================

Execution Time: $(date)
Checkpoint: $CHECKPOINT
Model Family: $MODEL_FAMILY
Trials per Task: $NUM_TRIALS
Seed: $SEED

Results Summary:
- Total Evaluations: $total_evaluations
- Successful: $successful_evaluations
- Failed: $failed_evaluations

Detailed Results:
EOF

# Add detailed results
if [[ -f "$SUMMARY_FILE" ]]; then
    echo "" >> "$SUMMARY_DETAILED"
    echo "Task Suite,Level,Success Rate,Successes,Total Episodes,Total Costs,Success Costs,Failure Costs,Log File" >> "$SUMMARY_DETAILED"
    tail -n +2 "$SUMMARY_FILE" >> "$SUMMARY_DETAILED"
fi

print_success "Summary saved to: $SUMMARY_DETAILED"
print_success "CSV results saved to: $SUMMARY_FILE"

# Display summary table
if [[ "$successful_evaluations" -gt 0 ]]; then
    print_info "Results Summary:"
    echo ""
    printf "%-25s %-8s %-12s %-10s %-10s %-12s %-12s %-12s\n" "Task Suite" "Level" "Success Rate" "Successes" "Total" "Total Costs" "Success Costs" "Failure Costs"
    printf "%-25s %-8s %-12s %-10s %-10s %-12s %-12s %-12s\n" "-------------------------" "--------" "------------" "----------" "----------" "------------" "------------" "------------"

    while IFS=',' read -r suite level success_rate successes total total_costs success_costs failure_costs; do
        if [[ "$success_rate" != "Success Rate" && "$success_rate" != "FAILED" ]]; then
            printf "%-25s %-8s %-12s %-10s %-10s %-12s %-12s %-12s\n" "$suite" "$level" "$success_rate" "$successes" "$total" "$total_costs" "$success_costs" "$failure_costs"
        fi
    done < "$SUMMARY_FILE"
fi

if [[ "$failed_evaluations" -gt 0 ]]; then
    print_warning "Some evaluations failed. Check the log files for details."
fi

print_success "Batch evaluation completed!"
