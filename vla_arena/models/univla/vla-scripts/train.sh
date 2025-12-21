# Set LD_LIBRARY_PATH for cuDNN if CUDNN_LIB_PATH is provided, otherwise try to find it dynamically
if [ -n "$CUDNN_LIB_PATH" ]; then
    export LD_LIBRARY_PATH="$CUDNN_LIB_PATH:$LD_LIBRARY_PATH"
elif command -v python &> /dev/null; then
    # Try to find cuDNN library path using Python
    CUDNN_PATH=$(python -c "import site; import os; paths = site.getsitepackages(); cudnn_path = None; [cudnn_path := os.path.join(p, 'nvidia', 'cudnn', 'lib') for p in paths if os.path.exists(os.path.join(p, 'nvidia', 'cudnn', 'lib'))]; print(cudnn_path if cudnn_path else '')" 2>/dev/null)
    if [ -n "$CUDNN_PATH" ] && [ -d "$CUDNN_PATH" ]; then
        export LD_LIBRARY_PATH="$CUDNN_PATH:$LD_LIBRARY_PATH"
    fi
fi
GPUS_PER_NODE=8
NNODES=4
MASTER_PORT=${MASTER_PORT:-28596}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
RANK=${RANK:-0}


# Run your training script with torchrun
torchrun --nproc_per_node ${GPUS_PER_NODE} --nnodes ${NNODES} --node_rank ${RANK} --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} train.py \
                                 --vla.type prism-dinosiglip-224px+mx-oxe-magic-soup-plus \
                                 --run_root_dir "vla_log" \
