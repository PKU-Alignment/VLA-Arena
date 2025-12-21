torchrun --standalone --nnodes 1 --nproc-per-node 8 main.py fit \
    --config config/lam-stage-1.yaml \
    2>&1 | tee lam-stage-1.log
