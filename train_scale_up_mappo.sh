#!/bin/bash

# Training script for SimCity Scale-Up Environment with MAPPO
# This script trains the model on the 8x8 grid with 4 agents and 5 building types

echo "Starting MAPPO training for SimCity Scale-Up Environment..."
echo "Environment: 8x8 grid, 4 agents, 5 building types"
echo "Algorithm: MAPPO"

# Activate conda environment
source /opt/anaconda3/bin/activate epymarl

# Navigate to src directory
cd /Volumes/Mac_Working/GitHub/6.S890_Course_Project/src

# Run training with MAPPO on scale-up environment
python3 main.py \
    --config=mappo \
    --env-config=simcity_scale_up \
    with env_args.time_limit=100 \
    t_max=2000000 \
    use_cuda=True \
    save_model=True \
    save_model_interval=10000 \
    test_interval=5000 \
    log_interval=1000 \
    runner_log_interval=1000 \
    learner_log_interval=1000 \
    buffer_size=5000 \
    batch_size=32 \
    batch_size_run=8

echo "Training completed!"
