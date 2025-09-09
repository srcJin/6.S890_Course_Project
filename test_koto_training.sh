#!/bin/bash

# Quick test training script for SimCity Scale-Up Koto Environment
# This runs a short training session to verify everything works

echo "🧪 Testing MAPPO training for SimCity Scale-Up Koto Environment..."
echo "Environment: 12x12 grid, 4 player archetypes, 8 building types"
echo "Algorithm: MAPPO (short test run)"

# Initialize conda and activate environment
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate epymarl

# Navigate to src directory
cd /Volumes/Mac_Working/GitHub/6.S890_Course_Project/src

# Run short test training (only 10000 timesteps)  
python main.py \
    --config=mappo \
    --env-config=simcity_scale_up_koto \
    with env_args.time_limit=50 \
    t_max=10000 \
    use_cuda=False \
    save_model=False \
    test_interval=2000 \
    log_interval=500 \
    runner_log_interval=500 \
    learner_log_interval=500 \
    buffer_size=1000 \
    batch_size=8 \
    batch_size_run=4 \
    common_reward=False

echo "✅ Koto test training completed!"
echo "If this worked, you can run the full training with: ./train_scale_up_koto_mappo.sh"