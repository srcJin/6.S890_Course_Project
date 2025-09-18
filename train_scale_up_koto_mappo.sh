#!/bin/bash

# Training script for SimCity Scale-Up Koto Environment with MAPPO
# This script trains the model on the 8x8 grid with 4 player archetypes and 6-parameter system
# Features:
# - 4 player archetypes: Altruistic, Balanced, Interest-Driven, Environmental Focused
# - 6-parameter urban resilience system (G, V, D, A, S, F)
# - Balanced resource system (Money=100, Reputation=70)

echo "Starting MAPPO training for SimCity Scale-Up Koto Environment..."
echo "Environment: 12x12 grid, 4 player archetypes, 8 building types"
echo "Parameters: G(Greenery), V(Vitality), D(Density), A(Adaptability), S(Sustainability), F(Flood_Resistance)"
echo "Algorithm: MAPPO"

# Initialize conda and activate environment
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate epymarl

# Navigate to src directory
cd /Volumes/Mac_Working/GitHub/6.S890_Course_Project/src

# Run training with MAPPO on koto scale-up environment  
python main.py \
    --config=mappo \
    --env-config=simcity_scale_up_koto \
    with env_args.time_limit=150 \
    t_max=100000000 \
    use_cuda=True \
    save_model=True \
    save_model_interval=10000 \
    test_interval=5000 \
    log_interval=1000 \
    runner_log_interval=1000 \
    learner_log_interval=1000 \
    buffer_size=10000 \
    batch_size=192 \
    batch_size_run=24 \
    epochs=3 \
    common_reward=False

echo "Koto training completed!"
echo "Check results in: results/models/ and results/tb_logs/"
