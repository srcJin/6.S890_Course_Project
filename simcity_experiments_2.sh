#!/bin/bash

# All simcity commands
commands=(
    "python src/main.py --config=maa2c --env-config=simcity"
    "python src/main.py --config=maa2c_ns --env-config=simcity"
    "python src/main.py --config=mappo --env-config=simcity"
    "python src/main.py --config=mappo_ns --env-config=simcity"
    "python src/main.py --config=pac --env-config=simcity"
    "python src/main.py --config=pac_ns --env-config=simcity"
)

# Number of parallel processes
num_parallel=6
session_name="simcity_run"

# Create a new tmux session
tmux new-session -d -s $session_name

# Run commands in tmux windows
for ((i = 0; i < ${#commands[@]}; i++)); do
    cmd="${commands[$i]}"
    window_index=$((i % num_parallel))

    # Check if the window exists, if not create it
    if ! tmux list-windows -t $session_name | grep -q "$window_index"; then
        tmux new-window -t $session_name -n "window_$window_index"
    fi

    # Run the command in the tmux window
    tmux send-keys -t $session_name:window_$window_index "$cmd" C-m
done

# Attach to the tmux session to monitor progress
tmux attach -t $session_name