#!/bin/bash

# All simcity_large commands
commands=(
    "python src/main.py --config=qmix --env-config=simcity_large"
    "python src/main.py --config=qmix_ns --env-config=simcity_large"
    "python src/main.py --config=vdn --env-config=simcity_large"
    "python src/main.py --config=vdn_ns --env-config=simcity_large"
    "python src/main.py --config=ippo --env-config=simcity_large"
    "python src/main.py --config=ippo_ns --env-config=simcity_large"
    "python src/main.py --config=maddpg --env-config=simcity_large"
    "python src/main.py --config=maddpg_ns --env-config=simcity_large"
)

# Function to run commands in 4 terminals
run_in_parallel() {
    num_parallel=4
    running_jobs=()

    for cmd in "${commands[@]}"; do
        # Wait for any terminal to free up
        while [ "${#running_jobs[@]}" -ge "$num_parallel" ]; do
            for i in "${!running_jobs[@]}"; do
                # Check if process has ended
                if ! kill -0 "${running_jobs[$i]}" 2>/dev/null; then
                    unset 'running_jobs[i]' # Remove finished process
                fi
            done
            running_jobs=("${running_jobs[@]}") # Re-index array
            sleep 1
        done

        # Run next command in a new terminal
        gnome-terminal -- bash -c "$cmd; exec bash" &
        running_jobs+=("$!") # Store process ID
    done

    # Wait for all jobs to complete
    wait
}

run_in_parallel
