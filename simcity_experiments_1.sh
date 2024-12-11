#!/bin/bash

# All simcity commands
commands=(
    "python src/main.py --config=coma --env-config=simcity"
    "python src/main.py --config=coma_ns --env-config=simcity"
    "python src/main.py --config=ia2c --env-config=simcity"
    "python src/main.py --config=ia2c_ns --env-config=simcity"
    "python src/main.py --config=iql --env-config=simcity"
    "python src/main.py --config=iql_ns --env-config=simcity"
)

running_jobs=()
num_parallel=6

# Function to clean up completed jobs
cleanup_jobs() {
    for i in "${!running_jobs[@]}"; do
        if ! kill -0 "${running_jobs[$i]}" 2>/dev/null; then
            unset 'running_jobs[i]' # Remove completed process
        fi
    done
    running_jobs=("${running_jobs[@]}") # Re-index array
}

# Run commands in parallel
for cmd in "${commands[@]}"; do
    # Wait for available slots
    while [ "${#running_jobs[@]}" -ge "$num_parallel" ]; do
        cleanup_jobs
        sleep 1
    done

    # Run the command in the background
    echo "Running: $cmd"
    eval "$cmd" &
    job_pid=$!

    # Check if the PID is valid
    if [[ "$job_pid" =~ ^[0-9]+$ ]]; then
        running_jobs+=("$job_pid") # Track the process ID
    else
        echo "Failed to start job: $cmd"
    fi
done

# Wait for all remaining jobs to finish
wait
echo "All experiments completed."