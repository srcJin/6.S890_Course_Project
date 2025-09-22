#!/bin/bash

# Convenient wrapper for resuming MAPPO training on SimCity Scale-Up Koto environment
# This script uses the enhanced resume_training_enhanced.py with full config inheritance
#
# Usage:
#   ./resume_koto_training.sh                    # Interactive mode
#   ./resume_koto_training.sh 0                  # Auto-resume run index 0 with latest checkpoint
#   ./resume_koto_training.sh 0 50000            # Auto-resume run index 0 from step 50000
#   ./resume_koto_training.sh 0 50000 200000000  # Auto-resume with custom t_max

echo "=== Enhanced MAPPO Training Resume Script ==="
echo "Environment: SimCity Scale-Up Koto (12x12 grid, 4 player archetypes)"
echo "Features: Full config inheritance from Sacred, automatic seed preservation"
echo ""

# Initialize conda and activate environment
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate epymarl

# Navigate to project root
cd /Volumes/Mac_Working/GitHub/6.S890_Course_Project

# Parse arguments
RUN_INDEX="$1"
STEP="$2"
T_MAX="${3:-200000000}"  # Default to 200M if not specified

if [ -z "$RUN_INDEX" ]; then
    # Interactive mode
    echo "Running in interactive mode..."
    python resume_training_enhanced.py --env-filter koto --t-max "$T_MAX"
else
    # Automated mode
    echo "Auto-resuming run index $RUN_INDEX"
    if [ -n "$STEP" ]; then
        echo "Using step: $STEP"
        python resume_training_enhanced.py --auto --run-index "$RUN_INDEX" --step "$STEP" --t-max "$T_MAX" --env-filter koto
    else
        echo "Using latest checkpoint"
        python resume_training_enhanced.py --auto --run-index "$RUN_INDEX" --t-max "$T_MAX" --env-filter koto
    fi
fi

echo ""
echo "Resume script completed!"
echo "Training will continue with all original parameters preserved."