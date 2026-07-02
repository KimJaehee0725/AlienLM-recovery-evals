#!/bin/bash

# N-gram Attack Evaluation - process cleanup script

echo "Cleaning up hanging processes..."

# Find and terminate run_evaluation.py processes
PIDS=$(ps aux | grep "run_evaluation.py" | grep -v grep | awk '{print $2}')

if [ -z "$PIDS" ]; then
    echo "No running run_evaluation.py processes found."
else
    echo "Found processes: $PIDS"
    echo "$PIDS" | xargs kill -9 2>/dev/null || true
    echo "Killed hanging processes."
fi

# Optionally check for dataset-related Python processes.
# Note: this may terminate other Python processes, so use with care.
if [ "$1" == "--aggressive" ]; then
    echo "Aggressive cleanup mode..."
    # Find processes related to the datasets/transformers libraries
    DATASET_PIDS=$(ps aux | grep python | grep -E "(datasets|transformers)" | grep -v grep | awk '{print $2}')
    if [ -n "$DATASET_PIDS" ]; then
        echo "Found dataset-related processes: $DATASET_PIDS"
        read -p "Kill these processes? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "$DATASET_PIDS" | xargs kill -9 2>/dev/null || true
            echo "Killed dataset-related processes."
        fi
    fi
fi

echo "Cleanup completed."
