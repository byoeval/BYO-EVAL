#!/bin/bash
set -e

# Dataset Evaluation and Diagnosis Script
# Wrapper for evaluate_diagnose_dataset.py

echo "=== Dataset Evaluation and Diagnosis ==="

# Set up environment
export PYTHONPATH="/app:$PYTHONPATH"

# Ensure output directory exists
mkdir -p /app/output

echo "Running dataset evaluation..."
echo "Output will be saved to: /app/output"
echo ""

# Run the evaluation script
cd /app
python evaluate_diagnose_dataset.py "$@"

echo ""
echo "Evaluation completed!"
echo "Check /app/output for results and reports." 