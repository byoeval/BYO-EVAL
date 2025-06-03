#!/bin/bash
set -e

# Diagnostic Tests Script
# Wrapper for diagnostic/run_diagnostic.py

echo "=== Diagnostic Tests ==="

# Check if required arguments are provided
if [ $# -lt 2 ]; then
    echo "Error: Missing required arguments"
    echo "Usage: $0 CONFIG_PATH DATASET_PATH [additional options]"
    echo ""
    echo "Example:"
    echo "  $0 /app/input/diagnostic_config.yml /app/data/my_dataset"
    exit 1
fi

# Set up environment
export PYTHONPATH="/app:$PYTHONPATH"

# Ensure output directory exists
mkdir -p /app/output

echo "Configuration file: $1"
echo "Dataset path: $2"
echo "Output will be saved to: /app/output"
echo ""

# Run the diagnostic script
cd /app
python diagnostic/run_diagnostic.py "$@"

echo ""
echo "Diagnostic tests completed!"
echo "Check /app/output for diagnostic results." 