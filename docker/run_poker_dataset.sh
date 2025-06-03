#!/bin/bash
set -e

# Poker Dataset Generation Script
# Wrapper for poker/dataset/generate_dataset.py with environment setup

echo "=== Poker Dataset Generation ==="

# Check if config file is provided
if [ $# -eq 0 ]; then
    echo "Error: No configuration file provided"
    echo "Usage: $0 CONFIG_PATH [--default-config DEFAULT_CONFIG]"
    echo ""
    echo "Example:"
    echo "  $0 /app/input/poker_config.yml"
    echo "  $0 /app/input/poker_config.yml --default-config /app/input/default_config.yml"
    exit 1
fi

# Set up Blender environment
export BLENDER_USER_SCRIPTS="/app/scripts"
export PYTHONPATH="/app:$PYTHONPATH"

# Ensure output directory exists
mkdir -p /app/output

# Set display for headless rendering
export DISPLAY=:99
if ! pgrep -x "Xvfb" > /dev/null; then
    echo "Starting virtual display..."
    Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &
    sleep 2
fi

echo "Configuration file: $1"
echo "Output will be saved to: /app/output"
echo ""

# Run the poker dataset generation
cd /app
python poker/dataset/generate_dataset.py "$@"

echo ""
echo "Poker dataset generation completed!"
echo "Check /app/output for generated files." 