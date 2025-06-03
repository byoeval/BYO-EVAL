#!/bin/bash
set -e

# BYO-EVAL Docker Entrypoint Script
# This script provides easy access to the four main tools in the project

echo "=== BYO-EVAL Docker Container ==="
echo "Available commands:"
echo "  chess-dataset    - Generate chess datasets"
echo "  poker-dataset    - Generate poker datasets"
echo "  evaluate         - Evaluate and diagnose datasets"
echo "  diagnostic       - Run diagnostic tests"
echo "  bash             - Interactive shell"
echo "=================================="

# Set up display for headless Blender rendering
export DISPLAY=:99
Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &

# Function to show usage
show_usage() {
    echo ""
    echo "Usage: docker run [docker-options] byo-eval [command] [arguments]"
    echo ""
    echo "Commands:"
    echo "  chess-dataset CONFIG_PATH [--default-config DEFAULT_CONFIG]"
    echo "      Generate chess image dataset from YAML configuration"
    echo ""
    echo "  poker-dataset CONFIG_PATH [--default-config DEFAULT_CONFIG]"
    echo "      Generate poker image dataset from YAML configuration"
    echo ""
    echo "  evaluate [options]"
    echo "      Evaluate VLM performance on generated datasets"
    echo ""
    echo "  diagnostic CONFIG_PATH DATASET_PATH [options]"
    echo "      Run diagnostic tests on datasets"
    echo ""
    echo "  bash"
    echo "      Start interactive bash shell"
    echo ""
    echo "Examples:"
    echo "  docker run -v \$(pwd)/configs:/app/input -v \$(pwd)/output:/app/output byo-eval chess-dataset /app/input/chess_config.yml"
    echo "  docker run -v \$(pwd)/data:/app/data byo-eval evaluate --dataset-path /app/data/my_dataset"
    echo ""
}

# Handle different commands
case "${1:-help}" in
    "chess-dataset")
        shift
        echo "Running Chess Dataset Generation..."
        python /app/chess/dataset/generate_dataset.py "$@"
        ;;
    "poker-dataset")
        shift
        echo "Running Poker Dataset Generation..."
        python /app/poker/dataset/generate_dataset.py "$@"
        ;;
    "evaluate")
        shift
        echo "Running Dataset Evaluation..."
        python /app/evaluate_diagnose_dataset.py "$@"
        ;;
    "diagnostic")
        shift
        echo "Running Diagnostic Tests..."
        python /app/diagnostic/run_diagnostic.py "$@"
        ;;
    "bash"|"shell")
        echo "Starting interactive shell..."
        exec /bin/bash
        ;;
    "help"|"--help"|"-h")
        show_usage
        ;;
    *)
        echo "Unknown command: $1"
        show_usage
        exit 1
        ;;
esac 