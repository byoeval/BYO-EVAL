#!/bin/bash
set -e

# BYO-EVAL Docker Build and Run Script
# This script helps you build and run the Docker container easily

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== BYO-EVAL Docker Setup ==="
echo "Project root: $PROJECT_ROOT"

# Function to create required directories
setup_directories() {
    echo "Creating required directories..."
    cd "$PROJECT_ROOT"
    mkdir -p configs data output cache datasets
    echo "  ✓ configs/ (place your YAML configuration files here)"
    echo "  ✓ data/ (place your input data here)"
    echo "  ✓ output/ (generated outputs will appear here)"
    echo "  ✓ cache/ (for model downloads and caching)"
    echo "  ✓ datasets/ (for additional datasets)"
}

# Function to build the Docker image
build_image() {
    echo "Building Docker image..."
    cd "$PROJECT_ROOT"
    docker-compose build
    echo "  ✓ Docker image built successfully"
}

# Function to show usage examples
show_examples() {
    echo ""
    echo "=== Usage Examples ==="
    echo ""
    echo "1. Generate chess dataset:"
    echo "   docker-compose run --rm byo-eval chess-dataset /app/input/your_config.yml"
    echo ""
    echo "2. Generate poker dataset:"
    echo "   docker-compose run --rm byo-eval poker-dataset /app/input/your_config.yml"
    echo ""
    echo "3. Evaluate dataset:"
    echo "   docker-compose run --rm byo-eval evaluate --dataset-path /app/data/your_dataset"
    echo ""
    echo "4. Run diagnostics:"
    echo "   docker-compose run --rm byo-eval diagnostic /app/input/config.yml /app/data/dataset"
    echo ""
    echo "5. Interactive shell:"
    echo "   docker-compose run --rm byo-eval bash"
    echo ""
    echo "For more details, see DOCKER_README.md"
}

# Function to test the installation
test_installation() {
    echo "Testing installation..."
    cd "$PROJECT_ROOT"
    
    echo "  Testing basic container startup..."
    if docker-compose run --rm byo-eval help > /dev/null 2>&1; then
        echo "  ✓ Container starts successfully"
    else
        echo "  ✗ Container startup failed"
        return 1
    fi
    
    echo "  Testing Blender Python API..."
    if docker-compose run --rm byo-eval bash -c "python -c 'import bpy; print(\"Blender OK\")'" > /dev/null 2>&1; then
        echo "  ✓ Blender Python API available"
    else
        echo "  ✗ Blender Python API not available"
        return 1
    fi
    
    echo "  ✓ Installation test passed"
}

# Main script logic
case "${1:-help}" in
    "setup"|"init")
        setup_directories
        ;;
    "build")
        build_image
        ;;
    "test")
        test_installation
        ;;
    "all"|"full")
        setup_directories
        build_image
        test_installation
        show_examples
        ;;
    "examples")
        show_examples
        ;;
    "help"|"--help"|"-h")
        echo "BYO-EVAL Docker Setup Script"
        echo ""
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  setup    - Create required directories"
        echo "  build    - Build the Docker image"
        echo "  test     - Test the installation"
        echo "  all      - Do setup, build, test, and show examples"
        echo "  examples - Show usage examples"
        echo "  help     - Show this help message"
        echo ""
        echo "Quick start: $0 all"
        ;;
    *)
        echo "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac 