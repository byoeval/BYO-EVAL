# BYO-EVAL Docker Setup

This directory contains Docker configuration files to easily run the BYO-EVAL project in a containerized environment.

## Overview

The Docker setup provides easy access to four main tools:
1. **Chess Dataset Generation** (`chess/dataset/generate_dataset.py`)
2. **Poker Dataset Generation** (`poker/dataset/generate_dataset.py`)
3. **Dataset Evaluation** (`evaluate_diagnose_dataset.py`)
4. **Diagnostic Tests** (`diagnostic/run_diagnostic.py`)

## Prerequisites

- Docker (version 20.10 or later)
- Docker Compose (version 2.0 or later)
- At least 8GB of available RAM
- At least 10GB of free disk space

### For GPU Support (Optional)
- NVIDIA GPU with CUDA support
- NVIDIA Docker runtime (`nvidia-docker2`)

## Quick Start

### 1. Build the Docker Image

```bash
# Build the image
docker-compose build

# Or build manually
docker build -t byo-eval .
```

### 2. Create Required Directories

```bash
mkdir -p configs data output cache datasets
```

### 3. Run the Container

```bash
# Show available commands
docker-compose run --rm byo-eval

# Or with docker directly
docker run --rm byo-eval
```

## Usage Examples

### Chess Dataset Generation

```bash
# Create a configuration file in ./configs/chess_config.yml
# Then run:
docker-compose run --rm byo-eval chess-dataset /app/input/chess_config.yml

# With custom default config:
docker-compose run --rm byo-eval chess-dataset \
    /app/input/chess_config.yml \
    --default-config /app/input/chess_default.yml
```

### Poker Dataset Generation

```bash
# Create a configuration file in ./configs/poker_config.yml
# Then run:
docker-compose run --rm byo-eval poker-dataset /app/input/poker_config.yml
```

### Dataset Evaluation

```bash
# Evaluate a dataset
docker-compose run --rm byo-eval evaluate \
    --dataset-path /app/data/my_dataset \
    --output-dir /app/output
```

### Diagnostic Tests

```bash
# Run diagnostics
docker-compose run --rm byo-eval diagnostic \
    /app/input/diagnostic_config.yml \
    /app/data/my_dataset
```

### Interactive Shell

```bash
# Start an interactive session
docker-compose run --rm byo-eval bash

# Inside the container, you can run commands directly:
python chess/dataset/generate_dataset.py --help
python evaluate_diagnose_dataset.py --help
```

## Directory Structure

The container uses the following directory mapping:

```
Host Directory    → Container Path → Purpose
./configs/        → /app/input/    → Configuration files (read-only)
./data/           → /app/data/     → Input datasets and data
./output/         → /app/output/   → Generated outputs
./cache/          → /app/cache/    → Cache for models/downloads
./datasets/       → /app/datasets/ → Additional datasets (read-only)
```

## Configuration Files

### Chess Dataset Configuration Example

Create `configs/chess_config.yml`:

```yaml
dataset:
  name: "my_chess_dataset"
  output_dir: "/app/output/chess"
  seed: 42
  piece_set: "default"

variables:
  chess.count_config:
    variate_type: "varying_all"
    variate_levels: [8, 16, 24, 32]
    n_images: 2
  
  chess.board_config.location:
    variate_type: "fixed"
    variate_levels: [0, 0, 0.9]
```

### Poker Dataset Configuration Example

Create `configs/poker_config.yml`:

```yaml
dataset:
  name: "my_poker_dataset"
  output_dir: "/app/output/poker"
  seed: 42

variables:
  poker.players.0.chip_count:
    variate_type: "varying_among_range"
    variate_levels: [100, 1000]
    n_images: 5
```

## GPU Support

To use GPU acceleration for VLM evaluation:

```bash
# Start with GPU support
docker-compose --profile gpu run --rm byo-eval-gpu bash

# Or with docker directly
docker run --gpus all --rm byo-eval bash
```

## Troubleshooting

### Common Issues

1. **Permission Errors**
   ```bash
   # Fix permissions for output directory
   sudo chown -R $USER:$USER output/
   ```

2. **Out of Memory**
   - Increase Docker memory limits in Docker Desktop
   - Or adjust resource limits in `docker-compose.yml`

3. **Display Issues**
   ```bash
   # The container automatically starts Xvfb for headless rendering
   # If you see display errors, check the logs
   docker-compose logs byo-eval
   ```

4. **Blender Import Errors**
   ```bash
   # Test Blender availability
   docker-compose run --rm byo-eval bash
   python -c "import bpy; print('Blender OK')"
   ```

### Debug Mode

Run with verbose logging:

```bash
docker-compose run --rm byo-eval bash
export PYTHONPATH=/app
python -v chess/dataset/generate_dataset.py your_config.yml
```

### Check Container Health

```bash
# Check if the container is healthy
docker-compose ps

# View health check logs
docker inspect byo-eval-container --format='{{.State.Health.Log}}'
```

## Performance Optimization

### Resource Allocation

Adjust in `docker-compose.yml`:

```yaml
deploy:
  resources:
    limits:
      cpus: '8.0'      # Increase for more CPU cores
      memory: 16G      # Increase for more RAM
```

### Volume Performance

For better I/O performance on macOS/Windows:

```yaml
volumes:
  - ./output:/app/output:delegated  # Use delegated mode
```

## Advanced Usage

### Custom Environment Variables

```bash
docker-compose run --rm \
  -e OPENAI_API_KEY=your_key \
  -e HF_TOKEN=your_token \
  byo-eval evaluate --dataset-path /app/data/dataset
```

### Custom Entry Point

```bash
# Skip the wrapper scripts and run directly
docker run --rm --entrypoint python byo-eval \
  chess/dataset/generate_dataset.py --help
```

### Development Mode

Mount the source code for development:

```bash
docker run --rm -it \
  -v $(pwd):/app \
  -v $(pwd)/output:/app/output \
  byo-eval bash
```

## Support

For issues specific to the Docker setup, check:

1. Docker logs: `docker-compose logs`
2. Container health: `docker-compose ps`
3. Resource usage: `docker stats`

For application-specific issues, refer to the main project documentation. 