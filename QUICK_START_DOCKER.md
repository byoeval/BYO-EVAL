# Quick Start Guide - Docker

Get the BYO-EVAL project running in Docker in 5 minutes!

## Prerequisites

- Docker and Docker Compose installed
- At least 8GB RAM and 10GB disk space

## Step 1: Setup and Build

```bash
# Run the automated setup
./docker/build_and_run.sh all
```

This will:
- Create necessary directories (`configs/`, `data/`, `output/`, etc.)
- Build the Docker image
- Test the installation
- Show usage examples

## Step 2: Create a Configuration File

### For Chess Dataset Generation

Create `configs/chess_config.yml`:

```yaml
dataset:
  name: "test_chess_dataset"
  output_dir: "/app/output/chess"
  seed: 42
  piece_set: "default"

variables:
  chess.count_config:
    variate_type: "varying_all"
    variate_levels: [8, 16]
    n_images: 2
```

### For Poker Dataset Generation

Create `configs/poker_config.yml`:

```yaml
dataset:
  name: "test_poker_dataset"
  output_dir: "/app/output/poker"
  seed: 42

variables:
  poker.players.0.chip_count:
    variate_type: "varying_among_range"
    variate_levels: [100, 500]
    n_images: 3
```

## Step 3: Generate Your First Dataset

### Chess Dataset

```bash
docker-compose run --rm byo-eval chess-dataset /app/input/chess_config.yml
```

### Poker Dataset

```bash
docker-compose run --rm byo-eval poker-dataset /app/input/poker_config.yml
```

## Step 4: Check Results

Your generated images and metadata will be in:
- `output/chess/` - Chess dataset outputs
- `output/poker/` - Poker dataset outputs

## Next Steps

### Evaluate a Dataset

```bash
# Put your dataset in the data/ directory first
docker-compose run --rm byo-eval evaluate \
    --dataset-path /app/data/your_dataset \
    --task counting \
    --vlm gpt-4-vision
```

### Run Diagnostics

```bash
# Create a diagnostic config first
docker-compose run --rm byo-eval diagnostic \
    /app/input/diagnostic_config.yml \
    /app/data/your_dataset
```

### Interactive Mode

```bash
# Start an interactive shell to explore
docker-compose run --rm byo-eval bash

# Inside the container:
python -c "import bpy; print('Blender available!')"
ls /app  # Explore the project structure
```

## Troubleshooting

### Permission Issues

```bash
sudo chown -R $USER:$USER output/ data/
```

### Memory Issues

Edit `docker-compose.yml` and increase memory limits:

```yaml
deploy:
  resources:
    limits:
      memory: 16G  # Increase as needed
```

### Can't Import Blender

```bash
# Test Blender availability
docker-compose run --rm byo-eval bash -c "python -c 'import bpy; print(\"OK\")'"
```

## Common Commands Reference

```bash
# Show help
docker-compose run --rm byo-eval help

# Build image
docker-compose build

# Interactive shell
docker-compose run --rm byo-eval bash

# Run with custom environment
docker-compose run --rm -e OPENAI_API_KEY=your_key byo-eval evaluate

# View logs
docker-compose logs

# Clean up
docker-compose down
docker system prune  # Remove unused containers/images
```

## File Structure

```
project/
├── configs/          # Your configuration files
├── cache/            # Model cache
├── docker/           # Docker configuration
├── docker-compose.yml
└── Dockerfile
```

That's it! You're ready to generate datasets and run evaluations with BYO-EVAL in Docker.

For more advanced usage, see `DOCKER_README.md`. 