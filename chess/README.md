# Chess Scene Generator

A Python library for generating 3D chess scenes using Blender. This library provides a flexible and configurable way to create photorealistic chess board renders with customizable pieces, materials, and scene settings.

## Features

- **Configurable Chess Board**
  - Customizable dimensions, materials, and patterns
  - Support for traditional checkerboard and random patterns
  - Configurable border width and thickness
  - Material customization for board frame and squares
  - Flexible board dimensions (custom rows and columns)

- **Chess Pieces**
  - Full set of chess pieces (pawn, rook, knight, bishop, queen, king)
  - Configurable materials and colors
  - Support for custom scaling and random rotation
  - Automatic positioning on the board
  - Random piece generation with difficulty presets

- **Random Generation**
  - Multiple difficulty presets (easy, medium, hard)
  - Customizable piece counts and distributions
  - Scale variations within and between piece types
  - Automatic board position randomization
  - Smart piece placement to avoid overcrowding

- **Scene Management**
  - Camera configuration with adjustable distance and angles
  - Customizable lighting setup
  - Environment settings (table, floor)
  - Noise and texture effects

- **Legend Generation**
  - Automatic generation of scene documentation
  - Board and piece position tracking
  - Support for both text and structured dictionary formats

## Installation

This module requires Blender to be installed in your system. It is designed to work with Blender's Python API.

## Usage

### Basic Scene Generation

Here's a basic example of how to generate a chess scene:

```python
from chess import generate_chess_image

# Configure the scene
scene_config = {
    "camera": {
        "distance": "medium",
        "angle": "medium",
        "horizontal_angle": 0.0,
        "randomize": True,
        "random_config": {
            "distance": True,
            "angle": True,
            "distance_std": 0.3,
            "angle_std": 5.0
        }
    },
    "render": {
        "engine": "CYCLES",
        "samples": 128,
        "resolution_x": 1280,
        "resolution_y": 720,
    },
    "lighting": {
        "lighting": "high"
    },
    "environment": {
        "table_shape": "elliptic",
        "table_length": 2.5,
        "table_width": 1.8,
        "table_texture": "wood",
        "floor_color": (0.2, 0.2, 0.2, 1.0),
        "floor_roughness": 0.7
    }
}

# Configure the board
board_config = {
    "length": 0.7,
    "width": 0.7,
    "thickness": 0.05,
    "location": (0, 0, 0.9),
    "border_width": 0.05,
    "rows": 8,
    "columns": 8
}

# Configure pieces
pieces_config = {
    "king_1": {
        "type": "king",
        "location": (0, 4),
        "color": (0.9, 0.9, 0.9, 1.0),  # White
        "scale": 0.08,
        "random_rotation": True,
        "max_rotation_angle": 10.0
    },
    "queen_1": {
        "type": "queen",
        "location": (0, 3),
        "color": (0.9, 0.9, 0.9, 1.0),  # White
        "scale": 0.08
    }
}

# Optional noise configuration
noise_config = {
    "blur": "none",           # Blur intensity preset
    "table_texture": "low"    # Table texture entropy preset
}

# Generate the scene
scene_path, legend_txt_path, legend_json_path, updated_scene_config, updated_board_config, updated_pieces_config, updated_noise_config = generate_chess_image(
    scene_config=scene_config,
    board_config=board_config,
    pieces_config=pieces_config,
    noise_config=noise_config,
    output_dir="renders",
    base_filename="example_scene"
)
```

### Random Scene Generation

You can also generate random chess scenes with different difficulty levels:

```python
from chess.config.random_generator import RandomConfigGenerator
from chess.config.models import (
    BoardRandomizationModel,
    PieceRandomizationModel,
    PieceCountRange,
    RandomizationRange
)

# Generate a scene with default difficulty (medium)
generator = RandomConfigGenerator()
board_config, pieces_config = generator.generate_all_configs()

# Generate a scene with custom difficulty
generator = RandomConfigGenerator(difficulty="hard")
board_config, pieces_config = generator.generate_all_configs()

# Generate a scene with custom configuration
board_config = {
    "location_bounds": [
        {"min_value": -0.5, "max_value": 0.5},  # x
        {"min_value": -0.5, "max_value": 0.5}   # y
    ],
    "pattern_randomization": False,
    "rows": 8,
    "columns": 8
}

piece_config = {
    "piece_counts": {
        "pawn": {"min_count": 4, "max_count": 6},
        "rook": {"min_count": 1, "max_count": 2},
        "knight": {"min_count": 1, "max_count": 2},
        "bishop": {"min_count": 1, "max_count": 2},
        "queen": {"min_count": 1, "max_count": 1},
        "king": {"min_count": 1, "max_count": 1}
    },
    "scale_range": {"min_value": 0.07, "max_value": 0.09, "step": 0.005},
    "allowed_colors": [(0.9, 0.9, 0.9, 1.0), (0.1, 0.1, 0.1, 1.0)],
    "intra_class_variation": True,
    "extra_class_variation": True
}

generator = RandomConfigGenerator(
    board_config=board_config,
    piece_config=piece_config,
    seed=42  # Optional: for reproducible results
)
board_config, pieces_config = generator.generate_all_configs()
```

## Configuration Options

### Scene Configuration

- **Camera**
  - `distance`: Camera distance from the scene ("low", "medium", "high" or float value)
  - `angle`: Camera viewing angle ("low", "medium", "high" or float value)
  - `horizontal_angle`: Rotation around the scene (degrees)
  - `randomize`: Enable random camera positioning
  - `random_config`: Settings for random camera positioning

- **Render**
  - `engine`: Render engine ("CYCLES" recommended)
  - `samples`: Number of render samples
  - `resolution_x`, `resolution_y`: Output resolution
  - `exposure`: Scene exposure adjustment
  - `file_format`: Output file format (default: "PNG")

- **Lighting**
  - `lighting`: Light intensity ("very_low", "low", "medium", "high", "very_high" or float multiplier)

- **Environment**
  - `table_shape`: Shape of the table ("elliptic", "rectangular")
  - `table_length`, `table_width`: Table dimensions
  - `table_texture`: Table material texture
  - `floor_color`: Floor color (RGBA tuple)
  - `floor_roughness`: Floor material roughness

### Board Configuration

- `length`, `width`: Board dimensions
- `thickness`: Board thickness
- `location`: Board position in 3D space (x, y, z)
- `border_width`: Width of the board border
- `rows`, `columns`: Number of squares (default: 8x8)
- `random_pattern`: Enable random board pattern
- `pattern_seed`: Seed for random pattern generation

### Piece Configuration

- `type`: Piece type ("king", "queen", "rook", "bishop", "knight", "pawn")
- `location`: Board position (row, column)
- `color`: Piece color (RGBA tuple)
- `scale`: Piece size multiplier
- `random_rotation`: Enable random rotation
- `max_rotation_angle`: Maximum random rotation angle

### Noise Configuration

- `blur`: Blur intensity preset ("none", "very_low", "low", "medium", "high", "very_high")
- `table_texture`: Table texture entropy preset ("none", "low", "medium", "high")

## Project Structure

```
chess/
├── __init__.py              # Package initialization
├── generate_chess_image.py  # Main scene generation
├── generate_legend.py       # Legend generation utilities
├── board.py                # Chess board creation
├── scene_chess.py          # Scene management
├── test_chess_generation.py # Test suite
├── config/                 # Configuration and models
│   ├── models.py          # Configuration models
│   ├── static_configs.py  # Predefined configurations
│   ├── difficulty_presets.py # Difficulty presets
│   └── random_generator.py # Random configuration generator
├── pieces/                 # Chess piece models
│   ├── pawn.py            # Pawn piece
│   ├── rook.py            # Rook piece
│   ├── knight.py          # Knight piece
│   ├── bishop.py          # Bishop piece
│   ├── queen.py           # Queen piece
│   └── king.py            # King piece
└── img/                    # Generated images
```

## Testing

The project includes a comprehensive test suite in `test_chess_generation.py`. Run the tests using:

```bash
python -m pytest test_chess_generation.py
```

The test suite covers:
- Board generation and configuration
- Piece placement and rotation
- Scene setup and rendering
- Noise effects
- Legend generation

