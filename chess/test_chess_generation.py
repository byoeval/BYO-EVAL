from typing import List, Optional, Union

from chess.config.models import (
    BoardRandomizationModel,
    PieceCountRange,
    PieceRandomizationModel,
    RandomizationRange,
)
from chess.config.random_generator import RandomConfigGenerator
from chess.generate_chess_image import generate_chess_image

# Basic scene configuration
scene_config = {
    "camera": {
        "distance": 3.0,  # Using preset instead of float
        "angle": 45.0,     # Using preset instead of float
        "horizontal_angle": 0.0,
        "randomize": False,
    },
    "render": {
        "engine": "CYCLES",
        "samples": 128,
        "resolution_x": 1280,
        "resolution_y": 720,
        "use_gpu": True
    },
    "lighting": {
        "lighting": "high"  # Using preset instead of individual light powers
    },
    "environment": {
        "table_shape": "circular",
        "table_length": 2.5,
        "table_width": 1.8,
        "table_texture": "wood",
        "floor_color": (0.9, 0.9, 0.9, 1.0),  # light gray
        "floor_roughness": 0.7,
        "table_height": 0.9
    }
}

# Basic board configuration with standard 8x8 grid
board_config = {
    "length": 0.7,
    "width": 0.7,
    "thickness": 0.05,
    "location": (0, 0, 0.9),
    "border_width": 0.05,
    "rows": 8,
    "columns": 8,
    "random_pattern": False
}

# Simple pieces configuration (just two pieces for testing)
pieces_config = {
    "king_white": {
        "type": "king",
        "location": (0, 4),  # E1 position
        "color": (0.9, 0.9, 0.9, 1.0),  # White
        "scale": 0.08
    },
    "king_black": {
        "type": "king",
        "location": (7, 4),  # E8 position
        "color": (0.1, 0.1, 0.1, 1.0),  # Black
        "scale": 0.08
    },
    "queen_white": {
        "type": "queen",
        "location": (0, 3),  # D1 position
        "color": (0.9, 0.9, 0.9, 1.0),  # White
        "scale": 0.08
    }
}

def test_random_generation(
    difficulty: str = "medium",
    output_dir: str = "test_renders/random",
    base_filename: str = "random_chess_scene",
    seed: int | None = None,
    gpus: str | int | list[int] | None = None
) -> None:
    """
    Test random chess scene generation with a specific difficulty.

    Args:
        difficulty: Difficulty level ('easy', 'medium', 'hard')
        output_dir: Directory to save renders
        base_filename: Base name for output files
        seed: Random seed for reproducibility
    """
    # Create random config generator
    generator = RandomConfigGenerator(difficulty=difficulty, seed=seed)

    # Generate random board and pieces configs
    board_config, pieces_config = generator.generate_all_configs()

    # Use the same scene config as the basic test
    scene_path, legend_path, _, _, _, _ = generate_chess_image(
        scene_config=scene_config,  # Using the existing scene_config
        board_config=board_config,
        pieces_config=pieces_config,
        output_dir=output_dir,
        base_filename=f"{base_filename}_{difficulty}",
        generate_legend=True,
        gpu=True,
        gpus=gpus
    )

    print(f"\nRandom generation test ({difficulty} difficulty):")
    print(f"Scene rendered to: {scene_path}")
    if legend_path:
        print(f"Legend generated to: {legend_path}")
        print("\nLegend contents:")
        with open(legend_path) as f:
            print(f.read())

def test_custom_random_generation(
    output_dir: str = "test_renders/random",
    base_filename: str = "custom_random_chess_scene",
    seed: int | None = None,
    rows: int = 6,  # Custom number of rows
    columns: int = 6,  # Custom number of columns
    gpus: str | int | list[int] | None = None
) -> None:
    """
    Test random chess scene generation with custom randomization config.

    Args:
        output_dir: Directory to save renders
        base_filename: Base name for output files
        seed: Random seed for reproducibility
        rows: Number of rows in the board (default: 6)
        columns: Number of columns in the board (default: 6)
        gpus: GPU selection for rendering (default: None, uses all available)
    """
    # Create custom randomization configs
    board_config = BoardRandomizationModel(
        location_bounds=(
            RandomizationRange(-0.5, 0.5),  # x
            RandomizationRange(-0.5, 0.5),  # y
        ),
        pattern_randomization=False,
        pattern_seed=seed
    )

    # Adjust piece counts for smaller board
    max_pieces = (rows * columns) // 2  # Use at most half the board spaces
    piece_config = PieceRandomizationModel(
        piece_counts={
            "pawn": PieceCountRange(2, max(2, max_pieces // 3)),
            "rook": PieceCountRange(1, 2),
            "knight": PieceCountRange(1, 2),
            "bishop": PieceCountRange(1, 2),
            "queen": PieceCountRange(1, 1),
            "king": PieceCountRange(1, 1)
        },
        scale_range=RandomizationRange(0.07, 0.09, 0.005),
        allowed_colors=[(0.9, 0.9, 0.9, 1.0), (0.1, 0.1, 0.1, 1.0)],
        intra_class_variation=True,
        extra_class_variation=True
    )

    # Create random config generator with custom configs and dimensions
    generator = RandomConfigGenerator(
        board_config=board_config,
        piece_config=piece_config,
        seed=seed,
        rows=rows,
        columns=columns
    )

    # Generate random board and pieces configs
    board_config, pieces_config = generator.generate_all_configs()

    # Generate the scene
    scene_path, legend_path, _, _, _, _ = generate_chess_image(
        scene_config=scene_config,  # Using the existing scene_config
        board_config=board_config,
        pieces_config=pieces_config,
        output_dir=output_dir,
        base_filename=base_filename,
        generate_legend=True,
        gpu=True,
        gpus=gpus
    )

    print("\nCustom random generation test:")
    print(f"Board dimensions: {rows}x{columns}")
    print(f"Scene rendered to: {scene_path}")
    if legend_path:
        print(f"Legend generated to: {legend_path}")
        print("\nLegend contents:")
        with open(legend_path) as f:
            print(f.read())



if __name__ == "__main__":
    # Test basic scene generation

    # Test level 1 configurations


    print("\nTesting basic scene generation:")
    scene_path, legend_path, updated_scene_config, updated_board_config, updated_pieces_config, noise_result = generate_chess_image(
        scene_config=scene_config,
        board_config=board_config,
        pieces_config=pieces_config,
        output_dir="test_renders/basic",
        base_filename="test_chess_scene",
        generate_legend=True,
        gpu=True,
        gpus=[1,2]
    )

    print(f"Scene rendered to: {scene_path}")
    if legend_path:
        print(f"Legend generated to: {legend_path}")


    # Test random generation with different difficulties
    for difficulty in ["hard"]:
        test_random_generation(
            difficulty=difficulty,
            seed=42,  # Use same seed for reproducibility
            gpus=[1,2]
        )

    # Print the cell positions from the updated board config
    if updated_board_config and "cell_positions" in updated_board_config:
        print("\nCell positions:")
        for cell, pos in updated_board_config["cell_positions"].items():
            print(f"{cell}: {pos}")
            break
