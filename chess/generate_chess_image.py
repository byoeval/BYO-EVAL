import os
from typing import Any

import bpy

from chess.board import ChessBoard
from chess.generate_legend import generate_legend
from chess.pieces.factories import PieceFactory
from noise.set_noise_config import build_noise_from_config
from scene_setup.general_setup import build_setup_from_config
from scene_setup.rendering import clear_scene


def generate_chess_image(
    scene_config: dict[str, Any],
    board_config: dict[str, Any] | ChessBoard,
    pieces_config: dict[str, dict[str, Any]],
    piece_factory: PieceFactory,
    noise_config: dict[str, Any] | None = None,
    output_dir: str | None = None,
    base_filename: str = "chess_scene"
) -> tuple[str, str | None, str | None, dict[str, Any], dict[str, Any], dict[str, dict[str, Any]], dict[str, Any] | None]:
    """
    Generate a chess scene image with optional text and JSON legends and noise effects.

    Args:
        scene_config: Configuration for the scene setup
        board_config: Either a configuration dictionary for the chess board or a ChessBoard instance
        pieces_config: Configuration for chess pieces
        piece_factory: An instance of PieceFactory to create the pieces
        output_dir: Directory to save the output files (default: current working directory)
        base_filename: Base name for output files (default: "chess_scene")
        noise_config: Configuration for noise effects (optional)

    Returns:
        Tuple containing:
        - Path to the rendered chess scene image
        - Path to the legend text file (None if not generated)
        - Path to the legend JSON file (None if not generated)
        - Updated scene configuration
        - Updated board configuration with cell positions
        - Updated pieces configuration with world positions
        - Updated noise configuration with applied settings (None if no noise was applied)
    """
    # Set up output paths
    if output_dir is None:
        output_dir = os.getcwd()

    # Create directory structure
    img_dir = os.path.join(output_dir, "img")
    legend_txt_dir = os.path.join(output_dir, "legend_txt")
    legend_json_dir = os.path.join(output_dir, "legend_json")

    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(legend_txt_dir, exist_ok=True)
    os.makedirs(legend_json_dir, exist_ok=True)

    scene_path = os.path.join(img_dir, f"{base_filename}.png")
    legend_txt_path = os.path.join(legend_txt_dir, f"{base_filename}.txt")
    legend_json_path = os.path.join(legend_json_dir, f"{base_filename}.json")

    # Update render path and GPU settings in scene config
    scene_config.setdefault("render", {})
    scene_config["render"]["output_path"] = scene_path

    # Ensure other required fields exist with empty dictionaries to avoid None errors
    scene_config.setdefault("background", {})
    scene_config.setdefault("lighting", {})
    scene_config.setdefault("resolution", {})
    scene_config.setdefault("camera", {})
    scene_config.setdefault("table", {})
    scene_config.setdefault("floor", {})

    # Make sure table has necessary defaults to prevent None errors
    if "material" not in scene_config["table"]:
        scene_config["table"]["material"] = {}

    # Clear any existing scene
    clear_scene()

    # Set up the scene using set_general_config with dictionary
    scene_result = build_setup_from_config(scene_config)

    # Extract final camera config (assuming it's updated within scene_result or scene_config)
    # If build_setup_from_config updates scene_config in-place, use that.
    # If it returns updates in scene_result, use that. Let's assume scene_result is preferred.
    final_camera_config = scene_result.get("camera") # Get the camera sub-dict
    if not final_camera_config:
        # Fallback if camera is not in scene_result, maybe it modified scene_config directly?
        final_camera_config = scene_config.get("camera")
        if not final_camera_config:
            print("Warning: Could not retrieve final camera configuration for legend.")
            final_camera_config = {}

    # Handle board creation based on input type
    if isinstance(board_config, ChessBoard):
        board = board_config
        board_obj, squares_collection = board.create()
        board_config_dict = board.config.to_dict()
    else:
        # board_config is a dictionary here
        # Pass the dictionary directly to ChessBoard constructor
        board = ChessBoard(board_config) # Pass the dict!
        board_obj, squares_collection = board.create()
        # Get the final config state from the ChessBoard instance if needed
        board_config_dict = board.config.to_dict()

    # Get cell positions and update board config
    cell_positions = board.get_cell_positions()
    board_config_dict["cell_positions"] = {
        k: (round(x, 4), round(y, 4))
        for k, (x, y) in cell_positions.items()
    }

    # Create pieces
    created_pieces = []
    piece_configs = {}  # Store updated piece configs for legend

    for piece_id, piece_config_dict in pieces_config.items():
        # Validate piece position against board dimensions
        row, col = piece_config_dict["location"][0], piece_config_dict["location"][1]
        if row < 0 or row >= board_config_dict["rows"] or col < 0 or col >= board_config_dict["columns"]:
            print(f"Warning: Piece {piece_id} position ({row}, {col}) is outside board dimensions ({board_config_dict['rows']}x{board_config_dict['columns']})")
            continue

        # Get cell position
        cell_pos = board.get_cell_position(row, col)
        if cell_pos is None:
            print(f"Warning: Invalid position for piece {piece_id} at row {row}, col {col}")
            continue

        # Calculate world position (x, y from cell, z from board height)
        board_height = board_config_dict["location"][2] + board_config_dict["thickness"]
        world_pos = (
            cell_pos[0],  # x from cell
            cell_pos[1],  # y from cell
            board_height  # z = board top surface
        )

        # Update piece location with world position
        piece_config_dict["location"] = world_pos

        # Use the piece factory to create the piece
        piece_type = piece_config_dict["type"].lower()
        try:
            piece = piece_factory.create_piece(piece_type, piece_config_dict)
        except Exception as factory_error:
            # Catch potential errors during factory creation itself
            print(f"Error using piece factory for type {piece_type}: {factory_error}")
            piece = None
            continue

        if piece is not None:
            created_pieces.append(piece)
            # Store updated config for legend
            piece_configs[piece_id] = {
                "type": piece_config_dict["type"],
                "board_position": (row, col),
                "world_position": world_pos,
                "color": piece_config_dict.get("color", "white"),
                "scale": piece_config_dict.get("scale", 1.0),
                "random_rotation": piece_config_dict.get("random_rotation", False),
                "max_rotation_angle": piece_config_dict.get("max_rotation_angle", 15.0)
            }

    # Apply noise configuration if provided
    noise_result = None
    if noise_config is not None:
        # Get the table object
        table_object = scene_result.get("table", {}).get("object")

        # Call build_noise_from_config regardless of table_object,
        # as some effects (like blur) don't need it.
        # The function itself should handle cases where table_object is None but needed.
        noise_result = build_noise_from_config(noise_config, table_object)

    # Set render output path and render
    bpy.context.scene.render.filepath = scene_path
    try:
        bpy.ops.render.render(write_still=True)
    except Exception as render_error:
        print(f"Warning: Error during render: {render_error}")
        # Decide how to handle render errors - maybe raise, maybe return indicating failure
        # For now, just print and continue to see if legend generation is reached
        # Continue execution for now

    # Generate legends
    generate_legend(
        board_config=board_config_dict,
        pieces_config=piece_configs,
        camera_config=final_camera_config, # Pass the extracted final camera config
        noise_config=noise_result, # Pass the final noise result dictionary
        txt_path=legend_txt_path,
        json_path=legend_json_path
    )

    # Clean up
    clear_scene()

    return scene_path, legend_txt_path, legend_json_path, scene_config, board_config_dict, piece_configs, noise_result

###########################################################
# Example usage
###########################################################

if __name__ == "__main__":
    # Example usage
    scene_config = {
        "camera": {
            "distance": 5.0,
            "angle": "medium",
            "horizontal_angle": 0.0,
            "randomize": False,
        },
        "render": {
            "engine": "CYCLES",
            "samples": 32,
            "resolution_x": 640,
            "resolution_y": 480,
            "gpu_enabled": True,
        },
        "lighting": {
            "lighting": "medium"
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

    board_config = {
        "length": 0.7,
        "width": 0.7,
        "thickness": 0.05,
        "location": (0, 0, 0.9),
        "border_width": 0.05
        # Other parameters will use defaults from BoardModel
    }

    pieces_config = {
        "king_1": {
            "type": "king",
            "location": (0, 4),
            "color": (0.9, 0.9, 0.9, 1.0),  # White color with alpha
            "scale": 0.08,
            "random_rotation": True,
            "max_rotation_angle": 10.0,
            "roughness": 0.3,  # Material property
            "material_name": "KingMaterial"  # Optional material name
        },
        "queen_1": {
            "type": "queen",
            "location": (0, 3),
            "color": (0.9, 0.9, 0.9, 1.0),
            "scale": 0.08,
            "random_rotation": False,
            "max_rotation_angle": 15.0,
        }
    }

    # Example noise configuration
    noise_config = {
        "blur": "none",           # Blur intensity preset
        "table_texture": "low"  # Table texture entropy preset
    }

    # Generate the scene
    scene_path, legend_txt_path, legend_json_path, updated_scene_config, updated_board_config, updated_pieces_config, updated_noise_config = generate_chess_image(
        scene_config=scene_config,
        board_config=board_config,
        pieces_config=pieces_config,
        # piece_factory=PieceFactory(),
        noise_config=noise_config,
        output_dir="renders",
        base_filename="example_scene"
    )

    print(f"Scene rendered to: {scene_path}")
    if legend_txt_path:
        print(f"Legend text generated to: {legend_txt_path}")
    if legend_json_path:
        print(f"Legend JSON generated to: {legend_json_path}")
    if updated_noise_config:
        print("Noise configuration applied successfully")
