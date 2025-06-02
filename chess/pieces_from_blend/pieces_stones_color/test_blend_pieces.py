"""Test script for blend-based chess pieces."""

import os
import sys
from pathlib import Path

# Redirect Blender's stdout to /dev/null while keeping our stderr output
if not sys.stderr.isatty():  # Only do this when not in interactive mode
    devnull = open(os.devnull, 'w')
    sys.stdout = devnull

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import bpy

from chess.board import ChessBoard
from chess.config.models import BoardModel
from chess.pieces_from_blend.pieces_stones_color import (
    create_bishop,
    create_king,
    create_knight,
    create_pawn,
    create_queen,
    create_rook,
)
from scene_setup.general_setup import build_setup_from_config
from scene_setup.models import SceneSetupModel
from scene_setup.rendering import clear_scene

# Scene configuration from generate_chess_image.py
scene_config = {
    "camera": {
        "distance": 2.5,
        "angle": "medium",
        "randomize": False,
    },
    "render": {
        "engine": "CYCLES",
        "samples": 512,
        "resolution_x": 1920,
        "resolution_y": 1080,
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

# Board configuration from generate_chess_image.py
board_config = {
    "length": 0.7,
    "width": 0.7,
    "thickness": 0.05,
    "location": (0, 0, 0.9),
    "border_width": 0.05,
    "rows": 8,
    "columns": 8
}

def clear_scene():
    """Clear all objects and collections from the scene."""
    print("Clearing scene...", file=sys.stderr)

    # Delete all objects
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    print("- Deleted all objects", file=sys.stderr)

    # Delete all collections except the master collection
    for collection in bpy.data.collections:
        bpy.data.collections.remove(collection)
    print("- Deleted all collections", file=sys.stderr)

    # Delete all materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material)
    print("- Deleted all materials", file=sys.stderr)

    # Delete all meshes
    for mesh in bpy.data.meshes:
        bpy.data.meshes.remove(mesh)
    print("- Deleted all meshes", file=sys.stderr)

def main():
    print("\n=== Starting Test Blend Pieces ===", file=sys.stderr)

    # Clear the scene
    clear_scene()

    # Set up the scene using general setup
    scene_setup_config = SceneSetupModel.from_dict(scene_config)
    scene_result = build_setup_from_config(scene_setup_config.to_dict())
    print("Scene setup completed", file=sys.stderr)

    # Create the chess board
    print("\nCreating chess board:", file=sys.stderr)
    board_config_obj = BoardModel(**board_config)
    board = ChessBoard(board_config_obj)
    board_obj, squares_collection = board.create()
    print("- Chess board created", file=sys.stderr)

    # Get cell positions from the board
    cell_positions = board.get_cell_positions()

    # Create pieces with different positions and colors
    print("\nCreating chess pieces:", file=sys.stderr)

    # Get board height for piece placement
    board_height = board_config["location"][2] + board_config["thickness"]

    # Define piece positions on the board (using chess notation)
    piece_positions = {
        # White pieces
        "white_pawn": (6, 0),    # Second row from bottom, first column
        "white_rook": (7, 0),    # Bottom row, first column
        "white_knight": (7, 1),  # Bottom row, second column
        "white_bishop": (7, 2),  # Bottom row, third column
        "white_queen": (7, 3),   # Bottom row, fourth column
        "white_king": (7, 4),    # Bottom row, fifth column

        # Black pieces
        "black_pawn": (1, 7),    # Second row from top, last column
        "black_rook": (0, 7),    # Top row, last column
        "black_knight": (0, 6),  # Top row, seventh column
        "black_bishop": (0, 5),  # Top row, sixth column
        "black_queen": (0, 3),   # Top row, fourth column
        "black_king": (0, 4),    # Top row, fifth column
    }

    # Create all pieces
    piece_creators = {
        "pawn": create_pawn,
        "rook": create_rook,
        "knight": create_knight,
        "bishop": create_bishop,
        "queen": create_queen,
        "king": create_king
    }

    # White pieces
    print("\nCreating white pieces:", file=sys.stderr)
    for piece_name, (row, col) in piece_positions.items():
        if piece_name.startswith("white_"):
            piece_type = piece_name.split("_")[1]
            pos = board.get_cell_position(row, col)
            config = {
                "type": piece_type,
                "location": (pos[0], pos[1], board_height),
                "geometry": {"scale": 1.0},
                "color": (0.9, 0.9, 0.9, 1)
            }
            piece = piece_creators[piece_type](config)
            print(f"- White {piece_type} created", file=sys.stderr)

    # Black pieces
    print("\nCreating black pieces:", file=sys.stderr)
    for piece_name, (row, col) in piece_positions.items():
        if piece_name.startswith("black_"):
            piece_type = piece_name.split("_")[1]
            pos = board.get_cell_position(row, col)
            config = {
                "type": piece_type,
                "location": (pos[0], pos[1], board_height),
                "geometry": {"scale": 1.0},
                "color": (0.2, 0.2, 0.2, 1)
            }
            piece = piece_creators[piece_type](config)
            print(f"- Black {piece_type} created", file=sys.stderr)

    # Set up render output
    print("\nSetting up render:", file=sys.stderr)
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    output_dir = os.path.join(project_root, "output")
    os.makedirs(output_dir, exist_ok=True)

    # Set up file paths for both render and blend file
    base_name = "test_pieces"
    render_path = os.path.join(output_dir, f"{base_name}_render.png")
    blend_path = os.path.join(output_dir, f"{base_name}.blend")

    bpy.context.scene.render.filepath = render_path
    print(f"- Output render path: {render_path}", file=sys.stderr)
    print(f"- Output blend path: {blend_path}", file=sys.stderr)

    # Print scene statistics
    print("\nScene statistics:", file=sys.stderr)
    print(f"- Objects in scene: {len(bpy.context.scene.objects)}", file=sys.stderr)
    print(f"- Collections: {len(bpy.data.collections)}", file=sys.stderr)
    print(f"- Materials: {len(bpy.data.materials)}", file=sys.stderr)
    print(f"- Meshes: {len(bpy.data.meshes)}", file=sys.stderr)

    # Save the blend file
    print("\nSaving blend file...", file=sys.stderr)
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)
    print(f"Blend file saved to {blend_path}", file=sys.stderr)

    # Render
    print("\nStarting render...", file=sys.stderr)
    bpy.ops.render.render(write_still=True)
    print(f"Render completed and saved to {render_path}", file=sys.stderr)

    print("\n=== Test Complete ===", file=sys.stderr)

if __name__ == "__main__":
    # Check if we're running inside Blender
    try:
        bpy.app.version  # This will raise AttributeError if not in Blender
        main()
    except (AttributeError, ImportError):
        # We're not in Blender, so launch Blender with this script
        import subprocess
        script_path = os.path.abspath(__file__)
        cmd = ["blender", "--background", "--python", script_path]
        subprocess.run(cmd, stdout=subprocess.DEVNULL)
