"""
Generate legend information from chess board and piece configurations.

This module provides a class to convert configuration dictionaries into formatted legend
information for chess boards and pieces, either as text or as structured dictionaries.
"""

import json
from typing import Any

from utils.color_utils import RGB_to_color


class config_to_legend:
    """
    Class for converting chess configurations into legend information.

    This class takes board and piece configuration dictionaries and provides
    methods to convert them into various formats for easier access and display.
    """

    def __init__(self,
                 board_config: dict[str, Any],
                 pieces_config: dict[str, dict[str, Any]],
                 camera_config: dict[str, Any] | None = None,
                 noise_config: dict[str, Any] | None = None):
        """
        Initialize the config_to_legend instance.

        Args:
            board_config: Configuration dictionary for the chess board
            pieces_config: Configuration dictionary for the chess pieces
            camera_config: Optional configuration dictionary for the camera's final state
            noise_config: Optional configuration dictionary for the final noise settings
        """
        self.board_config = board_config
        self.pieces_config = pieces_config
        self.camera_config = camera_config
        self.noise_config = noise_config

        # Extract common board properties for easier access
        self.board_location = board_config.get("location", (0, 0, 0))
        self.board_rows = board_config.get("rows", 8)
        self.board_columns = board_config.get("columns", 8)
        self.board_pattern = board_config.get("pattern", True)  # Checkered pattern
        self.board_white_color = board_config.get("square_w_color", (1.0, 1.0, 1.0, 1.0))
        self.board_black_color = board_config.get("square_b_color", (0.0, 0.0, 0.0, 1.0))

    def get_board_legend_dict(self) -> dict[str, Any]:
        """
        Extract and format board configuration into a dictionary.

        Returns:
            Dictionary containing formatted board legend information
        """
        board_legend = {
            "location": self.board_location,
            "dimensions": {
                "rows": self.board_rows,
                "columns": self.board_columns
            },
            "pattern": "checkered" if self.board_pattern else "uniform",
            "colors": {
                "white_cells": self.board_white_color,
                "black_cells": self.board_black_color
            }
        }

        # Add cell positions if available
        #if "cell_positions" in self.board_config:
        #    board_legend["cell_positions"] = self.board_config["cell_positions"]

        # Add materials if available
        if "chessboard_material" in self.board_config:
            board_legend["materials"] = {
                "board": self.board_config.get("chessboard_material", "default"),
                "white_squares": self.board_config.get("square_w_material", "default"),
                "black_squares": self.board_config.get("square_b_material", "default")
            }

        # Add dimensions if available
        if "length" in self.board_config and "width" in self.board_config:
            board_legend["size"] = {
                "length": self.board_config["length"],
                "width": self.board_config["width"],
                "thickness": self.board_config.get("thickness", 0.05),
                "border_width": self.board_config.get("board_border_width", 0.05)
            }

        return board_legend

    def get_board_legend_text(self) -> str:
        """
        Generate a text representation of the board configuration.

        Returns:
            Formatted string containing board legend information
        """
        legend = "CHESS BOARD LEGEND\n"
        legend += "=================\n\n"

        # Board location and dimensions
        legend += f"Board Position: ({self.board_location[0]}, {self.board_location[1]}, {self.board_location[2]})\n"
        legend += f"Board Size: {self.board_rows} x {self.board_columns} cells\n"

        # Physical dimensions if available
        if "length" in self.board_config and "width" in self.board_config:
            legend += f"Physical Size: {self.board_config['length']} x {self.board_config['width']} x {self.board_config.get('thickness', 0.05)}\n"

        # Pattern and colors
        legend += f"Pattern: {'Checkered' if self.board_pattern else 'Uniform'}\n"
        legend += f"White Cell Color: RGBA{self.board_white_color}\n"
        legend += f"Black Cell Color: RGBA{self.board_black_color}\n\n"

        # Materials if available
        if "chessboard_material" in self.board_config:
            legend += "Materials:\n"
            legend += f"  - Board Frame: {self.board_config.get('chessboard_material', 'default')}\n"
            legend += f"  - White Squares: {self.board_config.get('square_w_material', 'default')}\n"
            legend += f"  - Black Squares: {self.board_config.get('square_b_material', 'default')}\n\n"

        # Append cell positions (sample of first 4 to keep it manageable)
        if "cell_positions" in self.board_config:
            legend += "Cell Positions (sample):\n"
            cell_positions = self.board_config["cell_positions"]

            # Get four sample positions if available
            samples = list(cell_positions.items())[:4] if len(cell_positions) > 4 else list(cell_positions.items())
            for cell_key, pos in samples:
                legend += f"  - {cell_key}: ({pos[0]}, {pos[1]})\n"

            if len(cell_positions) > 4:
                legend += f"  - ... and {len(cell_positions) - 4} more positions\n"

        return legend

    def get_pieces_legend_dict(self) -> dict[str, dict[str, Any]]:
        """
        Prepare pieces configuration as a structured dictionary.

        Returns:
            Dictionary containing formatted piece information
        """
        # The pieces_config dictionary is already well-structured
        # Just ensure all pieces have consistent keys

        pieces_legend = {}
        for piece_id, piece_data in self.pieces_config.items():
            piece_entry = piece_data.copy()

            # Rename keys for clarity if needed
            if "location" in piece_entry:
                piece_entry["board_position"] = piece_entry["location"]
                # remove location from piece_entry
                del piece_entry["location"]

            if "world_position" in piece_entry:
                piece_entry["image_position"] = piece_entry["world_position"]
                #remove world_position from piece_entry
                del piece_entry["world_position"]

            # if color is a tuple, convert it to a string
            if isinstance(piece_entry["color"], tuple):
                piece_entry["color"] = RGB_to_color(piece_entry["color"][:3])

            pieces_legend[piece_id] = piece_entry

        return pieces_legend

    def get_pieces_legend_text(self) -> str:
        """
        Generate a text representation of the pieces configuration.

        Returns:
            Formatted string containing pieces legend information
        """
        legend = "CHESS PIECES LEGEND\n"
        legend += "==================\n\n"

        # Group pieces by type
        piece_types = {}
        for piece_id, piece_data in self.pieces_config.items():
            piece_type = piece_data.get("type", "unknown")
            if piece_type not in piece_types:
                piece_types[piece_type] = []
            piece_types[piece_type].append((piece_id, piece_data))

        # Generate text for each piece type
        for piece_type, pieces in piece_types.items():
            legend += f"{piece_type.upper()} PIECES ({len(pieces)})\n"
            legend += "-" * (len(piece_type) + 9 + len(str(len(pieces)))) + "\n"

            for piece_id, piece_data in pieces:
                # Extract position information
                board_pos = piece_data.get("board_position", piece_data.get("location", "N/A"))
                world_pos = piece_data.get("world_position", "N/A")

                # Format positions nicely
                if isinstance(board_pos, tuple | list) and len(board_pos) >= 2:
                    board_pos = f"row {board_pos[0]}, col {board_pos[1]}"

                if isinstance(world_pos, tuple | list) and len(world_pos) >= 3:
                    world_pos = f"({world_pos[0]:.4f}, {world_pos[1]:.4f}, {world_pos[2]:.4f})"

                # Extract color
                color = piece_data.get("color", "N/A")
                if isinstance(color, tuple):
                    if len(color) >= 3:
                        # Display both RGB values and approximate color name
                        color_name = RGB_to_color(color[:3])
                        color_info = f"RGBA{color} ({color_name})"
                    else:
                        color_info = f"RGB{color}"
                else:
                    color_info = color

                # Use the original piece_id (e.g., KING_1) in uppercase
                legend += f"· {piece_id.upper()}:\n"
                legend += f"  - Board Position: {board_pos}\n"
                legend += f"  - World Position: {world_pos}\n"
                legend += f"  - Color: {color_info}\n"
                legend += f"  - Scale: {piece_data.get('scale', 'N/A')}\n"

                # Add any additional properties
                for key, value in piece_data.items():
                    if key not in ["type", "location", "world_position", "color", "scale"]:
                        legend += f"  - {key}: {value}\n"

                legend += "\n"

        return legend

    def get_full_legend_dict(self) -> dict[str, Any]:
        """
        Get a complete legend dictionary with both board and pieces information.

        Returns:
            Dictionary containing all legend information
        """
        board_legend = self.get_board_legend_dict()
        pieces_legend = self.get_pieces_legend_dict()

        # Add Camera Info (if available)
        if self.camera_config:
            board_legend["camera"] = {
                "final_distance": self.camera_config.get("final_distance"),
                "final_angle": self.camera_config.get("final_angle"),
                "final_horizontal_angle": self.camera_config.get("final_horizontal_angle")
            }

        # Prepare the final dictionary
        serializable_noise_config = {}
        if self.noise_config:
            for noise_type, settings in self.noise_config.items():
                if not isinstance(settings, dict):
                    # Handle rare case where a noise setting isn't a dict
                    serializable_noise_config[noise_type] = settings
                    continue

                serializable_settings = {}
                for key, value in settings.items():
                    # Check if the value is a Blender object (specifically Material)
                    if hasattr(value, 'bl_rna'): # A common way to check for Blender types
                        if hasattr(value, 'name'):
                             serializable_settings[key] = f"<Blender {value.rna_type.identifier}: {value.name}>"
                        else:
                             serializable_settings[key] = f"<Blender {value.rna_type.identifier}>"
                    # Check if value is serializable (basic types, lists, dicts)
                    elif isinstance(value, str | int | float | bool | list | dict | tuple) or value is None:
                        serializable_settings[key] = value
                    else:
                        # Represent non-serializable types as strings
                        serializable_settings[key] = f"<Non-serializable: {type(value).__name__}>"
                serializable_noise_config[noise_type] = serializable_settings

        return {
            "board": board_legend,
            "pieces": pieces_legend,
            # Add noise info if available
            "noise": serializable_noise_config # Use the cleaned version
        }

    def get_full_legend_text(self) -> str:
        """
        Get a complete text legend with both board and pieces information.

        Returns:
            Formatted string containing all legend information
        """
        board_text = self.get_board_legend_text()
        pieces_text = self.get_pieces_legend_text()

        # Helper function to format numbers or return string directly (defined at method scope)
        def format_num_or_str(value, format_spec=".3f"):
            if isinstance(value, int | float):
                return f"{value:{format_spec}}"
            return str(value) # Return as string if not number

        # Start combining text parts
        full_text = board_text + "\n" + pieces_text

        # Add Camera Info (if available)
        if self.camera_config:
            full_text += "\n--- Camera ---\n"
            full_text += f"  Final Distance: {format_num_or_str(self.camera_config.get('final_distance', 'N/A'))}\n"
            full_text += f"  Final Vertical Angle: {format_num_or_str(self.camera_config.get('final_angle', 'N/A'))} degrees\n"
            full_text += f"  Final Horizontal Angle: {format_num_or_str(self.camera_config.get('final_horizontal_angle', 'N/A'))} degrees\n"

            pos = self.camera_config.get('final_position', 'N/A')
            if isinstance(pos, list | tuple):
                x_str = format_num_or_str(pos[0])
                y_str = format_num_or_str(pos[1])
                z_str = format_num_or_str(pos[2])
                full_text += f"  Final Position (X,Y,Z): ({x_str}, {y_str}, {z_str})\n"
            else:
                full_text += f"  Final Position (X,Y,Z): {pos}\n"

        # Add Noise Info (if available)
        if self.noise_config:
            full_text += "\n--- Noise Settings ---\n"
            for noise_type, settings in self.noise_config.items():
                full_text += f"  {noise_type.capitalize()}:\n"
                if isinstance(settings, dict):
                    for key, value in settings.items():
                        # Use formatter for potentially numeric values like fstop
                        full_text += f"    - {key}: {format_num_or_str(value)}\n"
                else: # Handle cases where noise config might not be a dict (though unlikely)
                    full_text += f"    {settings}\n"
            full_text += "\n"

        return full_text


# Example usage
if __name__ == "__main__":
    # Example board config
    board_config = {
        "rows": 8,
        "columns": 8,
        "length": 0.7,
        "width": 0.7,
        "thickness": 0.05,
        "location": (0, 0, 0.9),
        "pattern": True,
        "square_w_color": (0.9, 0.9, 0.9, 1.0),
        "square_b_color": (0.1, 0.1, 0.1, 1.0),
    }

    # Example pieces config - more chess-like
    pieces_config = {
        "king_1": {
            "type": "king",
            "location": (0, 4),
            "world_position": (0.1, 0.5, 0.95),
            "color": (0.9, 0.9, 0.9, 1.0),  # White
            "scale": 0.08
        },
        "queen_1": {
            "type": "queen",
            "location": (0, 3),
            "world_position": (0.1, 0.4, 0.95),
            "color": (0.9, 0.9, 0.9, 1.0),  # White
            "scale": 0.08
        },
        "rook_1": {
            "type": "rook",
            "location": (0, 0),
            "world_position": (0.1, 0.1, 0.95),
            "color": (0.9, 0.9, 0.9, 1.0),  # White
            "scale": 0.08
        },
        "king_2": {
            "type": "king",
            "location": (7, 4),
            "world_position": (0.8, 0.5, 0.95),
            "color": (0.1, 0.1, 0.1, 1.0),  # Black
            "scale": 0.08
        },
        "queen_2": {
            "type": "queen",
            "location": (7, 3),
            "world_position": (0.8, 0.4, 0.95),
            "color": (0.1, 0.1, 0.1, 1.0),  # Black
            "scale": 0.08
        },
        "pawn_1": {
            "type": "pawn",
            "location": (1, 0),
            "world_position": (0.2, 0.1, 0.95),
            "color": (0.9, 0.9, 0.9, 1.0),  # White
            "scale": 0.07
        }
    }

    # Create legend generator
    legend_generator = config_to_legend(board_config, pieces_config)

    # Get board legend as dictionary and text
    board_dict = legend_generator.get_board_legend_dict()
    board_text = legend_generator.get_board_legend_text()

    # Get pieces legend as dictionary and text
    pieces_dict = legend_generator.get_pieces_legend_dict()
    pieces_text = legend_generator.get_pieces_legend_text()

    # Get full legend
    full_dict = legend_generator.get_full_legend_dict()
    full_text = legend_generator.get_full_legend_text()

    # Print text legends
    print(full_text)

    # Print a sample of the dictionary legend (first piece only)
    print("\nSample of dictionary legend:")
    if pieces_dict:
        first_piece_id = list(pieces_dict.keys())[0]
        print(f"{first_piece_id}: {pieces_dict[first_piece_id]}")

def generate_legend(
    board_config: dict[str, Any],
    pieces_config: dict[str, dict[str, Any]],
    camera_config: dict[str, Any] | None,
    noise_config: dict[str, Any] | None,
    txt_path: str,
    json_path: str
):
    """
    Generate text and JSON legend files from configuration dictionaries.

    Args:
        board_config: Configuration dictionary for the chess board
        pieces_config: Configuration dictionary for the chess pieces
        camera_config: Optional configuration dictionary for the camera's final state
        noise_config: Optional configuration dictionary for the final noise settings
        txt_path: Path to save the text legend file
        json_path: Path to save the JSON legend file
    """
    try:
        # Create legend generator instance, passing camera_config
        legend_generator = config_to_legend(board_config, pieces_config, camera_config, noise_config)

        # Generate text legend
        txt_legend = legend_generator.get_full_legend_text()

        # Save text legend to file
        with open(txt_path, 'w') as f:
            f.write(txt_legend)

        # Generate JSON legend
        json_legend = legend_generator.get_full_legend_dict()

        # Save JSON legend to file
        with open(json_path, 'w') as f:
            json.dump(json_legend, f, indent=2)
    except Exception as e:
        print(f"Error generating legend: {e}")
