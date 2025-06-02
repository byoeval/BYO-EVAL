import random
from typing import Any

import bpy

from .config.models import BoardModel, MaterialModel


class ChessBoard:
    """
    A class to manage the chess board creation and state.

    This class handles:
    - Board creation with customizable dimensions and materials
    - Square pattern generation (traditional or random)
    - Cell position tracking
    - Material management
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize the chess board with the given configuration.

        Args:
            config: Configuration for the chess board, either as a BoardModel object
                   or a dictionary that can be converted to a BoardModel
        """
        # Convert dictionary to BoardModel if needed

        self.config = BoardModel.from_dict(config)
        self.cell_positions: dict[str, tuple[float, float]] = {}

    def _validate_config(self) -> None:
        """Validate board configuration parameters."""
        def is_power_of_two(n: int) -> bool:
            return n > 0 and (n & (n - 1)) == 0

        if not is_power_of_two(self.config.rows):
            self.config.rows = 2 ** max(1, int(self.config.rows).bit_length() - 1)
            print(f"Rows adjusted to nearest power of 2: {self.config.rows}")

        if not is_power_of_two(self.config.columns):
            self.config.columns = 2 ** max(1, int(self.config.columns).bit_length() - 1)
            print(f"Columns adjusted to nearest power of 2: {self.config.columns}")

    def _create_material(self, material_config: MaterialModel) -> bpy.types.Material:
        """Create a Blender material from configuration."""
        if material_config.custom_material:
            return material_config.custom_material

        material = bpy.data.materials.new(name=material_config.material_name or "ChessMaterial")
        material.use_nodes = True
        nodes = material.node_tree.nodes
        bsdf = nodes["Principled BSDF"]

        if isinstance(material_config.color, tuple):
            bsdf.inputs["Base Color"].default_value = material_config.color
        else:
            # Handle string color names if needed
            color_map = {
                "white": (0.9, 0.9, 0.9, 1.0),
                "black": (0.1, 0.1, 0.1, 1.0),
                # Add more color mappings as needed
            }
            bsdf.inputs["Base Color"].default_value = color_map.get(
                material_config.color.lower(),
                (0.8, 0.8, 0.8, 1.0)  # Default gray
            )

        bsdf.inputs["Roughness"].default_value = material_config.roughness
        return material

    def _generate_pattern(self) -> list[list[int]]:
        """Generate the board pattern (traditional checkerboard or random)."""
        pattern = []

        if self.config.random_pattern:
            if self.config.pattern_seed is not None:
                random.seed(self.config.pattern_seed)

            # Create a random symmetric pattern
            quarter_rows = self.config.rows // 2
            quarter_cols = self.config.columns // 2
            quarter_pattern = [
                [random.randint(0, 1) for _ in range(quarter_cols)]
                for _ in range(quarter_rows)
            ]

            # Expand to full pattern with symmetry
            for r in range(self.config.rows):
                row_pattern = []
                for c in range(self.config.columns):
                    qr = min(r, self.config.rows - r - 1) % quarter_rows
                    qc = min(c, self.config.columns - c - 1) % quarter_cols
                    cell = quarter_pattern[qr][qc]
                    row_pattern.append(cell)
                pattern.append(row_pattern)
        else:
            # Traditional checkerboard pattern
            pattern = [
                [(r + c) % 2 for c in range(self.config.columns)]
                for r in range(self.config.rows)
            ]

        return pattern

    def create(self) -> tuple[bpy.types.Object, bpy.types.Collection]:
        """
        Create the chess board in the scene.

        Returns:
            Tuple containing:
            - The board frame object
            - The squares collection
        """
        # Create or get collections
        if "Chessboard" not in bpy.data.collections:
            chess_collection = bpy.data.collections.new("Chessboard")
            bpy.context.scene.collection.children.link(chess_collection)
        else:
            chess_collection = bpy.data.collections["Chessboard"]

        if "ChessboardSquares" not in bpy.data.collections:
            squares_collection = bpy.data.collections.new("ChessboardSquares")
            bpy.context.scene.collection.children.link(squares_collection)
        else:
            squares_collection = bpy.data.collections["ChessboardSquares"]

        # Create board frame
        bpy.ops.mesh.primitive_cube_add(
            size=1,
            location=(
                self.config.location[0],
                self.config.location[1],
                self.config.location[2] + self.config.thickness / 2
            )
        )
        board = bpy.context.active_object
        board.name = "ChessboardFrame"
        board.scale.x = self.config.width + self.config.border_width
        board.scale.y = self.config.length + self.config.border_width
        board.scale.z = self.config.thickness

        # Create and apply materials using get_material_model method
        board_material = self._create_material(self.config.get_material_model('board'))
        white_material = self._create_material(self.config.get_material_model('white'))
        black_material = self._create_material(self.config.get_material_model('black'))

        if board.data.materials:
            board.data.materials[0] = board_material
        else:
            board.data.materials.append(board_material)

        # Generate board pattern
        pattern = self._generate_pattern()

        # Calculate square dimensions
        square_width = self.config.width / self.config.columns
        square_length = self.config.length / self.config.rows

        # Create squares
        for r in range(self.config.rows):
            for c in range(self.config.columns):
                # Calculate position
                pos_x = (self.config.location[0] - self.config.width/2 +
                        square_width/2 + c * square_width)
                pos_y = (self.config.location[1] - self.config.length/2 +
                        square_length/2 + r * square_length)
                pos_z = self.config.location[2] + self.config.thickness

                # Create square
                bpy.ops.mesh.primitive_cube_add(
                    size=1,
                    location=(pos_x, pos_y, pos_z + 0.001)  # Slight offset to avoid z-fighting
                )
                square = bpy.context.active_object
                square.name = f"ChessSquare_row_{r}_col_{c}"
                square.scale.x = square_width * 0.95
                square.scale.y = square_length * 0.95
                square.scale.z = 0.002

                # Store position
                self.cell_positions[f"cell_row_{r}_col_{c}"] = (pos_x, pos_y)

                # Apply material - FIXED: Now 0 means black and 1 means white
                material = black_material if pattern[r][c] == 0 else white_material
                if square.data.materials:
                    square.data.materials[0] = material
                else:
                    square.data.materials.append(material)

                # Add to squares collection
                if square.name in bpy.context.scene.collection.objects:
                    bpy.context.scene.collection.objects.unlink(square)
                squares_collection.objects.link(square)

        # Store references
        self.board_object = board
        self.squares_collection = squares_collection

        return board, squares_collection

    def get_cell_position(self, row: int, col: int) -> tuple[float, float] | None:
        """Get the world position of a cell by its row and column."""
        cell_key = f"cell_row_{row}_col_{col}"
        return self.cell_positions.get(cell_key)

    def get_cell_positions(self) -> dict[str, tuple[float, float]]:
        """Get all cell positions."""
        return self.cell_positions.copy()

    def cleanup(self) -> None:
        """Remove the board and squares from the scene."""
        if self.squares_collection:
            for obj in self.squares_collection.objects:
                bpy.data.objects.remove(obj, do_unlink=True)
            bpy.data.collections.remove(self.squares_collection)

        if self.board_object:
            bpy.data.objects.remove(self.board_object, do_unlink=True)
