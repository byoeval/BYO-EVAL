import math
import random
from abc import ABC, abstractmethod

import bpy

from chess.config.models import PieceModel


class ChessPiece(ABC):
    """Abstract base class for chess pieces."""

    def __init__(self, config: PieceModel):
        """
        Initialize a chess piece.

        Args:
            config: Configuration for the piece
        """
        self.config = config
        self.parts: list[bpy.types.Object] = []
        self.piece_object: bpy.types.Object | None = None

        # Ensure chess pieces collection exists
        if "ChessPieces" not in bpy.data.collections:
            self.collection = bpy.data.collections.new("ChessPieces")
            bpy.context.scene.collection.children.link(self.collection)
        else:
            self.collection = bpy.data.collections["ChessPieces"]

        # Set random seed if provided
        if self.config.geometry.seed is not None:
            random.seed(self.config.geometry.seed)

    def create(self) -> bpy.types.Object:
        """Create the chess piece and return the final object."""
        # Create piece geometry
        self.create_geometry()

        # Join all parts
        self.join_parts()

        # Apply material
        self.apply_material()

        # Apply random rotation if requested
        self.apply_rotation()

        # Add to collection
        self.add_to_collection()

        return self.piece_object

    @abstractmethod
    def create_geometry(self) -> None:
        """Create the piece-specific geometry. Must be implemented by subclasses."""

    def join_parts(self) -> None:
        """Join all parts into a single object."""
        if not self.parts:
            raise ValueError("No parts to join")

        bpy.ops.object.select_all(action='DESELECT')
        for part in self.parts:
            part.select_set(True)

        bpy.context.view_layer.objects.active = self.parts[0]
        bpy.ops.object.join()

        self.piece_object = bpy.context.active_object
        self.piece_object.name = f"{self.get_color_name()}{self.config.piece_type.capitalize()}"

    def apply_material(self) -> None:
        """Create and apply material to the piece."""
        if not self.piece_object:
            raise ValueError("No piece object to apply material to")

        material_config = self.config.material

        # Use provided material or create new one
        if material_config.custom_material:
            material = material_config.custom_material
        else:
            material_name = material_config.material_name or f"{self.get_color_name()}{self.config.piece_type.capitalize()}Material"
            material = bpy.data.materials.new(name=material_name)
            material.use_nodes = True
            nodes = material.node_tree.nodes
            bsdf = nodes["Principled BSDF"]
            bsdf.inputs["Base Color"].default_value = self.get_color_value()
            bsdf.inputs["Roughness"].default_value = material_config.roughness

        # Apply material
        if self.piece_object.data.materials:
            self.piece_object.data.materials[0] = material
        else:
            self.piece_object.data.materials.append(material)

    def apply_rotation(self) -> None:
        """Apply random rotation if configured."""
        if not self.piece_object:
            return

        if self.config.geometry.random_rotation:
            max_angle_rad = math.radians(self.config.geometry.max_rotation_angle)
            self.piece_object.rotation_euler.z = random.uniform(-max_angle_rad, max_angle_rad)

    def add_to_collection(self) -> None:
        """Add the piece to the chess pieces collection."""
        if not self.piece_object:
            return

        if self.piece_object.name in bpy.context.scene.collection.objects:
            bpy.context.scene.collection.objects.unlink(self.piece_object)
        self.collection.objects.link(self.piece_object)

    def get_color_name(self) -> str:
        """Get the color name for the piece."""
        if isinstance(self.config.material.color, str):
            return self.config.material.color.capitalize()
        else:
            # Determine if it's closer to white or black for naming
            brightness = sum(self.config.material.color[:3]) / 3
            return "Dark" if brightness < 0.5 else "Light"

    def get_color_value(self) -> tuple:
        """Get the RGBA color value for the piece."""
        if isinstance(self.config.material.color, str):
            return (0.9, 0.9, 0.85, 1.0) if self.config.material.color.lower() == "white" else (0.1, 0.1, 0.12, 1.0)
        return self.config.material.color
