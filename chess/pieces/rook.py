import math
from typing import Any

import bpy

from ..config.models import PieceModel
from .base import ChessPiece


class Rook(ChessPiece):
    """Chess rook (castle) piece implementation."""

    def create_geometry(self) -> None:
        """Create the rook geometry."""
        location = self.config.location
        scale = self.config.geometry.scale

        # Base of the rook
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=32,
            radius=0.3 * scale,
            depth=0.1 * scale,
            location=(location[0], location[1], location[2] + 0.05 * scale)
        )
        base = bpy.context.active_object
        base.name = f"{self.get_color_name()}Rook_Base"
        self.parts.append(base)

        # Body of the rook
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=32,
            radius=0.25 * scale,
            depth=0.6 * scale,
            location=(location[0], location[1], location[2] + 0.4 * scale)
        )
        body = bpy.context.active_object
        body.name = f"{self.get_color_name()}Rook_Body"
        self.parts.append(body)

        # Top of the rook (wider cylinder)
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=32,
            radius=0.28 * scale,
            depth=0.15 * scale,
            location=(location[0], location[1], location[2] + 0.775 * scale)
        )
        top = bpy.context.active_object
        top.name = f"{self.get_color_name()}Rook_Top"
        self.parts.append(top)

        # Create battlements (crenellations)
        num_battlements = 4
        battlement_width = 0.1 * scale
        battlement_height = 0.08 * scale
        top_radius = 0.28 * scale

        # Create the battlements at the corners
        for i in range(num_battlements):
            angle = (2 * math.pi * i) / num_battlements + (math.pi / 4)  # Offset by 45 degrees
            pos_x = location[0] + math.cos(angle) * (top_radius - battlement_width/2)
            pos_y = location[1] + math.sin(angle) * (top_radius - battlement_width/2)
            pos_z = location[2] + 0.85 * scale + battlement_height/2

            bpy.ops.mesh.primitive_cube_add(
                size=battlement_width,
                location=(pos_x, pos_y, pos_z)
            )
            battlement = bpy.context.active_object
            battlement.name = f"{self.get_color_name()}Rook_Battlement_{i}"
            battlement.scale = (1.0, 1.0, battlement_height / battlement_width)
            self.parts.append(battlement)


def create_rook(config: dict[str, Any]) -> bpy.types.Object:
    """
    Create a chess rook piece from a configuration dictionary.

    Args:
        config: Dictionary containing piece configuration

    Returns:
        The created rook object
    """
    piece_config = PieceModel.from_dict(config)
    rook = Rook(piece_config)
    return rook.create()
