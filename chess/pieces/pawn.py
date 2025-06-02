from typing import Any

import bpy

from ..config.models import PieceModel
from .base import ChessPiece


class Pawn(ChessPiece):
    """Chess pawn piece implementation."""

    def create_geometry(self) -> None:
        """Create the pawn geometry."""
        location = self.config.location
        scale = self.config.geometry.scale

        # Base of the pawn
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=32,
            radius=0.25 * scale,
            depth=0.1 * scale,
            location=(location[0], location[1], location[2] + 0.05 * scale)
        )
        base = bpy.context.active_object
        base.name = f"{self.get_color_name()}Pawn_Base"
        self.parts.append(base)

        # Body of the pawn
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=32,
            radius=0.2 * scale,
            depth=0.3 * scale,
            location=(location[0], location[1], location[2] + 0.25 * scale)
        )
        body = bpy.context.active_object
        body.name = f"{self.get_color_name()}Pawn_Body"
        self.parts.append(body)

        # Neck of the pawn
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=32,
            radius=0.15 * scale,
            depth=0.1 * scale,
            location=(location[0], location[1], location[2] + 0.45 * scale)
        )
        neck = bpy.context.active_object
        neck.name = f"{self.get_color_name()}Pawn_Neck"
        self.parts.append(neck)

        # Head of the pawn (sphere)
        bpy.ops.mesh.primitive_uv_sphere_add(
            segments=32,
            ring_count=16,
            radius=0.18 * scale,
            location=(location[0], location[1], location[2] + 0.6 * scale)
        )
        head = bpy.context.active_object
        head.name = f"{self.get_color_name()}Pawn_Head"
        self.parts.append(head)


def create_pawn(config: dict[str, Any]) -> bpy.types.Object:
    """
    Create a chess pawn piece from a configuration dictionary.

    Args:
        config: Dictionary containing piece configuration

    Returns:
        The created pawn object
    """
    piece_config = PieceModel.from_dict(config)
    pawn = Pawn(piece_config)
    return pawn.create()
