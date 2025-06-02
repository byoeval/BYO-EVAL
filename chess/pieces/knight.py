import bpy
from typing import Dict, Any

from .base import ChessPiece
from ..config.models import PieceModel

class Knight(ChessPiece):
    """Chess knight piece implementation."""
    
    def create_geometry(self) -> None:
        """Create the knight geometry."""
        location = self.config.location
        scale = self.config.geometry.scale
        
        # Base of the knight
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=32,
            radius=0.25 * scale,
            depth=0.1 * scale,
            location=(location[0], location[1], location[2] + 0.05 * scale)
        )
        base = bpy.context.active_object
        base.name = f"{self.get_color_name()}Knight_Base"
        self.parts.append(base)
        
        # Body of the knight
        bpy.ops.mesh.primitive_cube_add(
            size=0.4 * scale,
            location=(location[0], location[1], location[2] + 0.3 * scale)
        )
        body = bpy.context.active_object
        body.name = f"{self.get_color_name()}Knight_Body"
        self.parts.append(body)
        
        # Head of the knight (horse head shape)
        bpy.ops.mesh.primitive_uv_sphere_add(
            segments=32,
            ring_count=16,
            radius=0.2 * scale,
            location=(location[0], location[1], location[2] + 0.5 * scale)
        )
        head = bpy.context.active_object
        head.name = f"{self.get_color_name()}Knight_Head"
        self.parts.append(head)


def create_knight(config: Dict[str, Any]) -> bpy.types.Object:
    """
    Create a chess knight piece from a configuration dictionary.
    
    Args:
        config: Dictionary containing piece configuration
        
    Returns:
        The created knight object
    """
    piece_config = PieceModel.from_dict(config)
    knight = Knight(piece_config)
    return knight.create() 