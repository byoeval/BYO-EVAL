import bpy
import math
from typing import Dict, Any

from .base import ChessPiece
from ..config.models import PieceModel

class Queen(ChessPiece):
    """Chess queen piece implementation."""
    
    def create_geometry(self) -> None:
        """Create the queen geometry."""
        location = self.config.location
        scale = self.config.geometry.scale
        
        # Base of the queen
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=32,
            radius=0.3 * scale,
            depth=0.1 * scale,
            location=(location[0], location[1], location[2] + 0.05 * scale)
        )
        base = bpy.context.active_object
        base.name = f"{self.get_color_name()}Queen_Base"
        self.parts.append(base)
        
        # Body of the queen (tapered cylinder)
        bpy.ops.mesh.primitive_cone_add(
            vertices=32,
            radius1=0.25 * scale,
            radius2=0.2 * scale,
            depth=0.7 * scale,
            location=(location[0], location[1], location[2] + 0.45 * scale)
        )
        body = bpy.context.active_object
        body.name = f"{self.get_color_name()}Queen_Body"
        self.parts.append(body)
        
        # Crown base
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=32,
            radius=0.22 * scale,
            depth=0.15 * scale,
            location=(location[0], location[1], location[2] + 0.875 * scale)
        )
        crown_base = bpy.context.active_object
        crown_base.name = f"{self.get_color_name()}Queen_CrownBase"
        self.parts.append(crown_base)
        
        # Create crown points
        num_points = 8
        point_radius = 0.05 * scale
        point_height = 0.12 * scale
        crown_radius = 0.18 * scale
        
        for i in range(num_points):
            angle = (2 * math.pi * i) / num_points
            pos_x = location[0] + math.cos(angle) * crown_radius
            pos_y = location[1] + math.sin(angle) * crown_radius
            pos_z = location[2] + 0.95 * scale + point_height/2
            
            # Create a cone for each point
            bpy.ops.mesh.primitive_cone_add(
                vertices=16,
                radius1=point_radius,
                radius2=0,
                depth=point_height,
                location=(pos_x, pos_y, pos_z)
            )
            point = bpy.context.active_object
            point.name = f"{self.get_color_name()}Queen_Point_{i}"
            self.parts.append(point)
            
        # Add central orb
        bpy.ops.mesh.primitive_uv_sphere_add(
            radius=0.08 * scale,
            location=(location[0], location[1], location[2] + 1.0 * scale)
        )
        orb = bpy.context.active_object
        orb.name = f"{self.get_color_name()}Queen_Orb"
        self.parts.append(orb)


def create_queen(config: Dict[str, Any]) -> bpy.types.Object:
    """
    Create a chess queen piece from a configuration dictionary.
    
    Args:
        config: Dictionary containing piece configuration
        
    Returns:
        The created queen object
    """
    piece_config = PieceModel.from_dict(config)
    queen = Queen(piece_config)
    return queen.create() 