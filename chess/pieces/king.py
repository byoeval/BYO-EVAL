import bpy
import math
from typing import Dict, Any

from .base import ChessPiece
from ..config.models import PieceModel

class King(ChessPiece):
    """Chess king piece implementation."""
    
    def create_geometry(self) -> None:
        """Create the king geometry."""
        location = self.config.location
        scale = self.config.geometry.scale
        
        # Base of the king - starts at the specified z height
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=32,
            radius=0.3 * scale,
            depth=0.1 * scale,
            location=location  # Use exact location for base
        )
        base = bpy.context.active_object
        base.name = f"{self.get_color_name()}King_Base"
        self.parts.append(base)
        
        # Body of the king (tapered cylinder)
        body_height = 0.8 * scale
        bpy.ops.mesh.primitive_cone_add(
            vertices=32,
            radius1=0.25 * scale,
            radius2=0.22 * scale,
            depth=body_height,
            location=(
                location[0],
                location[1],
                location[2] + (body_height/2)  # Center the body above base
            )
        )
        body = bpy.context.active_object
        body.name = f"{self.get_color_name()}King_Body"
        self.parts.append(body)
        
        # Crown base
        crown_height = 0.15 * scale
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=32,
            radius=0.24 * scale,
            depth=crown_height,
            location=(
                location[0],
                location[1],
                location[2] + body_height + (crown_height/2)
            )
        )
        crown_base = bpy.context.active_object
        crown_base.name = f"{self.get_color_name()}King_CrownBase"
        self.parts.append(crown_base)
        
        # Create crown points
        num_points = 5
        point_radius = 0.06 * scale
        point_height = 0.1 * scale
        crown_radius = 0.2 * scale
        points_z = location[2] + body_height + crown_height + point_height/2
        
        for i in range(num_points):
            angle = (2 * math.pi * i) / num_points
            pos_x = location[0] + math.cos(angle) * crown_radius
            pos_y = location[1] + math.sin(angle) * crown_radius
            
            # Create a cone for each point
            bpy.ops.mesh.primitive_cone_add(
                vertices=16,
                radius1=point_radius,
                radius2=0,
                depth=point_height,
                location=(pos_x, pos_y, points_z)
            )
            point = bpy.context.active_object
            point.name = f"{self.get_color_name()}King_Point_{i}"
            self.parts.append(point)
        
        # Create cross
        cross_base_z = points_z + point_height/2
        cross_thickness = 0.06 * scale
        
        # Vertical part
        bpy.ops.mesh.primitive_cube_add(
            size=cross_thickness,
            location=(
                location[0],
                location[1],
                cross_base_z + 0.1 * scale
            )
        )
        cross_vert = bpy.context.active_object
        cross_vert.name = f"{self.get_color_name()}King_CrossVert"
        cross_vert.scale = (1.0, 1.0, 2.0)
        self.parts.append(cross_vert)
        
        # Horizontal part
        bpy.ops.mesh.primitive_cube_add(
            size=cross_thickness,
            location=(
                location[0],
                location[1],
                cross_base_z + 0.15 * scale
            )
        )
        cross_horz = bpy.context.active_object
        cross_horz.name = f"{self.get_color_name()}King_CrossHorz"
        cross_horz.scale = (1.8, 1.0, 0.8)
        self.parts.append(cross_horz)


def create_king(config: Dict[str, Any]) -> bpy.types.Object:
    """
    Create a chess king piece from a configuration dictionary.
    
    Args:
        config: Dictionary containing piece configuration
        
    Returns:
        The created king object
    """
    piece_config = PieceModel.from_dict(config)
    king = King(piece_config)
    return king.create() 