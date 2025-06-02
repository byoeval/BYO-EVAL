import bpy
from typing import Dict, Any
import math

from .base import ChessPiece
from ..config.models import PieceModel

class Bishop(ChessPiece):
    """Chess bishop piece implementation."""
    
    def create_geometry(self) -> None:
        """Create the bishop geometry."""
        location = self.config.location
        scale = self.config.geometry.scale
        
        # Base of the bishop
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=32,
            radius=0.3 * scale,
            depth=0.1 * scale,
            location=(location[0], location[1], location[2] + 0.05 * scale)
        )
        base = bpy.context.active_object
        base.name = f"{self.get_color_name()}Bishop_Base"
        self.parts.append(base)
        
        # Body of the bishop (tapered cylinder)
        bpy.ops.mesh.primitive_cone_add(
            vertices=32,
            radius1=0.25 * scale,
            radius2=0.15 * scale,
            depth=0.6 * scale,
            location=(location[0], location[1], location[2] + 0.4 * scale)
        )
        body = bpy.context.active_object
        body.name = f"{self.get_color_name()}Bishop_Body"
        self.parts.append(body)
        
        # Mitre base (wider at bottom)
        bpy.ops.mesh.primitive_cone_add(
            vertices=32,
            radius1=0.18 * scale,
            radius2=0.2 * scale,
            depth=0.2 * scale,
            location=(location[0], location[1], location[2] + 0.8 * scale)
        )
        mitre_base = bpy.context.active_object
        mitre_base.name = f"{self.get_color_name()}Bishop_MitreBase"
        self.parts.append(mitre_base)
        
        # Mitre top (pointed)
        bpy.ops.mesh.primitive_cone_add(
            vertices=32,
            radius1=0.2 * scale,
            radius2=0.02 * scale,
            depth=0.3 * scale,
            location=(location[0], location[1], location[2] + 1.05 * scale)
        )
        mitre_top = bpy.context.active_object
        mitre_top.name = f"{self.get_color_name()}Bishop_MitreTop"
        self.parts.append(mitre_top)
        
        # Add decorative band around mitre
        band_segments = 16
        band_radius = 0.21 * scale
        band_thickness = 0.03 * scale
        band_height = 0.85 * scale
        
        for i in range(band_segments):
            angle = (2 * math.pi * i) / band_segments
            pos_x = location[0] + math.cos(angle) * band_radius
            pos_y = location[1] + math.sin(angle) * band_radius
            
            bpy.ops.mesh.primitive_uv_sphere_add(
                segments=16,
                ring_count=8,
                radius=band_thickness,
                location=(pos_x, pos_y, location[2] + band_height)
            )
            band_part = bpy.context.active_object
            band_part.name = f"{self.get_color_name()}Bishop_Band_{i}"
            self.parts.append(band_part)
        
        # Add small sphere at the top
        bpy.ops.mesh.primitive_uv_sphere_add(
            segments=16,
            ring_count=8,
            radius=0.04 * scale,
            location=(location[0], location[1], location[2] + 1.25 * scale)
        )
        top_sphere = bpy.context.active_object
        top_sphere.name = f"{self.get_color_name()}Bishop_TopSphere"
        self.parts.append(top_sphere)


def create_bishop(config: Dict[str, Any]) -> bpy.types.Object:
    """
    Create a chess bishop piece from a configuration dictionary.
    
    Args:
        config: Dictionary containing piece configuration
        
    Returns:
        The created bishop object
    """
    piece_config = PieceModel.from_dict(config)
    bishop = Bishop(piece_config)
    return bishop.create() 