"""Chess pieces loaded from oldschool blend files."""

import os
import bpy
from typing import Dict, Any
from pathlib import Path

from chess.config.models import PieceModel
from chess.pieces.base import ChessPiece
from utils.blender_utils import extract_object_from_blend

class OldSchoolPiece(ChessPiece):
    """Base class for chess pieces loaded from the oldschool blend file."""
    
    piece_scale = 1.0  # Default scale factor multiplier
    
    def __init__(self, config: PieceModel, object_name: str):
        """
        Initialize a chess piece that loads from the oldschool blend file.
        
        Args:
            config: Configuration for the piece
            object_name: Name of the object in the blend file
        """
        super().__init__(config)
        # Get the project root directory using pathlib for more reliable path handling
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent.parent.absolute()
        self.blend_path = str(project_root / "blend_files" / "chess" / "chess_oldschool.blend")
        self.object_name = object_name
        self._extracted_object = None

    def create_geometry(self) -> None:
        """Load geometry from blend file instead of creating it."""
        
        if not self._extracted_object:
            self._extracted_object = extract_object_from_blend(
                self.blend_path,
                self.object_name,
                save=False
            )
            
        if not self._extracted_object:
            raise RuntimeError(f"Failed to load {self.object_name} from {self.blend_path}")
            
        # Create a copy of the object
        obj_copy = self._extracted_object.copy()
        obj_copy.data = self._extracted_object.data.copy()
        
        # Link to scene
        bpy.context.scene.collection.objects.link(obj_copy)
        
        # Add to parts list
        self.parts.append(obj_copy)
        
        # Set location
        obj_copy.location = self.config.location
        
        # Apply scale
        base_scale = self.config.geometry.scale
        final_scale = base_scale * self.piece_scale
        obj_copy.scale = (final_scale, final_scale, final_scale)
        
        # Create and assign material
        mat_name = f"{self.object_name}_material"
        mat = bpy.data.materials.new(name=mat_name)
        mat.use_nodes = True
        
        # Set up material nodes
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        
        # Clear existing nodes
        nodes.clear()
        
        # Create Principled BSDF node
        principled = nodes.new('ShaderNodeBsdfPrincipled')
        principled.location = (0, 0)
        
        # Create Material Output node
        output = nodes.new('ShaderNodeOutputMaterial')
        output.location = (300, 0)
        
        # Link nodes
        links.new(principled.outputs['BSDF'], output.inputs['Surface'])
        
        # Set material color and properties
        color = self.config.material.color
        if isinstance(color, tuple) and len(color) == 4:
            principled.inputs['Base Color'].default_value = color
        else:
            # Convert string color to RGBA
            if isinstance(color, str):
                if color.lower() == 'white':
                    color = (0.9, 0.9, 0.9, 1.0)
                else:  # black or any other color
                    color = (0.2, 0.2, 0.2, 1.0)
            else:
                color = (0.8, 0.8, 0.8, 1.0)
            principled.inputs['Base Color'].default_value = color
            
        principled.inputs['Metallic'].default_value = 0.0
        principled.inputs['Roughness'].default_value = self.config.material.roughness if hasattr(self.config.material, 'roughness') else 0.3

        # Assign material
        obj_copy.data.materials.clear()  # Remove existing materials
        obj_copy.data.materials.append(mat)
        
    def apply_material(self) -> None:
        """Override apply_material to do nothing since we handle materials in create_geometry."""
        pass


class Pawn(OldSchoolPiece):
    """Chess pawn piece from oldschool set."""
    
    piece_scale = 1.6  # Increase default pawn scale by 30%
    
    def __init__(self, config: PieceModel):
        """Initialize a pawn piece."""
        super().__init__(config, "Pawn.001")  # Using first pawn as template


class Rook(OldSchoolPiece):
    """Chess rook (elephant) piece from oldschool set."""
    
    def __init__(self, config: PieceModel):
        """Initialize a rook piece."""
        super().__init__(config, "Elephant.001")  # Using elephant as rook


class Knight(OldSchoolPiece):
    """Chess knight (plane) piece from oldschool set."""
    
    def __init__(self, config: PieceModel):
        """Initialize a knight piece."""
        super().__init__(config, "Plane.002")  # Using plane as knight


class Bishop(OldSchoolPiece):
    """Chess bishop (camal) piece from oldschool set."""
    
    def __init__(self, config: PieceModel):
        """Initialize a bishop piece."""
        super().__init__(config, "Camal.001")  # Using camal as bishop


class Queen(OldSchoolPiece):
    """Chess queen piece from oldschool set."""
    
    def __init__(self, config: PieceModel):
        """Initialize a queen piece."""
        super().__init__(config, "Queen.001")  # Using first queen as template


class King(OldSchoolPiece):
    """Chess king piece from oldschool set."""
    
    def __init__(self, config: PieceModel):
        """Initialize a king piece."""
        super().__init__(config, "King.001")  # Using first king as template


# Factory functions for creating pieces

def create_pawn(config: Dict[str, Any]) -> bpy.types.Object:
    """Create a chess pawn piece from a configuration dictionary."""
    piece_config = PieceModel.from_dict(config)
    pawn = Pawn(piece_config)
    return pawn.create()


def create_rook(config: Dict[str, Any]) -> bpy.types.Object:
    """Create a chess rook piece from a configuration dictionary."""
    piece_config = PieceModel.from_dict(config)
    rook = Rook(piece_config)
    return rook.create()


def create_knight(config: Dict[str, Any]) -> bpy.types.Object:
    """Create a chess knight piece from a configuration dictionary."""
    piece_config = PieceModel.from_dict(config)
    knight = Knight(piece_config)
    return knight.create()


def create_bishop(config: Dict[str, Any]) -> bpy.types.Object:
    """Create a chess bishop piece from a configuration dictionary."""
    piece_config = PieceModel.from_dict(config)
    bishop = Bishop(piece_config)
    return bishop.create()


def create_queen(config: Dict[str, Any]) -> bpy.types.Object:
    """Create a chess queen piece from a configuration dictionary."""
    piece_config = PieceModel.from_dict(config)
    queen = Queen(piece_config)
    return queen.create()


def create_king(config: Dict[str, Any]) -> bpy.types.Object:
    """Create a chess king piece from a configuration dictionary."""
    piece_config = PieceModel.from_dict(config)
    king = King(piece_config)
    return king.create() 