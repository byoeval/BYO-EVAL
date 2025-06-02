import abc
import logging
import math
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

import bpy

from utils.blender_utils import extract_object_from_blend

logger = logging.getLogger(__name__)

class PieceFactory(abc.ABC):
    """Abstract base class for creating chess pieces."""

    @abc.abstractmethod
    def create_piece(self, piece_type: str, config: dict[str, Any]) -> bpy.types.Object | None:
        """
        Creates a chess piece object based on its type and configuration.

        Args:
            piece_type: The type of the piece (e.g., 'pawn', 'king'). Lowercase.
            config: A dictionary containing configuration parameters for the piece
                    (e.g., world location, scale, material properties).
                    The 'location' key is expected to be the final world coordinates.

        Returns:
            A Blender object representing the created piece, or None if creation fails.
        """

    def _get_creator_func(self, piece_type: str) -> Callable | None:
        """Helper method to dynamically get the creation function based on type. Should be overridden by subclasses."""
        logger.warning(f"_get_creator_func not implemented for {self.__class__.__name__}")
        return None

    def _call_creator_func(self, creator_func: Callable, config: dict[str, Any]) -> bpy.types.Object | None:
        """Helper to call the creator function with error handling."""
        try:
            piece = creator_func(config)
            if piece is not None and isinstance(piece, bpy.types.Object):
                return piece
            else:
                logger.warning(f"Creator function {creator_func.__name__} did not return a valid Blender object.")
                return None
        except Exception as e:
            logger.error(f"Error calling piece creator function {creator_func.__name__}: {e}", exc_info=True)
            return None

# --- Concrete Factory Implementations ---

class DefaultPieceFactory(PieceFactory):
    """Creates pieces using the standard homemade generation logic from chess.pieces.*."""

    def _get_creator_func(self, piece_type: str) -> Callable | None:
        try:
            if piece_type == "pawn":
                from chess.pieces.pawn import create_pawn as func
            elif piece_type == "rook":
                from chess.pieces.rook import create_rook as func
            elif piece_type == "knight":
                from chess.pieces.knight import create_knight as func
            elif piece_type == "bishop":
                from chess.pieces.bishop import create_bishop as func
            elif piece_type == "queen":
                from chess.pieces.queen import create_queen as func
            elif piece_type == "king":
                from chess.pieces.king import create_king as func
            else:
                logger.warning(f"DefaultPieceFactory: Unknown piece type '{piece_type}'")
                return None
            return func
        except ImportError as e:
            logger.error(f"DefaultPieceFactory: Failed to import creation function for '{piece_type}': {e}")
            return None

    def create_piece(self, piece_type: str, config: dict[str, Any]) -> bpy.types.Object | None:
        creator_func = self._get_creator_func(piece_type.lower())
        if creator_func:
            return self._call_creator_func(creator_func, config)
        return None


class OldSchoolPieceFactory(PieceFactory):
    """
    Creates pieces using logic from chess.pieces_from_blend.old_school.pieces,
    with geometry caching for performance.
    """

    # Cache for extracted mesh data (piece_type -> mesh)
    _geometry_cache: dict[str, bpy.types.Mesh] = {}
    # Cache for extracted template objects (to avoid re-extraction within _get_or_extract_mesh)
    _extracted_template_objects: dict[str, bpy.types.Object] = {}

    # Mapping from piece type to object name in the blend file
    _piece_type_to_blend_object_map: dict[str, str] = {
        "pawn": "Pawn.016",
        "rook": "Elephant.001", # Using elephant as rook
        "knight": "Plane.002",   # Using plane as knight
        "bishop": "Camal.001",  # Using camal as bishop
        "queen": "Queen.001",
        "king": "King.001",
    }

    # Specific scaling factors for each piece type
    _piece_type_to_scale_map: dict[str, float] = {
        "pawn": 1.7,
        "rook": 1.2,
        "knight": 1.2,
        "bishop": 1.4,
        "queen": 1.3,
        "king": 1.4,
    }

    def __init__(self):
        """Initialize the factory and clear the cache for this instance."""
        # Clear the static cache when a new factory is potentially created (e.g., for a new image)
        # Note: This assumes a new factory instance is created per top-level generation task (e.g., per image).
        # If the factory were a true singleton across multiple images, this clearing logic would be wrong.
        OldSchoolPieceFactory._geometry_cache.clear()
        OldSchoolPieceFactory._extracted_template_objects.clear()
        logger.debug("OldSchoolPieceFactory initialized, geometry cache cleared.")

        # Get the project root directory using pathlib for more reliable path handling
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent.absolute()
        self.blend_path = str(project_root / "blend_files" / "chess" / "chess_oldschool.blend")


    def _get_or_extract_mesh(self, piece_type: str) -> bpy.types.Mesh | None:
        """
        Retrieves mesh data from cache or extracts it from the blend file.

        Args:
            piece_type: The type of piece (e.g., 'pawn').

        Returns:
            The mesh data, or None if extraction fails.
        """
        if piece_type in OldSchoolPieceFactory._geometry_cache:
            logger.debug(f"Cache hit for mesh: {piece_type}")
            return OldSchoolPieceFactory._geometry_cache[piece_type]

        logger.debug(f"Cache miss for mesh: {piece_type}. Extracting...")
        object_name = self._piece_type_to_blend_object_map.get(piece_type)
        if not object_name:
            logger.warning(f"OldSchoolPieceFactory: Unknown piece type '{piece_type}' in mapping.")
            return None

        # Check if we already extracted the template object to avoid redundant blend file access
        if object_name in OldSchoolPieceFactory._extracted_template_objects:
            template_object = OldSchoolPieceFactory._extracted_template_objects[object_name]
            logger.debug(f"Reusing already extracted template object '{object_name}' for mesh data.")
        else:
            try:
                template_object = extract_object_from_blend(
                    self.blend_path,
                    object_name,
                    save=False  # Don't save the template itself
                )
                if template_object:
                    # Store the template object temporarily
                    OldSchoolPieceFactory._extracted_template_objects[object_name] = template_object
                    logger.debug(f"Extracted template object '{object_name}' from blend file.")
                else:
                     logger.error(f"Failed to extract template object '{object_name}' from {self.blend_path}")
                     return None
            except Exception as e:
                logger.error(f"Error extracting '{object_name}' from {self.blend_path}: {e}", exc_info=True)
                return None

        if template_object and template_object.data:
            # Important: Cache a COPY of the mesh data
            mesh_copy = template_object.data.copy()
            OldSchoolPieceFactory._geometry_cache[piece_type] = mesh_copy
            logger.info(f"Cached mesh data for: {piece_type}")

            # Clean up the extracted template object from the scene after caching its data
            # Ensure it's actually in the scene's objects before trying to remove
            if template_object.name in bpy.data.objects:
                 bpy.data.objects.remove(template_object, do_unlink=True)
                 logger.debug(f"Removed temporary template object '{object_name}' after caching mesh.")
            # Also remove from our temporary dict
            del OldSchoolPieceFactory._extracted_template_objects[object_name]

            return mesh_copy
        else:
            logger.warning(f"Extracted template object '{object_name}' has no mesh data.")
            # Clean up if extraction failed partially
            if template_object and template_object.name in bpy.data.objects:
                 bpy.data.objects.remove(template_object, do_unlink=True)
            if object_name in OldSchoolPieceFactory._extracted_template_objects:
                 del OldSchoolPieceFactory._extracted_template_objects[object_name]
            return None

    def _create_material(self, base_name: str, config: dict[str, Any]) -> bpy.types.Material:
        """Creates a new material based on the piece config."""
        mat_name = f"{base_name}_{config.get('color', 'default')}_{random.random():.4f}" # Add random suffix for uniqueness
        mat = bpy.data.materials.new(name=mat_name)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()

        principled = nodes.new('ShaderNodeBsdfPrincipled')
        principled.location = (0, 0)
        output = nodes.new('ShaderNodeOutputMaterial')
        output.location = (300, 0)
        links.new(principled.outputs['BSDF'], output.inputs['Surface'])

        # Color
        color_value = config.get('color', 'white') # Default to white if not specified
        if isinstance(color_value, str):
            color_value = color_value.lower()
            if color_value == 'white':
                rgba = (0.9, 0.9, 0.9, 1.0)
            elif color_value == 'black':
                rgba = (0.2, 0.2, 0.2, 1.0)
            else: # Default fallback
                rgba = (0.8, 0.8, 0.8, 1.0)
        elif isinstance(color_value, list | tuple) and len(color_value) == 4:
            rgba = tuple(color_value)
        else: # Default fallback
            rgba = (0.8, 0.8, 0.8, 1.0)
            logger.warning(f"Invalid color format '{color_value}', using default gray.")

        principled.inputs['Base Color'].default_value = rgba

        # Other properties
        principled.inputs['Metallic'].default_value = config.get('metallic', 0.0)
        principled.inputs['Roughness'].default_value = config.get('roughness', 0.3)

        return mat

    def create_piece(self, piece_type: str, config: dict[str, Any]) -> bpy.types.Object | None:
        """Creates a piece instance using cached geometry and applies config."""

        print("--------------------------------")
        print("CONFIG", config)
        print("INITIAL LOCATION IN CONFIG", config['location'])
        print("--------------------------------")
        piece_type_lower = piece_type.lower()
        cached_mesh = self._get_or_extract_mesh(piece_type_lower)

        if not cached_mesh:
            logger.error(f"Could not get mesh for piece type: {piece_type_lower}")
            return None

        # Create new object linked to the cached mesh
        instance_name = f"{piece_type_lower}_instance_{random.random():.4f}"
        new_object = bpy.data.objects.new(name=instance_name, object_data=cached_mesh)

        # Link to scene
        bpy.context.collection.objects.link(new_object)

        # Apply transformations
        # Location (expects world coordinates from the caller)
        new_object.location = config['location']
        print("NEW OBJECT LOCATION", new_object.location)
        # ensure z is 0.95
        # if not pawn, location at z=0.95  else 1.0
        new_object.location.z = 0.95

        # Scale
        DEFAULT_BASE_SCALE = 0.08 # Set a sensible default matching previous behaviour
        base_scale = config.get('scale', DEFAULT_BASE_SCALE) # Base scale from config, fallback to default
        type_specific_scale = self._piece_type_to_scale_map.get(piece_type_lower, 1.0)
        final_scale = base_scale * type_specific_scale
        new_object.scale = (final_scale, final_scale, final_scale)

        # --- Adjust Z-location based on object origin vs. base ---
        # Ensure dependency graph is updated to get correct dimensions after scaling
        bpy.context.view_layer.update()


        # ---------------------------------------------------------

        # Rotation
        if config.get('random_rotation', False):
            max_angle_deg = config.get('max_rotation_angle', 15.0)
            angle_rad = random.uniform(-math.radians(max_angle_deg), math.radians(max_angle_deg))
            # Rotate randomly around Z-axis
            new_object.rotation_euler = (0, 0, angle_rad)
        else:
            # Apply default rotation based on piece type if random_rotation is False
            if piece_type_lower == "knight":
                # Default rotation for knight (90 degrees around Z)
                new_object.rotation_euler = (0, 0, math.pi / 2)
            else:
                # Default for other pieces
                new_object.rotation_euler = (0, 0, 0.1) # Explicitly set to (near) zero

        # Apply Material
        material = self._create_material(instance_name, config)
        if new_object.data.materials:
            new_object.data.materials[0] = material # Replace default/existing
        else:
            new_object.data.materials.append(material) # Add if none exist

        logger.debug(f"Created piece instance '{instance_name}' from cached mesh '{piece_type_lower}'")

        print("--------------------------------")
        print("FINAL WITHIN FUNCTION FACTORY LOCATION", new_object.location)
        print("--------------------------------")
        return new_object

    # _get_creator_func is no longer needed for this factory
    # _call_creator_func is no longer needed for this factory


class StonesColorPieceFactory(PieceFactory):
    """Creates pieces using logic from chess.pieces_from_blend.pieces_stones_color.pieces."""
    # TODO: Implement caching similar to OldSchoolPieceFactory if this factory is used and needs optimization.
    # For now, it keeps the old import-based logic.

    def _get_creator_func(self, piece_type: str) -> Callable | None:
        try:
             # Assuming the function names match the piece types in pieces_stones_color/pieces.py
            if piece_type == "pawn":
                from chess.pieces_from_blend.pieces_stones_color.pieces import (
                    create_pawn as func,
                )
            elif piece_type == "rook":
                from chess.pieces_from_blend.pieces_stones_color.pieces import (
                    create_rook as func,
                )
            elif piece_type == "knight":
                 from chess.pieces_from_blend.pieces_stones_color.pieces import (
                     create_knight as func,
                 )
            elif piece_type == "bishop":
                 from chess.pieces_from_blend.pieces_stones_color.pieces import (
                     create_bishop as func,
                 )
            elif piece_type == "queen":
                 from chess.pieces_from_blend.pieces_stones_color.pieces import (
                     create_queen as func,
                 )
            elif piece_type == "king":
                 from chess.pieces_from_blend.pieces_stones_color.pieces import (
                     create_king as func,
                 )
            else:
                logger.warning(f"StonesColorPieceFactory: Unknown piece type '{piece_type}'")
                return None
            return func
        except ImportError as e:
            logger.error(f"StonesColorPieceFactory: Failed to import creation function for '{piece_type}': {e}")
            return None
        except AttributeError as e:
             logger.error(f"StonesColorPieceFactory: Creation function not found for '{piece_type}' in pieces_stones_color.pieces: {e}")
             return None

    def create_piece(self, piece_type: str, config: dict[str, Any]) -> bpy.types.Object | None:
        creator_func = self._get_creator_func(piece_type.lower())
        if creator_func:
             # Potentially adapt config if needed for stones_color functions
            return self._call_creator_func(creator_func, config)
        return None
