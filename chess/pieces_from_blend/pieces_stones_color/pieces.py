"""Chess pieces loaded from blend files."""

import os
from typing import Any

import bpy

from utils.blender_utils import extract_object_from_blend

from ...config.models import PieceConfig
from ...pieces.base import ChessPiece


class BlendPiece(ChessPiece):
    """Base class for chess pieces loaded from blend files."""

    piece_scale = 0.02  # Default scale factor for chess pieces

    def __init__(self, config: PieceConfig, blend_path: str, collection_name: str):
        """
        Initialize a chess piece that loads from a blend file.

        Args:
            config: Configuration for the piece
            blend_path: Path to the blend file containing the piece
            collection_name: Name of the collection containing the piece
        """
        super().__init__(config)
        self.blend_path = blend_path
        self.collection_name = collection_name
        self._extracted_collection = None
        print(f"\nInitializing {collection_name} piece:")
        print(f"- Scale: {config.geometry.scale}")
        print(f"- Color: {config.material.color}")
        print(f"- Location: {config.location}")

    def create_geometry(self) -> None:
        """Load geometry from blend file instead of creating it."""
        print(f"\nCreating geometry for {self.collection_name}:")

        if not self._extracted_collection:
            print(f"- Extracting from {self.blend_path}")
            self._extracted_collection = extract_object_from_blend(
                self.blend_path,
                self.collection_name,
                save=False
            )

        if not self._extracted_collection:
            raise RuntimeError(f"Failed to load {self.collection_name} from {self.blend_path}")

        print(f"- Successfully loaded collection with {len(self._extracted_collection.objects)} objects")

        # Link the collection to the scene and view layer
        scene = bpy.context.scene
        view_layer = bpy.context.view_layer

        # Create a new collection for this piece
        piece_collection = bpy.data.collections.new(f"{self.collection_name}_instance")
        scene.collection.children.link(piece_collection)
        print(f"- Created new collection: {piece_collection.name}")

        # Link the collection to the scene
        mesh_count = 0
        for obj in self._extracted_collection.objects:
            if obj.type == 'MESH':
                mesh_count += 1
                print(f"\n  Processing mesh object {obj.name}:")

                # Create a copy of the object
                obj_copy = obj.copy()
                obj_copy.data = obj.data.copy()
                print(f"  - Created copy: {obj_copy.name}")

                # Link to our piece collection
                piece_collection.objects.link(obj_copy)
                print("  - Linked to collection")

                # Add to our parts list
                self.parts.append(obj_copy)

                # Set location
                obj_copy.location = self.config.location

                # Apply scale (multiply by the piece's base scale)
                base_scale = self.config.geometry.scale
                final_scale = base_scale * self.piece_scale
                obj_copy.scale = (final_scale, final_scale, final_scale)
                print(f"  - Applied scale: {final_scale} (base: {base_scale} * piece: {self.piece_scale})")

                # Remove all existing materials from the object
                while obj_copy.data.materials:
                    obj_copy.data.materials.pop()

                # Create a new material for this piece
                mat_name = f"{self.collection_name}_{obj_copy.name}_material"
                mat = bpy.data.materials.new(name=mat_name)
                mat.use_nodes = True
                print(f"  - Created new material: {mat_name}")

                # Set up the material nodes
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

                # Set material color
                principled.inputs['Base Color'].default_value = self.config.material.color
                principled.inputs['Metallic'].default_value = 0.0
                principled.inputs['Roughness'].default_value = 0.3
                print(f"  - Set material color to {self.config.material.color}")

                # Assign the material to the object
                obj_copy.data.materials.append(mat)
                print("  - Assigned new material to object")

                # Apply color to all materials
                if obj_copy.material_slots:
                    print(f"  - Processing {len(obj_copy.material_slots)} material slots:")
                    for i, slot in enumerate(obj_copy.material_slots):
                        if slot.material:
                            # Create a copy of the material
                            mat = slot.material.copy()
                            slot.material = mat
                            print(f"    Slot {i}: Created material copy {mat.name}")

                            # Ensure material uses nodes
                            if not mat.use_nodes:
                                mat.use_nodes = True
                                print(f"    Slot {i}: Enabled nodes")

                            # Get or create Principled BSDF
                            principled = None
                            for node in mat.node_tree.nodes:
                                if node.type == 'BSDF_PRINCIPLED':
                                    principled = node
                                    break

                            if not principled:
                                principled = mat.node_tree.nodes.new('ShaderNodeBsdfPrincipled')
                                print(f"    Slot {i}: Created new Principled BSDF node")
                            else:
                                print(f"    Slot {i}: Using existing Principled BSDF node")

                            # Set base color
                            principled.inputs['Base Color'].default_value = self.config.material.color
                            print(f"    Slot {i}: Set color to {self.config.material.color}")
                else:
                    print("  - No material slots found!")

        print(f"\nFinished processing {mesh_count} mesh objects for {self.collection_name}")


class Pawn(BlendPiece):
    """Chess pawn piece loaded from blend file."""

    piece_scale = 0.02  # Keep original scale for pawn

    def __init__(self, config: PieceConfig):
        """
        Initialize a pawn piece.

        Args:
            config: Configuration for the piece
        """
        # Get the project root directory (3 levels up from this file)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        blend_path = os.path.join(project_root, "blend_files", "chess", "chess_classique.blend")
        super().__init__(config, blend_path, "front")


class Rook(BlendPiece):
    """Chess rook piece loaded from blend file."""

    piece_scale = 0.02  # Keep original scale for rook

    def __init__(self, config: PieceConfig):
        """
        Initialize a rook piece.

        Args:
            config: Configuration for the piece
        """
        # Get the project root directory (3 levels up from this file)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        blend_path = os.path.join(project_root, "blend_files", "chess", "chess_classique.blend")
        super().__init__(config, blend_path, "rook")


class Queen(BlendPiece):
    """Chess queen piece loaded from blend file."""

    piece_scale = 0.14  # Larger scale for queen

    def __init__(self, config: PieceConfig):
        """
        Initialize a queen piece.

        Args:
            config: Configuration for the piece
        """
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        blend_path = os.path.join(project_root, "blend_files", "chess", "chess_classique.blend")
        super().__init__(config, blend_path, "Collection 7")


class King(BlendPiece):
    """Chess king piece loaded from blend file."""

    piece_scale = 0.14  # Larger scale for king

    def __init__(self, config: PieceConfig):
        """
        Initialize a king piece.

        Args:
            config: Configuration for the piece
        """
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        blend_path = os.path.join(project_root, "blend_files", "chess", "chess_classique.blend")
        super().__init__(config, blend_path, "Collection 8")


class Bishop(BlendPiece):
    """Chess bishop piece loaded from blend file."""

    piece_scale = 0.12  # Larger scale for bishop

    def __init__(self, config: PieceConfig):
        """
        Initialize a bishop piece.

        Args:
            config: Configuration for the piece
        """
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        blend_path = os.path.join(project_root, "blend_files", "chess", "chess_classique.blend")
        super().__init__(config, blend_path, "Officer")


class Knight(BlendPiece):
    """Chess knight piece loaded from blend file."""

    piece_scale = 0.02  # Base scale for knight

    def __init__(self, config: PieceConfig):
        """
        Initialize a knight piece.

        Args:
            config: Configuration for the piece
        """
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        blend_path = os.path.join(project_root, "blend_files", "chess", "chess_classique.blend")
        super().__init__(config, blend_path, "horse")

    def create_geometry(self) -> None:
        """Load geometry from blend file with custom xyz scaling."""
        print(f"\nCreating geometry for {self.collection_name}:")

        if not self._extracted_collection:
            print(f"- Extracting from {self.blend_path}")
            self._extracted_collection = extract_object_from_blend(
                self.blend_path,
                self.collection_name,
                save=False
            )

        if not self._extracted_collection:
            raise RuntimeError(f"Failed to load {self.collection_name} from {self.blend_path}")

        print(f"- Successfully loaded collection with {len(self._extracted_collection.objects)} objects")

        # Link the collection to the scene and view layer
        scene = bpy.context.scene
        view_layer = bpy.context.view_layer

        # Create a new collection for this piece
        piece_collection = bpy.data.collections.new(f"{self.collection_name}_instance")
        scene.collection.children.link(piece_collection)
        print(f"- Created new collection: {piece_collection.name}")

        # Link the collection to the scene
        mesh_count = 0
        for obj in self._extracted_collection.objects:
            if obj.type == 'MESH':
                mesh_count += 1
                print(f"\n  Processing mesh object {obj.name}:")

                # Create a copy of the object
                obj_copy = obj.copy()
                obj_copy.data = obj.data.copy()
                print(f"  - Created copy: {obj_copy.name}")

                # Link to our piece collection
                piece_collection.objects.link(obj_copy)
                print("  - Linked to collection")

                # Add to our parts list
                self.parts.append(obj_copy)

                # Set location
                obj_copy.location = self.config.location

                # Apply custom xyz scale for knight
                base_scale = self.config.geometry.scale
                obj_copy.scale = (
                    base_scale * 0.04,  # x scale
                    base_scale * 0.04,  # y scale
                    base_scale * 0.02   # z scale
                )
                print(f"  - Applied custom xyz scale: ({obj_copy.scale[0]}, {obj_copy.scale[1]}, {obj_copy.scale[2]})")

                # Remove all existing materials from the object
                while obj_copy.data.materials:
                    obj_copy.data.materials.pop()

                # Create a new material for this piece
                mat_name = f"{self.collection_name}_{obj_copy.name}_material"
                mat = bpy.data.materials.new(name=mat_name)
                mat.use_nodes = True
                print(f"  - Created new material: {mat_name}")

                # Set up the material nodes
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

                # Set material color
                principled.inputs['Base Color'].default_value = self.config.material.color
                principled.inputs['Metallic'].default_value = 0.0
                principled.inputs['Roughness'].default_value = 0.3
                print(f"  - Set material color to {self.config.material.color}")

                # Assign the material to the object
                obj_copy.data.materials.append(mat)
                print("  - Assigned new material to object")

                # Apply color to all materials
                if obj_copy.material_slots:
                    print(f"  - Processing {len(obj_copy.material_slots)} material slots:")
                    for i, slot in enumerate(obj_copy.material_slots):
                        if slot.material:
                            # Create a copy of the material
                            mat = slot.material.copy()
                            slot.material = mat
                            print(f"    Slot {i}: Created material copy {mat.name}")

                            # Ensure material uses nodes
                            if not mat.use_nodes:
                                mat.use_nodes = True
                                print(f"    Slot {i}: Enabled nodes")

                            # Get or create Principled BSDF
                            principled = None
                            for node in mat.node_tree.nodes:
                                if node.type == 'BSDF_PRINCIPLED':
                                    principled = node
                                    break

                            if not principled:
                                principled = mat.node_tree.nodes.new('ShaderNodeBsdfPrincipled')
                                print(f"    Slot {i}: Created new Principled BSDF node")
                            else:
                                print(f"    Slot {i}: Using existing Principled BSDF node")

                            # Set base color
                            principled.inputs['Base Color'].default_value = self.config.material.color
                            print(f"    Slot {i}: Set color to {self.config.material.color}")
                else:
                    print("  - No material slots found!")

        print(f"\nFinished processing {mesh_count} mesh objects for {self.collection_name}")


def create_pawn(config: dict[str, Any]) -> bpy.types.Object:
    """
    Create a chess pawn piece from a configuration dictionary.

    Args:
        config: Dictionary containing piece configuration

    Returns:
        The created pawn object
    """
    piece_config = PieceConfig.from_dict(config)
    pawn = Pawn(piece_config)
    return pawn.create()


def create_rook(config: dict[str, Any]) -> bpy.types.Object:
    """
    Create a chess rook piece from a configuration dictionary.

    Args:
        config: Dictionary containing piece configuration

    Returns:
        The created rook object
    """
    piece_config = PieceConfig.from_dict(config)
    rook = Rook(piece_config)
    return rook.create()


def create_queen(config: dict[str, Any]) -> bpy.types.Object:
    """
    Create a chess queen piece from a configuration dictionary.

    Args:
        config: Dictionary containing piece configuration

    Returns:
        The created queen object
    """
    piece_config = PieceConfig.from_dict(config)
    queen = Queen(piece_config)
    return queen.create()


def create_king(config: dict[str, Any]) -> bpy.types.Object:
    """
    Create a chess king piece from a configuration dictionary.

    Args:
        config: Dictionary containing piece configuration

    Returns:
        The created king object
    """
    piece_config = PieceConfig.from_dict(config)
    king = King(piece_config)
    return king.create()


def create_bishop(config: dict[str, Any]) -> bpy.types.Object:
    """
    Create a chess bishop piece from a configuration dictionary.

    Args:
        config: Dictionary containing piece configuration

    Returns:
        The created bishop object
    """
    piece_config = PieceConfig.from_dict(config)
    bishop = Bishop(piece_config)
    return bishop.create()


def create_knight(config: dict[str, Any]) -> bpy.types.Object:
    """
    Create a chess knight piece from a configuration dictionary.

    Args:
        config: Dictionary containing piece configuration

    Returns:
        The created knight object
    """
    piece_config = PieceConfig.from_dict(config)
    knight = Knight(piece_config)
    return knight.create()
