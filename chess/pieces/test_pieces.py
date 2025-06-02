import os
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

import bpy
import mathutils

from scene_setup.rendering import setup_render
from utils.blender_utils import enable_gpu_rendering

from ..pieces import (
    create_bishop,
    create_king,
    create_knight,
    create_pawn,
    create_queen,
    create_rook,
)


def setup_lighting():
    """Set up three-point lighting for the scene."""
    # Remove default light if it exists
    if "Light" in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects["Light"], do_unlink=True)

    # Key light (main light)
    bpy.ops.object.light_add(type='AREA', location=(3, -2, 4))
    key_light = bpy.context.active_object
    key_light.data.energy = 4000
    key_light.data.size = 1
    key_light.name = "KeyLight"
    key_light.rotation_euler = (0.5, 0.2, 0.3)

    # Fill light (softer, from opposite side)
    bpy.ops.object.light_add(type='AREA', location=(-3, -1, 3))
    fill_light = bpy.context.active_object
    fill_light.data.energy = 2000
    fill_light.data.size = 2
    fill_light.name = "FillLight"
    fill_light.rotation_euler = (0.5, -0.2, -0.3)

    # Back light (rim light)
    bpy.ops.object.light_add(type='AREA', location=(0, 3, 4))
    back_light = bpy.context.active_object
    back_light.data.energy = 3000
    back_light.data.size = 1.5
    back_light.name = "BackLight"
    back_light.rotation_euler = (-0.5, 0, 0)

def setup_camera(location=(5, -5, 3), target_point=(1.5, 0, 0)):
    """Set up camera with given location and target point."""
    if "Camera" in bpy.data.objects:
        camera = bpy.data.objects["Camera"]
    else:
        bpy.ops.object.camera_add()
        camera = bpy.context.active_object

    camera.location = location

    # Point camera at target using Blender's Vector class
    direction = mathutils.Vector((
        target_point[0] - camera.location[0],
        target_point[1] - camera.location[1],
        target_point[2] - camera.location[2]
    ))

    # Calculate rotation to point at target
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()

    # Set as active camera
    bpy.context.scene.camera = camera

    return camera

def setup_scene():
    """Set up the scene with a simple ground plane."""
    # Create ground plane
    bpy.ops.mesh.primitive_plane_add(size=10, location=(0, 0, -0.001))
    plane = bpy.context.active_object

    # Add material to plane
    mat = bpy.data.materials.new(name="GroundMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.8, 0.8, 0.8, 1)
    nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.7

    plane.data.materials.append(mat)

def clear_scene():
    """Clear all objects and collections from the scene."""
    # Remove all objects
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # Remove all collections except the default Scene Collection
    for collection in bpy.data.collections:
        bpy.data.collections.remove(collection)

def test_all_pieces(output_dir: str = "tests/images/piece_test_renders"):
    """Test creating all chess pieces with different configurations."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Clear the scene first
    clear_scene()

    # Setup scene, lighting, and camera
    setup_scene()
    setup_lighting()
    setup_camera((8, -8, 5), (2.5, 0, 0))  # Adjusted camera position for better view

    # Setup render settings
    setup_render(resolution="medium", samples=128)

    # Dictionary mapping piece types to their creation functions
    piece_creators = {
        'pawn': create_pawn,
        'rook': create_rook,
        'knight': create_knight,
        'bishop': create_bishop,
        'queen': create_queen,
        'king': create_king
    }

    # Create two rows of pieces: white and black
    created_pieces = []
    spacing = 1.2  # Space between pieces

    # Create white pieces
    for i, (piece_type, create_func) in enumerate(piece_creators.items()):
        config = {
            'type': piece_type,
            'location': (i * spacing, 0, 0),
            'color': 'white',
            'scale': 1.0
        }
        piece = create_func(config)
        created_pieces.append(piece)
        print(f"Created white {piece_type}: {piece.name}")
        print(f"Location: {piece.location}")
        print(f"Material: {piece.data.materials[0].name}")
        print("-" * 40)

    # Create black pieces
    for i, (piece_type, create_func) in enumerate(piece_creators.items()):
        config = {
            'type': piece_type,
            'location': (i * spacing, 1.5, 0),  # Second row
            'color': 'black',
            'scale': 1.0
        }
        piece = create_func(config)
        created_pieces.append(piece)
        print(f"Created black {piece_type}: {piece.name}")
        print(f"Location: {piece.location}")
        print(f"Material: {piece.data.materials[0].name}")
        print("-" * 40)

    # Render the scene
    output_path = os.path.join(output_dir, "all_pieces.png")
    bpy.context.scene.render.filepath = output_path

    # Enable GPU rendering if available
    enable_gpu_rendering()

    # Render
    bpy.ops.render.render(write_still=True)
    print(f"Rendered all pieces to {output_path}")

    return created_pieces

if __name__ == "__main__":
    pieces = test_all_pieces()
    print(f"\nSuccessfully created {len(pieces)} chess pieces and rendered them")
