from typing import Any

import bpy

from scene_setup.models import RenderModel
from utils.blender_utils import enable_gpu_rendering


def clear_scene():
    """Remove all objects and orphaned data blocks from the scene."""
    # Remove objects
    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False) # Use use_global=False for scripts

    # Remove orphaned data blocks (meshes, materials, textures, images, node groups)
    # Iterate multiple times to handle dependencies
    for _ in range(5): # Repeat a few times to catch dependencies
        for block in bpy.data.meshes:
            if block.users == 0:
                bpy.data.meshes.remove(block)

        for block in bpy.data.materials:
            if block.users == 0:
                bpy.data.materials.remove(block)

        for block in bpy.data.textures:
            if block.users == 0:
                bpy.data.textures.remove(block)

        for block in bpy.data.images:
            if block.users == 0:
                bpy.data.images.remove(block)

        # Include node groups
        for block in bpy.data.node_groups:
            if block.users == 0:
                bpy.data.node_groups.remove(block)

        # Add other common data types
        for block in bpy.data.cameras:
            if block.users == 0:
                bpy.data.cameras.remove(block)

        for block in bpy.data.lights:
            if block.users == 0:
                bpy.data.lights.remove(block)

        for block in bpy.data.worlds:
            if block.users == 0:
                bpy.data.worlds.remove(block)

        # Actions can also hold references
        for block in bpy.data.actions:
            if block.users == 0:
                bpy.data.actions.remove(block)


def setup_render(config: dict[str, Any]):
    """
    Setup the render settings from a RenderModel or configuration dictionary.

    Args:
        config: Either a RenderModel object or a dictionary with render configuration:
               - resolution: ResolutionModel, dict, string preset ("low", "medium", "high"),
                            or tuple of (width, height)
               - file_format: The file format of the render (default: "PNG")
               - engine: The engine of the render (default: "CYCLES")
               - samples: The number of samples of the render (default: 128)
               - exposure: The exposure of the render (default: 0.0)
               - gpu_enabled: Whether to enable GPU rendering (default: True)
               - gpus: Which GPUs to use for rendering (None for all)
    """
    scene = bpy.context.scene

    # Convert config to RenderModel if it's a dictionary
    render_model = RenderModel.from_dict(config)

    # Get resolution from the render model
    resolution_model = render_model.resolution

    # Set render settings from the model
    scene.render.image_settings.file_format = render_model.file_format
    scene.render.resolution_x = resolution_model.width
    scene.render.resolution_y = resolution_model.height
    scene.render.resolution_percentage = resolution_model.resolution_percentage
    scene.render.pixel_aspect_x = resolution_model.pixel_aspect_x
    scene.render.pixel_aspect_y = resolution_model.pixel_aspect_y
    scene.render.engine = render_model.engine
    scene.cycles.samples = render_model.samples
    scene.view_settings.exposure = render_model.exposure

    # Enable GPU rendering if requested
    if render_model.gpu_enabled:
        enable_gpu_rendering(gpus=render_model.gpus)


def setup_render_from_config(config: dict[str, Any]):
    """
    Setup the render settings from a configuration dictionary.

    Args:
        config: A dictionary with render configuration:
               - resolution: ResolutionModel, dict, string preset ("low", "medium", "high"),
                            or tuple of (width, height)
               - file_format: The file format of the render (default: "PNG")
               - engine: The engine of the render (default: "CYCLES")
               - samples: The number of samples of the render (default: 128)
    """

    # Set up the render
    setup_render(config)


