from typing import Any

import bpy  # Add bpy import if not already present

from scene_setup.camera import build_camera_from_config
from scene_setup.environment.background import create_background
from scene_setup.environment.floor import create_floor_from_config
from scene_setup.environment.standard_lighting import build_lighting_from_config
from scene_setup.environment.table import create_table_from_config
from scene_setup.rendering import clear_scene, setup_render_from_config
from scene_setup.resolution import build_resolution_from_config


def build_setup_from_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    Build a complete scene setup from a configuration dictionary.

    Args:
        config: Dictionary with configuration parameters for all scene elements:
               - camera: Dict with camera configuration
               - resolution: Dict with resolution configuration
               - table: Dict with table configuration
               - render: Dict with render configuration (optional)
               - floor: Dict with floor configuration (optional)
               - grid: Dict with grid configuration (optional - NOT USED BY THIS FUNCTION DIRECTLY ANYMORE for drawing)

    Returns:
        Dict: Updated configuration dictionary including a key 'created_objects'
              with references to created Blender objects (e.g., table, camera).
    """
    # Clear the scene first
    clear_scene()

    # Dictionary to store created objects
    created_objects: dict[str, bpy.types.Object | None] = {
        "table": None,
        "floor": None,
        "camera": None,
    }

    # Process resolution configuration
    resolution_config = config.get("resolution", {"resolution": "medium"})
    config["resolution"] = build_resolution_from_config(resolution_config)

    # Set up render with the resolution
    render_config = config.get("render", {})

    # Set up the render
    setup_render_from_config(render_config)

    # Create background (currently does nothing)
    create_background()

    # Create floor if configured
    floor_config = config.get("floor", {})
    if floor_config:
        floor = create_floor_from_config(floor_config)
        created_objects["floor"] = floor # Store floor reference

    # Create table
    table_config = config.get("table", {})
    table = create_table_from_config(table_config)
    created_objects["table"] = table # Store table reference

    # Process camera configuration
    camera_config = config.get("camera", {"distance": "medium", "angle": "medium"})
    camera, camera_config = build_camera_from_config(camera_config)
    created_objects["camera"] = camera # Store camera reference
    # Update the main config with the final camera parameters
    config["camera"] = camera_config

    # Set up lighting
    lighting_config = config.get("lighting", {"lighting": "medium"})
    lighting_config = build_lighting_from_config(lighting_config)
    config["lighting"] = lighting_config # Store lighting config back

    # Grid configuration is no longer processed here for handler registration.
    # The calling script will use it with add_grid_to_image_file if needed.
    if "grid" in config:
        print("Info (build_setup_from_config): 'grid' configuration found. "
              "Note: Grid drawing is now a separate post-render step "
              "using 'add_grid_to_image_file'.")


    # Add created objects to the config dictionary before returning
    config["created_objects"] = created_objects

    return config
