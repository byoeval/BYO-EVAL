import math
from typing import Any

import bpy
from mathutils import Euler

from scene_setup.models import LightingModel


def create_lighting(intensity="medium"):
    """
    Create basic three-point lighting setup with configurable intensity.

    Args:
        intensity: Either:
            - a string preset ("very_low", "low", "medium", "high", "very_high")
            - a float multiplier for the base energy values
            - a dict with a "lighting" key containing either of the above

    Returns:
        tuple: The three created lights (key_light, fill_light, back_light)
    """
    # Process intensity setting
    if isinstance(intensity, dict):
        intensity = intensity.get("lighting", "medium")

    # Create a lighting model instance
    lighting_model = LightingModel(lighting=intensity)

    # Get the intensity factor from presets if it's a string
    if isinstance(intensity, str):
        lighting_presets = LightingModel.LIGHTING_PRESETS
        if intensity not in lighting_presets:
            raise ValueError(
                f"Invalid lighting intensity: {intensity}. "
                f"Must be one of {list(lighting_presets.keys())} or a float multiplier"
            )
        intensity_factor = lighting_presets[intensity]
    else:
        intensity_factor = intensity

    # Key light
    bpy.ops.object.light_add(type='AREA', location=(5, -5, 8))
    key_light = bpy.context.active_object
    key_light.data.energy = lighting_model.key_light_power * intensity_factor
    key_light.rotation_euler = Euler((math.radians(60), 0, math.radians(45)))

    # Fill light
    bpy.ops.object.light_add(type='AREA', location=(-6, -4, 5))
    fill_light = bpy.context.active_object
    fill_light.data.energy = lighting_model.fill_light_power * intensity_factor
    fill_light.rotation_euler = Euler((math.radians(45), 0, math.radians(-45)))

    # Back light
    bpy.ops.object.light_add(type='AREA', location=(0, 6, 6))
    back_light = bpy.context.active_object
    back_light.data.energy = lighting_model.back_light_power * intensity_factor
    back_light.rotation_euler = Euler((math.radians(-45), 0, math.radians(180)))

    return key_light, fill_light, back_light

def build_lighting_from_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    Process a lighting configuration dictionary and create lights.

    Args:
        config: Dictionary with lighting configuration parameter:
               - "lighting": str ("very_low", "low", "medium", "high", "very_high")
                            or float multiplier

    Returns:
        Dict: Updated configuration dictionary with final values and created lights
    """
    # Create a lighting model from config
    lighting_model = LightingModel.from_dict(config)

    # Get lighting intensity from the model
    lighting_option = lighting_model.lighting

    # Create the lights
    key_light, fill_light, back_light = create_lighting(lighting_option)

    # Process intensity value for config
    if isinstance(lighting_option, str):
        intensity_factor = LightingModel.LIGHTING_PRESETS[lighting_option]
    else:
        intensity_factor = lighting_option

    # Store the final values in the config
    config["final_lighting_factor"] = intensity_factor
    config["lights"] = {
        "key_light": key_light,
        "fill_light": fill_light,
        "back_light": back_light
    }

    return config
