import bpy
import math
from mathutils import Euler
from typing import Dict, Any

from Noise.models import LightNoiseModel


def remove_all_lights():
    """
    Remove all existing lights from the scene.
    """
    # First deselect all objects
    bpy.ops.object.select_all(action='DESELECT')
    
    # Select only light objects
    for obj in bpy.context.scene.objects:
        if obj.type == 'LIGHT':
            obj.select_set(True)
    
    # Delete all selected objects (lights)
    if bpy.context.selected_objects:
        bpy.ops.object.delete()


def create_lighting(intensity="medium"):
    """
    Create basic three-point lighting setup with configurable intensity.
    
    Args:
        intensity: Either a string preset ("very_low", "low", "medium", "high", "very_high")
                  or a float multiplier for the base energy values
    
    Returns:
        tuple: The three created lights (key_light, fill_light, back_light)
    """
    # Remove any existing lights
    remove_all_lights()
    
    # Process intensity setting
    if isinstance(intensity, str):
        if intensity not in LightNoiseModel.LIGHTING_PRESETS:
            raise ValueError(
                f"Invalid lighting intensity: {intensity}. "
                f"Must be one of {list(LightNoiseModel.LIGHTING_PRESETS.keys())} or a float multiplier"
            )
        intensity_factor = LightNoiseModel.LIGHTING_PRESETS[intensity]
    else:
        intensity_factor = intensity
    
    # Key light
    bpy.ops.object.light_add(type='AREA', location=(5, -5, 8))
    key_light = bpy.context.active_object
    key_light.data.energy = LightNoiseModel.BASE_KEY_LIGHT_ENERGY * intensity_factor
    key_light.rotation_euler = Euler((math.radians(60), 0, math.radians(45)))
    
    # Fill light
    bpy.ops.object.light_add(type='AREA', location=(-6, -4, 5))
    fill_light = bpy.context.active_object
    fill_light.data.energy = LightNoiseModel.BASE_FILL_LIGHT_ENERGY * intensity_factor
    fill_light.rotation_euler = Euler((math.radians(45), 0, math.radians(-45)))
    
    # Back light
    bpy.ops.object.light_add(type='AREA', location=(0, 6, 6))
    back_light = bpy.context.active_object
    back_light.data.energy = LightNoiseModel.BASE_BACK_LIGHT_ENERGY * intensity_factor
    back_light.rotation_euler = Euler((math.radians(-45), 0, math.radians(180)))
    
    return key_light, fill_light, back_light


def build_lighting_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a lighting configuration dictionary and create lights.
    
    Args:
        config: Dictionary with lighting configuration parameter:
               - "lighting": str ("very_low", "low", "medium", "high", "very_high") 
                            or float multiplier
    
    Returns:
        Dict: Updated configuration dictionary with final values and created lights
    """
    # Create light model from config
    light_model = LightNoiseModel.from_dict(config)
    
    # Create the lights
    key_light, fill_light, back_light = create_lighting(light_model.lighting)
    
    # Process intensity value for config
    if isinstance(light_model.lighting, str):
        intensity_factor = LightNoiseModel.LIGHTING_PRESETS[light_model.lighting]
    else:
        intensity_factor = light_model.lighting
    
    # Store the final values in the config
    config["final_lighting_factor"] = intensity_factor
    config["lights"] = {
        "key_light": key_light,
        "fill_light": fill_light,
        "back_light": back_light
    }
    
    return config