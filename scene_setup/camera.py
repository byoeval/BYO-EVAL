import bpy
import math
import random
from mathutils import Euler, Vector
from typing import Dict, Tuple, Any
from scene_setup.models import CameraModel


def create_camera_standard(location=(0, -35, 20), 
                           rotation=(math.radians(60), 0, 0)):
    """Create and position the camera."""
    # Move camera further back (y) and higher up (z)
    bpy.ops.object.camera_add(location=location)
    camera = bpy.context.active_object
    # Adjust angle to look more downward (increase x rotation)
    camera.rotation_euler = Euler(rotation)
    bpy.context.scene.camera = camera
    
    return camera


def build_camera_location(radius: float = 35.0, vertical_angle: float = 60.0, horizontal_angle: float = 0.0) -> Tuple[float, float, float]:
    """
    Calculate camera position on a sphere looking at (0,0,1).
    
    Args:
        radius: Distance from the center point (0,0,1)
        vertical_angle: Angle from horizontal plane in degrees (0=horizontal, 90=vertical)
        horizontal_angle: Angle around vertical axis in degrees
    
    Returns:
        tuple[float, float, float]: (x, y, z) coordinates of the camera
    """
    # Convert angles to radians
    vertical_rad = math.radians(vertical_angle)
    horizontal_rad = math.radians(horizontal_angle)
    
    # Calculate position using spherical coordinates
    x = radius * math.cos(vertical_rad) * math.cos(horizontal_rad)
    y = radius * math.cos(vertical_rad) * math.sin(horizontal_rad)
    z = radius * math.sin(vertical_rad) + 1 
    
    return (x, y, z)


def create_camera_spherical(radius: float = 35.0, 
                            vertical_angle: float = 60.0, 
                            horizontal_angle: float = 0.0) -> bpy.types.Object:
    """
    Create and position a camera on a sphere looking at point (0,0,1).
    
    Args:
        radius: Distance from the center point (0,0,1)
        vertical_angle: Angle from horizontal plane in degrees (0=horizontal, 90=vertical)
        horizontal_angle: Angle around vertical axis in degrees
    
    Returns:
        bpy.types.Object: The created camera object
    """
    location = build_camera_location(radius, vertical_angle, horizontal_angle)
    
    # Create the camera at the calculated position
    bpy.ops.object.camera_add(location=location)
    camera = bpy.context.active_object
    
    # Point camera at (0,0,1)
    direction = Vector((0, 0, 1)) - Vector(location)
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()
    
    # Set as active camera
    bpy.context.scene.camera = camera
    
    return camera


def get_camera_parameters(camera_model: CameraModel) -> Tuple[float, float, float]:
    """
    Determines the final camera parameters (distance, vertical angle, horizontal angle)
    based on the CameraModel, handling presets and applying randomization if enabled.

    Args:
        camera_model: The CameraModel instance containing configuration.

    Returns:
        tuple[float, float, float]: The final radius, vertical angle (degrees),
                                   and horizontal angle (degrees) for the camera.
    """
    # --- Get Base Values (Handle Presets) ---
    distance_option = camera_model.distance
    angle_option = camera_model.angle
    horizontal_angle = camera_model.horizontal_angle # Assuming no presets for horizontal

    # Distance
    if isinstance(distance_option, str):
        distance = CameraModel.DISTANCE_PRESETS.get(distance_option, CameraModel.DISTANCE_PRESETS["medium"])
    else:
        distance = float(distance_option) # Ensure float

    # Vertical Angle
    if isinstance(angle_option, str):
        vertical_angle = CameraModel.ANGLE_PRESETS.get(angle_option, CameraModel.ANGLE_PRESETS["medium"])
    else:
        vertical_angle = float(angle_option) # Ensure float

    # --- Apply Uniform Randomization Independently --- 
    
    # Randomize distance if enabled
    if camera_model.randomize_distance:
        pct = camera_model.randomize_distance_percentage
        min_dist = distance * (1 - pct)
        max_dist = distance * (1 + pct)
        distance = random.uniform(min_dist, max_dist)
        print(f"  Randomized Camera Distance: {distance:.3f} (Range: [{min_dist:.3f}, {max_dist:.3f}])")

    # Randomize angles if enabled
    if camera_model.randomize_angle:
        # Vertical angle
        pct_angle = camera_model.randomize_angle_percentage
        min_angle = vertical_angle * (1 - pct_angle)
        max_angle = vertical_angle * (1 + pct_angle)
        vertical_angle = max(1.0, min(89.0, random.uniform(min_angle, max_angle)))
        print(f"  Randomized Camera Angle: {vertical_angle:.3f} (Range: [{min_angle:.3f}, {max_angle:.3f}])")
        

    return distance, vertical_angle, horizontal_angle


def build_camera_from_config(config: Dict[str, Any]) -> Tuple[bpy.types.Object, Dict[str, Any]]:
    """
    Build a camera from a configuration dictionary.
    
    Args:
        config: Dictionary with camera configuration parameters.
    
    Returns:
        Tuple containing:
        - bpy.types.Object: The created camera
        - Dict: Updated configuration dictionary with model, model_dict and final values
    """
    # Create a camera model from config
    # This model now contains the separate randomize flags read from the config dict
    camera_model = CameraModel.from_dict(config) 
    
    # Get final parameters (handling presets and randomization internally)
    radius, vertical_angle, horizontal_angle = get_camera_parameters(camera_model)
    
    # Store final calculated values back into the config dict for reference/metadata
    config['final_distance'] = radius
    config['final_angle'] = vertical_angle
    config['final_horizontal_angle'] = horizontal_angle
    config['final_position'] = build_camera_location(radius, vertical_angle, horizontal_angle)
    
    # Create and return the camera
    camera = create_camera_spherical(
        radius=radius,
        vertical_angle=vertical_angle,
        horizontal_angle=horizontal_angle
    )
    
    return camera, config
