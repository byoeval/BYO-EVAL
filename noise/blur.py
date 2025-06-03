from typing import Any

import bpy

from noise.models import BlurNoiseModel


def setup_blur(intensity="none"):
    """
    Setup depth of field blur for the active camera.

    Args:
        intensity: Either a string preset ("none", "very_low", "low", "medium", "high", "very_high")
                  or a float value for the f-stop (lower values = more blur)

    Returns:
        float: The actual f-stop value used, or None if blur is disabled
    """
    # Get the active camera
    camera = bpy.context.scene.camera
    if not camera:
        raise ValueError("No active camera in the scene")

    # Process intensity setting
    if isinstance(intensity, str):
        if intensity not in BlurNoiseModel.BLUR_PRESETS:
            raise ValueError(
                f"Invalid blur intensity: {intensity}. "
                f"Must be one of {list(BlurNoiseModel.BLUR_PRESETS.keys())} or a float value"
            )
        f_stop = BlurNoiseModel.BLUR_PRESETS[intensity]
    else:
        f_stop = intensity if isinstance(intensity, int | float) else None

    # Handle the "none" case by disabling depth of field
    if f_stop is None:
        camera.data.dof.use_dof = False
        return None

    # Enable depth of field and set parameters
    camera.data.dof.use_dof = True
    camera.data.dof.aperture_fstop = max(0.1, f_stop)  # Ensure f-stop is never 0

    # Set a reasonable focus distance if not already set
    # TODO: Potentially set focus distance based on camera target or scene bounds
    if camera.data.dof.focus_distance == 0:
        camera.data.dof.focus_distance = 10.0

    return f_stop


def build_blur_from_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    Process a blur configuration dictionary and set up camera depth of field.

    Args:
        config: Dictionary with blur configuration parameter:
               - "blur": str ("none", "very_low", "low", "medium", "high", "very_high")
                        or float value for f-stop

    Returns:
        Dict: Updated configuration dictionary with final values
    """
    # Create blur model from config
    blur_model = BlurNoiseModel.from_dict(config)

    # Set up the blur
    f_stop = setup_blur(blur_model.intensity)

    # Store the final values in the config
    config["final_blur_fstop"] = f_stop
    config["blur_enabled"] = f_stop is not None

    return config
