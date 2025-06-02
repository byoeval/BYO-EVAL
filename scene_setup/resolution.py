from typing import Dict, Any
from scene_setup.models import ResolutionModel

def build_resolution_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a resolution configuration dictionary.
    
    Args:
        config: Dictionary with resolution configuration parameter:
               - "resolution": str ("low", "medium", "high") or tuple(int, int)
    
    Returns:
        Dict: Updated configuration dictionary with ResolutionModel and final values
    """
    # Get resolution from config or use default
    resolution_option = config.get("resolution", "medium")
    
    # Process based on type
    if isinstance(resolution_option, str):
        resolution_presets = ResolutionModel.RESOLUTION_PRESETS
        if resolution_option not in resolution_presets:
            raise ValueError(
                f"Invalid resolution option: {resolution_option}. "
                f"Must be one of {list(resolution_presets.keys())} or a tuple of (width, height)"
            )
        width, height = resolution_presets[resolution_option]
    else:
        # If it's already a tuple, use it directly
        width, height = resolution_option
    
    # Create a resolution model
    resolution_model = ResolutionModel(width=width, height=height)
    
    # Store the model and its dict representation in the config
    config["final_resolution"] = (resolution_model.width, resolution_model.height)
    
    return config
