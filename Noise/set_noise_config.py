import bpy
from typing import Dict, Any, Optional

from Noise.light import build_lighting_from_config
from Noise.table_texture import build_table_texture_from_config
from Noise.blur import build_blur_from_config
from Noise.models import NoiseConfigModel


def build_noise_from_config(
    config: Dict[str, Any], 
    table_object: Optional[bpy.types.Object] = None
) -> Dict[str, Any]:
    """
    Build noise elements (lighting, table texture, blur) from a configuration dictionary.
    
    Args:
        config: Dictionary with noise configuration parameters:
               - blur: String preset or float value for blur intensity
               - light: Dict with lighting noise configuration
               - table_texture: Dict with table texture noise configuration
        table_object: The table object to apply texture to (if provided)
    
    Returns:
        Dict: Updated configuration dictionary with all created noise elements and final values
    """
    # Convert raw config to NoiseConfigModel
    noise_config_model = NoiseConfigModel.from_dict(config)
    result_config = {}
    
    # Process lighting configuration
    if noise_config_model.light:
        result_config["light"] = build_lighting_from_config(noise_config_model.light.to_dict())
    
    # Process table texture configuration
    if noise_config_model.table_texture:
        result_config["table_texture"] = build_table_texture_from_config(noise_config_model.table_texture.to_dict())
        
        # Apply the material to the table if provided
        if table_object and "table_material" in result_config["table_texture"]:
            material = result_config["table_texture"]["table_material"]
            
            # Assign material to table
            if table_object.data.materials:
                table_object.data.materials[0] = material
            else:
                table_object.data.materials.append(material)
    
    # Process blur configuration
    if noise_config_model.blur:
        blur_arg = {"blur": noise_config_model.blur.intensity}
        result_config["blur"] = build_blur_from_config(blur_arg)
    
    return result_config