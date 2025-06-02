import bpy
from scene_setup.models import FloorModel, MaterialModel
from typing import Dict, Any


# Create a simple floor
def create_floor(color=(0.8, 0.8, 0.8, 1.0), roughness=0.5):
    """
    Create a floor plane with specified material properties.
    
    Args:
        color: RGBA color tuple for the floor
        roughness: Roughness value for the floor material
        
    Returns:
        The created floor object
    """
    # Create a floor model
    material_model = MaterialModel(color=color, roughness=roughness)
    floor_model = FloorModel(color=color, roughness=roughness, material=material_model)
    
    # Create floor plane
    bpy.ops.mesh.primitive_plane_add(size=10, location=(0, 0, 0))
    floor = bpy.context.active_object
    floor.name = "Floor"
    
    # Create floor material
    material = bpy.data.materials.new(name="FloorMaterial")
    material.use_nodes = True
    bsdf = material.node_tree.nodes["Principled BSDF"]

    # Set material properties from model
    bsdf.inputs["Base Color"].default_value = floor_model.color
    bsdf.inputs["Roughness"].default_value = floor_model.roughness
    
    # Assign material to floor
    if floor.data.materials:
        floor.data.materials[0] = material
    else:
        floor.data.materials.append(material)
    
    return floor


def create_floor_from_config(config: Dict[str, Any]) -> bpy.types.Object:
    """
    Create a floor based on the provided configuration.
    
    Args:
        config: A dictionary containing the floor configuration:
            - color: RGBA color tuple for the floor
            - roughness: Roughness value for the floor material
        
    Returns:
        The created floor object
    """
    # Create floor model from config
    floor_model = FloorModel.from_dict(config)
    
    # Create the floor using the base function
    return create_floor(
        color=floor_model.color,
        roughness=floor_model.roughness
    )
