import random
from typing import Any

import bpy

from Noise.models import TableTextureNoiseModel


def create_table_material(entropy="low", name="TableMaterial"):
    """
    Create a material for a table with specified texture entropy.

    Args:
        entropy: Either a string preset ("low", "medium", "high")
                or an integer value (0, 1, 2)
        name: Name for the created material

    Returns:
        bpy.types.Material: The created material
    """
    # Process entropy setting
    if isinstance(entropy, str):
        if entropy not in TableTextureNoiseModel.TEXTURE_ENTROPY_PRESETS:
            raise ValueError(
                f"Invalid texture entropy: {entropy}. "
                f"Must be one of {list(TableTextureNoiseModel.TEXTURE_ENTROPY_PRESETS.keys())} or an integer (0-2)"
            )
        entropy_level = TableTextureNoiseModel.TEXTURE_ENTROPY_PRESETS[entropy]
    else:
        entropy_level = entropy

    # Create a new material
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links

    # Clear default nodes
    for node in nodes:
        nodes.remove(node)

    # Create output node
    output_node = nodes.new(type='ShaderNodeOutputMaterial')
    output_node.location = (600, 0)

    # Create principled BSDF node
    bsdf_node = nodes.new(type='ShaderNodeBsdfPrincipled')
    bsdf_node.location = (400, 0)

    # Link BSDF to output
    links.new(bsdf_node.outputs["BSDF"], output_node.inputs["Surface"])

    # Set up material based on entropy level
    if entropy_level == 0:
        # Low entropy: Simple monochrome
        color = random.choice(TableTextureNoiseModel.LOW_ENTROPY_COLORS)
        bsdf_node.inputs["Base Color"].default_value = color
        bsdf_node.inputs["Roughness"].default_value = random.uniform(0.3, 0.7)

    elif entropy_level == 1:
        # Medium entropy: Noise texture with color variation
        # Add noise texture
        noise_node = nodes.new(type='ShaderNodeTexNoise')
        noise_node.location = (0, 0)
        noise_node.inputs["Scale"].default_value = random.uniform(2.0, 5.0)
        noise_node.inputs["Detail"].default_value = random.uniform(2.0, 6.0)

        # Add color ramp for more control
        ramp_node = nodes.new(type='ShaderNodeValToRGB')
        ramp_node.location = (200, 0)

        # Set random colors for the ramp
        color1 = random.choice(TableTextureNoiseModel.LOW_ENTROPY_COLORS)
        color2 = random.choice(TableTextureNoiseModel.LOW_ENTROPY_COLORS)
        ramp_node.color_ramp.elements[0].color = color1
        ramp_node.color_ramp.elements[1].color = color2

        # Link nodes
        links.new(noise_node.outputs["Fac"], ramp_node.inputs["Fac"])
        links.new(ramp_node.outputs["Color"], bsdf_node.inputs["Base Color"])

        # Add some roughness variation
        links.new(noise_node.outputs["Fac"], bsdf_node.inputs["Roughness"])

    else:  # entropy_level == 2
        # High entropy: Complex texture with shapes and colors
        # Add two noise textures for more complexity
        noise1_node = nodes.new(type='ShaderNodeTexNoise')
        noise1_node.location = (-200, 100)
        noise1_node.inputs["Scale"].default_value = random.uniform(3.0, 8.0)
        noise1_node.inputs["Detail"].default_value = random.uniform(4.0, 8.0)

        noise2_node = nodes.new(type='ShaderNodeTexNoise')
        noise2_node.location = (-200, -100)
        noise2_node.inputs["Scale"].default_value = random.uniform(10.0, 20.0)
        noise2_node.inputs["Detail"].default_value = random.uniform(2.0, 4.0)

        # Add color ramps
        ramp1_node = nodes.new(type='ShaderNodeValToRGB')
        ramp1_node.location = (0, 100)

        ramp2_node = nodes.new(type='ShaderNodeValToRGB')
        ramp2_node.location = (0, -100)

        # Set random colors for the ramps with more vibrant options
        ramp1_node.color_ramp.elements[0].color = (
            random.uniform(0.2, 0.8),
            random.uniform(0.2, 0.8),
            random.uniform(0.2, 0.8),
            1.0
        )
        ramp1_node.color_ramp.elements[1].color = (
            random.uniform(0.2, 0.8),
            random.uniform(0.2, 0.8),
            random.uniform(0.2, 0.8),
            1.0
        )

        ramp2_node.color_ramp.elements[0].color = (
            random.uniform(0.2, 0.8),
            random.uniform(0.2, 0.8),
            random.uniform(0.2, 0.8),
            1.0
        )
        ramp2_node.color_ramp.elements[1].color = (
            random.uniform(0.2, 0.8),
            random.uniform(0.2, 0.8),
            random.uniform(0.2, 0.8),
            1.0
        )

        # Mix the two textures
        mix_node = nodes.new(type='ShaderNodeMixRGB')
        mix_node.location = (200, 0)
        mix_node.blend_type = random.choice(['MIX', 'ADD', 'MULTIPLY', 'OVERLAY'])
        mix_node.inputs["Fac"].default_value = random.uniform(0.3, 0.7)

        # Link nodes
        links.new(noise1_node.outputs["Fac"], ramp1_node.inputs["Fac"])
        links.new(noise2_node.outputs["Fac"], ramp2_node.inputs["Fac"])
        links.new(ramp1_node.outputs["Color"], mix_node.inputs[1])
        links.new(ramp2_node.outputs["Color"], mix_node.inputs[2])
        links.new(mix_node.outputs["Color"], bsdf_node.inputs["Base Color"])

        # Add bump for more texture
        bump_node = nodes.new(type='ShaderNodeBump')
        bump_node.location = (200, -200)
        links.new(noise2_node.outputs["Fac"], bump_node.inputs["Height"])
        links.new(bump_node.outputs["Normal"], bsdf_node.inputs["Normal"])

    return material


def build_table_texture_from_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    Process a table texture configuration dictionary and create a material.

    Args:
        config: Dictionary with texture configuration parameter:
               - "table_texture": str ("low", "medium", "high") or int (0, 1, 2)

    Returns:
        Dict: Updated configuration dictionary with final values and created material
    """
    # Create table texture model from config
    texture_model = TableTextureNoiseModel.from_dict(config)

    # Create the material
    material = create_table_material(texture_model.table_texture)

    # Process entropy value for config
    if isinstance(texture_model.table_texture, str):
        entropy_level = TableTextureNoiseModel.TEXTURE_ENTROPY_PRESETS[texture_model.table_texture]
    else:
        entropy_level = texture_model.table_texture

    # Store the final values in the config
    config["final_table_texture_level"] = entropy_level
    config["table_material"] = material

    return config
