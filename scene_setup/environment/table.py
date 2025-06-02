from typing import Any

import bpy

from scene_setup.models import MaterialModel, TableModel, TableShape, TableTexture

# Clear existing objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()


# Create a simple table
def create_table(shape="rectangular", length=2.0, width=1.5, height=0.9,
                 texture="wood", custom_material=None):
    """
    Create a table with specified shape, dimensions, and texture.

    Args:
        shape: The shape of the table ("rectangular", "circular", "elliptic")
        length: The length of the table (Y dimension)
        width: The width of the table (X dimension)
        height: The height of the table
        texture: The texture type ("wood", "marble", "metal")
        custom_material: Optional custom material to use instead of built-in textures

    Returns:
        The created table object
    """
    # Create a table model
    material_model = MaterialModel(custom_material=custom_material)
    table_model = TableModel(
        shape=TableShape(shape),
        length=length,
        width=width,
        height=height,
        texture=TableTexture(texture),
        material=material_model
    )

    # Table dimensions from model
    table_height = table_model.height
    table_top_thickness = 0.03

    # Create table top based on shape
    if table_model.shape == TableShape.CIRCULAR:
        # Create a circular table
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=32,
            radius=table_model.width/2,
            depth=table_top_thickness,
            location=(0, 0, table_height - table_top_thickness/2)
        )
        table_top = bpy.context.active_object

    elif table_model.shape == TableShape.ELLIPTIC:
        # Create a circular table and scale it to make it elliptical
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=32,
            radius=1.0,
            depth=table_top_thickness,
            location=(0, 0, table_height - table_top_thickness/2)
        )
        table_top = bpy.context.active_object
        table_top.scale.x = table_model.width/2
        table_top.scale.y = table_model.length/2

    else:  # Default to rectangular
        bpy.ops.mesh.primitive_cube_add(
            size=1,
            location=(0, 0, table_height - table_top_thickness/2)
        )
        table_top = bpy.context.active_object
        table_top.scale.x = table_model.width
        table_top.scale.y = table_model.length
        table_top.scale.z = table_top_thickness

    table_top.name = "TableTop"

    # Use custom material if provided, otherwise create one based on texture type
    if table_model.material.custom_material:
        material = table_model.material.custom_material
    else:
        material = bpy.data.materials.new(name="TableMaterial")
        material.use_nodes = True
        nodes = material.node_tree.nodes
        links = material.node_tree.links
        bsdf = nodes["Principled BSDF"]

        if table_model.texture == TableTexture.WOOD:
            # Wood texture
            bsdf.inputs["Base Color"].default_value = (0.6, 0.3, 0.1, 1.0)
            bsdf.inputs["Roughness"].default_value = 0.3

            # Add noise texture for wood grain
            tex_coord = nodes.new(type='ShaderNodeTexCoord')
            mapping = nodes.new(type='ShaderNodeMapping')
            noise = nodes.new(type='ShaderNodeTexNoise')
            color_ramp = nodes.new(type='ShaderNodeValToRGB')

            # Position nodes
            tex_coord.location = (-800, 0)
            mapping.location = (-600, 0)
            noise.location = (-400, 0)
            color_ramp.location = (-200, 0)

            # Configure noise for wood grain
            noise.inputs["Scale"].default_value = 20.0
            noise.inputs["Detail"].default_value = 10.0
            noise.inputs["Distortion"].default_value = 0.2

            # Configure color ramp for wood tones
            color_ramp.color_ramp.elements[0].color = (0.4, 0.2, 0.05, 1.0)
            color_ramp.color_ramp.elements[1].color = (0.7, 0.4, 0.15, 1.0)

            # Link nodes
            links.new(tex_coord.outputs["Object"], mapping.inputs["Vector"])
            links.new(mapping.outputs["Vector"], noise.inputs["Vector"])
            links.new(noise.outputs["Fac"], color_ramp.inputs["Fac"])
            links.new(color_ramp.outputs["Color"], bsdf.inputs["Base Color"])

        elif table_model.texture == TableTexture.MARBLE:
            # Marble texture
            bsdf.inputs["Base Color"].default_value = (0.9, 0.9, 0.9, 1.0)
            bsdf.inputs["Roughness"].default_value = 0.1
            bsdf.inputs["Specular"].default_value = 0.8

            # Add noise texture for marble pattern
            tex_coord = nodes.new(type='ShaderNodeTexCoord')
            mapping = nodes.new(type='ShaderNodeMapping')
            noise = nodes.new(type='ShaderNodeTexVoronoi')
            color_ramp = nodes.new(type='ShaderNodeValToRGB')

            # Position nodes
            tex_coord.location = (-800, 0)
            mapping.location = (-600, 0)
            noise.location = (-400, 0)
            color_ramp.location = (-200, 0)

            # Configure noise for marble pattern
            noise.inputs["Scale"].default_value = 5.0

            # Configure color ramp for marble tones
            color_ramp.color_ramp.elements[0].color = (0.8, 0.8, 0.8, 1.0)
            color_ramp.color_ramp.elements[1].color = (0.95, 0.95, 0.95, 1.0)

            # Link nodes
            links.new(tex_coord.outputs["Object"], mapping.inputs["Vector"])
            links.new(mapping.outputs["Vector"], noise.inputs["Vector"])
            links.new(noise.outputs["Distance"], color_ramp.inputs["Fac"])
            links.new(color_ramp.outputs["Color"], bsdf.inputs["Base Color"])

        elif table_model.texture == TableTexture.METAL:
            # Metal texture
            bsdf.inputs["Base Color"].default_value = (0.8, 0.8, 0.9, 1.0)
            bsdf.inputs["Roughness"].default_value = 0.1
            bsdf.inputs["Metallic"].default_value = 1.0
            bsdf.inputs["Specular"].default_value = 0.9

        else:  # Default to plain
            # Plain red color (as in original)
            bsdf.inputs["Base Color"].default_value = (1.0, 0.0, 0.0, 1.0)
            bsdf.inputs["Roughness"].default_value = 0.2

    # Assign material to table
    if table_top.data.materials:
        table_top.data.materials[0] = material
    else:
        table_top.data.materials.append(material)

    return table_top


def create_table_from_config(config: dict[str, Any]) -> bpy.types.Object:
    """
    Create a table based on the provided configuration.

    Args:
        config: A dictionary containing the table configuration:
            - shape: The shape of the table ("rectangular", "circular", "elliptic")
            - length: The length of the table (Y dimension)
            - width: The width of the table (X dimension)
            - texture: The texture type ("wood", "marble", "metal")
            - material: Optional custom material to use
            - height: Optional table height (default: 0.9)

    Returns:
        The created table object
    """
    # Create table model from config
    table_model = TableModel.from_dict(config)

    # Create the table using the base function
    return create_table(
        shape=table_model.shape.value,
        length=table_model.length,
        width=table_model.width,
        height=table_model.height,
        texture=table_model.texture.value,
        custom_material=table_model.material.custom_material
    )


# Main function
def test_table():

    from scene_setup import (
        build_camera_from_config,
        create_floor,
        setup_lighting,
        setup_render,
    )

    # Create the scene elements
    table_config = {
        "shape": "elliptic",
        "length": 2.5,
        "width": 1.8,
        "texture": "wood"
    }
    table = create_table_from_config(table_config)

    create_floor(color=(0.6, 0.6, 0.6, 1), roughness=0.6)

    # Set up camera using scene_setup module
    camera_config = {
        "distance": 5.,  # Medium distance from the table
        "angle": 60.,     # Medium angle (around 60 degrees from horizontal)
    }

    build_camera_from_config(camera_config)
    setup_lighting(powers=(20000, 10000, 10000))
    setup_render(gpu_enabled=True)  # GPU settings are now handled by setup_render

    # Add water drops to the table
    # Set the active object to the table
    bpy.context.view_layer.objects.active = table
    table.select_set(True)

    print("Scene created successfully!")

    # render the scene
    bpy.context.scene.render.filepath = "table_scene.png"
    bpy.ops.render.render(write_still=True)



