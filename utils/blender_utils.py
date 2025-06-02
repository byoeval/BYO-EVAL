import bpy
import os
from typing import Union, List, Optional

def enable_gpu_rendering(rendering_engine: str = "OPTIX", gpus: Optional[Union[str, int, List[int]]] = None) -> None:
    """
    Enable GPU rendering in Cycles with specific GPU selection.
    
    Args:
        rendering_engine (str): The rendering engine to use.
            - "OPTIX" : Use OptiX rendering.
            - "CUDA" : Use CUDA rendering.
            - "NONE" : Use CPU rendering.
        gpus (Optional[Union[str, int, List[int]]], optional): GPUs to use for rendering.
            - None or "all": Use all available GPUs (default)
            - int: Use a single GPU by index
            - List[int]: Use multiple GPUs by their indices
            
    Raises:
        ValueError: If invalid GPU indices are provided or if no GPUs are available
        
    Examples:
        >>> enable_gpu_rendering()  # Use all available GPUs
        >>> enable_gpu_rendering(gpus="all")  # Explicitly use all GPUs
        >>> enable_gpu_rendering(gpus=0)  # Use only the first GPU
        >>> enable_gpu_rendering(gpus=[0, 1])  # Use first two GPUs
    """
    # Get cycles preferences
    cycles_prefs = bpy.context.preferences.addons["cycles"].preferences
    
    # Set compute device type
    cycles_prefs.compute_device_type = rendering_engine

    # Set the device and feature set
    bpy.context.scene.cycles.device = "GPU" if rendering_engine != "NONE" else "CPU"

    # Get devices
    cycles_prefs.refresh_devices()
    devices = cycles_prefs.devices

    if not devices:
        raise ValueError("No compatible devices found for rendering")

    # Convert single GPU index to list for uniform processing
    if isinstance(gpus, int):
        gpus = [gpus]
    
    # Create a mapping of GPU indices to their actual devices
    gpu_devices = []
    for device in devices:
        if device.type in ['CUDA', 'OPTIX']:
            # Extract the PCI ID from the device ID to identify unique GPUs
            if hasattr(device, 'id') and '_' in device.id:
                pci_id = device.id.split('_')[2]  # Get the PCI ID part
                gpu_devices.append((device, pci_id))
    
    # Get unique PCI IDs to count actual GPUs
    unique_pci_ids = list(set(pci_id for _, pci_id in gpu_devices))
    
    # Validate GPU indices if specific GPUs are requested
    if isinstance(gpus, list):
        if any(idx >= len(unique_pci_ids) for idx in gpus):
            raise ValueError(
                f"Invalid GPU indices: {gpus}. "
                f"Available GPU indices are: {list(range(len(unique_pci_ids)))}"
            )
    
    # Enable/disable devices based on selection
    print("\nConfiguring devices:")
    print("-" * 50)
    for device in devices:
        if gpus is None or gpus == "all":
            # Enable all GPU devices, disable CPU
            device.use = device.type in ['CUDA', 'OPTIX']
        else:
            # For manual GPU selection, check if this device's PCI ID matches selected indices
            if device.type in ['CUDA', 'OPTIX'] and hasattr(device, 'id'):
                pci_id = device.id.split('_')[2]
                selected_pci_ids = [unique_pci_ids[idx] for idx in gpus]
                device.use = pci_id in selected_pci_ids
            else:
                device.use = False
                
        print(f"{device.name} ({device.type}): {'Enabled' if device.use else 'Disabled'}")
    print("-" * 50)


def detect_object_from_blend(blend_file_path, verbose=False):
    """
    Lists all objects and collections from a .blend file.
    
    Args:
        blend_file_path (str): Path to the .blend file
        verbose (bool): Whether to print detailed information
        
    Returns:
        tuple: (list of object names, list of collection names) found in the blend file
    """
    # Lists to store object and collection information
    object_info = []
    collection_info = []
    
    print(f"Scanning blend file: {blend_file_path}")
    
    # Load the blend file data without linking
    with bpy.data.libraries.load(blend_file_path, link=False) as (data_from, data_to):
        print(f"Found {len(data_from.objects)} objects and {len(data_from.collections)} collections in {blend_file_path}")
        
        # Store object names
        for obj_name in data_from.objects:
            object_info.append(obj_name)
            
        # Store collection names
        for coll_name in data_from.collections:
            collection_info.append(coll_name)
    
    if verbose:
        # Print detailed list of objects
        print("\nObjects in blend file:")
        for i, obj_name in enumerate(object_info):
            print(f"{i+1}. {obj_name}")
            
        # Print detailed list of collections
        print("\nCollections in blend file:")
        for i, coll_name in enumerate(collection_info):
            print(f"{i+1}. {coll_name}")
    
    return object_info, collection_info


def extract_object_from_blend(blend_file_path, 
                              object_name, 
                              save=False, 
                              output_path=None):
    """
    Extracts an object or collection from a blend file with all its properties, materials, 
    children, etc. Optionally saves it to a new blend file.
    
    Args:
        blend_file_path (str): Path to source .blend file
        object_name (str): Name of the object or collection to extract
        save (bool): Whether to save the extracted object to a separate file
        output_path (str): Path to save the extracted object (if save=True)
                           If None, will use object_name.blend in current directory
    
    Returns:
        bpy.types.Object or bpy.types.Collection or None: The extracted object/collection or None if not found
    """
    # Store current file state
    current_file = bpy.data.filepath
    current_scene_name = bpy.context.scene.name if bpy.context.scene else None
    
    print(f"Extracting {object_name} from {blend_file_path}")
    
    # Create a temporary scene for extraction
    temp_scene = bpy.data.scenes.new("TempExtractionScene")
    bpy.context.window.scene = temp_scene
    
    # Load all necessary data from the blend file
    with bpy.data.libraries.load(blend_file_path, link=False) as (data_from, data_to):
        # Check if it's a collection
        is_collection = object_name in data_from.collections
        
        # Check if it's an object
        is_object = object_name in data_from.objects
        
        if not is_collection and not is_object:
            print(f"'{object_name}' not found as object or collection in blend file.")
            print("Available objects:", ", ".join(data_from.objects))
            print("Available collections:", ", ".join(data_from.collections))
            bpy.data.scenes.remove(temp_scene)
            return None
        
        # Load all data to ensure complete hierarchy
        data_to.objects = data_from.objects
        data_to.materials = data_from.materials
        data_to.textures = data_from.textures
        data_to.images = data_from.images
        data_to.collections = data_from.collections
        data_to.meshes = data_from.meshes
        data_to.node_groups = data_from.node_groups
    
    target = None
    
    if is_collection:
        # Find the collection
        for collection in bpy.data.collections:
            if collection.name == object_name:
                target = collection
                # Link the collection to the scene
                temp_scene.collection.children.link(collection)
                break
    else:
        # Find the object
        for obj in bpy.data.objects:
            if obj.name == object_name or obj.name.startswith(f"{object_name}."):
                target = obj
                # Link the object to the scene
                temp_scene.collection.objects.link(obj)
                break
    
    if not target:
        print(f"Could not find '{object_name}' after loading data")
        bpy.data.scenes.remove(temp_scene)
        return None
    
    # Process materials
    def process_materials(obj):
        if obj.material_slots:
            for slot in obj.material_slots:
                if slot.material:
                    material = slot.material
                    
                    # Make sure material uses nodes
                    if not material.use_nodes:
                        material.use_nodes = True
                    
                    # Verify material node setup
                    if material.node_tree:
                        # Get or create essential nodes
                        principled_node = None
                        output_node = None
                        
                        for node in material.node_tree.nodes:
                            if node.type == 'BSDF_PRINCIPLED':
                                principled_node = node
                            elif node.type == 'OUTPUT_MATERIAL':
                                output_node = node
                        
                        # Create nodes if missing
                        if not principled_node:
                            principled_node = material.node_tree.nodes.new(type="ShaderNodeBsdfPrincipled")
                            principled_node.location = (0, 0)
                            
                        if not output_node:
                            output_node = material.node_tree.nodes.new(type="ShaderNodeOutputMaterial")
                            output_node.location = (300, 0)
                        
                        # Connect nodes if not connected
                        if not material.node_tree.links:
                            material.node_tree.links.new(
                                principled_node.outputs["BSDF"],
                                output_node.inputs["Surface"]
                            )
    
    # Process materials for all objects
    if is_collection:
        for obj in target.objects:
            process_materials(obj)
    else:
        process_materials(target)
        # Process child objects recursively
        for child in target.children_recursive:
            process_materials(child)
    
    # Save to a new blend file if requested
    if save:
        if not output_path:
            # Create default output path using object name
            output_path = f"{object_name}.blend"
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Remove all objects/collections except the target and its children
        if is_collection:
            for coll in list(temp_scene.collection.children):
                if coll != target:
                    temp_scene.collection.children.unlink(coll)
        else:
            for obj in list(temp_scene.collection.objects):
                if obj != target and obj not in target.children_recursive:
                    temp_scene.collection.objects.unlink(obj)
        
        # Save the file
        bpy.ops.wm.save_as_mainfile(filepath=output_path)
        print(f"Saved extracted {'collection' if is_collection else 'object'} to {output_path}")
    
    print(f"Successfully extracted {object_name} with all materials and children")
    
    # Clean up the temporary scene
    bpy.data.scenes.remove(temp_scene)
    
    # Restore original scene if there was one
    if current_scene_name and current_scene_name in bpy.data.scenes:
        bpy.context.window.scene = bpy.data.scenes[current_scene_name]
    
    return target


def render_scene(output_path_arg):
        bpy.context.scene.render.filepath = output_path_arg
        bpy.ops.render.render(write_still=True)

# lets test the functions
if __name__ == "__main__":
    # test detect_object_from_blend
    blend_file_path = "blend_files/chess/chess_classique.blend"
    objects, collections = detect_object_from_blend(blend_file_path)
    print(f"Objects: {objects}")
    print(f"Collections: {collections}")

    # test extract_object_from_blend
    object_name = "horse"
    extract_object_from_blend(blend_file_path, object_name, save=True, output_path="horse.blend")
    