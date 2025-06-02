import bpy
import os
import sys
import random
import math
from typing import Optional, Dict, Any, Tuple, Union, List

# Ensure workspace root is in path for sibling imports
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if workspace_root not in sys.path:
    sys.path.append(workspace_root)

# Import necessary functions and models
from utils.blender_utils import render_scene
from scene_setup.general_setup import build_setup_from_config
from poker.config.models import ChipModel, ChipPileModel # Import models

OUTPUT_IMAGE_PATH = "poker/img/chip_render.png" # Define output path
_DEFAULT_CHIPS_BLEND_FILE = "poker/blend_files/CHIPS.blend" 

class PokerChipLoader:
    """Manages loading poker chip assets from a blend file (Placeholder)."""

    def __init__(self, blend_file_path: Optional[str] = None):
        self.blend_file_path = blend_file_path or _DEFAULT_CHIPS_BLEND_FILE
        self.loaded_chips: Dict[str, bpy.types.Object] = {}
        self._ensure_file_exists()
        print(f"[PokerChipLoader] Initialized (Placeholder) with: {self.blend_file_path}")

    def _ensure_file_exists(self):
        abs_path = os.path.abspath(self.blend_file_path)
        if not os.path.exists(abs_path):
            print(f"Warning: Chip blend file not found at {abs_path}. Chip loading will likely fail.")
            # In a real scenario, might raise FileNotFoundError or create a dummy file

    def link_chip(self, chip_name: str) -> Optional[bpy.types.Object]:
        """Links a chip object from the blend file (Placeholder)."""
        print(f"  [PokerChipLoader] Attempting to link chip: '{chip_name}' (Placeholder)")
        # In a real implementation:
        # 1. Check if chip_name is already loaded (cached)
        # 2. If not, use bpy.data.libraries.load() to load the object
        # 3. Handle errors (object not found in blend file)
        # 4. Create a linked copy (bpy.data.objects.new linked to loaded data)
        # 5. Cache the original loaded object data if needed for efficiency
        
        # Placeholder: Create a simple cylinder as a dummy chip
        try:
            # Ensure the dummy chip has a unique name to avoid conflicts
            dummy_name = f"{chip_name}_linked_dummy_{random.randint(1000,9999)}"
            bpy.ops.mesh.primitive_cylinder_add(radius=0.04, depth=0.01, location=(0,0,-10)) # Create off-screen
            dummy_chip = bpy.context.object
            dummy_chip.name = dummy_name
            print(f"    Created dummy cylinder '{dummy_chip.name}'")
            return dummy_chip
        except Exception as e:
            print(f"    Error creating dummy chip: {e}")
            return None

    def get_chip_dimensions(self, chip_name: str) -> Optional[tuple[float, float, float]]:
         """Gets the dimensions of a chip (Placeholder)."""
         print(f"  [PokerChipLoader] Getting dimensions for: '{chip_name}' (Placeholder)")
         # Real implementation: Load/link the chip, get dimensions, maybe cache
         # Placeholder: Return fixed dimensions matching dummy cylinder
         return (0.08, 0.08, 0.01)

# ---------------- Internal Chip Loader (build_chip_from_config calls this) -------------

def load_chip(
    chip_object_name: str,
    location: Tuple[float, float, float],
    scale_tuple: Tuple[float, float, float], 
    color: Optional[Tuple[float, float, float, float]], 
    blend_file_path: str 
) -> Optional[bpy.types.Object]:
    """
    Internal function: Loads a specific chip object, makes it local, sets its transform and color.
    This function is intended to be called multiple times and should create independent objects.

    Args:
        chip_object_name: Name of the chip object to load.
        location: Target (x, y, z) location for the chip.
        scale_tuple: Target (x, y, z) scale for the chip.
        color: Target (r, g, b, a) base color override. If None, no color change applied.
        blend_file_path: Path to the .blend file.

    Returns:
        The processed chip object if successful, None otherwise.
    """
    abs_blend_file_path = os.path.abspath(blend_file_path)
    if not os.path.exists(abs_blend_file_path):
        print(f"Error: Blend file not found at {abs_blend_file_path}")
        return None

    
    # Get object names before linking
    before_object_names = set(o.name for o in bpy.data.objects)
    
    try:
        with bpy.data.libraries.load(abs_blend_file_path, link=True) as (data_from, data_to):
            if chip_object_name in data_from.objects:
                data_to.objects = [chip_object_name]
                if not data_to.objects:
                     print(f"Error: Failed to stage object data '{chip_object_name}' for linking.")
                     return None
            else:
                print(f"Error: Object '{chip_object_name}' not found in library '{abs_blend_file_path}'")
                return None
    except Exception as e:
        print(f"Error during bpy.data.libraries.load: {e}")
        return None

    # Get object names after linking
    after_object_names = set(o.name for o in bpy.data.objects)
    new_names = after_object_names - before_object_names

    linked_obj_instance = None
    if len(new_names) == 1:
        new_name = list(new_names)[0]
        linked_obj_instance = bpy.data.objects.get(new_name)
        
    elif len(new_names) == 0:
        # This might happen if Blender reused an existing instance AND data block without creating anything new
        # Or if the link operation silently failed to create an instance.
        # Let's try finding an existing unlinked instance from the library as a fallback.

        found_existing = False
        for obj in bpy.data.objects:
            if obj.name.startswith(chip_object_name) and obj.library and obj.library.filepath == abs_blend_file_path:
                 # Check if it's NOT linked to the current scene collection yet
                 if not obj.users_collection or not any(coll == bpy.context.collection for coll in obj.users_collection):
                      linked_obj_instance = obj
                      found_existing = True
                      break 
        if not found_existing:
             print(f"Error: Linking '{chip_object_name}' created no new instance and no suitable existing instance found.")
             return None
    else:
        # Multiple new objects were created - unexpected scenario
        print(f"Error: Linking '{chip_object_name}' unexpectedly created multiple new objects: {new_names}")
        return None

    # Check if we successfully found an instance
    if not linked_obj_instance:
        print(f"Error: Failed to identify the linked object instance for '{chip_object_name}'.")
        return None

    # Link the instance to the current scene's collection (if not already)
    if not any(coll == bpy.context.collection for coll in linked_obj_instance.users_collection):
        try:
            bpy.context.collection.objects.link(linked_obj_instance)
        except RuntimeError as e:
            try: bpy.data.objects.remove(linked_obj_instance, do_unlink=True)
            except: pass
            return None

    # --- Make Local --- 
    if not linked_obj_instance:
         print("Error: Lost reference to object instance before making local.")
         return None
    local_chip_object = None 
    if linked_obj_instance.library:
        # print(f"Making '{linked_obj_instance.name}' and its data local...")
        bpy.ops.object.select_all(action='DESELECT')
        bpy.context.view_layer.objects.active = linked_obj_instance
        linked_obj_instance.select_set(True)
        try:
            bpy.ops.object.make_local(type='ALL') # Crucial: Make Object, Data, and Materials local
            local_chip_object = bpy.context.view_layer.objects.active 
            if local_chip_object.name != linked_obj_instance.name:
                 print(f"Warning: Object name changed after make_local() to {local_chip_object.name}")
        except Exception as e:
            print(f"Error during make_local: {e}")
            bpy.ops.object.select_all(action='DESELECT') # Ensure deselection on error
            return None
    else:
         # print(f"Object '{linked_obj_instance.name}' is already local.")
         local_chip_object = linked_obj_instance
    bpy.ops.object.select_all(action='DESELECT')
    if not local_chip_object:
        print("Error: Failed to get local object reference.")
        return None

    # --- Transform --- 
    # print(f"Setting location to {location} and scale to {scale_tuple}...") # Reduced verbosity
    local_chip_object.location = location
    local_chip_object.scale = scale_tuple

    # --- Change Color (Optional) --- 
    if color:
        # print(f"Attempting to set Base Color to {color} for all materials...")
        if not local_chip_object.material_slots:
            print("  Warning: Loaded object has no material slots.")
        else:
            modified_count = 0
            for slot in local_chip_object.material_slots:
                mat = slot.material
                if not mat: continue
                if not mat.use_nodes: mat.use_nodes = True
                principled_node = next((n for n in mat.node_tree.nodes if n.type == 'BSDF_PRINCIPLED'), None)
                if principled_node:
                    try:
                        principled_node.inputs["Base Color"].default_value = color
                        modified_count += 1 
                    except KeyError: pass # Ignore if no base color input
                    except Exception as e:
                        print(f"  Error setting Base Color for material '{mat.name}': {e}")
                # else: print(f"  Warning: Could not find Principled BSDF node in material '{mat.name}'.")
            # print(f"Finished processing materials. Set base color for {modified_count} material(s).")
    # else: print("Skipping color change as no color was provided.")

    return local_chip_object

# ---------------- Single Chip Builder (used by pile builder) ----------------------

def build_chip_from_config(
    config: Union[Dict[str, Any], ChipModel]
) -> Optional[bpy.types.Object]:
    """
    Loads and configures a single chip based on a ChipModel or dictionary config.

    Args:
        config: A ChipModel instance or a dictionary conforming to its structure.

    Returns:
        The loaded and processed chip object, or None if an error occurs.
    """
    if isinstance(config, dict):
        try:
            chip_model = ChipModel.from_dict(config)
        except (TypeError, ValueError) as e:
             print(f"Error parsing chip config dictionary: {e}")
             return None
    elif isinstance(config, ChipModel):
        chip_model = config
    else:
        raise TypeError(f"Expected dict or ChipModel, got {type(config)}")

    blend_file = chip_model.blend_file_path or _DEFAULT_CHIPS_BLEND_FILE
    scale_input = chip_model.scale
    if isinstance(scale_input, (int, float)):
        scale_tuple = (float(scale_input), float(scale_input), float(scale_input))
    elif isinstance(scale_input, (list, tuple)) and len(scale_input) == 3:
         scale_tuple = tuple(float(v) for v in scale_input)
    else:
         print(f"Error: Invalid scale format '{scale_input}'. Using default (1,1,1).")
         scale_tuple = (1.0, 1.0, 1.0)

    # print(f"Building chip: {chip_model.chip_object_name} at {chip_model.location} with scale {scale_tuple}...") # Reduced verbosity
    return load_chip(
        blend_file_path=blend_file,
        chip_object_name=chip_model.chip_object_name,
        location=chip_model.location,
        scale_tuple=scale_tuple,
        color=chip_model.color
    )

# ---------------- Chip Pile Builder ---------------------------------------------

def build_pile_from_config(
    config: Union[Dict[str, Any], ChipPileModel]
) -> List[bpy.types.Object]:
    """
    Builds a pile (stack) of chips based on a configuration.

    Args:
        config: A ChipPileModel instance or a dictionary conforming to its structure.

    Returns:
        A list of the created Blender chip objects in the pile, or an empty list on error.
    """
    if isinstance(config, dict):
        try:
            pile_model = ChipPileModel.from_dict(config)
        except (TypeError, ValueError) as e:
             print(f"Error parsing chip pile config dictionary: {e}")
             return []
    elif isinstance(config, ChipPileModel):
        pile_model = config
    else:
        raise TypeError(f"Expected dict or ChipPileModel, got {type(config)}")

    if pile_model.random_seed is not None:
        print(f"Setting random seed to: {pile_model.random_seed}")
        random.seed(pile_model.random_seed)

    # --- Measure Chip Dimensions --- 
    chip_height = None
    chip_width = None
    temp_chip_obj = None
    try:
        print("Measuring chip dimensions...")
        # Create a base config copy, ensure no color override for measurement
        measure_config = pile_model.base_chip_config
        if isinstance(measure_config, ChipModel):
             measure_config = measure_config.to_dict()
        measure_config = measure_config.copy() # Make a copy to modify
        measure_config['color'] = None # Don't waste time coloring the temp chip
        measure_config['location'] = (0,0,-100) # Place it far away

        temp_chip_obj = build_chip_from_config(measure_config)
        if not temp_chip_obj:
            raise RuntimeError("Failed to create temporary chip for measurement.")
        
        # Ensure dimensions are up-to-date
        bpy.context.view_layer.update()
        chip_height = temp_chip_obj.dimensions.z
        chip_width = temp_chip_obj.dimensions.x # Assuming x is the relevant width
        print(f"Measured chip dimensions (H x W): {chip_height:.4f} x {chip_width:.4f}")
        if chip_height <= 0:
            print("Warning: Measured chip height is zero or negative. Using fallback 0.01")
            chip_height = 0.01 # Fallback height

    except Exception as e:
        print(f"Error measuring chip dimensions: {e}")
        return [] # Cannot proceed without dimensions
    finally:
        # --- Clean up temporary chip --- 
        if temp_chip_obj:
            try:
                 bpy.data.objects.remove(temp_chip_obj, do_unlink=True)
                 print("Cleaned up temporary measurement chip.")
            except Exception as e_clean:
                 print(f"Error cleaning up temporary chip: {e_clean}")
    
    if chip_height is None or chip_width is None:
         print("Error: Failed to determine chip dimensions.")
         return []

    # --- Build the Pile --- 
    created_chips = []
    pile_base_location = pile_model.location
    max_spread_radius = pile_model.spread_factor * chip_width * 0.1 # 10% of width at max spread

    print(f"Building pile of {pile_model.n_chips} chips at {pile_base_location}...")
    for i in range(pile_model.n_chips):
        # Calculate position for this chip
        z_pos = pile_base_location[2] + (i * (chip_height + pile_model.vertical_gap))
        
        x_offset = 0.0
        y_offset = 0.0
        if pile_model.spread_factor > 0:
            radius = random.uniform(0, max_spread_radius)
            angle = random.uniform(0, 2 * math.pi)
            x_offset = radius * math.cos(angle)
            y_offset = radius * math.sin(angle)
            
        current_location = (pile_base_location[0] + x_offset, pile_base_location[1] + y_offset, z_pos)
        
        # Create config for this specific chip instance
        chip_instance_config = pile_model.base_chip_config
        if isinstance(chip_instance_config, ChipModel):
             chip_instance_config = chip_instance_config.to_dict()
        chip_instance_config = chip_instance_config.copy()
        chip_instance_config['location'] = current_location
        # Ensure scale/color/etc from base are preserved unless overridden
        
        # Build the chip
        chip_obj = build_chip_from_config(chip_instance_config)
        if chip_obj:
            created_chips.append(chip_obj)
        else:
            print(f"Warning: Failed to create chip {i+1} in the pile.")
            # Optionally stop here: return []

    print(f"Finished building pile. Created {len(created_chips)} chips.")
    return created_chips


if __name__ == "__main__":
    # --- Configuration ---
    # Define the base chip type
    base_chip = {
        "chip_object_name": "Cylinder001", # Use a common cylinder type
        "scale": 0.1,
        "color": (0.9, 0.9, 0.1, 1.0) # Yellowish
    }

    # Define configurations for multiple piles
    pile_configs = [
        {
            "n_chips": 5,
            "base_chip_config": base_chip,
            "location": (-0.3, 0, 0.91), # Pile 1 location
            "spread_factor": 0.0, # Perfectly stacked
            "vertical_gap": 0.005
        },
        {
            "n_chips": 8,
            "base_chip_config": {**base_chip, "color": (0.1, 0.8, 0.9, 1.0)}, # Teal chips
            "location": (0.0, 0.1, 0.91), # Pile 2 location
            "spread_factor": 0.5, # Medium spread
            "random_seed": 123
        },
        {
            "n_chips": 3,
            "base_chip_config": {**base_chip, "color": (0.9, 0.2, 0.7, 1.0)}, # Pink chips
            "location": (0.3, -0.1, 0.91), # Pile 3 location
            "spread_factor": 1.0, # Max spread (within 10% radius)
             "vertical_gap": 0.0
        },
    ]

    # Scene setup configuration
    scene_setup_config = {
        "camera": {"distance": 3.0, "angle": 50}, # Adjust for piles
        "lighting": {"lighting": "medium"},
        "table": {"diameter": 1.2, "felt_color": (0.2, 0.5, 0.2, 1.0)},
        "render": {"engine": "CYCLES", "samples": 64} 
    }
    # --- End Configuration ---

    if bpy.context is None:
        print("Error: This script must be run from within Blender or using 'blender --python'")
    else:
        print("--- Starting Chip Pile Loading and Scene Setup ---")
        all_piles_built = True
        all_created_objects = []
        try:
            # 1. Build General Scene Setup
            print("Building scene setup from config...")
            build_setup_from_config(scene_setup_config)
            print("Scene setup complete.")

            # 2. Build all chip piles
            for i, pile_conf in enumerate(pile_configs):
                print(f"\nAttempting to build pile {i+1} from config...")
                pile_objects = build_pile_from_config(pile_conf)
                if pile_objects:
                    print(f"Successfully built pile {i+1} with {len(pile_objects)} chips.")
                    all_created_objects.extend(pile_objects)
                else:
                    print(f"Failed to build pile {i+1}.")
                    all_piles_built = False
                    # break # Optionally stop if a pile fails
            
            if not all_created_objects:
                 print("Error: No chips were successfully created in any pile.")
                 all_piles_built = False

            # 3. Render the scene only if piles were built
            if all_piles_built:
                print(f"\nRendering scene with {len(all_created_objects)} total chip(s) to {OUTPUT_IMAGE_PATH}...")
                render_scene(OUTPUT_IMAGE_PATH)
                print(f"Rendering complete. Output saved to {OUTPUT_IMAGE_PATH}")
            else:
                print("\nSkipping render because one or more piles failed to build.")

            print("--- Script finished ---")

        except Exception as e:
            print(f"An error occurred during the main execution: {e}")
            import traceback
            traceback.print_exc() 