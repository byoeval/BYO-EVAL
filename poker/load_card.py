import bpy
import os
import math
from typing import Tuple, Dict, List, Optional, Union, Any

# Only import CardModel needed here now
from poker.config.models import CardModel

_POKER_DECK_BLEND_FILE = "poker/blend_files/poker_deck.blend"
_CARD_BACK_MATERIAL_NAME = "CardBackMaterial"

def _get_or_create_card_back_material() -> bpy.types.Material:
    """
    Retrieves the shared card back material, creating it if it doesn't exist.

    Returns:
        bpy.types.Material: The card back material.
    """
    # Check if the material already exists
    mat = bpy.data.materials.get(_CARD_BACK_MATERIAL_NAME)

    if mat is None:
        # Create material
        mat = bpy.data.materials.new(name=_CARD_BACK_MATERIAL_NAME)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        principled_bsdf = nodes.get("Principled BSDF")

        if principled_bsdf:
            # Set base color (e.g., dark blue)
            principled_bsdf.inputs["Base Color"].default_value = (0.02, 0.03, 0.2, 1.0)
            # Set roughness
            principled_bsdf.inputs["Roughness"].default_value = 0.6
            # Explicitly set alpha to 1.0 for full opacity
            principled_bsdf.inputs["Alpha"].default_value = 1.0
        print(f"Created new material: '{_CARD_BACK_MATERIAL_NAME}'")
    else:
        print(f"Reusing existing material: '{_CARD_BACK_MATERIAL_NAME}'")

    return mat

class PokerCardLoader:
    """
    Manages loading poker card objects from a specified .blend file into the scene.

    Attributes:
        blend_file_path (str): The path to the .blend file containing the card assets.
    """

    def __init__(self, blend_file_path: str = _POKER_DECK_BLEND_FILE):
        """
        Initializes the PokerCardLoader.

        Args:
            blend_file_path (str, optional): Path to the blend file containing the cards.
                                             Defaults to the value of _POKER_DECK_BLEND_FILE.

        Raises:
            FileNotFoundError: If the specified blend file does not exist.
        """
        if not os.path.exists(blend_file_path):
            raise FileNotFoundError(f"Error: Blend file not found at '{blend_file_path}'")
        self.blend_file_path = blend_file_path
        print(f"PokerCardLoader initialized with blend file: '{self.blend_file_path}'")

    def load_card(
        self,
        card_name: str,
        position: Tuple[float, float, float],
        scale: Union[float, Tuple[float, float, float]] = 1.0,
        face_up: bool = True,
        rotation_euler: Optional[Tuple[float, float, float]] = None,
    ) -> Optional[bpy.types.Object]:
        """
        Load a specific card object from the blend file and place it in the scene.

        Args:
            card_name (str): The name of the card (e.g., "AS", "10H", "KC").
                             Assumes object name in blend file is "Card_{card_name}".
            position (Tuple[float, float, float]): World coordinates (x, y, z) for the card.
            scale (Union[float, Tuple[float, float, float]], optional): Scale factor.
                                                                          Defaults to 1.0.
            face_up (bool, optional): If True (default), card is placed face up.
                                      If False, card is placed face down (rotated 180 deg on X).
                                      This is overridden by `rotation_euler` if provided.
            rotation_euler (Optional[Tuple[float, float, float]], optional): Specific rotation
                                                                              in radians (X, Y, Z).
                                                                              If provided, overrides `face_up`.
                                                                              Defaults to None.

        Returns:
            Optional[bpy.types.Object]: The loaded card object, or None if loading failed.
        """
        object_name_in_blend = f"Card_{card_name}"
        print(f"Attempting to load object '{object_name_in_blend}' from '{self.blend_file_path}'...")

        # --- Get object names before loading --- 
        existing_object_names = set(bpy.data.objects.keys())
        loaded_obj = None
        new_object_name = None

        # --- Always load the object from the blend file --- 
        # Blender will handle renaming if object_name_in_blend already exists (e.g., Card_9H.001)
        try:
            with bpy.data.libraries.load(self.blend_file_path, link=False) as (data_from, data_to):
                if object_name_in_blend in data_from.objects:
                    data_to.objects = [object_name_in_blend]
                    # Don't print success yet, wait until we confirm it's loaded below
                else:
                    print(f"Error: Object '{object_name_in_blend}' not found inside the blend file '{self.blend_file_path}'")
                    print(f"Available objects in blend file: {data_from.objects}")
                    return None
        except Exception as e:
            print(f"Error loading library '{self.blend_file_path}': {e}")
            return None

        # --- Find the newly added object --- 
        current_object_names = set(bpy.data.objects.keys())
        newly_added_names = current_object_names - existing_object_names

        if not newly_added_names:
            # This might happen if the object existed AND load failed silently, or some other issue
            print(f"Error: No new object found in bpy.data.objects after attempting to load '{object_name_in_blend}'.")
            # Check if the original name exists, maybe it was reused despite our intent?
            if object_name_in_blend in bpy.data.objects and object_name_in_blend not in existing_object_names:
                 print(f"Warning: Original name '{object_name_in_blend}' appeared unexpectedly. Attempting to use it.")
                 new_object_name = object_name_in_blend
            else:
                print("Could not identify the newly loaded object.")
                return None # Failed to load or identify
        elif len(newly_added_names) > 1:
            # Should not happen if we only tried to load one object
            print(f"Warning: Multiple objects ({newly_added_names}) added unexpectedly. Attempting to find one starting with the target name.")
            # Try to find the one most likely related to our loaded object
            possible_names = [name for name in newly_added_names if name.startswith(object_name_in_blend)]
            if len(possible_names) == 1:
                 new_object_name = possible_names[0]
                 print(f"Identified newly loaded object as: '{new_object_name}'")
            else:
                 print(f"Error: Could not disambiguate newly loaded object among {possible_names}. Aborting.")
                 # Clean up potentially unwanted new objects?
                 return None
        else:
            # Exactly one new object was added - this is the expected case
            new_object_name = newly_added_names.pop()
            print(f"Identified newly loaded object as: '{new_object_name}'")

        # Get the actual new object reference
        loaded_obj = bpy.data.objects.get(new_object_name)

        if not loaded_obj:
            print(f"Error: Failed to get the bpy.data.object for '{new_object_name}' after loading.")
            return None

        # --- Link the *new* object to the scene's collection --- 
        try:
            # Ensure we link the identified new object, even if name is like .001
            # Check if it's already linked (shouldn't be, but belt-and-suspenders)
            if loaded_obj.name not in bpy.context.collection.objects:
                 bpy.context.collection.objects.link(loaded_obj)
                 print(f"Successfully linked '{loaded_obj.name}' to the scene.")
            else:
                # This case should ideally not happen with the new logic
                print(f"Info: Newly loaded object '{loaded_obj.name}' was already linked? Continuing.")
        except Exception as e:
             print(f"Error linking '{loaded_obj.name}' to scene: {e}")
             # If linking fails, remove the object we just loaded to avoid clutter
             if new_object_name in bpy.data.objects:
                 bpy.data.objects.remove(bpy.data.objects[new_object_name], do_unlink=True)
             return None

        # --- Ensure Recto Card Material is Opaque --- 
        if face_up:
            for slot in loaded_obj.material_slots:
                if slot.material:
                    slot.material.blend_method = 'OPAQUE'
                    # slot.material.shadow_method = 'OPAQUE' # Removed: Attribute deprecated/changed in Blender 4+
                    print(f"Set material '{slot.material.name}' on '{loaded_obj.name}' to OPAQUE blend mode.")
                else:
                    print(f"Warning: Material slot empty on '{loaded_obj.name}'. Cannot set blend mode.")

        # --- Apply Card Back Material if face_down --- 
        if not face_up:
            if loaded_obj.material_slots:
                card_back_material = _get_or_create_card_back_material()
                # Assign to the first material slot
                loaded_obj.material_slots[0].material = card_back_material
                print(f"Applied '{_CARD_BACK_MATERIAL_NAME}' to '{loaded_obj.name}'")
            else:
                print(f"Warning: Cannot apply card back material to '{loaded_obj.name}' because it has no material slots.")

        # --- Set position, scale, and rotation for the new object --- 
        loaded_obj.location = position

        if isinstance(scale, (int, float)):
            loaded_obj.scale = (scale, scale, scale)
        elif isinstance(scale, tuple) and len(scale) == 3:
             loaded_obj.scale = scale
        else:
            print(f"Warning: Invalid scale format for '{loaded_obj.name}'. Using (1,1,1).")
            loaded_obj.scale = (1.0, 1.0, 1.0)

        # Apply rotation ONLY if explicitly provided or face_down is False
        # Removed default Z rotation logic
        if rotation_euler is not None:
            # Use explicitly provided rotation
            loaded_obj.rotation_euler = rotation_euler
            print(f"Applied explicit rotation {rotation_euler} to '{loaded_obj.name}'.")
        elif not face_up:
            # Default face-down: Rotate 180 on X ONLY
            loaded_obj.rotation_euler = (math.pi, 0, 0)
            print(f"Applied default face-down rotation (X-flip only) to '{loaded_obj.name}'.")
        else:
            # Default face-up: No rotation applied by default
            loaded_obj.rotation_euler = (0, 0, 0)

        print(f"Placed '{loaded_obj.name}' at {position} with scale {loaded_obj.scale[:]} and rotation {loaded_obj.rotation_euler[:]}")
        return loaded_obj

    def build_card_from_config(
        self,
        card_config: Union[Dict[str, Any], CardModel]
    ) -> Optional[bpy.types.Object]:
        """
        Loads and places a card based on a configuration dictionary or CardModel.

        Args:
            card_config: Either a dictionary conforming to CardModel structure
                         or a CardModel instance.

        Returns:
            Optional[bpy.types.Object]: The loaded card object, or None if loading failed.
        """
        if isinstance(card_config, dict):
            try:
                model = CardModel.from_dict(card_config)
            except (ValueError, TypeError) as e:
                print(f"Error creating CardModel from dictionary: {e}")
                print(f"Input config: {card_config}")
                return None
        elif isinstance(card_config, CardModel):
            model = card_config
        else:
            print(f"Error: Invalid card_config type. Expected dict or CardModel, got {type(card_config)}")
            return None

        # Call the main loading function with parameters from the model
        return self.load_card(
            card_name=model.card_name,
            position=model.location,
            scale=model.scale,
            face_up=model.face_up,
            rotation_euler=model.rotation_euler
        )

# --- Test Section ---
if __name__ == "__main__":
    import sys
    workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if workspace_root not in sys.path:
        sys.path.append(workspace_root)

    try:
        from scene_setup.general_setup import build_setup_from_config
        # Import the standalone river builder function
        from poker.river_builder import build_river_from_config
    except ImportError as e:
        print(f"Error importing necessary modules: {e}")
        print("Please ensure the script is run from the workspace root or PYTHONPATH is set correctly.")
        sys.exit(1)

    print("--- Running Test Setup ---")

    # 1. Basic Scene Setup
    test_config = {
        "camera": {"distance": 4.5}, # Slightly further back
        "lighting": {"lighting": "medium"},
        "table": {"diameter": 1.8, "felt_color": (0.1, 0.4, 0.15, 1.0)}, # Larger table
        "render": {"engine": "CYCLES", "samples": 64} # Lower samples for faster test
    }

    try:
        print("Building scene setup from config...")
        build_setup_from_config(test_config)
        print("Scene setup complete.")
    except Exception as e:
        print(f"Error during scene setup: {e}")
        sys.exit(1)

    # 2. Initialize Card Loader
    try:
        card_loader = PokerCardLoader()
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)
    except Exception as e:
        print(f"Error initializing PokerCardLoader: {e}")
        sys.exit(1)

    # 3. Define card configurations using build_card_from_config (Individual cards)
    # Keep a couple for verification
    card_z = 0.91 # On the table surface
    individual_card_configs = [
        {'card_name': "AS", 'location': (-0.6, 0.4, card_z), 'scale': 0.1, 'face_up': True},
        {'card_name': "KC", 'location': (0.6, 0.4, card_z), 'scale': 0.1, 'face_up': False}, # Face down
    ]

    print("\nLoading individual cards...")
    loaded_individual_cards = []
    for config in individual_card_configs:
        obj = card_loader.build_card_from_config(config)
        if obj:
            loaded_individual_cards.append(obj)

    # 4. Define River/Pile configurations using build_river_from_config
    river_layout_config = {
        'card_names': ['QH', 'JD', '10S', '9C', '8H'],
        'n_cards': 5,
        'n_columns': 5,
        'column_spacing': 0.15, # Space out horizontally
        'start_location': (-0.3, 0, card_z),
        'scale': 0.1,
        'n_verso': 0, # All face up
        'card_gap': {'base_gap_y': 0.12}, # Default gap is fine for single row
    }

    burn_pile_config = {
        'card_names': ['2D', '3C', '4S', '5H', '6D', '7C'],
        'n_cards': 6,
        'n_columns': 1, # Single column
        'start_location': (0.5, -0.3, card_z),
        'scale': 0.09,
        'n_verso': 6, # All face down
        'verso_loc': 'ordered', # Not really needed here, but for consistency
        'card_gap': {'base_gap_y': 0.005, 'random_gap': True, 'random_percentage': 0.5},
        'random_seed': 42
    }

    random_verso_pile_config = {
        'card_names': ['AD', 'KH', 'QC', 'JS'],
        'n_cards': 4,
        'n_columns': 2,
        'start_location': (-0.4, -0.4, card_z),
        'scale': 0.1,
        'n_verso': 2, # Two face down
        'verso_loc': 'random', # Place them randomly
        'column_spacing': 0.15,
        'card_gap': {'base_gap_y': 0.12, 'random_gap': False},
        'random_seed': 12345
    }

    # Use the imported standalone function, passing the loader instance
    print("\nBuilding river layout...")
    loaded_river_cards = build_river_from_config(card_loader, river_layout_config)

    print("\nBuilding burn pile...")
    loaded_burn_cards = build_river_from_config(card_loader, burn_pile_config)

    print("\nBuilding random verso pile...")
    loaded_random_verso_cards = build_river_from_config(card_loader, random_verso_pile_config)

    total_loaded_count = len(loaded_individual_cards) + len(loaded_river_cards) + len(loaded_burn_cards) + len(loaded_random_verso_cards)
    if total_loaded_count == 0:
         print("\nError: No cards were loaded successfully in any step.")
         sys.exit(1)

    print(f"\nSuccessfully loaded a total of {total_loaded_count} cards.")

    # 5. Render the scene
    output_filename = "card_river_test.png"
    output_dir = "poker/img" # Relative to workspace root
    output_path = os.path.join(output_dir, output_filename)

    os.makedirs(output_dir, exist_ok=True)

    print(f"\nSetting render output path to: {output_path}")
    bpy.context.scene.render.filepath = output_path
    bpy.context.scene.render.image_settings.file_format = 'PNG'

    try:
        print("Starting render...")
        bpy.ops.render.render(write_still=True)
        print(f"Render finished! Image saved to {output_path}")
    except Exception as e:
        print(f"Error during rendering: {e}")

    print("--- Test Finished ---") 