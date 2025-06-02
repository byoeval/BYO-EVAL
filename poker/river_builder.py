import bpy
import random
import math
from typing import Union, List, Dict, Any

# Models and Loader
from poker.config.models import RiverModel, CardModel
from poker.load_card import PokerCardLoader

def build_river_from_config(
    card_loader: PokerCardLoader,
    river_config: Union[Dict[str, Any], RiverModel]
) -> List[bpy.types.Object]:
    """
    Builds a layout of multiple cards (river, pile, etc.) based on configuration,
    using the provided PokerCardLoader instance.

    Args:
        card_loader: An initialized instance of PokerCardLoader.
        river_config: Either a dictionary conforming to RiverModel structure
                      or a RiverModel instance.

    Returns:
        A list of the loaded card bpy.types.Object instances.
        Returns an empty list if configuration is invalid or loading fails.
    """
    loaded_cards = []
    try:
        if isinstance(river_config, dict):
            model = RiverModel.from_dict(river_config)
        elif isinstance(river_config, RiverModel):
            model = river_config
        else:
            raise TypeError(f"Expected dict or RiverModel, got {type(river_config)}")

        if model.random_seed is not None:
            random.seed(model.random_seed)

        # Determine which card indices should be face down
        face_down_indices = set()
        if model.n_verso > 0:
            all_indices = list(range(model.n_cards))
            if model.verso_loc == 'ordered':
                face_down_indices = set(all_indices[:model.n_verso])
            elif model.verso_loc == 'random':
                face_down_indices = set(random.sample(all_indices, model.n_verso))

        print(f"\nBuilding river/pile: {model.n_cards} cards, {model.n_verso} verso ('{model.verso_loc}')")

        # NEW: Calculate total horizontal offset based on number of cards
        total_y_offset = 0.1 * (model.n_cards // 5)
        adjusted_start_y = model.start_location[0] - total_y_offset # Apply negative offset
        adjusted_start_location = (adjusted_start_y, model.start_location[1], model.start_location[2])
        print(f"  Calculated total y-offset: -{total_y_offset:.4f}")
        print(f"  Adjusted start location: ({adjusted_start_location[0]:.4f}, {adjusted_start_location[1]:.4f}, {adjusted_start_location[2]:.4f})")

        # Layout calculation loop
        for i in range(model.n_cards):
            card_name = model.card_names[i]
            is_face_up = i not in face_down_indices

            # --- Simplified Horizontal Layout Logic ---
            # X position progresses horizontally based on index and gap, using adjusted start Y
            y_pos = adjusted_start_location[0] + i * model.card_gap.get('base_gap_x', 0.15)
            # Y position is constant, plus optional jitter, using adjusted start X
            x_pos = adjusted_start_location[1]

            # Debug print for calculated position before jitter
            print(f"  Card {i}: Base calc_x={x_pos:.4f}, calc_y={y_pos:.4f} (before jitter)")

            # Apply random VERTICAL gap (y-jitter) if enabled
            if model.card_gap.get('random_gap', False):
                # Use base_gap_y for jitter magnitude
                base_gap_y_jitter = model.card_gap.get('base_gap_y', 0.005)
                percentage = model.card_gap.get('random_percentage', 0.2)
                deviation = base_gap_y_jitter * percentage
                y_offset = random.uniform(-deviation, deviation)
                # Jitter affects the vertical position (x_pos in this layout)
                x_pos += y_offset # Apply jitter to x_pos
                print(f"  Card {i}: Applied random y_offset: {y_offset:.4f}, final y={y_pos:.4f}")

            # Add tiny Z offset to prevent z-fighting in piles, using adjusted start Z
            z_pos = adjusted_start_location[2] + i * 0.0005

            # Final location (NO SWAP)
            location = (x_pos, y_pos, z_pos)
            print(f"  Card {i}: Final location set to: ({location[0]:.4f}, {location[1]:.4f}, {location[2]:.4f})")

            # Determine final rotation: Base flip + Z orientation
            z_orientation = math.pi / 2 # 90 degrees for vertical cards in horizontal layout
            if is_face_up:
                # Face up: Only Z orientation needed
                final_rotation = (0, 0, z_orientation)
            else:
                # Face down: X flip + Z orientation
                final_rotation = (math.pi, 0, z_orientation)

            # Create individual card config using the CardModel
            card_model_config = CardModel(
                card_name=card_name,
                location=location,
                scale=model.scale,
                face_up=is_face_up, # Keep for semantics
                rotation_euler=final_rotation # Pass the explicitly calculated rotation
            )

            # Use the loader's method to load the individual card
            print(f"  Loading card {i}: '{card_name}' face_up={is_face_up}")
            # Use the card_loader instance passed to this function
            loaded_obj = card_loader.build_card_from_config(card_model_config)
            if loaded_obj:
                loaded_cards.append(loaded_obj)
            else:
                print(f"Warning: Failed to load card '{card_name}' at index {i}")

    except (ValueError, TypeError) as e:
        print(f"Error processing river configuration: {e}")
        print(f"Input config: {river_config}")
        return []

    return loaded_cards

# --- Test Section ---
if __name__ == "__main__":
    import sys
    import os

    # Ensure the workspace root is in the Python path
    workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if workspace_root not in sys.path:
        sys.path.append(workspace_root)

    try:
        import bpy
        from scene_setup.general_setup import build_setup_from_config
        from poker.load_card import PokerCardLoader
    except ImportError as e:
        print(f"Error importing necessary modules: {e}")
        print("Please ensure the script is run from the workspace root or PYTHONPATH is set correctly.")
        sys.exit(1)

    print("--- Running River Builder Test Setup (Simplified Horizontal Layout) ---")

    # 1. Basic Scene Setup
    scene_setup_config = {
        "camera": {"distance": 3.0},
        "lighting": {"lighting": "medium"},
        "table": {"diameter": 1.5, "felt_color": (0.2, 0.5, 0.2, 1.0)},
        "render": {"engine": "CYCLES", "samples": 64, "gpus": [0]}
    }

    try:
        print("Building scene setup from config...")
        build_setup_from_config(scene_setup_config)
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

    # 3. Define Horizontal Layout configurations
    card_z = 0.91

    # 5 cards spread horizontally
    river_layout_config = {
        'card_names': ['QH', 'JD', '10S', '9C', '5H', '9H'],
        'n_cards': 6,
        # No n_columns/column_spacing needed
        'start_location': (-0.4, 0.1, card_z),
        'scale': 0.1,
        'n_verso': 0,
        'card_gap': {
            'base_gap_x': 0.15, # Horizontal spacing
            'base_gap_y': 0.0,  # No vertical base gap
            'random_gap': False # No jitter
            },
    }

    # Horizontal Pile (4 cards close together, random y-jitter)
    burn_pile_config = {
        'card_names': ['2D', '3C', '4S', '9H'],
        'n_cards': 4,
        # No n_columns/column_spacing needed
        'start_location': (-0.2, -0.2, card_z),
        'scale': 0.09,
        'n_verso': 4,
        'verso_loc': 'ordered',
        'card_gap': {
            'base_gap_x': 0.01, # Small horizontal spacing for pile effect
            'base_gap_y': 0.005, # Base for random y-jitter magnitude
            'random_gap': True,
            'random_percentage': 0.8
            },
        'random_seed': 99
    }

    # 4. Build the layouts
    print("\nBuilding horizontal river layout...")
    loaded_river_cards = build_river_from_config(card_loader, river_layout_config)

    print("\nBuilding horizontal burn pile...")
    loaded_burn_cards = build_river_from_config(card_loader, burn_pile_config)

    total_loaded_count = len(loaded_river_cards) + len(loaded_burn_cards)
    if total_loaded_count == 0:
         print("\nError: No cards were loaded successfully.")
         sys.exit(1)

    print(f"\nSuccessfully loaded a total of {total_loaded_count} cards.")

    # 5. Render the scene
    output_filename = "river_builder_final_horizontal_test.png" # Final name
    output_dir = os.path.join(workspace_root, "poker/img")
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

    print("--- River Builder Test Finished ---") 