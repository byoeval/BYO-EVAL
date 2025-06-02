import os

# Ensure workspace root is in path for sibling imports
import sys
from typing import Any

import bpy
from mathutils import Vector

workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if workspace_root not in sys.path:
    sys.path.append(workspace_root)

# Poker specific models and builders
from poker.chip_area_builder import build_chip_area_from_config
from poker.config.models import PlayerModel
from poker.load_card import PokerCardLoader  # Import from correct location
from poker.player_hand_builder import build_player_hand_from_config

# Scene setup (for testing)
from scene_setup.general_setup import build_setup_from_config


def build_player_from_config(
    config: dict[str, Any] | PlayerModel,
    card_loader: PokerCardLoader
) -> dict[str, list[bpy.types.Object]]:
    """
    Builds a player's components (hand and chip area) based on a configuration.
    Uses standalone builder functions.

    Args:
        config: A PlayerModel instance or a dictionary conforming to its structure.
        card_loader: An initialized PokerCardLoader instance.

    Returns:
        A dictionary containing lists of created Blender objects:
        {'cards': [card_objects], 'chips': [chip_objects]}
        Returns {'cards': [], 'chips': []} on error or if components are not defined.
    """
    player_cards: list[bpy.types.Object] = []
    player_chips: list[bpy.types.Object] = []
    hand_center: Vector | None = None
    player_model: PlayerModel | None = None

    try:
        if isinstance(config, dict):
            player_model = PlayerModel.from_dict(config)
        elif isinstance(config, PlayerModel):
            player_model = config
        else:
            raise TypeError(f"Expected dict or PlayerModel, got {type(config)}")

        print(f"\n--- Building Player: {player_model.player_id} ---")

        # 1. Build Player Hand
        if player_model.hand_config:
            print("Building player hand...")
            # build_player_hand_from_config returns (list_of_cards, center_vector | None)
            hand_objects, hand_center = build_player_hand_from_config(
                card_loader=card_loader,
                hand_config=player_model.hand_config
            )
            if hand_objects:
                 player_cards.extend(hand_objects)
            if hand_center:
                 print(f"  Hand center calculated at: ({hand_center.x:.2f}, {hand_center.y:.2f}, {hand_center.z:.2f})")
            else:
                 print("  Warning: Hand center could not be determined from hand build.")
        else:
            print("No hand configuration provided for this player.")

        # 2. Build Chip Area (relative to hand center or player location)
        if player_model.chip_area_config:
            chip_area_ref_location = None
            if hand_center is not None:
                print("Building chip area relative to calculated hand center...")
                chip_area_ref_location = tuple(hand_center)
            elif hasattr(player_model, 'hand_config') and player_model.hand_config.location:
                print("Building chip area relative to player hand location (fallback due to missing hand center)...")
                chip_area_ref_location = tuple(player_model.hand_config.location)

            if chip_area_ref_location:
                 # Call the function from chip_area_builder.py
                 chip_objects = build_chip_area_from_config(
                     config=player_model.chip_area_config,
                     area_center_location=chip_area_ref_location, # Use determined reference location
                     resolved_pile_configs=player_model._resolved_pile_configs # Pass the resolved pile data
                 )
                 if chip_objects:
                      player_chips.extend(chip_objects)
            else:
                 print("Warning: Cannot build chip area because no reference location (hand center or hand location) was determined. Define chips independently if needed.")
        else:
             print("No chip area configuration provided for this player.")

    except Exception as e:
        player_id_str = config.get('player_id', 'Unknown') if isinstance(config, dict) else getattr(player_model, 'player_id', 'Unknown')
        print(f"Error building player '{player_id_str}': {e}")
        import traceback
        traceback.print_exc()
        # Return empty lists on error during player build
        return {'cards': [], 'chips': []}

    print(f"--- Finished Building Player: {player_model.player_id} ---")
    return {'cards': player_cards, 'chips': player_chips}


# --- Main execution block for testing ---
if __name__ == "__main__":
    print("Running player_builder.py directly for testing.")

    # --- Test Configuration ---
    # Example Base Chip / Pile for Player
    player_base_chip = {
        "chip_object_name": "Cylinder001", # Use a name likely in default chip file
        "scale": 0.12,
        "color": (0.1, 0.1, 0.1, 1.0) # Black chip base
    }
    player_base_pile = {
         "n_chips": 6,
         "base_chip_config": player_base_chip,
         "spread_factor": 0.2,
         "vertical_gap": 0.004
    }
    # Example Chip Area for Player
    player_chip_area = {
         "base_pile_config": player_base_pile,
         "n_piles": 2,
         "n_chips_per_pile": [6, 8], # Override n_chips
         "pile_colors": [ (0.9, 0.1, 0.1, 1.0), None], # Red pile, Black pile
         "pile_spreads": [0.1, 0.4], # Tighter spread, looser spread
         "pile_gap_h": 0.07,
         "pile_gap_random_factor": 0.6,
         "pile_gap_v_range": 0.01,
         "placement_offset_from_cards": 0.35,
         "random_seed": 789
    }
    # Example Hand for Player
    player_hand = {
        "card_names": ["AS", "KH"], # Ace Spade, King Heart
        "n_cards": 2,
        "location": (0.0, 0.7, 0.91), # Player position (e.g., bottom center)
        "scale": 0.15,
        "spread_factor_h": 0.4,
        "spread_factor_v": 0.05,
        "random_seed": 111
    }
    # Example Player Config
    test_player_config = {
         "player_id": "Test Player 1",
         "hand_config": player_hand,
         "chip_area_config": player_chip_area
    }

    # Example Scene Setup
    scene_setup_config = {
        "camera": {"distance": 3.0, "angle": 50},
        "lighting": {"lighting": "medium"},
        "table": {"diameter": 1.5, "felt_color": (0.1, 0.4, 0.1, 1.0)},
        "render": {"engine": "CYCLES", "samples": 64}
    }
    # --- End Test Configuration ---

    if bpy.context is None:
        print("Error: Must run from within Blender.")
    else:
         print("--- Test: Building Player --- ")
         try:
            print("Setting up scene...")
            build_setup_from_config(scene_setup_config)
            print("Scene setup done.")

            print("Initializing card loader...")
            # Need to initialize card loader for the hand builder
            try:
                 card_loader = PokerCardLoader() # Assumes default deck path exists
            except FileNotFoundError as e:
                 print(f"Error: Default deck file not found. {e}")
                 raise # Stop test if deck is missing
            print("Card loader initialized.")

            # Note: build_chip_area_from_config will use its internal logic
            # which relies on default chip file paths in poker/load_chip.py
            # No need to pass a chip_loader here.

            print("Building player...")
            player_objects = build_player_from_config(test_player_config, card_loader)

            n_cards = len(player_objects.get('cards', []))
            n_chips = len(player_objects.get('chips', []))

            if n_cards > 0 or n_chips > 0:
                 print(f"\nSuccessfully created player objects: {n_cards} card(s), {n_chips} chip(s).")
                 # Render
                 render_output_path = "player_builder_test_render.png" # New output name
                 abs_render_path = os.path.join(workspace_root, render_output_path)
                 render_dir = os.path.dirname(abs_render_path)
                 os.makedirs(render_dir, exist_ok=True)
                 print(f"Rendering scene to {abs_render_path}...")
                 bpy.context.scene.render.filepath = abs_render_path
                 bpy.ops.render.render(write_still=True)
                 print("Render complete.")
            else:
                 print("\nPlayer building failed or produced no objects.")

         except Exception as e:
            print(f"An error occurred during player builder testing: {e}")
            import traceback
            traceback.print_exc()
