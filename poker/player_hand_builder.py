import math
import random
from typing import Any

import bpy
from mathutils import Vector

# Models and Loader
from poker.config.models import (
    DEFAULT_CARD_HEIGHT,
    DEFAULT_CARD_WIDTH,
    DEFAULT_Z_ROTATION,
    MAX_CENTER_GAP_FACTOR_X,
    MAX_CENTER_GAP_FACTOR_Y,
    MIN_CENTER_GAP_FACTOR_X,
    MIN_CENTER_GAP_FACTOR_Y,
    CardModel,
    PlayerHandModel,
)
from poker.load_card import PokerCardLoader

# --- Constants ---
TINY_Z_STEP = 0.0007 # To prevent z-fighting (Increased step for testing overlap artifacts)

def _calculate_gap(spread_factor: float, base_size: float, scale: float,
                   min_factor: float, max_factor: float,
                   randomize: bool = False) -> float:
    """
    Calculates the target distance between card centers based on spread factor and scale.
    Optionally randomizes the gap uniformly between 0 and the calculated maximum.

    Args:
        spread_factor: Value from 0 (minimal) to 1 (maximal).
        base_size: The base size (width or height) of the card in Blender units.
        scale: The scale factor being applied to the cards.
        min_factor: Minimum gap as a factor of card size.
        max_factor: Maximum gap as a factor of card size.
        randomize: If True, return a random value between 0 and the calculated gap.

    Returns:
        The calculated (or randomized) distance between card centers.
    """
    scaled_size = base_size * scale
    min_gap = scaled_size * min_factor
    max_gap = scaled_size * max_factor
    calculated_gap = min_gap + (max_gap - min_gap) * spread_factor

    final_gap = max(0, calculated_gap)

    if randomize:
        # Note: random.uniform(a, b) can return b, so this includes 0 and the max gap.
        return random.uniform(0, final_gap)
    else:
        return final_gap

def _calculate_center_facing_rotation(x: float, y: float) -> float:
    """
    Calculate the rotation angle needed for a card to face the center (0,0),
    attenuated based on the angle of the hand's location relative to the Y-axis.

    Args:
        x: X coordinate of the card (vertical axis in user's perspective)
        y: Y coordinate of the card (horizontal axis in user's perspective)

    Returns:
        The attenuated rotation angle in radians.
    """
    # Calculate the angle needed to point directly at the center
    angle_to_center = math.atan2(-y, -x)  # Negative because we want to face towards center


    attenuation_factor = abs((angle_to_center % (math.pi / 4.0)) - (math.pi / 4.0)) / (math.pi / 4.0)
    attenuation_factor = min(1.0, attenuation_factor) # Clamp to [0, 1]

    # modify angle_to_center with attenuation factor depending on the distance to the nearest axis
    # modify only if the angle is lower than a threshold at 75% of pi/4
    if (angle_to_center % (math.pi / 4.0)) > (math.pi / 4.0 * 0.25):
        new_angle_to_center = angle_to_center  - (angle_to_center * attenuation_factor / 5)
    else:
        new_angle_to_center = angle_to_center

    original_target_rotation = DEFAULT_Z_ROTATION + angle_to_center

    # Adjust the offset based on the attenuation factor
    attenuated_rotation = DEFAULT_Z_ROTATION + new_angle_to_center
    print(" \n \n --------------------------------")
    print("angle to center: ", math.degrees(angle_to_center), "attenuation factor: ", attenuation_factor, "original target rotation: ", math.degrees(original_target_rotation), "attenuated rotation: ", math.degrees(attenuated_rotation))
    print("-------------------------------- \n \n   ")

    return attenuated_rotation

def build_player_hand_from_config(
    card_loader: PokerCardLoader,
    hand_config: dict[str, Any] | PlayerHandModel
) -> tuple[list[bpy.types.Object], Vector | None]:
    """
    Builds a player's hand of cards, centered around a location, with horizontal and vertical spread.
    Note: X is vertical (up/down), Y is horizontal (left/right) in the user's perspective.

    Args:
        card_loader: An initialized instance of PokerCardLoader.
        hand_config: Either a dictionary conforming to PlayerHandModel structure
                     or a PlayerHandModel instance.

    Returns:
        Tuple containing:
         - A list of the loaded card bpy.types.Object instances.
         - A Vector representing the center location of the hand (model.location), or None on error.
        Returns ([], None) if configuration is invalid or loading fails.
    """
    loaded_cards = []
    hand_center = None # Initialize hand center
    try:
        if isinstance(hand_config, dict):
            model = PlayerHandModel.from_dict(hand_config)
        elif isinstance(hand_config, PlayerHandModel):
            model = hand_config
        else:
            raise TypeError(f"Expected dict or PlayerHandModel, got {type(hand_config)}")

        if model.n_cards == 0:
            print("\nBuilding player hand: 0 cards requested.")
            return [], None

        if model.random_seed is not None:
            random.seed(model.random_seed)

        # Determine face down indices
        face_down_indices = set()
        if model.n_verso > 0:
            all_indices = list(range(model.n_cards))
            if model.verso_loc == 'ordered':
                face_down_indices = set(all_indices[:model.n_verso])
            elif model.verso_loc == 'random':
                face_down_indices = set(random.sample(all_indices, model.n_verso))

        print(f"\nBuilding player hand near {model.location}: {model.n_cards} cards, {model.n_verso} verso ('{model.verso_loc}')")

        # --- Determine Orientation (Parallel to Y-axis or not) ---
        loc_x, loc_y = model.location[0], model.location[1]

        # Avoid division by zero or undefined angle at origin
        if abs(loc_x) < 1e-6 and abs(loc_y) < 1e-6:
            angle_to_y_axis = 0.0 # Default for origin case
        else:
            angle_to_y_axis = math.atan2(loc_x, loc_y) % math.pi  # modulo pi
        # Check if the angle is within +/- 45 degrees (pi/4 rad) of the Y axis
        is_y_parallel = abs(angle_to_y_axis) < (math.pi / 4.0)
        print(f"  Hand Location Angle (vs +Y): {math.degrees(angle_to_y_axis):.2f} deg -> is_y_parallel: {is_y_parallel}")

        # --- Determine target scale ---
        target_scale: float
        if isinstance(model.scale, int | float):
            target_scale = float(model.scale)
        elif isinstance(model.scale, tuple) and len(model.scale) == 3:
            target_scale = float(model.scale[0])  # Use X component
        else:
            print(f"Warning: Invalid scale format in hand config ({model.scale}). Using default scale 1.0")
            target_scale = 1.0

        print(f"  Using Target Scale: {target_scale:.4f}")
        print(f"  Spread Factors - Horizontal: {model.spread_factor_h:.2f}, Vertical: {model.spread_factor_v:.2f}")

        # --- Calculate Gaps (considering scale) ---
        # Note: Y is horizontal spread, X is vertical spread in user's perspective
        gap_y = _calculate_gap(
            model.spread_factor_h,  # Horizontal spread affects Y axis
            DEFAULT_CARD_WIDTH,
            target_scale,
            MIN_CENTER_GAP_FACTOR_X,
            MAX_CENTER_GAP_FACTOR_X,
            randomize=model.randomize_gap_h
        )

        gap_x = _calculate_gap(
            model.spread_factor_v,  # Vertical spread affects X axis
            DEFAULT_CARD_HEIGHT,
            target_scale,
            MIN_CENTER_GAP_FACTOR_Y,
            MAX_CENTER_GAP_FACTOR_Y,
            randomize=model.randomize_gap_v
        )
        print("VERTICAL GAP: ", gap_x)


        # Conditionally swap gaps if the orientation is Y-parallel
        if is_y_parallel:
            gap_x, gap_y = gap_y, gap_x # Swap the calculated gaps
            print(f"  Y-Parallel Orientation: Swapping gaps. Effective Gap X (Vertical): {gap_x:.4f}, Effective Gap Y (Horizontal): {gap_y:.4f}")
        else:
            print(f"  Standard Orientation: Gap X (Vertical): {gap_x:.4f}, Gap Y (Horizontal): {gap_y:.4f}")

        # --- Centered Layout Calculation ---
        total_width = (model.n_cards - 1) * gap_y  # Total horizontal width (Y axis)
        start_y_offset = -total_width / 2.0  # Center horizontally
        print(f"  Start Y Offset: {start_y_offset:.4f}")

        # --- Card Placement Loop ---
        cumulative_offset_x = 0
        cumulative_offset_y = 0
        for i in range(model.n_cards):
            card_name = model.card_names[i]
            is_face_up = i not in face_down_indices
            z_pos = model.location[2] + i * TINY_Z_STEP  # Consistent Z offset


            random_direction = random.choice([1, -1])
            if not is_y_parallel:
                # Calculate position relative to the hand's center location
                current_y_offset = start_y_offset + i * gap_y  # Horizontal spread
                current_x_offset = cumulative_offset_x + random_direction * gap_x  # Vertical spread
                cumulative_offset_x += random_direction * gap_x
            else:
                current_y_offset = cumulative_offset_y + random_direction * gap_y  # Horizontal spread
                cumulative_offset_y += random_direction * gap_y
                current_x_offset = i * gap_x  # Vertical spread

            # Apply offsets to base location

            x_pos = model.location[0] + current_x_offset  # Vertical position
            y_pos = model.location[1] + current_y_offset  # Horizontal position

            location = (x_pos, y_pos, z_pos)
            print(f"  Card {i}: Offsets=(vertical={current_x_offset:.4f}, horizontal={current_y_offset:.4f}), "
                  f"Final location=({location[0]:.4f}, {location[1]:.4f}, {location[2]:.4f})")

            # --- Calculate Center-Facing Rotation ---
            # Pass the angle relative to the nearest Y-axis (mapped to [0, pi/2])
            # to attenuate the rotation.
            base_rotation = _calculate_center_facing_rotation(x_pos, y_pos)

            # Apply random rotation variation if specified
            if model.rotation_std_dev > 0:
                rotation_variation = random.gauss(0, model.rotation_std_dev)
                final_z_rotation = base_rotation + rotation_variation
            else:
                final_z_rotation = base_rotation

            # Apply face up/down rotation
            if is_face_up:
                final_rotation = (0, 0, final_z_rotation)
            else:
                final_rotation = (math.pi, 0, final_z_rotation)

            # --- Create and Load Card ---
            card_model_config = CardModel(
                card_name=card_name,
                location=location,
                scale=model.scale,
                face_up=is_face_up,
                rotation_euler=final_rotation
            )

            print(f"  Loading card {i}: '{card_name}' face_up={is_face_up}, "
                  f"rotation={[math.degrees(r) for r in final_rotation]}")
            loaded_obj = card_loader.build_card_from_config(card_model_config)
            if loaded_obj:
                loaded_cards.append(loaded_obj)
            else:
                print(f"Warning: Failed to load card '{card_name}' at index {i}")

        # Set the hand center if successful
        hand_center = Vector(model.location)

    except (ValueError, TypeError, FileNotFoundError) as e:
        print(f"Error processing player hand configuration: {e}")
        print(f"Input config snippet: {str(hand_config)[:200]}...")
        return [], None
    except Exception as e:
        print(f"An unexpected error occurred during hand generation: {e}")
        import traceback
        traceback.print_exc()
        return [], None

    return loaded_cards, hand_center

# --- Test Section ---
if __name__ == "__main__":
    import os
    import sys

    # Ensure the workspace root is in the Python path
    workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if workspace_root not in sys.path:
        sys.path.append(workspace_root)

    try:
        import bpy

        from poker.load_card import PokerCardLoader
        from scene_setup.general_setup import build_setup_from_config
    except ImportError as e:
        print(f"Error importing necessary modules: {e}")
        print("Please ensure the script is run from the workspace root or PYTHONPATH is set correctly.")
        sys.exit(1)

    print("--- Running Player Hand Builder Test Setup (Scale-Aware Layout) ---")

    # 1. Basic Scene Setup
    scene_setup_config = {
        "camera": {"distance": 4.0, "angle": 50},
        "lighting": {"lighting": "medium"},
        "table": {"diameter": 1.5, "felt_color": (0.1, 0.4, 0.15, 1.0)},
        "render": {"engine": "CYCLES", "samples": 64}
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

    # 3. Define Player Hand configurations with various spreads
    card_z = 0.91

    player1_hand_config = {
        'card_names': ['AS', 'AC'],
        'n_cards': 2,
        'location': (-0.5, 0, card_z),  # Vertical position (X) changed
        'scale': 0.1,
        'spread_factor_h': 0.6,  # Horizontal spread (affects Y axis)
        'spread_factor_v': 0.0,  # No vertical spread (X axis)
        'first_card_std_dev': 0.0,
        'rotation_std_dev': 0.1,  # Small random variation in rotation
        'n_verso': 0,
        'random_seed': 555
    }

    player2_hand_config = {
        'card_names': ['KD', 'KH'],
        'n_cards': 2,
        'location': (0.5, 0, card_z),  # Vertical position (X) changed
        'scale': 0.1,
        'n_verso': 2,
        'verso_loc': 'ordered',
        'spread_factor_h': 0.1,  # Small horizontal spread (Y axis)
        'spread_factor_v': 0.2,  # Slight vertical spread (X axis)
        'first_card_std_dev': 0.0,
        'rotation_std_dev': 0.0,  # No rotation randomization
        'random_seed': 666
    }

    player3_hand_config = {
        'card_names': ['QC', 'JC', '10C'],
        'n_cards': 3,
        'location': (0, -0.5, card_z),  # Vertical position (X) changed
        'scale': 0.1,
        'spread_factor_h': 1.0,  # Full horizontal spread (Y axis)
        'spread_factor_v': 0.3,  # Moderate vertical spread (X axis)
        'first_card_std_dev': 0.0,
        'rotation_std_dev': 0.2,  # Moderate rotation randomization
        'n_verso': 0,
        'random_seed': 777
    }

    # 4. Build the hands using the function from this file
    print("\nBuilding Player 1 hand...")
    loaded_hand1_cards, hand_center1 = build_player_hand_from_config(card_loader, player1_hand_config)

    print("\nBuilding Player 2 hand...")
    loaded_hand2_cards, hand_center2 = build_player_hand_from_config(card_loader, player2_hand_config)

    print("\nBuilding Player 3 hand...")
    loaded_hand3_cards, hand_center3 = build_player_hand_from_config(card_loader, player3_hand_config)

    total_loaded_count = len(loaded_hand1_cards) + len(loaded_hand2_cards) + len(loaded_hand3_cards)
    if total_loaded_count == 0:
         print("\nError: No cards were loaded successfully.")
         sys.exit(1)

    print(f"\nSuccessfully loaded a total of {total_loaded_count} player hand cards.")

    # 5. Render the scene
    output_filename = "player_hand_scale_aware_test.png"
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

    print("--- Player Hand Builder Test Finished ---")
