import bpy
import random
import math
import logging
from mathutils import Vector
from typing import Union, List, Dict, Any, Optional, Tuple

# Ensure workspace root is in path for sibling imports
import sys
import os
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if workspace_root not in sys.path:
    sys.path.append(workspace_root)

# Models and Loader
from poker.config.models import (
    CardOverlapModel, LayoutMode, CardModel, CardTypeModel, CardTypeMode,
    DEFAULT_CARD_WIDTH, DEFAULT_CARD_HEIGHT, DEFAULT_CARD_NAMES
)
from poker.load_card import PokerCardLoader

# Constants
TINY_Z_OFFSET = 0.0005 # To prevent z-fighting

logger = logging.getLogger(__name__)

# --- Helper Function ---
def _distribute_items_randomly(total_items: int, num_bins: int, rng: random.Random, min_per_bin: int = 0) -> List[int]:
    """Distributes a total number of items randomly into bins, ensuring a minimum per bin.
    
    Args:
        total_items: Total number of items to distribute.
        num_bins: Number of bins.
        rng: Random number generator instance.
        min_per_bin: Minimum number of items required in each bin.
        
    Returns:
        A list where each element is the count of items in the corresponding bin.
    """
    if num_bins <= 0:
        logger.warning("Cannot distribute items into 0 bins.")
        return []
    if total_items < 0:
        logger.warning("Cannot distribute negative items.")
        return [0] * num_bins
    if total_items == 0:
        return [0] * num_bins
    if min_per_bin < 0:
        logger.warning("min_per_bin cannot be negative. Setting to 0.")
        min_per_bin = 0

    required_minimum_total = min_per_bin * num_bins
    if total_items < required_minimum_total:
        logger.warning(f"Total items ({total_items}) is less than required minimum ({required_minimum_total}) for {num_bins} bins with min {min_per_bin} each. Distribution might not meet minimum.")
        # Fallback to simple random distribution without minimum guarantee in this case
        counts = [0] * num_bins
        for _ in range(total_items):
            chosen_bin = rng.randrange(num_bins)
            counts[chosen_bin] += 1
        return counts

    # Allocate minimum to each bin first
    counts = [min_per_bin] * num_bins
    remaining_items = total_items - required_minimum_total

    # Distribute remaining items randomly
    for _ in range(remaining_items):
        chosen_bin = rng.randrange(num_bins)
        counts[chosen_bin] += 1

    return counts

# --- Main Builder Function ---
def build_card_overlap_layout_from_config(
    config: Union[Dict[str, Any], CardOverlapModel],
    card_loader: PokerCardLoader
) -> List[bpy.types.Object]:
    """
    Builds a general layout of cards with overlap control based on configuration.

    Args:
        config: A CardOverlapModel instance or a dictionary conforming to its structure.
        card_loader: An initialized instance of PokerCardLoader.

    Returns:
        A list of the loaded card bpy.types.Object instances.
        Returns an empty list if configuration is invalid or loading fails.
    """
    loaded_cards = []
    model: CardOverlapModel

    try:
        # --- 1. Parse and Validate Config ---
        if isinstance(config, dict):
            model = CardOverlapModel.from_dict(config)
        elif isinstance(config, CardOverlapModel):
            model = config
        else:
            raise TypeError(f"Expected dict or CardOverlapModel, got {type(config)}")

        if model.overall_cards == 0:
            logger.info("build_card_overlap_layout: 0 overall_cards requested, returning empty list.")
            return []

        rng = random.Random(model.random_seed) if model.random_seed is not None else random.Random()

        # --- 2. Deal Card Names ---
        card_type_model = CardTypeModel.from_dict(model.card_type_config)
        allow_repetition = card_type_model.allow_repetition
        source_deck = list(DEFAULT_CARD_NAMES) # Start with full deck

        if card_type_model.mode == CardTypeMode.SUBSET_N:
            n = card_type_model.subset_n
            if n is not None and 0 < n <= len(source_deck):
                 source_deck = rng.sample(source_deck, n)
            else:
                 logger.warning(f"Invalid subset_n ({n}) for deck size {len(source_deck)}. Using full deck for sampling.")
        elif card_type_model.mode == CardTypeMode.EXPLICIT_LIST:
            if card_type_model.card_list:
                 source_deck = list(card_type_model.card_list)
            else:
                 logger.warning("Mode is EXPLICIT_LIST but card_list is empty. Using full deck for sampling.")
        elif card_type_model.mode == CardTypeMode.SUIT_ONLY:
            target_suit_char = card_type_model.suit[0].upper() if card_type_model.suit else ''
            if target_suit_char in ['S', 'H', 'D', 'C']:
                 source_deck = [card for card in source_deck if card.endswith(target_suit_char)]
            else:
                 logger.warning(f"Invalid suit '{card_type_model.suit}'. Using full deck for sampling.")
        # FULL_DECK mode uses the default source_deck

        if not source_deck:
             logger.error("Source deck for card dealing is empty. Cannot proceed.")
             return []

        dealt_card_names: List[str] = []
        if allow_repetition:
            dealt_card_names = rng.choices(source_deck, k=model.overall_cards)
        else:
            if model.overall_cards > len(source_deck):
                 logger.error(f"Cannot deal {model.overall_cards} unique cards from a deck of size {len(source_deck)}.")
                 return []
            dealt_card_names = rng.sample(source_deck, k=model.overall_cards)
        
        logger.info(f"Dealt {len(dealt_card_names)} card names (repetition={allow_repetition}).")


        # --- 3. Determine Face Down Indices ---
        face_down_indices = set()
        if model.n_verso > 0:
            all_indices = list(range(model.overall_cards))
            if model.verso_loc == 'ordered':
                face_down_indices = set(all_indices[:model.n_verso])
            elif model.verso_loc == 'random':
                if model.n_verso > len(all_indices):
                    logger.warning(f"Requested n_verso ({model.n_verso}) > overall_cards ({model.overall_cards}). Setting all cards face down.")
                    face_down_indices = set(all_indices)
                else:
                    face_down_indices = set(rng.sample(all_indices, model.n_verso))
        logger.info(f"Determined {len(face_down_indices)} face-down cards ('{model.verso_loc}' mode).")


        # --- 4. Distribute Cards into Lines/Columns ---
        items_per_bin: List[int] = []
        num_bins = 0
        if model.layout_mode == LayoutMode.HORIZONTAL:
            num_bins = model.n_lines
            items_per_bin = _distribute_items_randomly(model.overall_cards, num_bins, rng, min_per_bin=2)
            logger.info(f"Distributed cards into {num_bins} lines (min 2 each if possible): {items_per_bin}")
        elif model.layout_mode == LayoutMode.VERTICAL:
            num_bins = model.n_columns
            items_per_bin = _distribute_items_randomly(model.overall_cards, num_bins, rng, min_per_bin=2)
            logger.info(f"Distributed cards into {num_bins} columns (min 2 each if possible): {items_per_bin}")
        
        if sum(items_per_bin) != model.overall_cards:
             logger.error(f"Card distribution failed: Sum of cards per bin ({sum(items_per_bin)}) != overall cards ({model.overall_cards})")
             return [] # Should not happen with _distribute_items_randomly


        # --- 5. Calculate Placement Parameters ---
        target_scale: float
        if isinstance(model.scale, (int, float)):
            target_scale = float(model.scale)
        elif isinstance(model.scale, tuple) and len(model.scale) == 3:
            target_scale = float(model.scale[0]) # Use X component for simplicity
        else:
            logger.warning(f"Invalid scale format ({model.scale}). Using default 0.1")
            target_scale = 0.1

        scaled_width = DEFAULT_CARD_WIDTH * target_scale
        scaled_height = DEFAULT_CARD_HEIGHT * target_scale
        # horizontal_step = scaled_width * (1.0 - model.horizontal_overlap_factor) # Old: Used for X step
        # vertical_step = scaled_height * (1.0 - model.vertical_overlap_factor) # Used in Vertical mode
        logger.debug(f"Scale={target_scale:.3f}, Scaled W={scaled_width:.3f}, H={scaled_height:.3f}")
        # logger.debug(f"Steps: H={horizontal_step:.3f} (overlap={model.horizontal_overlap_factor:.2f}), V={vertical_step:.3f} (overlap={model.vertical_overlap_factor:.2f})") # Old debug


        # --- 6. Placement Loop ---
        card_list_idx = 0
        center_vec = Vector(model.center_location)

        if model.layout_mode == LayoutMode.HORIZONTAL:
            # Lines are distributed horizontally (along X), cards within lines are vertical (along Y)
            logger.info("Placing cards in HORIZONTAL mode (lines along X, cards along Y)...")

            # Step for cards within a line (along Y axis), based on height and horizontal_overlap
            in_line_step_y = scaled_height * (1.0 - model.horizontal_overlap_factor)
            logger.debug(f"Horizontal layout using vertical step (height overlap): {in_line_step_y:.3f} (factor={model.horizontal_overlap_factor:.2f})")

            # Gap between lines (along X axis)
            line_gap_x = model.line_gap # Assume line_gap defines distance between centerlines of lines
            logger.debug(f"Horizontal layout line gap (along X): {line_gap_x:.3f}")

            # Calculate overall layout width (extent along X)
            total_layout_width = (num_bins - 1) * line_gap_x if num_bins > 1 else 0
            start_x = center_vec.x - total_layout_width / 2.0

            for line_idx in range(num_bins): # Outer loop controls X position (lines)
                current_x = start_x + line_idx * line_gap_x
                num_cards_in_line = items_per_bin[line_idx]
                if num_cards_in_line == 0: continue

                # Center the line vertically (along Y)
                line_length_y = (num_cards_in_line - 1) * in_line_step_y if num_cards_in_line > 1 else 0
                line_start_y = center_vec.y - line_length_y / 2.0
                logger.debug(f"  Line {line_idx}: Start X={current_x:.3f}, Start Y={line_start_y:.3f}, Cards={num_cards_in_line}")

                for card_in_line_idx in range(num_cards_in_line): # Inner loop controls Y position (cards in line)
                    if card_list_idx >= model.overall_cards:
                        logger.error("Card index out of bounds during placement loop!")
                        break # Safety break

                    card_name = dealt_card_names[card_list_idx]
                    is_face_up = card_list_idx not in face_down_indices

                    # Position card along Y axis within the line
                    x_pos = current_x
                    y_pos = line_start_y + card_in_line_idx * in_line_step_y
                    z_pos = center_vec.z + card_list_idx * TINY_Z_OFFSET

                    # Rotation for horizontal layout (cards standing vertically, long edge along Y)
                    rotation = (math.pi if not is_face_up else 0, 0, math.pi / 2)

                    card_model_config = CardModel(
                        card_name=card_name,
                        location=(x_pos, y_pos, z_pos),
                        scale=model.scale,
                        face_up=is_face_up,
                        rotation_euler=rotation
                    )
                    # logger.debug(f"    Placing card {card_list_idx}: '{card_name}' at ({x_pos:.3f}, {y_pos:.3f}, {z_pos:.3f})")
                    loaded_obj = card_loader.build_card_from_config(card_model_config)
                    if loaded_obj:
                        loaded_cards.append(loaded_obj)
                    else:
                        logger.warning(f"Failed to load card '{card_name}' (index {card_list_idx})")
                    
                    card_list_idx += 1
                if card_list_idx >= model.overall_cards: break # Break outer loop if done

        elif model.layout_mode == LayoutMode.VERTICAL:
            logger.info("Placing cards in VERTICAL mode (stacks along X, distributed along Y)...")
            # Distribute columns horizontally (along Y)
            total_layout_length = (num_bins - 1) * model.column_gap if num_bins > 1 else 0
            start_y = center_vec.y - total_layout_length / 2.0
            # Calculate step based on width and vertical_overlap_factor
            step_x = scaled_width * (1.0 - model.vertical_overlap_factor)
            logger.debug(f"Vertical layout using horizontal step (width overlap): {step_x:.3f}")

            for row_idx in range(num_bins): # Outer loop controls Y position (rows/columns)
                current_y = start_y + row_idx * model.column_gap
                num_cards_in_stack = items_per_bin[row_idx]
                if num_cards_in_stack == 0: continue

                # Center the stack vertically (along X)
                stack_width = (num_cards_in_stack - 1) * step_x if num_cards_in_stack > 1 else 0
                stack_start_x = center_vec.x - stack_width / 2.0
                logger.debug(f"  Stack {row_idx}: Start Y={current_y:.3f}, Start X={stack_start_x:.3f}, Cards={num_cards_in_stack}")

                for card_in_stack_idx in range(num_cards_in_stack): # Inner loop controls X position (cards in stack)
                    if card_list_idx >= model.overall_cards:
                        logger.error("Card index out of bounds during placement loop!")
                        break # Safety break

                    card_name = dealt_card_names[card_list_idx]
                    is_face_up = card_list_idx not in face_down_indices

                    # Position card along X axis within the stack
                    x_pos = stack_start_x + card_in_stack_idx * step_x
                    y_pos = current_y
                    z_pos = center_vec.z + card_list_idx * TINY_Z_OFFSET

                    # Rotation for vertical layout (cards standing vertically, rotated 90 deg)
                    rotation = (math.pi if not is_face_up else 0, 0, math.pi / 2)

                    card_model_config = CardModel(
                        card_name=card_name,
                        location=(x_pos, y_pos, z_pos),
                        scale=model.scale,
                        face_up=is_face_up,
                        rotation_euler=rotation
                    )
                    # logger.debug(f"    Placing card {card_list_idx}: '{card_name}' at ({x_pos:.3f}, {y_pos:.3f}, {z_pos:.3f})")
                    loaded_obj = card_loader.build_card_from_config(card_model_config)
                    if loaded_obj:
                        loaded_cards.append(loaded_obj)
                    else:
                        logger.warning(f"Failed to load card '{card_name}' (index {card_list_idx})")

                    card_list_idx += 1
                if card_list_idx >= model.overall_cards: break # Break outer loop if done

        if card_list_idx != model.overall_cards:
             logger.warning(f"Placement loop finished but only placed {card_list_idx}/{model.overall_cards} cards.")

    except (ValueError, TypeError) as e:
        logger.error(f"Error processing card overlap layout configuration: {e}")
        import traceback
        print("--- TRACEBACK for Config Processing Error ---")
        traceback.print_exc()
        print("---------------------------------------------")
        # logger.error(f"Input config: {config}") # Be careful logging full config
        return []
    except Exception as e:
        logger.exception(f"An unexpected error occurred during card overlap layout generation: {e}")
        return []

    logger.info(f"Finished building card overlap layout. Created {len(loaded_cards)} cards.")
    return loaded_cards


# --- Test Section ---
if __name__ == "__main__":
    print("Running card_overlap_builder.py directly for testing.")
    logging.basicConfig(level=logging.DEBUG) # Show debug messages for testing

    try:
        from scene_setup.general_setup import build_setup_from_config
        from utils.blender_utils import render_scene
        from scene_setup.rendering import clear_scene
    except ImportError as e:
        print(f"Error importing necessary modules: {e}")
        print("Please ensure the script is run from the workspace root or PYTHONPATH is set correctly.")
        sys.exit(1)

    # --- Test Scene Setup ---
    scene_setup_config = {
        "camera": {"distance": 3.0, "angle": 55, "horizontal_angle": 0},
        "lighting": {"lighting": "medium"},
        "table": {"shape": "rectangular", "width": 2.0, "length": 1.5, "felt_color": (0.1, 0.3, 0.15, 1.0)},
        "render": {"engine": "CYCLES", "samples": 16}
    }
    try:
        build_setup_from_config(scene_setup_config)
    except Exception as e:
        print(f"Error during scene setup: {e}")
        sys.exit(1)

    # --- Card Loader ---
    try:
        card_loader = PokerCardLoader()
    except Exception as e:
        print(f"Error initializing PokerCardLoader: {e}")
        sys.exit(1)

    # --- Test Config 1: Horizontal Layout ---
    test_config_h = {
        "layout_id": "Test_Horizontal_3Lines",
        "overall_cards": 15,
        "layout_mode": "horizontal", # Use string value
        "n_lines": 3,
        "card_type_config": {
            "mode": "full_deck",
            "allow_repetition": False
        },
        "center_location": (0.0, 0.0, 0.91),
        "scale": 0.12,
        "horizontal_overlap_factor": 0.9, # Less overlap
        "line_gap": 0.2, # Larger gap between lines
        "n_verso": 0,
        "verso_loc": "random",
        "random_seed": 123
    }

    # --- Test Config 2: Vertical Layout ---
    test_config_v = {
        "layout_id": "Test_Vertical_4Cols",
        "overall_cards": 12,
        "layout_mode": "vertical", # Use string value
        "n_columns": 4,
        "card_type_config": {
            "mode": "suit_only",
            "suit": "Spades",
            "allow_repetition": True
        },
        "center_location": (0.0, 0.0, 0.91),
        "scale": 0.1,
        "vertical_overlap_factor": 0.9, # More overlap
        "column_gap": 0.15,
        "n_verso": 0,
        "verso_loc": "ordered",
        "random_seed": 456
    }

    # --- Build and Render Horizontal Layout ---
    print("\n--- Building Horizontal Layout --- ")
    built_cards_h = []
    try:
        # Setup scene for horizontal
        build_setup_from_config(scene_setup_config) 
        card_loader = PokerCardLoader() # Initialize loader
        built_cards_h = build_card_overlap_layout_from_config(test_config_h, card_loader)
        if built_cards_h:
             output_filename_h = "card_overlap_builder_test_h.png"
             output_dir = os.path.join(workspace_root, "poker/img")
             output_path_h = os.path.join(output_dir, output_filename_h)
             os.makedirs(output_dir, exist_ok=True)
             bpy.context.scene.render.filepath = output_path_h
             bpy.context.scene.render.image_settings.file_format = 'PNG'
             print(f"\nRendering Horizontal Layout to {output_path_h}...")
             bpy.ops.render.render(write_still=True)
             print("Horizontal Render finished.")
        else:
             print("Skipping horizontal render as no cards were built.")
    except Exception as e:
        print(f"Error during Horizontal test: {e}")
        import traceback
        traceback.print_exc()

    # --- Clear Scene --- 
    print("\n--- Clearing Scene ---")
    clear_scene()

    # --- Build and Render Vertical Layout --- 
    print("\n--- Building Vertical Layout --- ")
    built_cards_v = []
    try:
        # Re-setup scene after clearing
        build_setup_from_config(scene_setup_config) 
        card_loader = PokerCardLoader() # Re-initialize loader
        built_cards_v = build_card_overlap_layout_from_config(test_config_v, card_loader)
        if built_cards_v:
             output_filename_v = "card_overlap_builder_test_v.png"
             output_dir = os.path.join(workspace_root, "poker/img") # Redundant, but safe
             output_path_v = os.path.join(output_dir, output_filename_v)
             os.makedirs(output_dir, exist_ok=True)
             bpy.context.scene.render.filepath = output_path_v
             bpy.context.scene.render.image_settings.file_format = 'PNG'
             print(f"\nRendering Vertical Layout to {output_path_v}...")
             bpy.ops.render.render(write_still=True)
             print("Vertical Render finished.")
        else:
             print("Skipping vertical render as no cards were built.")
    except Exception as e:
        print(f"Error during Vertical test: {e}")
        import traceback
        traceback.print_exc()

    print("--- Card Overlap Builder Test Finished ---")