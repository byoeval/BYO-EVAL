import os
import random

# Ensure workspace root is in path for sibling imports
import sys
from typing import Any

import bpy
from mathutils import Vector

workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if workspace_root not in sys.path:
    sys.path.append(workspace_root)

from poker.config.models import ChipAreaConfig, ChipModel, ChipPileModel

# Revert to original imports
from poker.load_chip import _DEFAULT_CHIPS_BLEND_FILE, load_chip
from scene_setup.general_setup import build_setup_from_config
from utils.blender_utils import render_scene

DEFAULT_TABLE_HEIGHT = 0.9 # Assuming table surface is around this Z level

def build_chip_area_from_config(
    config: dict | ChipAreaConfig,
    area_center_location: tuple[float, float, float],
    resolved_pile_configs: list[dict[str, Any]] | None = None # List of resolved configs per pile
) -> list[bpy.types.Object]:
    """
    Builds a chip area (one or more piles) based on configuration.
    (Reverted to calling load_chip per chip for now)

    Args:
        config: A ChipAreaConfig instance or a dictionary conforming to its structure.
        area_center_location: The (x, y, z) coordinates representing the desired
                              centerline for the chip area (usually derived from hand).
        resolved_pile_configs: If provided by distribution logic, this list dictates
                               the number of piles and provides property overrides
                               (n_chips, color, scale) for each pile.

    Returns:
        A list of created Blender chip objects, or an empty list on error.
    """
    all_chips = []
    area_config: ChipAreaConfig | None = None

    try:
        # Parse overall area config
        if isinstance(config, dict):
            area_config = ChipAreaConfig.from_dict(config)
        elif isinstance(config, ChipAreaConfig):
            area_config = config
        else:
            raise TypeError(f"Expected dict or ChipAreaConfig, got {type(config)}")

        # --- Determine the number of piles and get specs ---
        num_piles_to_create = area_config.n_piles # Default from config
        pile_specs = None # Holds the list of override dicts
        if resolved_pile_configs is not None:
            num_piles_to_create = len(resolved_pile_configs)
            pile_specs = resolved_pile_configs
            print(f"  Using {num_piles_to_create} piles based on resolved distribution configs.")
        else:
            print(f"  Using n_piles from config: {area_config.n_piles}")

        if num_piles_to_create <= 0:
             print("  Number of piles to create is 0. Skipping chip area.")
             return []

        # --- Ensure we have a parsed base_pile_model (needed for defaults and layout) ---
        base_pile_model: ChipPileModel | None = None
        if isinstance(area_config.base_pile_config, ChipPileModel):
            base_pile_model = area_config.base_pile_config
        elif isinstance(area_config.base_pile_config, dict):
            try:
                base_pile_model = ChipPileModel.from_dict(area_config.base_pile_config)
            except (ValueError, TypeError) as e:
                print(f"  Error parsing base_pile_config dict: {e}. Cannot determine base chip.")
                return [] # Cannot proceed without base pile config
        else:
            print(f"  Invalid type for base_pile_config ({type(area_config.base_pile_config)}). Cannot determine base chip.")
            return []

        # Prepare RNG for layout randomness
        rng = random.Random(area_config.random_seed) if area_config.random_seed is not None else random.Random()

        # --- Calculate Layout Vectors ---
        area_center_vec = Vector(area_center_location)
        table_center_vec = Vector((0.0, 0.0, area_center_vec.z))
        direction_to_center = (table_center_vec - area_center_vec)
        if direction_to_center.length > 1e-6:
            direction_to_center.normalize()
        else:
            direction_to_center = Vector((0.0, -1.0, 0.0))

        chip_area_baseline_center = area_center_vec + direction_to_center * area_config.placement_offset_from_cards
        tangent_vec = Vector((-direction_to_center.y, direction_to_center.x, 0.0))
        if tangent_vec.length < 1e-6:
            tangent_vec = Vector((1.0, 0.0, 0.0))
        tangent_vec.normalize()

        total_width = (num_piles_to_create - 1) * area_config.pile_gap_h if num_piles_to_create > 1 else 0
        start_pos_on_tangent = chip_area_baseline_center - tangent_vec * (total_width / 2.0)
        # --- End Layout Vector Calculation ---

        # --- Build each pile ---
        current_pile_center = start_pos_on_tangent.copy()

        for i in range(num_piles_to_create):
            current_pile_spec = pile_specs[i] if pile_specs and i < len(pile_specs) else {}

            # Determine n_chips for this pile
            n_chips_this_pile = current_pile_spec.get('n_chips', base_pile_model.n_chips)
            if area_config.n_chips_per_pile and i < len(area_config.n_chips_per_pile):
                 n_chips_override = area_config.n_chips_per_pile[i]
                 if isinstance(n_chips_override, int) and n_chips_override >= 0:
                     n_chips_this_pile = n_chips_override
            if n_chips_this_pile <= 0:
                 print(f"  Skipping pile {i} as final n_chips is <= 0.")
                 continue

            # Determine color for this pile
            pile_color_override = current_pile_spec.get('color')
            if pile_color_override is None and area_config.pile_colors and i < len(area_config.pile_colors):
                 pile_color_override = area_config.pile_colors[i]

            # Determine scale for this pile
            pile_scale_override = current_pile_spec.get('scale')

            # Determine spread factor for this pile
            pile_spread_factor = base_pile_model.spread_factor
            if area_config.pile_spreads and i < len(area_config.pile_spreads):
                 spread_override = area_config.pile_spreads[i]
                 if isinstance(spread_override, float | int) and 0.0 <= spread_override <= 1.0:
                     pile_spread_factor = float(spread_override)

            # Determine base chip config properties for load_chip
            chip_object_name = "DefaultChip"
            chip_base_scale = 0.06
            chip_base_color = None
            chip_blend_path = _DEFAULT_CHIPS_BLEND_FILE
            base_chip_config_dict = None
            if isinstance(base_pile_model.base_chip_config, ChipModel):
                 base_chip_config_dict = base_pile_model.base_chip_config.to_dict()
            elif isinstance(base_pile_model.base_chip_config, dict):
                 base_chip_config_dict = base_pile_model.base_chip_config

            if base_chip_config_dict:
                 chip_object_name = base_chip_config_dict.get('chip_object_name', chip_object_name)
                 chip_base_scale = base_chip_config_dict.get('scale', chip_base_scale)
                 chip_base_color = base_chip_config_dict.get('color') # None if not present
                 chip_blend_path = base_chip_config_dict.get('blend_file_path') or _DEFAULT_CHIPS_BLEND_FILE

            # Finalize color and scale
            final_color = pile_color_override if pile_color_override is not None else chip_base_color
            final_scale = pile_scale_override if pile_scale_override is not None else chip_base_scale

            # Calculate scaled diameter and max offset
            current_scale_val = 1.0
            if isinstance(final_scale, int | float): current_scale_val = final_scale
            elif isinstance(final_scale, tuple | list) and len(final_scale) > 0: current_scale_val = final_scale[0]
            chip_diameter_scaled = 2.0 * current_scale_val
            min_required_gap = chip_diameter_scaled * 0.8
            max_offset = chip_diameter_scaled * 0.5 * pile_spread_factor

            # Calculate pile position
            if i > 0:
                 configured_gap_h = area_config.pile_gap_h * (1.0 + rng.uniform(-area_config.pile_gap_random_factor, area_config.pile_gap_random_factor))
                 actual_gap_h = max(min_required_gap, configured_gap_h)
                 current_pile_center += tangent_vec * actual_gap_h

            pile_base_pos = current_pile_center.copy()
            vertical_jitter = rng.uniform(-area_config.pile_gap_v_range, area_config.pile_gap_v_range)
            pile_base_pos.z += vertical_jitter + 0.001 # Add slight Z offset

            pile_x, pile_y, pile_base_z = pile_base_pos.x, pile_base_pos.y, pile_base_pos.z
            print(f"  Building pile {i}: {n_chips_this_pile} chips at ~({pile_x:.2f}, {pile_y:.2f}), spread={pile_spread_factor:.2f}")

            # Convert final_scale to tuple for load_chip
            if isinstance(final_scale, list | tuple) and len(final_scale) == 3:
                scale_tuple = tuple(final_scale)
            elif isinstance(final_scale, int | float):
                scale_tuple = (final_scale, final_scale, final_scale)
            else:
                print(f"    Warning: Invalid scale value ({final_scale}) for pile {i}. Using (1,1,1).")
                scale_tuple = (1.0, 1.0, 1.0)

            for j in range(n_chips_this_pile):
                chip_z = pile_base_z + j * base_pile_model.vertical_gap
                offset_x = rng.uniform(-max_offset, max_offset)
                offset_y = rng.uniform(-max_offset, max_offset)
                chip_x = pile_x + offset_x
                chip_y = pile_y + offset_y

                # Call load_chip for each individual chip
                try:
                    chip = load_chip(
                        chip_object_name=chip_object_name,
                        location=(chip_x, chip_y, chip_z),
                        scale_tuple=scale_tuple, # Use final scale for this pile
                        color=final_color,  # Use final color for this pile
                        blend_file_path=chip_blend_path
                    )
                    if chip:
                        all_chips.append(chip)
                    else:
                        print(f"    Warning: load_chip returned None for chip {j}, pile {i}.")
                except Exception as load_e:
                    print(f"    ERROR calling load_chip for chip {j}, pile {i}: {load_e}")
                    continue # Continue to next chip/pile on error

    except Exception as e:
        print(f"Error building chip area: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        return []

    print(f"Finished building chip area. Created {len(all_chips)} total chips.")
    return all_chips


# --- Main execution block for testing ---
if __name__ == "__main__":
    print("Running chip_area_builder.py directly for testing.")

    # Example Test Config
    base_chip_for_area = {
        "chip_object_name": "Cylinder001",
        "scale": 0.1,
        "color": (0.5, 0.5, 0.5, 1.0) # Default Grey
    }
    base_pile_for_area = {
         "n_chips": 5,
         "base_chip_config": base_chip_for_area,
         "spread_factor": 0.1,
         "vertical_gap": 0.005
    }
    test_chip_area_config = {
         "base_pile_config": base_pile_for_area,
         "n_piles": 3,
         "n_chips_per_pile": [3, 5, 4], # Override n_chips
         "pile_colors": [ (0.8, 0.1, 0.1, 1.0), None, (0.1, 0.8, 0.1, 1.0) ], # Red, Default Grey, Green
         "pile_spreads": [0.0, 0.5, None], # Stacked, Medium Spread, Default Spread
         "pile_gap_h": 0.08,
         "pile_gap_random_factor": 0.8,
         "pile_gap_v_range": 0.03,
         "placement_offset_from_cards": 0.25,
         "random_seed": 456
    }

    # Example scene setup
    scene_setup_config = {
        "camera": {"distance": 3.5, "angle": 55},
        "lighting": {"lighting": "high"},
        "table": {"diameter": 1.5, "felt_color": (0.1, 0.3, 0.1, 1.0)},
        "render": {"engine": "CYCLES", "samples": 64}
    }

    # Example area center (replace with actual player hand center later)
    example_area_center = (0.8, 0.5, DEFAULT_TABLE_HEIGHT)

    if bpy.context is None:
        print("Error: Must run from within Blender.")
    else:
         print("--- Test: Building Chip Area ---")
         try:
            print("Setting up scene...")
            build_setup_from_config(scene_setup_config)
            print("Scene setup done.")

            print("Building chip area...")
            created_objects = build_chip_area_from_config(test_chip_area_config, example_area_center)

            if created_objects:
                 print(f"\nSuccessfully created {len(created_objects)} chip objects in the area.")
                 # Render
                 render_output_path = "chip_area_render.png"
                 print(f"Rendering scene to {render_output_path}...")
                 render_scene(render_output_path)
                 print("Render complete.")
            else:
                 print("\nChip area building failed or produced no objects.")

         except Exception as e:
            print(f"An error occurred during testing: {e}")
            import traceback
            traceback.print_exc()
