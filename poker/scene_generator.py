import logging  # Added logging
import math
import os
import random
import sys
import traceback  # Added traceback
from pathlib import Path
from typing import Any

import bpy
import mathutils  # Added for math utilities like Vector

# Ensure workspace root is in path for sibling imports
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if workspace_root not in sys.path:
    sys.path.append(workspace_root)

# Models and Builders
from noise.set_noise_config import build_noise_from_config  # Import noise module
from poker.card_overlap_builder import (
    build_card_overlap_layout_from_config,  # Import the overlap builder
)
from poker.config.models import PokerSceneModel
from poker.generate_legend import generate_poker_legend
from poker.load_card import _POKER_DECK_BLEND_FILE, PokerCardLoader
from poker.player_builder import build_player_from_config
from poker.river_builder import build_river_from_config
from scene_setup.general_setup import build_setup_from_config
from scene_setup.grid import add_grid_to_image_file  # Added for drawing grid

logger = logging.getLogger(__name__) # Added logger instance


def world_to_camera_view(scene, obj, coord):
    """
    Returns the camera space coords for a 3d point.
    (also known as: normalized device coordinates - NDC).

    Where (0, 0) is the bottom left and (1, 1)
    is the top right of the camera frame.
    values outside 0-1 are also supported.
    A negative 'z' value means the point is behind the camera.

    Takes shift-x/y, lens angle and sensor size into account
    as well as perspective/ortho projections.

    :arg scene: Scene to use for frame size.
    :type scene: :class:`bpy.types.Scene`
    :arg obj: Camera object.
    :type obj: :class:`bpy.types.Object`
    :arg coord: World space location.
    :type coord: :class:`mathutils.Vector`
    :return: a vector where X and Y map to the view plane and
       Z is the depth on the view axis.
    :rtype: :class:`mathutils.Vector`
    """
    from mathutils import Vector

    co_local = obj.matrix_world.normalized().inverted() @ coord
    z = -co_local.z

    camera = obj.data
    frame = list(camera.view_frame(scene=scene)[:3])
    if camera.type != 'ORTHO':
        if z == 0.0:
            return Vector((0.5, 0.5, 0.0))
        else:
            frame = [-(v / (v.z / z)) for v in frame]

    min_x, max_x = frame[2].x, frame[1].x
    min_y, max_y = frame[1].y, frame[0].y

    x = (co_local.x - min_x) / (max_x - min_x)
    y = (co_local.y - min_y) / (max_y - min_y)

    return Vector((x, y, z))



def get_object_pixel_bbox(scene: bpy.types.Scene, camera: bpy.types.Object, obj: bpy.types.Object, rendered_width: int, rendered_height: int) -> tuple[int, int, int, int] | None:
    """
    Calculates the 2D pixel bounding box of an object in the rendered image.

    Args:
        scene: The current Blender scene.
        camera: The active camera object.
        obj: The object whose bounding box is to be calculated.
        rendered_width: The width of the rendered image in pixels.
        rendered_height: The height of the rendered image in pixels.

    Returns:
        A tuple (min_x, min_y, max_x, max_y) representing the pixel bounding box,
        or None if the object is not in the camera view or an error occurs.
    """
    if not obj or not camera or not scene:
        logger.warning("get_object_pixel_bbox: Invalid scene, camera, or object provided.")
        return None

    # Get the 8 corners of the object's bounding box in world space
    bbox_corners_world = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]

    # Project these corners to 2D camera space (coordinates in [0, 1] range)
    coords_2d = []
    for corner_world in bbox_corners_world:
        # world_to_camera_view returns a 3D vector (x, y, depth)
        # x, y are [0, 1] if in view, depth is distance from camera
        co_2d = world_to_camera_view(scene, camera, corner_world)
        if co_2d.z > 0:  # Check if the corner is in front of the camera
            coords_2d.append(co_2d)

    if not coords_2d:
        logger.debug(f"Object '{obj.name}' is likely not in camera view or behind it.")
        return None

    # Convert to pixel coordinates and find min/max
    min_x, max_x = rendered_width, 0
    min_y, max_y = rendered_height, 0

    for co in coords_2d:
        # Convert [0,1] range to pixel coordinates
        px = int(round(co.x * rendered_width))
        py = int(round(co.y * rendered_height))

        # Blender's (0,0) for image coordinates is bottom-left,
        # but often image processing top-left. We'll use Blender's standard for now.
        # If top-left is needed, py = rendered_height - py

        min_x = min(min_x, px)
        max_x = max(max_x, px)
        min_y = min(min_y, py)
        max_y = max(max_y, py)

    # Clamp values to be within image dimensions, as projection can sometimes go slightly outside
    min_x = max(0, min_x)
    max_x = min(rendered_width -1 , max_x)
    min_y = max(0, min_y)
    max_y = min(rendered_height -1, max_y)

    if max_x < min_x or max_y < min_y:
        logger.debug(f"Object '{obj.name}' projected bounding box is invalid (max < min).")
        return None

    return min_x, min_y, max_x, max_y


def generate_poker_scene_from_config(
    scene_model: dict[str, Any] | PokerSceneModel,
    output_dir: str,
    base_filename: str
) -> dict[str, Any]:
    """
    Generates a poker scene, renders it to an 'img' subdir, and saves legends
    to 'legend_txt' and 'legend_json' subdirs within the output_dir.

    Args:
        scene_model: Resolved PokerSceneModel instance or config dict.
        output_dir: The base directory where subdirs for img, legend_txt,
                    and legend_json will be created.
        base_filename: Base filename for image and legends.

    Returns:
        A dictionary containing generation results:
        {
            'image_path': Optional[str],
            'legend_txt_path': Optional[str],
            'legend_json_path': Optional[str],
            'final_scene_config': Optional[Dict],
            'noise_config': Optional[Dict],
            'created_objects': Dict[str, List[bpy.types.Object]]
        }
    """
    all_loaded_cards: list[bpy.types.Object] = []
    all_loaded_chips: list[bpy.types.Object] = []
    config_model: PokerSceneModel | None = None
    rendered_image_path: str | None = None
    legend_txt_path: str | None = None
    legend_json_path: str | None = None
    final_config_dict: dict[str, Any] | None = None
    noise_result: dict[str, Any] | None = None
    card_grid_locations: dict[str, dict[str, Any]] = {} # To store card grid locations

    # Define base output path and subdirectories
    base_output_path = Path(output_dir)
    img_output_dir_path = base_output_path / "img"
    # Legend paths will be handled by generate_poker_legend

    # Ensure output directories exist (img dir specifically needed here)
    try:
        img_output_dir_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create image output directory '{img_output_dir_path}': {e}")
        return { # Return empty structure on directory failure
            'image_path': None, 'legend_txt_path': None, 'legend_json_path': None,
            'final_scene_config': None, 'noise_config': None,
            'created_objects': {'cards': [], 'chips': []},
            'card_grid_locations': {}
        }

    try:
        # 1. Parse and Validate Config
        if isinstance(scene_model, dict):
            config_model = PokerSceneModel.from_dict(scene_model)
        elif isinstance(scene_model, PokerSceneModel):
            config_model = scene_model
        else:
            raise TypeError(f"Expected dict or PokerSceneModel, got {type(scene_model)}")

        final_config_dict = config_model.to_dict()
        logger.info(f"--- Generating Poker Scene: {config_model.n_players} players for {base_filename} ---")

        # Generate Legends (passing only base output dir)
        logger.info(f"Generating legends for {base_filename} (inside {output_dir})...")
        legend_txt_path, legend_json_path = generate_poker_legend(
            scene_model=config_model,
            output_dir=str(base_output_path), # Pass base dir
            base_filename=base_filename
        )
        if not legend_txt_path:
             logger.warning("Legend generation failed or returned None paths.")

        # 2. Set Global Random Seed (if provided)
        if config_model.random_seed is not None:
            random.seed(config_model.random_seed)
            logger.info(f"Set global random seed to: {config_model.random_seed}")

        # 3. Perform General Scene Setup (Table, Camera, Lighting, Render)
        logger.info("Building general scene setup...")
        setup_info = build_setup_from_config(config_model.scene_setup)
        table_conf = config_model.scene_setup.get('table', {})
        if table_conf.get('shape', 'circular') == 'circular':
            table_radius = table_conf.get('diameter', 2.0) / 2.0
        else: # Rectangular approx
            table_radius = max(table_conf.get('length', 2.0), table_conf.get('width', 1.2)) / 2.0 * 0.8 # Approx radius
        logger.info("General scene setup complete.")

        # 4. Initialize Card Loader
        logger.debug("Initializing card loader...")
        loader_args = {}
        blend_file_to_use = config_model.deck_blend_file or _POKER_DECK_BLEND_FILE # Determine correct path
        loader_args['blend_file_path'] = blend_file_to_use
        card_loader = PokerCardLoader(**loader_args)
        logger.debug(f"Using deck blend file: {blend_file_to_use}")

        # 5. Build General Card Overlap Layout (if configured)
        logger.info("Checking for general card overlap layout config...")
        if config_model.card_overlap_config:
             logger.info("Building general card overlap layout...")
             overlap_cards = build_card_overlap_layout_from_config(
                 config=config_model.card_overlap_config, # Pass the resolved CardOverlapModel
                 card_loader=card_loader
             )
             all_loaded_cards.extend(overlap_cards)
             logger.info(f"Finished building general layout. Added {len(overlap_cards)} cards.")
        else:
             logger.info("No general card overlap layout configured.")

        # 6. Build Players (Hands and Chips)
        logger.info("Building players...")
        # Access the resolved player list, which might have been generated in __post_init__
        resolved_players = getattr(config_model, '_resolved_players', config_model.players)
        if not resolved_players:
            logger.info("No players defined or generated in configuration.")
        else:
            # Iterate over the resolved player list
            for _i, player_conf in enumerate(resolved_players):
                # player_conf is already guaranteed to be a PlayerModel instance
                # due to the logic in __post_init__

                player_id_str = player_conf.player_id # Use player_id from model
                logger.info(f"Building assets for {player_id_str}...")

                # Call the standalone function from player_builder.py
                created_player_objects = build_player_from_config(
                    config=player_conf, # Pass the PlayerModel instance directly
                    card_loader=card_loader
                )

                # Collect objects
                all_loaded_cards.extend(created_player_objects.get('cards', []))
                all_loaded_chips.extend(created_player_objects.get('chips', []))
                logger.info(f"Finished building assets for {player_id_str}.")

        # 7. Build Community Cards (River/Board)
        logger.info("Building community cards...")
        # Access the resolved community cards model, which might have been generated/modified in __post_init__
        resolved_community_model = getattr(config_model, '_resolved_community_cards', config_model.community_cards)
        if resolved_community_model:
            logger.info(f"Building river with {resolved_community_model.n_cards} cards: {resolved_community_model.card_names}")
            community_cards_objects = build_river_from_config(card_loader, resolved_community_model)
            all_loaded_cards.extend(community_cards_objects)
        else:
            logger.info("No community cards defined or generated in configuration.")

        # 8. Apply noise effects if provided
        logger.info("Checking for noise configuration...")
        if config_model.noise_config is not None:
            # Get the table object from setup_info
            table_object = setup_info.get("table", {}).get("object")

            # Apply noise configuration
            logger.info(f"Applying noise effects with config: {config_model.noise_config}")
            try:
                noise_result = build_noise_from_config(config_model.noise_config, table_object)
                logger.info("Noise effects applied successfully.")
            except Exception as noise_e:
                logger.error(f"Error applying noise effects: {noise_e}", exc_info=True)
                # Continue with rendering even if noise application fails
        else:
            logger.info("No noise configuration provided.")

        # 9. Render the Scene
        logger.info("Preparing to render scene...")
        render_config = config_model.scene_setup.get('render', {})
        img_extension = '.png'
        # MODIFIED: Save image to 'img' subdirectory
        abs_output_path = img_output_dir_path / f"{base_filename}{img_extension}"
        logger.info(f"Render output path set to: {abs_output_path}")
        bpy.context.scene.render.filepath = str(abs_output_path) # Use string path

        try:
            logger.info("Starting render...")
            bpy.ops.render.render(write_still=True)
            logger.info(f"Render finished! Image saved to {abs_output_path}")
            rendered_image_path = str(abs_output_path) # Store path on success
        except Exception as render_e:
            logger.error(f"Error during rendering: {render_e}", exc_info=True)

        # --- Draw Grid on Rendered Image (if configured) ---
        if rendered_image_path and isinstance(config_model.scene_setup, dict) and config_model.scene_setup.get('grid'):
            logger.info(f"Drawing grid on image: {rendered_image_path}")
            grid_draw_config_dict = config_model.scene_setup.get('grid') # Get dict directly
            if grid_draw_config_dict: # Ensure it's not None
                try:
                    success_drawing_grid = add_grid_to_image_file(
                        image_filepath=rendered_image_path,
                        grid_config=grid_draw_config_dict,
                        output_filepath=rendered_image_path # Overwrite the original image
                    )
                    if success_drawing_grid:
                        logger.info("Successfully drew grid on the image.")
                    else:
                        logger.warning("Failed to draw grid on the image.")
                except Exception as grid_draw_e:
                    logger.error(f"Error drawing grid on image: {grid_draw_e}", exc_info=True)
            else:
                logger.warning("Grid configuration found but is empty/None.")
        elif rendered_image_path and hasattr(config_model.scene_setup, 'grid') and config_model.scene_setup.grid: # Object access fallback
            logger.info(f"Drawing grid on image (object access): {rendered_image_path}")
            grid_draw_config_dict = config_model.scene_setup.grid.to_dict()
            try:
                success_drawing_grid = add_grid_to_image_file(
                    image_filepath=rendered_image_path,
                    grid_config=grid_draw_config_dict,
                    output_filepath=rendered_image_path # Overwrite the original image
                )
                if success_drawing_grid:
                    logger.info("Successfully drew grid on the image.")
                else:
                    logger.warning("Failed to draw grid on the image.")
            except Exception as grid_draw_e:
                logger.error(f"Error drawing grid on image: {grid_draw_e}", exc_info=True)
        else:
            logger.info("Skipping grid drawing: No image rendered or no grid config in scene_setup.")

        # --- Calculate Card Grid Locations (after rendering and grid drawing) ---
        logger.info("Calculating card grid locations...")
        granularity = 0
        grid_config_for_calc = None

        if isinstance(config_model.scene_setup, dict):
            grid_config_for_calc = config_model.scene_setup.get('grid')
        elif hasattr(config_model.scene_setup, 'grid'): # Object access
            grid_object_for_calc = config_model.scene_setup.grid
            if grid_object_for_calc: # If it's an object, get its granularity attribute
                 grid_config_for_calc = grid_object_for_calc # Keep for consistency if needed, or just get granularity
                 granularity = grid_object_for_calc.granularity

        if rendered_image_path and grid_config_for_calc:
            if isinstance(grid_config_for_calc, dict) and not granularity: # If it was a dict, get granularity from it
                granularity = grid_config_for_calc.get('granularity', 0)

            # Ensure granularity is now set if grid_config_for_calc was valid
            if not granularity and hasattr(grid_config_for_calc, 'granularity'): # Check if it was an object after all
                 granularity = grid_config_for_calc.granularity

            if granularity > 0:
                scene = bpy.context.scene
                camera = scene.camera # Assumes camera is set correctly
                render_settings = scene.render
                rendered_width = int(render_settings.resolution_x * (render_settings.resolution_percentage / 100.0))
                rendered_height = int(render_settings.resolution_y * (render_settings.resolution_percentage / 100.0))

                if camera and granularity > 0 and rendered_width > 0 and rendered_height > 0:
                    cell_width = rendered_width / granularity
                    cell_height = rendered_height / granularity

                    for card_obj in all_loaded_cards:
                        if not card_obj or not hasattr(card_obj, 'name'):
                            logger.warning("Skipping invalid card object in all_loaded_cards.")
                            continue

                        pixel_bbox = get_object_pixel_bbox(scene, camera, card_obj, rendered_width, rendered_height)
                        if pixel_bbox:
                            min_x_px, min_y_px, max_x_px, max_y_px = pixel_bbox

                            # Determine grid cells the card overlaps
                            # Note: Blender pixel coords are (0,0) at bottom-left.
                            # If grid drawing assumes (0,0) top-left, y-coordinates need inversion for cell mapping.
                            # For now, assume grid drawing also uses bottom-left (0,0) or pixel data access handles it.

                            start_col = math.floor(min_x_px / cell_width)
                            end_col = math.floor(max_x_px / cell_width)
                            # For y-coordinates, Blender y increases upwards.
                            # If grid is drawn from top (row 0) to bottom, then convert Blender y to grid row.
                            # Let's assume row 0 is at the bottom for consistency with Blender pixel y for now.
                            start_row = math.floor(min_y_px / cell_height)
                            end_row = math.floor(max_y_px / cell_height)

                            # Ensure indices are within bounds
                            start_col = max(0, min(start_col, granularity - 1))
                            end_col = max(0, min(end_col, granularity - 1))
                            start_row = max(0, min(start_row, granularity - 1))
                            end_row = max(0, min(end_row, granularity - 1))

                            occupied_cells = []
                            for r in range(start_row, end_row + 1):
                                for c in range(start_col, end_col + 1):
                                    occupied_cells.append((r, c))

                            card_grid_locations[card_obj.name] = {
                                "pixel_bbox": (min_x_px, min_y_px, max_x_px, max_y_px),
                                "grid_cells": occupied_cells,
                                "center_grid_cell": (math.floor((min_y_px + max_y_px) / 2 / cell_height), math.floor((min_x_px + max_x_px) / 2 / cell_width))
                            }
                            logger.debug(f"Card '{card_obj.name}' occupies cells: {occupied_cells}")
                        else:
                            logger.debug(f"Could not get pixel bounding box for card '{card_obj.name}'.")
                else:
                    logger.warning("Skipping card grid location calculation: Missing camera, zero granularity, or zero render dimensions.")
            else:
                logger.info("Skipping card grid location calculation: No image rendered or no grid config.")
        else:
            logger.info("Skipping card grid location calculation: No image rendered or no grid config.")

        total_objects = len(all_loaded_cards) + len(all_loaded_chips)
        logger.info(f"--- Poker Scene Generation Complete for {base_filename} ---")
        logger.info(f"  Total Players Configured: {len(config_model.players)}")
        logger.info(f"  Total Cards Created: {len(all_loaded_cards)}")
        logger.info(f"  Total Chips Created: {len(all_loaded_chips)}")
        logger.info(f"  Total Objects Created: {total_objects}")
        logger.info(f"  Image Rendered: {'Yes' if rendered_image_path else 'No'}")

    except (ValueError, TypeError, FileNotFoundError) as e:
        logger.error(f"Error generating poker scene from configuration for {base_filename}: {e}")
        logger.error(f"Input config snippet: {str(scene_model)[:500]}...")
        logger.error(traceback.format_exc())
        # Return paths collected so far, including potentially failed legends
        return {
            'image_path': rendered_image_path,
            'legend_txt_path': legend_txt_path,
            'legend_json_path': legend_json_path,
            'final_scene_config': final_config_dict,
            'noise_config': noise_result,
            'created_objects': {'cards': all_loaded_cards, 'chips': all_loaded_chips},
            'card_grid_locations': card_grid_locations # Add new info to results
        }
    except Exception as e:
        logger.error(f"An unexpected error occurred during scene generation for {base_filename}: {e}")
        logger.error(traceback.format_exc())
        return {
            'image_path': rendered_image_path,
            'legend_txt_path': legend_txt_path,
            'legend_json_path': legend_json_path,
            'final_scene_config': final_config_dict,
            'noise_config': noise_result,
            'created_objects': {'cards': all_loaded_cards, 'chips': all_loaded_chips},
            'card_grid_locations': card_grid_locations # Add new info to results
        }

    # Return results
    return {
        'image_path': rendered_image_path,
        'legend_txt_path': legend_txt_path,
        'legend_json_path': legend_json_path,
        'final_scene_config': final_config_dict,
        'noise_config': noise_result,
        'created_objects': {'cards': all_loaded_cards, 'chips': all_loaded_chips},
        'card_grid_locations': card_grid_locations # Add new info to results
    }

# --- Test Section ---
if __name__ == "__main__":

    # Setup basic logging for the test
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logger.info("--- Running Poker Scene Generator Test ---")

    # Define default table height for use in test config locations
    table_height = 0.9

    # --- Example Chip Configs (for reuse in players) ---
    base_chip_blue = {"chip_object_name": "Cylinder001", "scale": 0.06, "color": (0.1, 0.2, 0.8, 1)}
    base_chip_red = {"chip_object_name": "Cylinder001", "scale": 0.06, "color": (0.8, 0.1, 0.1, 1)}
    base_pile_blue = {"n_chips": 8, "base_chip_config": base_chip_blue, "spread_factor": 0.1}
    base_pile_red = {"n_chips": 12, "base_chip_config": base_chip_red, "spread_factor": 0.2}
    chip_area_1 = {
        "base_pile_config": base_pile_blue,
        "n_piles": 2,
        "n_chips_per_pile": [8, 10],
        "pile_colors": [None, (0.2, 0.8, 0.2, 1)], # Blue (from base), Green
        "pile_spreads": [None, 0.3],
        "random_seed": 1001
    }
    chip_area_2 = {
        "base_pile_config": base_pile_red,
        "n_piles": 1,
        "random_seed": 1002
    }
    # --- End Chip Configs ---

    # Define test output parameters
    test_output_dir = "poker/img/test_output"
    test_base_filename = "poker_scene_test_render"

    # Define a complete scene configuration dictionary
    test_scene_config = {
        "n_players": 6,
        "random_seed": 12345,
        "deck_blend_file": None, # Use default
        "scene_setup": {
            "camera": {"distance": 3.7, "angle": 72, "horizontal_angle": 0}, # Adjusted distance
            "lighting": {"lighting": "medium"},
            "table": {
                "shape": "rectangular",
                "width": 1.5,
                "length": 2.4,
                "felt_color": (0.1, 0.4, 0.15, 1.0)
            },
            "render": {
                "engine": "CYCLES",
                "samples": 12, # Reduced samples for faster test
                "resolution": {"width": 512, "height": 384}, # Smaller resolution for test
                # Output path is now constructed by the function based on args
                # "output_path": "poker/img/poker_scene_6_player_v2_test.png",
                "gpus_enabled" : False,
            },
            "grid": { # Added grid configuration
                "granularity": 3,
                "line_thickness": 2,
                "line_color_rgba": (0.0, 0.0, 0.0, 0.8)
            }
        },
        "players": [
            {
                'player_id': 'Alice',
                'hand_config': {
                    'card_names': ['AS'],
                    'n_cards': 1,
                    'location': (-0.6, 0.0, table_height + 0.01),
                    'scale': 0.1,
                    'spread_factor_h': 0.2,
                    'spread_factor_v': 0.05,
                    'n_verso': 0,
                    'random_seed': 101
                },
                'chip_area_config': chip_area_1
            },
            {
                'player_id': 'Bob',
                'hand_config': {
                    'card_names': ['KD', 'KS'],
                    'n_cards': 2,
                    'location': (0.6, 0.0, table_height + 0.01),
                    'scale': 0.1,
                    'spread_factor_h': 0.3,
                    'spread_factor_v': 0.1,
                    'n_verso': 1,
                    'verso_loc': 'random',
                    'random_seed': 102
                },
                'chip_area_config': chip_area_2
            },
            {
                'player_id': 'Charlie',
                'hand_config': {
                    'card_names': ['QD', 'QS'],
                    'n_cards': 2,
                    'location': (0.0, 1.0, table_height + 0.01),
                    'scale': 0.1,
                    'spread_factor_h': 0.2,
                    'spread_factor_v': 0.3,
                    'n_verso': 2,
                    'random_seed': 103
                },
            },
            {
                'player_id': 'Diana',
                'hand_config': {
                    'card_names': ['JD', 'JS'],
                    'n_cards': 2,
                    'location': (0.0, -1.0, table_height + 0.01),
                    'scale': 0.1,
                    'spread_factor_h': 0.2,
                    'spread_factor_v': 0.3,
                    'n_verso': 1,
                    'verso_loc': 'ordered',
                    'random_seed': 104
                }
            },
            {
                'player_id': 'Eve',
                'hand_config': {
                    'card_names': ['10D', '10S'],
                    'n_cards': 2,
                    'location': (-0.5, 0.6, table_height + 0.01),
                    'scale': 0.1,
                    'spread_factor_h': 0.9,
                    'spread_factor_v': 0.1,
                    'n_verso': 1,
                    'random_seed': 105
                },
                'chip_area_config': chip_area_1
            },
            {
                'player_id': 'Frank',
                'hand_config': {
                    'card_names': ['9D', '9S'],
                    'n_cards': 2,
                    'location': (0.6, -0.7, table_height + 0.01),
                    'scale': 0.1,
                    'spread_factor_h': 0.2,
                    'spread_factor_v': 0.2,
                    'n_verso': 1,
                    'random_seed': 106
                }
            }
        ],
        "community_cards": {
            'card_names': ['4C', '4H', '4D', '4S', '5C'], 'n_cards': 5,
            'start_location': (-0.3, 0, 0.9 + 0.01),
            'scale': 0.1,
            'n_verso': 0,
            'card_gap': {'base_gap_x': 0.15, 'base_gap_y': 0.005, 'random_gap': False} # Added base_gap_y for model compliance
        }
    }

    # Generate the scene and render
    generation_result = generate_poker_scene_from_config(
        scene_model=test_scene_config,
        output_dir=test_output_dir,
        base_filename=test_base_filename
        )

    logger.info(f"Function returned: {generation_result}")

    # Check results
    if generation_result and generation_result.get('image_path'):
        logger.info("Scene generation and rendering likely succeeded.")
        logger.info(f"Image saved to: {generation_result['image_path']}")
        # Further checks could be added here (e.g., object counts)
    else:
        logger.error("Scene generation or rendering failed.")
        sys.exit(1)

    logger.info("--- Poker Scene Generator Test Finished ---")
