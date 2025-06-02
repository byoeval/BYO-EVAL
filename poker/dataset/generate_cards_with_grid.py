import bpy
import random
import os
import sys
import math
import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# Ensure workspace root is in path for sibling imports
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if workspace_root not in sys.path:
    sys.path.append(workspace_root)

try:
    from poker.scene_generator import generate_poker_scene_from_config
except ImportError as e:
    print(f"Error importing project modules: {e}. Ensure PYTHONPATH is set correctly or run from project root.")
    sys.exit(1)

# --- Configuration Section ---
NUM_IMAGES = 50  # Number of images to generate
NUM_RANDOM_CARDS = 1  # Number of random cards to place in each image
OUTPUT_DIR = "poker/output/one_card_loc_grid_3x3"
BASE_FILENAME = "output_path_img"
GLOBAL_RANDOM_SEED: Optional[int] = None  # None for true randomization across runs

TABLE_SHAPE: str = "rectangular"  # "rectangular" or "circular"
TABLE_DIMENSIONS: Dict[str, float] = {"width": 1.5, "length": 2.4} # if rectangular
TABLE_HEIGHT: float = 0.90  # Z-coordinate of the table surface
TABLE_FELT_COLOR: Tuple[float, float, float, float] = (0.1, 0.4, 0.15, 1.0)

CARD_SCALE: float = 0.15
CARD_Z_OFFSET: float = 0.01 # Small offset to place card slightly above table

CAMERA_DISTANCE: float = 4.0
CAMERA_ANGLE: float = 75
CAMERA_HORIZONTAL_ANGLE: float = 0

RENDER_ENGINE: str = "CYCLES"
RENDER_SAMPLES: int = 4096  # Set lower for faster tests, higher for quality
RENDER_RESOLUTION: Dict[str, int] = {"width": 1920, "height": 1080}
GPUS_ENABLED: bool = True  # Use GPU rendering if available

GRID_GRANULARITY: int = 3
GRID_LINE_THICKNESS: int = 2
GRID_LINE_COLOR_RGBA: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.8) # Black, mostly opaque

# Available card faces (Abbreviated: Suit then Rank: S=Spades, H=Hearts, D=Diamonds, C=Clubs; A,K,Q,J,T(10),9-2)
AVAILABLE_CARD_FACES: List[str] = ['QS', 'QH', 'QD', 'QC', 'KS', 'KH', 'KD', 'KC', 'JS', 'JH', 'JD', 'JC', 'AS', 'AH', 'AD', 'AC', '9S', '9H', '9D', '9C', '8S', '8H', '8D', '8C', '7S', '7H', '7D', '7C', '6S', '6H', '6D', '6C', '5S', '5H', '5D', '5C', '4S', '4H', '4D', '4C', '3S', '3H', '3D', '3C', '2S', '2H', '2D', '2C', '10S', '10H', '10D', '10C']

# --- End Configuration Section ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _calculate_random_position_on_table(
    table_shape: str,
    table_dimensions: Dict[str, float],
    table_height: float,
    card_z_offset: float
) -> Tuple[float, float, float]:
    """Calculates a random (x, y, z) position on the table surface."""
    loc_z = table_height + card_z_offset
    loc_x, loc_y = 0.0, 0.0
    edge_inset_factor = 0.9  # Place cards within 60% of the table's central area

    if table_shape == "rectangular":
        width = table_dimensions.get("width", 1.0)
        length = table_dimensions.get("length", 1.0)
        half_width = width / 2.0 * edge_inset_factor
        half_length = length / 2.0 * edge_inset_factor
        loc_y = random.uniform(-half_length, half_length)
        loc_x = random.uniform(-half_width, half_width)
    elif table_shape == "circular":
        diameter = table_dimensions.get("diameter", 1.0)
        radius = diameter / 2.0 * edge_inset_factor
        angle = random.uniform(0, 2 * math.pi)
        dist = random.uniform(0, radius) # Distribute points uniformly in a circle
        loc_x = dist * math.cos(angle)
        loc_y = dist * math.sin(angle)
    else:
        logger.warning(f"Unsupported table shape '{table_shape}'. Defaulting to (0,0) for x,y.")

    return loc_x, loc_y, loc_z


def _generate_single_card_player_config(
    player_id: str,
    table_shape: str,
    table_dimensions: Dict[str, float],
    table_height: float,
    card_scale: float,
    card_z_offset: float,
    all_card_faces: List[str],
    used_cards: set # Keep track of used cards to avoid duplicates if desired
) -> Dict[str, Any]:
    """
    Generates the configuration dictionary for a single "player"
    which represents one randomly placed card.
    """
    position = _calculate_random_position_on_table(
        table_shape, table_dimensions, table_height, card_z_offset
    )

    # Select a card, try to avoid reuse if possible
    chosen_card = None
    if not all_card_faces:
        logger.error("No card faces available to choose from!")
        return {} # Should not happen with default list

    available_to_pick = [card for card in all_card_faces if card not in used_cards]
    if not available_to_pick: # All cards used, allow reuse (should only happen if NUM_RANDOM_CARDS > 52)
        logger.warning("All unique card faces used, allowing reuse.")
        available_to_pick = all_card_faces
    
    chosen_card = random.choice(available_to_pick)
    used_cards.add(chosen_card)

    # Random Z-axis rotation for the card
    random_rotation_z_degrees = random.uniform(0, 360)
    card_rotation_euler_rad = (0, 0, math.radians(random_rotation_z_degrees))

    player_conf = {
        "player_id": player_id,
        "hand_config": {
            "card_names": [chosen_card] if chosen_card else ["AS"], # Fallback to Ace of Spades
            "n_cards": 1,
            "location": position,
            "scale": card_scale,
            "rotation_euler": card_rotation_euler_rad, # Added rotation
            "spread_factor_h": 0, # No spread for a single card
            "spread_factor_v": 0,
            "n_verso": 0 # Card is face up
        },
        "chip_area_config": None  # No chips for these random cards
    }
    return player_conf


def _update_legend_with_grid_info(legend_json_path: str, card_grid_locations: Dict[str, Any]) -> bool:
    """
    Updates the legend JSON file with card grid location information.
    
    Args:
        legend_json_path: Path to the legend JSON file
        card_grid_locations: Dictionary containing card grid location data
        
    Returns:
        True if update was successful, False otherwise
    """
    if not legend_json_path or not os.path.exists(legend_json_path):
        logger.error(f"Legend JSON file not found: {legend_json_path}")
        return False
        
    try:
        # Read the existing JSON legend
        with open(legend_json_path, 'r') as f:
            legend_data = json.load(f)
            
        # Add card grid location information
        legend_data['card_grid_locations'] = card_grid_locations
        
        # Format card information better for readability
        if 'players' in legend_data:
            for player in legend_data['players']:
                if 'hand_config' in player and 'card_names' in player['hand_config']:
                    card_names = player['hand_config']['card_names']
                    player_id = player.get('player_id', 'Unknown')
                    
                    # Match card objects to their grid locations
                    for card_name in card_names:
                        for obj_name, grid_info in card_grid_locations.items():
                            # Card objects often include the card name in their object name
                            if card_name in obj_name:
                                # Add a reference to this card's grid location directly in the player info
                                if 'grid_locations' not in player:
                                    player['grid_locations'] = {}
                                player['grid_locations'][card_name] = grid_info
                                logger.info(f"Added grid location for card {card_name} to player {player_id}")
        
        # Write the updated JSON back to the file
        with open(legend_json_path, 'w') as f:
            json.dump(legend_data, f, indent=2)
            
        logger.info(f"Successfully updated legend JSON with grid information: {legend_json_path}")
        return True
    except Exception as e:
        logger.error(f"Error updating legend JSON with grid information: {e}", exc_info=True)
        return False


def generate_random_card_scene(image_index: int, output_dir_path: Path, seed_offset: int) -> Dict[str, Any]:
    """
    Generate a single random card scene with configured parameters.
    
    Args:
        image_index: Index of the image being generated (used for filename)
        output_dir_path: Path to output directory
        seed_offset: Offset to add to global seed for this specific image
        
    Returns:
        Dictionary with generation results
    """
    current_filename = f"{BASE_FILENAME}_{image_index:05d}"
    
    # Set up the random seed for this image if a global seed was specified
    current_seed = None
    if GLOBAL_RANDOM_SEED is not None:
        current_seed = GLOBAL_RANDOM_SEED + seed_offset
        random.seed(current_seed)
        logger.info(f"Using random seed: {current_seed} for image {image_index}")
    
    # Generate player configurations (each player is a single card)
    players_config_list: List[Dict[str, Any]] = []
    used_cards_for_this_scene = set()
    for i in range(NUM_RANDOM_CARDS):
        player_id = f"RandomCard_{i+1}"
        player_conf = _generate_single_card_player_config(
            player_id=player_id,
            table_shape=TABLE_SHAPE,
            table_dimensions=TABLE_DIMENSIONS,
            table_height=TABLE_HEIGHT,
            card_scale=CARD_SCALE,
            card_z_offset=CARD_Z_OFFSET,
            all_card_faces=AVAILABLE_CARD_FACES,
            used_cards=used_cards_for_this_scene
        )
        if player_conf:
            players_config_list.append(player_conf)

    if not players_config_list:
        logger.error("No player configurations were generated. Skipping this image.")
        return {}

    # Construct the full scene configuration dictionary
    scene_config: Dict[str, Any] = {
        "n_players": len(players_config_list), # Reflects number of "card players"
        "random_seed": current_seed,
        "deck_blend_file": None,  # Use default deck
        "scene_setup": {
            "camera": {
                "distance": CAMERA_DISTANCE,
                "angle": CAMERA_ANGLE,
                "horizontal_angle": CAMERA_HORIZONTAL_ANGLE
            },
            "lighting": {"lighting": "medium"},
            "table": {
                "shape": TABLE_SHAPE,
                **TABLE_DIMENSIONS, # Unpack width/length or diameter
                "height": TABLE_HEIGHT, # Add height to table config as it's used by setup
                "felt_color": TABLE_FELT_COLOR
            },
            "render": {
                "engine": RENDER_ENGINE,
                "samples": RENDER_SAMPLES,
                "resolution": RENDER_RESOLUTION,
                "gpus_enabled": GPUS_ENABLED,
                "gpus":[1,2]
            },
            "grid": { # Added grid configuration
                "granularity": GRID_GRANULARITY,
                "line_thickness": GRID_LINE_THICKNESS,
                "line_color_rgba": GRID_LINE_COLOR_RGBA
            }
        },
        "players": players_config_list,
        "community_cards": None, # No community cards for this specific test
        "card_overlap_config": None, # No general overlap for this test
        "noise_config": None # No noise for this test
    }

    # Call the main scene generator function
    logger.info(f"Generating scene {image_index}/{NUM_IMAGES}: {current_filename}")
    try:
        generation_result = generate_poker_scene_from_config(
            scene_model=scene_config, # Pass the dictionary directly
            output_dir=str(output_dir_path), # scene_generator expects string path
            base_filename=current_filename
        )

        if generation_result and generation_result.get('image_path'):
            logger.info(f"SUCCESS: Image generated at: {generation_result['image_path']}")
            
            # Check for legend JSON path and card grid locations
            legend_json_path = generation_result.get('legend_json_path')
            card_grid_locations = generation_result.get('card_grid_locations', {})
            
            if legend_json_path and card_grid_locations:
                update_success = _update_legend_with_grid_info(legend_json_path, card_grid_locations)
                if update_success:
                    logger.info(f"Legend JSON updated with grid location information: {legend_json_path}")
                    
                    # Create text version of the legend
                    try:
                        # Read the JSON data
                        with open(legend_json_path, 'r') as f:
                            legend_data = json.load(f)
                        
                        # Create a text version of the legend - path should already be correct due to our updates
                        txt_legend_path = legend_json_path.replace('.json', '.txt')
                        with open(txt_legend_path, 'w') as f:
                            f.write(f"Legend for image: {current_filename}.png\n")
                            f.write("=" * 40 + "\n\n")
                            
                            # Write main scene info
                            f.write(f"Cards placed: {NUM_RANDOM_CARDS}\n")
                            f.write(f"Grid granularity: {GRID_GRANULARITY}x{GRID_GRANULARITY}\n\n")
                            
                            # Write card information
                            f.write("CARD INFORMATION:\n")
                            f.write("-" * 40 + "\n")
                            for player in legend_data.get('players', []):
                                player_id = player.get('player_id', 'Unknown')
                                f.write(f"Player ID: {player_id}\n")
                                
                                if 'hand_config' in player and 'card_names' in player['hand_config']:
                                    cards = player['hand_config']['card_names']
                                    f.write(f"  Cards: {', '.join(cards)}\n")
                                    
                                    # Add grid locations if available
                                    if 'grid_locations' in player:
                                        for card, grid_info in player.get('grid_locations', {}).items():
                                            row = grid_info.get('row', 'Unknown')
                                            col = grid_info.get('col', 'Unknown')
                                            f.write(f"  {card} grid location: row {row}, col {col}\n")
                                f.write("\n")
                        
                        logger.info(f"Created text legend file: {txt_legend_path}")
                    except Exception as e:
                        logger.error(f"Failed to create text legend file: {e}")
                else:
                    logger.error(f"Failed to update legend JSON with grid location information: {legend_json_path}")
            
            # Return generation info for summary
            return {
                'filename': current_filename,
                'image_path': generation_result.get('image_path'),
                'legend_json_path': legend_json_path,
                'card_grid_locations': card_grid_locations
            }
        else:
            logger.error(f"FAILURE: Scene generation failed for {current_filename}")
            return {}
    except Exception as e:
        logger.error(f"Error generating scene {current_filename}: {e}", exc_info=True)
        return {}


def create_dataset_summary(output_dir_path: Path, generation_results: List[Dict[str, Any]]):
    """
    Creates a summary JSON file for the entire dataset.
    
    Args:
        output_dir_path: Path to output directory
        generation_results: List of results from all image generations
    """
    successful_generations = [r for r in generation_results if r.get('image_path')]
    
    summary = {
        "dataset_info": {
            "total_images": len(successful_generations),
            "total_attempted": NUM_IMAGES,
            "image_resolution": RENDER_RESOLUTION,
            "grid_granularity": GRID_GRANULARITY,
            "cards_per_image": NUM_RANDOM_CARDS,
        },
        "images": []
    }
    
    for result in successful_generations:
        image_path = result.get('image_path', '')
        legend_json_path = result.get('legend_json_path', '')
        
        # Extract just the filename components for cleaner summary
        image_filename = os.path.basename(image_path) if image_path else ''
        legend_filename = os.path.basename(legend_json_path) if legend_json_path else ''
        
        summary['images'].append({
            "filename": result.get('filename', ''),
            "image_file": image_filename,
            "legend_file": legend_filename,
            "has_grid_info": bool(result.get('card_grid_locations'))
        })
    
    # Save the summary
    summary_path = output_dir_path / "dataset_summary.json"
    try:
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Dataset summary written to {summary_path}")
    except Exception as e:
        logger.error(f"Failed to write dataset summary: {e}", exc_info=True)


def main():
    """
    Main function to generate a dataset of random card scenes with grid information.
    """
    logger.info(f"=== Generating dataset of {NUM_IMAGES} random card scenes with grid information ===")

    # Ensure output directory exists
    output_dir_path = Path(workspace_root) / OUTPUT_DIR
    try:
        output_dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir_path.resolve()}")
    except OSError as e:
        logger.error(f"Failed to create output directory '{output_dir_path}': {e}")
        return

    # Set the global random seed if specified
    if GLOBAL_RANDOM_SEED is not None:
        random.seed(GLOBAL_RANDOM_SEED)
        logger.info(f"Global random seed set to: {GLOBAL_RANDOM_SEED}")
    else:
        logger.info("Using non-deterministic random seed for dataset (true randomization)")

    # Generate all images
    all_results = []
    for i in range(NUM_IMAGES):
        image_index = i + 1  # 1-based indexing for filenames
        seed_offset = i  # Ensure each image has a unique random seed if global seed is set
        
        result = generate_random_card_scene(
            image_index=image_index,
            output_dir_path=output_dir_path,
            seed_offset=seed_offset
        )
        
        if result:
            all_results.append(result)
            logger.info(f"Completed image {image_index}/{NUM_IMAGES}")
        else:
            logger.warning(f"Failed to generate image {image_index}/{NUM_IMAGES}")
    
    # Create a summary of the dataset
    if all_results:
        create_dataset_summary(output_dir_path, all_results)
        logger.info(f"Successfully generated {len(all_results)}/{NUM_IMAGES} images with grid information")
    else:
        logger.error("No images were successfully generated")

    logger.info("=== Dataset generation complete ===")


if __name__ == "__main__":
    try:
        import bpy
        if bpy.context is None:
            logger.error("Blender context is None. This script needs a full Blender environment.")
            logger.error("Try running from Blender's scripting tab or via: blender --background --python your_script.py")
        # Proceed with main() regardless, as the code might be importable in non-Blender environments
        # for module usage, but generate_poker_scene_from_config will fail if context is missing.
        main()
    except ImportError:
        logger.error("Failed to import bpy. This script must be run within Blender or an environment where 'bpy' is available.")
        logger.error("Example: blender --background --python poker/examples/generate_random_cards_dataset.py")
        sys.exit(1) 