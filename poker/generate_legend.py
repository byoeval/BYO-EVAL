"""
Generate legend information from resolved PokerSceneModel configurations.

This module provides a class to convert configuration dictionaries into formatted legend
information for poker scenes, including table, cards, players, and chips.
"""

import json
import logging

from pathlib import Path
from typing import Any


from poker.config.models import (
    CardOverlapModel,
    ChipAreaConfig,
    PlayerHandModel,
    PlayerModel,
    PokerSceneModel,
    RiverModel,
)

logger = logging.getLogger(__name__)

class PokerConfigToLegend:
    """
    Class for converting resolved PokerSceneModel configurations into legend information.
    """

    def __init__(self, scene_model: PokerSceneModel):
        """
        Initialize the PokerConfigToLegend instance.

        Args:
            scene_model: The resolved PokerSceneModel instance containing the final scene configuration.
        """
        if not isinstance(scene_model, PokerSceneModel):
             raise TypeError("Input must be a PokerSceneModel instance.")
        self.scene_model = scene_model

    def _format_vector(self, vec: tuple | list | None, precision: int = 3) -> str:
        """Helper to format tuples/lists of numbers nicely."""
        if vec is None:
            return "N/A"
        try:
            if isinstance(vec, list | tuple):
                 formatted = ", ".join([f"{v:.{precision}f}" if isinstance(v, int | float) else str(v) for v in vec])
                 return f"({formatted})"
            elif isinstance(vec, int | float): # Handle single number scale
                return f"{vec:.{precision}f}"
            else:
                return str(vec)
        except Exception:
            return str(vec) # Fallback

    # --- Scene Setup ---
    def get_scene_setup_dict(self) -> dict[str, Any]:
        """Extract and format general scene setup info into a dictionary."""
        # Use the raw scene_setup dict from the model
        return self.scene_model.scene_setup

    def get_scene_setup_text(self) -> str:
        """Generate a text representation of the general scene setup."""
        setup = self.scene_model.scene_setup
        text = "SCENE SETUP\n===========\n"
        text += f"Random Seed (Top Level): {self.scene_model.random_seed}\n"
        text += f"Deck Blend File: {self.scene_model.deck_blend_file or 'Default'}\n"

        # Camera
        cam = setup.get('camera', {})
        text += "\nCamera:\n"
        text += f"  Distance: {self._format_vector(cam.get('distance'))}\n"
        text += f"  Vertical Angle: {self._format_vector(cam.get('angle'))} deg\n"
        text += f"  Horizontal Angle: {self._format_vector(cam.get('horizontal_angle'))} deg\n"

        # Lighting
        light = setup.get('lighting', {})
        text += "\nLighting:\n"
        text += f"  Type: {light.get('lighting', 'N/A')}\n"

        # Table
        table = setup.get('table', {})
        text += "\nTable:\n"
        text += f"  Shape: {table.get('shape', 'N/A')}\n"
        if table.get('shape') == 'circular':
            text += f"  Diameter: {self._format_vector(table.get('diameter'))}\n"
        elif table.get('shape') == 'rectangular':
            text += f"  Length: {self._format_vector(table.get('length'))}\n"
            text += f"  Width: {self._format_vector(table.get('width'))}\n"
        text += f"  Height: {self._format_vector(table.get('height'))}\n"
        text += f"  Felt Color: {self._format_vector(table.get('felt_color'))}\n"

        # Render
        render = setup.get('render', {})
        text += "\nRender:\n"
        text += f"  Engine: {render.get('engine', 'N/A')}\n"
        text += f"  Samples: {render.get('samples', 'N/A')}\n"
        res = render.get('resolution', {})
        text += f"  Resolution: {res.get('width', 'N/A')}x{res.get('height', 'N/A')}\n"
        text += f"  GPUs Enabled: {render.get('gpus_enabled', 'N/A')}\n"

        return text

    # --- Card Overlap Layout ---
    def get_card_overlap_dict(self) -> dict[str, Any] | None:
        """Extract info from the card overlap model."""
        overlap_model = self.scene_model.card_overlap_config
        if overlap_model and isinstance(overlap_model, CardOverlapModel):
            return overlap_model.to_dict()
        return None

    def get_card_overlap_text(self) -> str:
        """Generate a text representation of the card overlap layout."""
        overlap_model = self.scene_model.card_overlap_config
        text = "CARD OVERLAP LAYOUT\n===================\n"
        if not overlap_model or not isinstance(overlap_model, CardOverlapModel):
            text += "None\n"
            return text

        text += f"Number of Cards: {overlap_model.overall_cards}\n"
        text += f"Layout Mode: {overlap_model.layout_mode.value}\n"

        if overlap_model.layout_mode.value == 'horizontal':
            text += f"Number of Lines: {overlap_model.n_lines}\n"
        else:  # vertical
            text += f"Number of Columns: {overlap_model.n_columns}\n"

        text += f"Center Location: {self._format_vector(overlap_model.center_location)}\n"
        text += f"Scale: {self._format_vector(overlap_model.scale)}\n"
        text += f"Horizontal Overlap Factor: {overlap_model.horizontal_overlap_factor}\n"
        text += f"Vertical Overlap Factor: {overlap_model.vertical_overlap_factor}\n"

        if overlap_model.n_verso > 0:
            text += f"Face Down Cards: {overlap_model.n_verso} ('{overlap_model.verso_loc}' placement)\n"
        else:
            text += "Face Down Cards: None\n"

        # Card type info summary
        card_type = overlap_model.card_type_config
        if card_type and isinstance(card_type, dict):
            mode = card_type.get('mode', 'full_deck')
            text += f"Card Type: {mode}\n"

        text += f"Random Seed: {overlap_model.random_seed}\n"
        return text

    # --- Community Cards ---
    def get_community_cards_dict(self) -> dict[str, Any] | None:
        """Extract info from the resolved community cards model."""
        community_model = getattr(self.scene_model, '_resolved_community_cards', None)
        if community_model and isinstance(community_model, RiverModel):
            return community_model.to_dict()
        return None # Return None if no community cards

    def get_community_cards_text(self) -> str:
        """Generate a text representation of the community cards (river)."""
        community_model = getattr(self.scene_model, '_resolved_community_cards', None)
        text = "COMMUNITY CARDS (RIVER)\n=======================\n"
        if not community_model or not isinstance(community_model, RiverModel):
            text += "None\n"
            return text

        text += f"Number of Cards: {community_model.n_cards}\n"
        card_names_str = ", ".join(community_model.card_names) if community_model.card_names else "None"
        text += f"Cards: {card_names_str}\n"
        text += f"Start Location: {self._format_vector(community_model.start_location)}\n"
        text += f"Scale: {self._format_vector(community_model.scale)}\n"
        gap = community_model.card_gap
        text += f"Card Gap: BaseX={self._format_vector(gap.get('base_gap_x'))}, BaseY={self._format_vector(gap.get('base_gap_y'))}, Random={gap.get('random_gap')}, Rand%={self._format_vector(gap.get('random_percentage'))}\n"
        text += f"Random Seed: {community_model.random_seed}\n"
        return text

    # --- Players ---
    def _get_hand_dict(self, hand_model: PlayerHandModel) -> dict[str, Any]:
        """Helper to get dict for a PlayerHandModel."""
        return hand_model.to_dict()

    def _get_hand_text(self, hand_model: PlayerHandModel) -> str:
        """Helper to get text for a PlayerHandModel."""
        text =  f"    Hand Location: {self._format_vector(hand_model.location)}\n"
        text += f"    Hand Card Count: {hand_model.n_cards}\n"
        card_names_str = ", ".join(hand_model.card_names) if hand_model.card_names else "None"
        text += f"    Hand Cards: {card_names_str}\n"
        text += f"    Hand Scale: {self._format_vector(hand_model.scale)}\n"
        text += f"    Hand Spread (H/V): {self._format_vector(hand_model.spread_factor_h)} / {self._format_vector(hand_model.spread_factor_v)}\n"
        text += f"    Hand Random Seed: {hand_model.random_seed}\n"
        # Add other hand details if needed
        return text

    def _get_chip_area_dict(self, chip_area_model: ChipAreaConfig | None, resolved_piles: list[dict[str, Any]] | None) -> dict[str, Any] | None:
        """Helper to get dict for a ChipAreaConfig and its resolved piles."""
        if not chip_area_model or not isinstance(chip_area_model, ChipAreaConfig):
            return None

        area_dict = chip_area_model.to_dict()
        # Add the resolved pile specifications (which contain counts, colors, etc.)
        area_dict['resolved_piles'] = resolved_piles or []
        return area_dict

    def _get_chip_area_text(self, chip_area_model: ChipAreaConfig | None, resolved_piles: list[dict[str, Any]] | None) -> str:
        """Helper to get text for a ChipAreaConfig and its resolved piles."""
        if not chip_area_model or not isinstance(chip_area_model, ChipAreaConfig):
            return "    Chip Area: None\n"

        text =  "    Chip Area:\n"
        text += f"      Number of Piles (Resolved): {len(resolved_piles) if resolved_piles else 0}\n"
        text += f"      Offset from Cards: {self._format_vector(chip_area_model.placement_offset_from_cards)}\n"
        text += f"      Pile Gap (H/Rand): {self._format_vector(chip_area_model.pile_gap_h)} / {self._format_vector(chip_area_model.pile_gap_random_factor)}\n"
        text += f"      Layout Random Seed: {chip_area_model.random_seed}\n"

        if resolved_piles:
            text += f"      Resolved Piles ({len(resolved_piles)} total):\n"
            # Show details for first few piles
            for i, pile_spec in enumerate(resolved_piles[:3]):
                n_chips = pile_spec.get('n_chips', 'Default')
                color = self._format_vector(pile_spec.get('color', 'Default'))
                scale = self._format_vector(pile_spec.get('scale', 'Default'))
                text += f"        - Pile {i+1}: n_chips={n_chips}, color={color}, scale={scale}\n"
            if len(resolved_piles) > 3:
                text += f"        - ... ({len(resolved_piles) - 3} more piles)\n"
        else:
            text += "      Resolved Piles: None\n"

        # Add base pile config details if needed
        # base_pile = chip_area_model.base_pile_config
        # if isinstance(base_pile, ChipPileModel):
        #    text += f"      Base Pile Config: n_chips={base_pile.n_chips}, spread={base_pile.spread_factor}\n"

        return text

    def get_player_dict(self, player_model: PlayerModel) -> dict[str, Any]:
        """Format a single player's resolved info into a dictionary."""
        hand_dict = self._get_hand_dict(player_model.hand_config)
        chip_area_dict = self._get_chip_area_dict(
            player_model.chip_area_config,
            getattr(player_model, '_resolved_pile_configs', None) # Get resolved piles
        )

        return {
            'player_id': player_model.player_id,
            'hand_config': hand_dict,
            'chip_area_config': chip_area_dict
        }

    def get_player_text(self, player_model: PlayerModel) -> str:
        """Generate a text representation for a single player."""
        text = f"Player: {player_model.player_id}\n"
        text += "------" + "-"*len(player_model.player_id) + "\n"
        text += self._get_hand_text(player_model.hand_config)
        text += self._get_chip_area_text(
            player_model.chip_area_config,
            getattr(player_model, '_resolved_pile_configs', None)
        )
        return text

    def get_players_dict(self) -> list[dict[str, Any]]:
        """Get a list of dictionaries, one for each resolved player."""
        resolved_players = getattr(self.scene_model, '_resolved_players', [])
        return [self.get_player_dict(p) for p in resolved_players]

    def get_players_text(self) -> str:
        """Generate text legend for all resolved players."""
        resolved_players = getattr(self.scene_model, '_resolved_players', [])
        text = f"PLAYERS ({len(resolved_players)})\n=========\n"
        if not resolved_players:
            text += "None\n"
            return text

        for player_model in resolved_players:
            text += self.get_player_text(player_model) + "\n"
        return text

    # --- Noise Config ---
    def get_noise_config_dict(self) -> dict[str, Any] | None:
        """Extract and format noise configuration info into a dictionary."""
        return self.scene_model.noise_config

    def get_noise_config_text(self) -> str:
        """Generate a text representation of the noise configuration."""
        noise_config = self.scene_model.noise_config
        text = "NOISE CONFIGURATION\n===================\n"
        if not noise_config:
            text += "None\n"
            return text

        for key, value in noise_config.items():
            text += f"{key}: {value}\n"

        return text

    # --- Full Legend ---
    def get_full_legend_dict(self) -> dict[str, Any]:
        """Get a complete legend dictionary with all resolved information."""
        legend_dict = {
            "scene_setup": self.get_scene_setup_dict(),
            "community_cards": self.get_community_cards_dict(),
            "players": self.get_players_dict()
        }

        # Add card overlap information if present
        card_overlap_dict = self.get_card_overlap_dict()
        if card_overlap_dict:
            legend_dict["card_overlap_layout"] = card_overlap_dict

        # Add noise configuration if present
        noise_config_dict = self.get_noise_config_dict()
        if noise_config_dict:
            legend_dict["noise_config"] = noise_config_dict

        return legend_dict

    def get_full_legend_text(self) -> str:
        """Get a complete text legend with all resolved information."""
        text = self.get_scene_setup_text() + "\n"
        # Add card overlap information if present
        if self.scene_model.card_overlap_config:
            text += self.get_card_overlap_text() + "\n"
        text += self.get_community_cards_text() + "\n"
        text += self.get_players_text() + "\n"
        # Add noise configuration if present
        if self.scene_model.noise_config:
            text += self.get_noise_config_text()
        return text

# --- Top-Level Function ---

def generate_poker_legend(
    scene_model: PokerSceneModel,
    output_dir: str,
    base_filename: str
) -> tuple[str | None, str | None]:
    """
    Generates text and JSON legend files from a resolved PokerSceneModel,
    saving them into 'legend_txt' and 'legend_json' subdirectories.

    Args:
        scene_model: The resolved PokerSceneModel instance.
        output_dir: The base directory where 'legend_txt' and 'legend_json'
                    subdirectories will be created.
        base_filename: The base filename for the output files (e.g., 'poker_scene_001').

    Returns:
        A tuple containing the paths to the generated files: (txt_path, json_path).
        Returns (None, None) if an error occurs.
    """
    base_output_path = Path(output_dir)
    # MODIFIED: Define subdirectories
    txt_output_dir_path = base_output_path / "legend_txt"
    json_output_dir_path = base_output_path / "legend_json"

    # Construct full paths within subdirectories
    txt_path = txt_output_dir_path / f"{base_filename}_legend.txt"
    json_path = json_output_dir_path / f"{base_filename}_legend.json"

    try:
        # Ensure subdirectories exist
        txt_output_dir_path.mkdir(parents=True, exist_ok=True)
        json_output_dir_path.mkdir(parents=True, exist_ok=True)

        legend_generator = PokerConfigToLegend(scene_model)

        # Generate and save text legend
        txt_legend = legend_generator.get_full_legend_text()
        with open(txt_path, 'w') as f_txt:
            f_txt.write(txt_legend)
        logger.info(f"Generated text legend: {txt_path}")

        # Generate and save JSON legend
        json_legend = legend_generator.get_full_legend_dict()
        with open(json_path, 'w') as f_json:
            json.dump(json_legend, f_json, indent=2, default=str)
        logger.info(f"Generated JSON legend: {json_path}")

        return str(txt_path), str(json_path)

    except Exception as e:
        logger.error(f"Error generating poker legend for {base_filename}: {e}", exc_info=True)
        if txt_path.exists(): txt_path.unlink(missing_ok=True) # Use missing_ok=True
        if json_path.exists(): json_path.unlink(missing_ok=True)
        return None, None

# --- Example Usage (if run directly) ---
if __name__ == "__main__":
    # Setup basic logging for the test
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logger.info("--- Running Poker Legend Generator Test ---")

    # Create a dummy PokerSceneModel instance (or load from a test YAML)
    # This requires a valid config dict structure
    test_config_dict = {
        "n_players": 2,
        "random_seed": 42,
        "scene_setup": {
            "camera": {"distance": 3.8, "angle": 50},
            "table": {"shape": "circular", "diameter": 2.0, "height": 0.9},
            "render": {"samples": 16}
        },
        # Provide some initial player config for PokerSceneModel to resolve
        "default_player_config": {
            "hand_config": {"scale": 0.1},
            "chip_area_config": {"n_piles": 1, "base_pile_config": {"n_chips": 5, "base_chip_config": {"chip_object_name": "Chip"}}}}
        ,
        # Provide card distribution inputs for PokerSceneModel to resolve
        "card_distribution_inputs": {
            "river_cards": 3,
            "player_cards": 2
        },
        "chip_distribution_inputs": {
            "overall_piles": 4,
            "n_chips_options": [5, 10, 15]
        }
    }

    try:
        # PokerSceneModel will resolve cards/chips in __post_init__
        test_scene_model = PokerSceneModel.from_dict(test_config_dict)

        # Define output for the test - now just the base dir
        test_base_output_dir = "poker/img/test_output"
        test_base_filename = "poker_legend_test"
        # No need to create subdirs here, function handles it

        # Generate legends using only the base dir
        txt_file, json_file = generate_poker_legend(
            scene_model=test_scene_model,
            output_dir=test_base_output_dir, # Just pass base dir
            base_filename=test_base_filename
        )

        if txt_file and json_file:
             logger.info("Successfully generated legends:")
             logger.info(f"  TXT: {txt_file}") # Will be inside legend_txt subdir
             logger.info(f"  JSON: {json_file}") # Will be inside legend_json subdir
        else:
             logger.error("Legend generation failed.")

    except Exception as e:
        logger.error(f"Error during test: {e}", exc_info=True)

    logger.info("--- Poker Legend Generator Test Finished ---")
