# poker/chip_dealer.py

import random
from typing import List, Dict, Optional, Any
import logging

# Ensure workspace root is in path for sibling imports
import os
import sys
workspace_root_dealer = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if workspace_root_dealer not in sys.path:
    sys.path.append(workspace_root_dealer)

# Use ChipDistributionInput for type hinting
from poker.config.models import ChipDistributionInput

logger = logging.getLogger(__name__)

def _distribute_randomly(total_items: int, num_bins: int, rng: random.Random) -> List[int]:
    """Distributes a total number of items randomly into bins using multinomial logic."""
    if num_bins <= 0:
        return []
    if total_items <= 0:
        return [0] * num_bins
        
    # Simple approach: Assign one by one randomly
    counts = [0] * num_bins
    for _ in range(total_items):
        chosen_bin = rng.randrange(num_bins)
        counts[chosen_bin] += 1
    return counts


def deal_chip_piles(
    chip_distribution_inputs: Optional[Dict[str, Any]], # Raw dict like {"overall_piles": 10}
    num_players: int,
    random_seed: Optional[int] = None
) -> List[int]:
    """
    Resolves chip *pile* counts for each player based on distribution inputs.

    Args:
        chip_distribution_inputs: A dictionary containing distribution parameters like
                                  'overall_piles', 'piles_per_player', 'min_piles_per_player'.
                                  Corresponds to ChipDistributionInput structure.
        num_players: The number of player hands to deal for.
        random_seed: Optional random seed for distribution randomness.

    Returns:
        A list of integers, where each integer represents the number of chip *piles*
        allocated to the corresponding player. Returns a list of zeros if
        resolution fails or no inputs are provided.

    Raises:
        ValueError: If inputs are invalid.
    """
    # --- Resolve Chip Pile Counts from Inputs ---
    if chip_distribution_inputs is None or num_players <= 0:
        logger.info("No chip distribution inputs or no players. Returning zero pile counts.")
        return [0] * num_players

    try:
        inputs = ChipDistributionInput(**chip_distribution_inputs)
        inputs.n_players = num_players # Set contextually
        # Basic validation on min_piles_per_player
        if inputs.min_piles_per_player is not None and (not isinstance(inputs.min_piles_per_player, int) or inputs.min_piles_per_player < 0):
            logger.warning(f"Invalid min_piles_per_player ({inputs.min_piles_per_player}), ignoring it.")
            inputs.min_piles_per_player = None
    except TypeError as e:
        raise ValueError(f"Invalid structure in chip_distribution_inputs: {e}")

    player_pile_counts = [0] * num_players
    rng = random.Random(random_seed) if random_seed is not None else random.Random()

    # == Primary Logic Branch: overall_piles specified ==
    if isinstance(inputs.overall_piles, int) and inputs.overall_piles >= 0:
        overall_piles = inputs.overall_piles
        logger.debug(f"Resolving pile counts using overall_piles = {overall_piles}")

        min_player_piles = inputs.min_piles_per_player or 0
        max_player_piles = inputs.max_piles_per_player # Can be None

        # Validate max_piles_per_player
        if max_player_piles is not None and (not isinstance(max_player_piles, int) or max_player_piles < 0):
            logger.warning(f"Invalid max_piles_per_player ({max_player_piles}), ignoring it.")
            max_player_piles = None
        
        # Ensure max is not less than min if both are specified
        if max_player_piles is not None and min_player_piles > max_player_piles:
             logger.warning(f"min_piles_per_player ({min_player_piles}) > max_piles_per_player ({max_player_piles}). Ignoring max constraint.")
             max_player_piles = None
        
        logger.debug(f"Constraints: min_piles={min_player_piles}, max_piles={max_player_piles}")

        # --- Distribution with min and max constraints ---
        player_pile_counts = [min_player_piles] * num_players
        piles_assigned_so_far = sum(player_pile_counts)
        
        if piles_assigned_so_far > overall_piles:
            logger.warning(f"Minimum player piles total ({piles_assigned_so_far}) exceeds overall_piles ({overall_piles}). Distributing overall piles randomly, ignoring minimum and maximum.")
            # Fallback to simple random distribution if minimums already exceed total
            player_pile_counts = _distribute_randomly(overall_piles, num_players, rng)
        else:
            remaining_piles_to_assign = overall_piles - piles_assigned_so_far
            logger.debug(f"Assigning minimums first ({piles_assigned_so_far} piles). Distributing remaining {remaining_piles_to_assign} piles iteratively with max constraint.")
            
            for _ in range(remaining_piles_to_assign):
                # Find players eligible to receive another pile (below max, if max is set)
                eligible_player_indices = [
                    idx for idx, count in enumerate(player_pile_counts)
                    if max_player_piles is None or count < max_player_piles
                ]

                if not eligible_player_indices:
                    logger.warning(f"Could not assign all {overall_piles} piles. Stopped assignment with {remaining_piles_to_assign - _} piles remaining because all players reached the max limit ({max_player_piles}).")
                    break # Stop assigning if no player can receive more

                # Choose a random eligible player and give them a pile
                chosen_player_index = rng.choice(eligible_player_indices)
                player_pile_counts[chosen_player_index] += 1
            
            piles_actually_assigned = sum(player_pile_counts)
            if piles_actually_assigned != overall_piles and not eligible_player_indices: # Check if loop broke early
                logger.info(f"Final assigned piles ({piles_actually_assigned}) differs from requested overall_piles ({overall_piles}) due to max_piles_per_player constraint.")

        # Warn if explicit player spec was ignored because overall_piles was dominant
        if inputs.piles_per_player is not None: logger.warning("'overall_piles' is set, ignoring 'piles_per_player'.")

    # == Fallback Branch: overall_piles NOT specified ==
    else:
        if inputs.overall_piles is not None: logger.warning(f"Invalid overall_piles value '{inputs.overall_piles}', ignoring it.")
        logger.debug("Resolving counts using specific player pile inputs.")

        # Use explicit piles_per_player if provided (list or int)
        if isinstance(inputs.piles_per_player, list):
            if len(inputs.piles_per_player) == num_players:
                valid_counts = [c if isinstance(c, int) and c >= 0 else 0 for c in inputs.piles_per_player]
                if any(c == 0 and oc != 0 for c, oc in zip(valid_counts, inputs.piles_per_player)): logger.warning(f"Invalid counts in piles_per_player list {inputs.piles_per_player}, invalid set to 0.")
                player_pile_counts = valid_counts
            else:
                logger.warning(f"piles_per_player list length ({len(inputs.piles_per_player)}) != num_players ({num_players}). Ignoring list.")
                # Apply minimum if available as fallback
                if inputs.min_piles_per_player is not None:
                     logger.debug(f"Applying min_piles_per_player ({inputs.min_piles_per_player}) as fallback.")
                     player_pile_counts = [inputs.min_piles_per_player] * num_players
        elif isinstance(inputs.piles_per_player, int) and inputs.piles_per_player >= 0:
            logger.debug(f"Applying uniform piles_per_player ({inputs.piles_per_player})")
            player_pile_counts = [inputs.piles_per_player] * num_players
        # Apply minimum if specified AND no explicit player spec given/valid
        elif inputs.min_piles_per_player is not None:
            logger.debug(f"Applying min_piles_per_player ({inputs.min_piles_per_player}) as no explicit piles_per_player given.")
            player_pile_counts = [inputs.min_piles_per_player] * num_players
        elif inputs.piles_per_player is not None:
            logger.warning(f"Invalid piles_per_player value '{inputs.piles_per_player}', ignoring.")

    # Final cleanup
    player_pile_counts = [max(0, c) for c in player_pile_counts]
    logger.info(f"Resolved chip pile counts per player: {player_pile_counts}")

    return player_pile_counts 