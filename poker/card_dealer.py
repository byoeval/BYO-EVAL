# poker/card_dealer.py

import logging

# Ensure workspace root is in path for sibling imports
# (This might be needed if this module is run standalone or imported differently)
import os
import random
import sys
from typing import Any

workspace_root_dealer = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if workspace_root_dealer not in sys.path:
    sys.path.append(workspace_root_dealer)

# Use CardDistributionInput to type hint the raw config dict
from poker.config.models import DEFAULT_CARD_NAMES, CardDistributionInput

logger = logging.getLogger(__name__)

def _distribute_randomly(num_items: int, num_bins: int, rng: random.Random) -> list[int]:
    """Distributes num_items randomly into num_bins, ensuring non-negative counts."""
    if num_bins <= 0:
        if num_items > 0:
             logger.warning(f"Cannot distribute {num_items} items into {num_bins} bins.")
        return []
    if num_items < 0:
        logger.warning(f"Cannot distribute negative items ({num_items}). Returning zeros.")
        return [0] * num_bins

    # Generate num_bins - 1 random split points between 0 and num_items (inclusive)
    split_points = sorted([rng.randint(0, num_items) for _ in range(num_bins - 1)])

    counts = []
    last_point = 0
    for point in split_points:
        counts.append(point - last_point)
        last_point = point
    counts.append(num_items - last_point) # Add the last segment

    return counts

def deal_cards(
    distribution_inputs: dict[str, Any] | None, # Raw dict like {"river_cards": 3, "player_cards": 2}
    num_players: int, # The actual number of players being generated/used
    deck: list[str] = DEFAULT_CARD_NAMES,
    random_seed: int | None = None,
    allow_repetition: bool = True # NEW flag
) -> dict[str, list[list[str]]]: # Changed return type slightly
    """
    Resolves card counts from inputs, shuffles a deck (if no repetition),
    and deals cards (sequentially or with replacement). Enforces maximum counts.

    Args:
        distribution_inputs: A dictionary containing distribution parameters like
                             'river_cards', 'player_cards', 'overall_cards',
                             'max_player_cards', 'max_river_cards', etc.
                             Corresponds to CardDistributionInput structure.
        num_players: The number of player hands to deal for.
        deck: The list of card names to use as the source deck.
        random_seed: Optional random seed for shuffling or random choice.
        allow_repetition: If True, deal by randomly choosing from the deck with
                          replacement. If False, shuffle the deck once and deal
                          sequentially without replacement.

    Returns:
        A dictionary containing:
        {
            'river': List[str],     # List of card names for the river
            'players': List[List[str]] # List of lists, where each inner list contains
                                     # card names for a player.
        }
        Returns empty lists if dealing fails (e.g., not enough cards).

    Raises:
        ValueError: If the total number of cards required exceeds the deck size
        when repetition is not allowed. Or if inputs are invalid.
    """
    # --- Resolve Card Counts from Inputs ---
    if distribution_inputs is None:
        logger.warning("No card distribution inputs provided. Returning empty deal.")
        return {'river': [], 'players': [[] for _ in range(num_players)]}

    # Use CardDistributionInput for easier access and potential future validation
    try:
        inputs = CardDistributionInput(**distribution_inputs)
        # Set the actual number of players based on the scene context
        inputs.n_players = num_players
        # Basic validation on min_cards_per_player
        if inputs.min_cards_per_player is not None and (not isinstance(inputs.min_cards_per_player, int) or inputs.min_cards_per_player < 0):
            logger.warning(f"Invalid min_cards_per_player ({inputs.min_cards_per_player}), ignoring it.")
            inputs.min_cards_per_player = None
        # Basic validation on max counts
        if inputs.max_player_cards is not None and (not isinstance(inputs.max_player_cards, int) or inputs.max_player_cards < 0):
            logger.warning(f"Invalid max_player_cards ({inputs.max_player_cards}), ignoring it.")
            inputs.max_player_cards = None # Or set to a very high number? Using None disables capping.
        if inputs.max_river_cards is not None and (not isinstance(inputs.max_river_cards, int) or inputs.max_river_cards < 0):
            logger.warning(f"Invalid max_river_cards ({inputs.max_river_cards}), ignoring it.")
            inputs.max_river_cards = None
    except TypeError as e:
        raise ValueError(f"Invalid structure in card_distribution_inputs: {e}")

    # Get max limits (use None if not set, indicating no limit)
    max_p_cards = inputs.max_player_cards
    max_r_cards = inputs.max_river_cards

    # Logic moved from CardDistributionModel.from_inputs
    river_count = 0
    player_counts = [0] * num_players

    # Initialize RNG for potential random distribution
    rng = random.Random(random_seed) if random_seed is not None else random.Random()

    # == Primary Logic Branch: overall_cards specified ==
    if isinstance(inputs.overall_cards, int) and inputs.overall_cards >= 0:
        overall_cards = inputs.overall_cards
        logger.debug(f"Attempting card count resolution using overall_cards = {overall_cards}")

        # Cap min_player_cards immediately by max_player_cards
        min_player_cards = inputs.min_cards_per_player or 0
        if max_p_cards is not None and min_player_cards > max_p_cards:
            logger.warning(f"Requested min_cards_per_player ({min_player_cards}) exceeds max_player_cards ({max_p_cards}). Capping minimum at {max_p_cards}.")
            min_player_cards = max_p_cards
        min_player_total = num_players * min_player_cards

        # --- Case: overall + river specified --- (Takes precedence over min_player_cards)
        if isinstance(inputs.river_cards, int) and inputs.river_cards >= 0:
            logger.debug(f"Handling 'overall_cards' + 'river_cards' ({inputs.river_cards})")
            requested_river = inputs.river_cards

            # Apply max river card limit
            if max_r_cards is not None and requested_river > max_r_cards:
                logger.warning(f"Requested river_cards ({requested_river}) exceeds max_river_cards ({max_r_cards}). Capping at {max_r_cards}.")
                river_count = max_r_cards
            else:
                river_count = requested_river

            if river_count > overall_cards:
                logger.warning(f"Requested/Capped river_cards ({river_count}) exceeds overall_cards ({overall_cards}). Setting river count to 0.")
                river_count = 0

            remaining_for_players = overall_cards - river_count
            if remaining_for_players < 0: remaining_for_players = 0

            if num_players > 0:
                logger.debug(f"Distributing {remaining_for_players} cards randomly among {num_players} players.")
                player_counts = _distribute_randomly(remaining_for_players, num_players, rng)
            elif remaining_for_players > 0:
                logger.warning(f"Have {remaining_for_players} cards left, but num_players is 0.")

            # Warn if other player specs were ignored
            if inputs.player_cards is not None: logger.warning("'overall_cards' and 'river_cards' are set, ignoring 'player_cards'.")
            if inputs.min_cards_per_player is not None: logger.warning("'overall_cards' and 'river_cards' are set, ignoring 'min_cards_per_player'.")

        # --- Case: overall + min_cards_per_player specified ---
        elif inputs.min_cards_per_player is not None:
            logger.debug(f"Handling 'overall_cards' + 'min_cards_per_player' ({min_player_cards})")
            if min_player_total > overall_cards:
                logger.warning(f"Minimum player cards total ({min_player_total}, possibly capped by max_player_cards) exceeds overall_cards ({overall_cards}). Attempting minimum only.")
                river_count = 0
                player_counts = [min_player_cards] * num_players # Already capped by max_p_cards
            else:
                player_counts = [min_player_cards] * num_players # Start with minimum (already capped)
                remaining_overall = overall_cards - min_player_total

                # Randomly assign remaining cards between river and extra player cards
                # Ensure river doesn't exceed max_river_cards
                potential_river = rng.randint(0, remaining_overall)
                if max_r_cards is not None and potential_river > max_r_cards:
                    logger.debug(f"Random river assignment ({potential_river}) capped by max_river_cards ({max_r_cards}).")
                    river_count = max_r_cards
                else:
                    river_count = potential_river

                extra_player_cards = remaining_overall - river_count # Cards remaining after river allocation
                logger.debug(f"Assigned {river_count} to river, {extra_player_cards} extra for players.")

                if num_players > 0 and extra_player_cards > 0:
                    extra_distribution = _distribute_randomly(extra_player_cards, num_players, rng)
                    for i in range(num_players):
                        player_counts[i] += extra_distribution[i]

            # Warn if other player specs were ignored
            if inputs.player_cards is not None: logger.warning("'overall_cards' and 'min_cards_per_player' are set, ignoring 'player_cards'.")
            if inputs.river_cards is not None: logger.warning("'overall_cards' and 'min_cards_per_player' are set, ignoring 'river_cards'.")

        # --- Case: overall only specified ---
        else:
            logger.debug("Handling 'overall_cards' only.")
            # Distribute overall_cards randomly between river and players
            potential_river = rng.randint(0, overall_cards)
            # Cap river count
            if max_r_cards is not None and potential_river > max_r_cards:
                logger.debug(f"Random river assignment ({potential_river}) capped by max_river_cards ({max_r_cards}).")
                river_count = max_r_cards
            else:
                river_count = potential_river

            remaining_for_players = overall_cards - river_count
            if remaining_for_players < 0: remaining_for_players = 0 # Should not happen if logic is correct

            if num_players > 0:
                logger.debug(f"Distributing {remaining_for_players} cards randomly among {num_players} players (river got {river_count}).")
                player_counts = _distribute_randomly(remaining_for_players, num_players, rng)
            elif remaining_for_players > 0:
                logger.warning(f"Have {remaining_for_players} cards left, but num_players is 0.")

            # Warn if other specs were ignored
            if inputs.player_cards is not None: logger.warning("'overall_cards' is set, ignoring 'player_cards'.")
            if inputs.river_cards is not None: logger.warning("'overall_cards' is set, ignoring 'river_cards'.")
            if inputs.min_cards_per_player is not None: logger.warning("'overall_cards' is set, ignoring 'min_cards_per_player'.")

    # == Fallback Branch: overall_cards NOT specified ==
    else:
        if inputs.overall_cards is not None: logger.warning(f"Invalid overall_cards value '{inputs.overall_cards}', ignoring it.")
        logger.debug("Resolving counts using river/player specific inputs.")
        # Use explicit river_cards if provided, respecting max_river_cards
        requested_river = inputs.river_cards if isinstance(inputs.river_cards, int) and inputs.river_cards >= 0 else 0
        if inputs.river_cards is not None and requested_river == 0 and inputs.river_cards != 0:
            logger.warning(f"Invalid river_cards value '{inputs.river_cards}', resolving to 0.")

        if max_r_cards is not None and requested_river > max_r_cards:
            logger.warning(f"Requested river_cards ({requested_river}) exceeds max_river_cards ({max_r_cards}). Capping at {max_r_cards}.")
            river_count = max_r_cards
        else:
            river_count = requested_river

        # Use explicit player_cards if provided (list or int), respecting max_player_cards
        if isinstance(inputs.player_cards, list):
            if len(inputs.player_cards) == num_players:
                valid_counts = []
                for i, c in enumerate(inputs.player_cards):
                    original_c = c
                    if not isinstance(c, int) or c < 0:
                        logger.warning(f"Invalid count ({c}) in player_cards list at index {i}, setting to 0.")
                        c = 0
                    # Apply max player card limit
                    if max_p_cards is not None and c > max_p_cards:
                        logger.warning(f"Player card count ({c}) at index {i} exceeds max_player_cards ({max_p_cards}). Capping at {max_p_cards}.")
                        c = max_p_cards
                    valid_counts.append(c)
                player_counts = valid_counts
            else:
                logger.warning(f"player_cards list length ({len(inputs.player_cards)}) != num_players ({num_players}). Ignoring list.")
        elif isinstance(inputs.player_cards, int) and inputs.player_cards >= 0:
            requested_p_cards = inputs.player_cards
            # Apply max player card limit
            if max_p_cards is not None and requested_p_cards > max_p_cards:
                 logger.warning(f"Requested player_cards ({requested_p_cards}) exceeds max_player_cards ({max_p_cards}). Capping at {max_p_cards}.")
                 capped_p_cards = max_p_cards
            else:
                 capped_p_cards = requested_p_cards
            player_counts = [capped_p_cards] * num_players
        elif inputs.player_cards is not None: logger.warning(f"Invalid player_cards value '{inputs.player_cards}', ignoring.")
        # Apply minimum if specified AND no explicit player_cards were given
        elif inputs.min_cards_per_player is not None:
             min_p_cards = inputs.min_cards_per_player # Already validated >= 0
             # Apply max player card limit to the minimum
             if max_p_cards is not None and min_p_cards > max_p_cards:
                 logger.warning(f"Requested min_cards_per_player ({min_p_cards}) exceeds max_player_cards ({max_p_cards}). Capping minimum at {max_p_cards}.")
                 capped_min = max_p_cards
             else:
                 capped_min = min_p_cards
             logger.debug(f"Applying capped min_cards_per_player ({capped_min}) as no explicit player_cards given.")
             player_counts = [capped_min] * num_players

    # --- Final Count Capping & Validation ---
    river_count = max(0, river_count) # Ensure non-negative

    # Final cap on player counts (important for cases where distribution happened)
    capped_player_counts = []
    for i, p_count in enumerate(player_counts):
        final_p_count = max(0, p_count) # Ensure non-negative
        if max_p_cards is not None and final_p_count > max_p_cards:
            logger.warning(f"Final calculated count for player {i} ({final_p_count}) exceeds max_player_cards ({max_p_cards}). Capping.")
            final_p_count = max_p_cards
        capped_player_counts.append(final_p_count)
    player_counts = capped_player_counts # Replace with capped counts

    logger.debug(f"Final resolved card counts (after potential capping): river={river_count}, players={player_counts}")
    # --- End Count Resolution ---

    total_cards_needed = river_count + sum(player_counts)

    # Only check deck size and shuffle if repetition is NOT allowed
    shuffled_deck = None
    if not allow_repetition:
        if total_cards_needed > len(deck):
            raise ValueError(
                f"Repetition not allowed, and not enough cards in the deck ({len(deck)}) to deal "
                f"{total_cards_needed} cards (river={river_count}, players={player_counts})"
            )
        # Create a copy of the deck to shuffle
        shuffled_deck = deck[:]
        logger.debug(f"Shuffling deck (repetition=False) with seed: {random_seed}")
        random.Random(random_seed).shuffle(shuffled_deck)
    # else: If repetition allowed, we don't need to shuffle or check size

    # Initialize RNG for card choices if repetition is allowed
    rng_choice = random.Random(random_seed)

    # Deal cards
    dealt_cards: dict[str, list[list[str]]] = {
        'river': [],
        'players': [[] for _ in range(num_players)]
    }

    card_index = 0 # Only used if allow_repetition is False

    # Deal river cards
    print("DEALING RIVER CARDS WITH: ", river_count)
    if river_count > 0:
        if allow_repetition:
            dealt_cards['river'] = [rng_choice.choice(deck) for _ in range(river_count)]
        elif shuffled_deck: # Should always be true if not allow_repetition and count > 0
            dealt_cards['river'] = shuffled_deck[card_index : card_index + river_count]
            card_index += river_count
        logger.debug(f"Dealt {river_count} cards to river: {dealt_cards['river']}")

    print("DEALT CARDS TO RIVER: ", dealt_cards['river'])
    print("\n---------")
    # Deal player cards
    for i in range(num_players):
        num_player_cards = player_counts[i]
        if num_player_cards > 0:
            if allow_repetition:
                 dealt_cards['players'][i] = [rng_choice.choice(deck) for _ in range(num_player_cards)]
            elif shuffled_deck:
                 dealt_cards['players'][i] = shuffled_deck[card_index : card_index + num_player_cards]
                 card_index += num_player_cards
            logger.debug(f"Dealt {num_player_cards} cards to player {i}: {dealt_cards['players'][i]}")
        else:
            logger.debug(f"Player {i} receives 0 cards.")

    total_dealt = river_count + sum(p_count for p_count in player_counts)
    logger.info(f"Finished dealing cards. Total dealt/assigned: {total_dealt}")

    print("OVERALL DEALT CARDS: ", dealt_cards)
    return dealt_cards
