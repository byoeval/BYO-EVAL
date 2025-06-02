import random
from typing import List, Dict, Union

# --- Card Constants ---
RANKS: List[str] = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
SUITS: List[str] = ['S', 'D', 'H', 'C'] # Spades, Diamonds, Hearts, Clubs
HEADS: List[str] = ['A', 'K', 'Q', 'J', '10'] # Ace, King, Queen, Jack, 10
NUMBERS: List[str] = ['2', '3', '4', '5', '6', '7', '8', '9']

# --- Default Deck ---
DEFAULT_CARD_NAMES: List[str] = [f"{rank}{suit}" for suit in SUITS for rank in RANKS]

# --- Card Presets ---
CARD_PRESETS: Dict[str, List[str]] = {
    "default": DEFAULT_CARD_NAMES,
    "red": [card for card in DEFAULT_CARD_NAMES if card.endswith('D') or card.endswith('H')],
    "black": [card for card in DEFAULT_CARD_NAMES if card.endswith('S') or card.endswith('C')],
    "spades": [card for card in DEFAULT_CARD_NAMES if card.endswith('S')],
    "diamonds": [card for card in DEFAULT_CARD_NAMES if card.endswith('D')],
    "hearts": [card for card in DEFAULT_CARD_NAMES if card.endswith('H')],
    "clubs": [card for card in DEFAULT_CARD_NAMES if card.endswith('C')],
    "heads": [card for card in DEFAULT_CARD_NAMES if any(card.startswith(h) for h in HEADS)],
    "numbers": [card for card in DEFAULT_CARD_NAMES if any(card.startswith(n) for n in NUMBERS)],
    # Add more presets as needed
}

def get_card_pool(config_value: Union[str, int, List[str]]) -> List[str]:
    """
    Determines the pool of available cards based on the configuration value.

    Args:
        config_value: The value from the 'poker.card_pool_config' variable.
                      - int: Sample N unique cards randomly from the default deck.
                      - str: Use a predefined preset name (e.g., 'red', 'heads').
                      - List[str]: Use the provided list directly as the pool.

    Returns:
        A list of card names representing the available pool.

    Raises:
        ValueError: If the preset name is unknown or N is invalid for sampling.
        TypeError: If the config_value type is unsupported.
    """
    if isinstance(config_value, int):
        n = config_value
        if not (0 < n <= len(DEFAULT_CARD_NAMES)):
            raise ValueError(f"Cannot sample {n} unique cards from a deck of {len(DEFAULT_CARD_NAMES)}.")
        return random.sample(DEFAULT_CARD_NAMES, k=n)
    elif isinstance(config_value, str):
        preset_name = config_value.lower()
        if preset_name not in CARD_PRESETS:
            raise ValueError(f"Unknown card preset: '{preset_name}'. Available: {list(CARD_PRESETS.keys())}.")
        return CARD_PRESETS[preset_name]
    elif isinstance(config_value, list):
        # Validate if it's a list of strings (basic check)
        if not all(isinstance(item, str) for item in config_value):
            raise TypeError("If 'config_value' is a list, it must contain only strings (card names).")
        # Check if cards are valid? Optional, could be slow. For now, accept the list.
        return config_value
    else:
        raise TypeError(f"Unsupported type for card pool configuration: {type(config_value)}.")

def sample_cards_from_pool(pool: List[str], n_cards: int) -> List[str]:
    """
    Samples cards from a given pool with replacement.

    Args:
        pool: The list of card names to sample from.
        n_cards: The number of cards to sample.

    Returns:
        A list of sampled card names. Returns an empty list if pool is empty or n_cards is 0.
    """
    if not pool or n_cards <= 0:
        return []
    return random.choices(pool, k=n_cards)

# Example Usage:
# pool1 = get_card_pool(10) # Get 10 random unique cards
# pool2 = get_card_pool("red") # Get all red cards
# pool3 = get_card_pool(["AS", "KD", "QH"]) # Use a specific list

# hand = sample_cards_from_pool(pool2, 5) # Sample 5 cards (with replacement) from red cards 