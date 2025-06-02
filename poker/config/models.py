import logging  # Add logging import if not already present
import math

# Ensure workspace root is in path for sibling imports
# (Assuming this logic might be needed if models.py is run directly or imported elsewhere)
import os
import random
import sys
import traceback  # Added import
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

workspace_root_models = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')) # Go up two levels
if workspace_root_models not in sys.path:
    sys.path.append(workspace_root_models)

# Import the newly created function
# from poker.player_location_builder import calculate_player_locations, PlayerDistribution

# Need copy for deepcopying default player config
import copy

# For card dealing
# from poker.card_dealer import deal_cards # Removed top-level import

logger = logging.getLogger(__name__)

# default card names:
DEFAULT_CARD_NAMES: list[str] = ["AS", "2S", "3S", "4S", "5S", "6S", "7S", "8S", "9S", "10S", "JS", "QS", "KS", "AD", "2D", "3D", "4D", "5D", "6D", "7D", "8D", "9D", "10D", "JD", "QD", "KD", "AH", "2H", "3H", "4H", "5H", "6H", "7H", "8H", "9H", "10H", "JH", "QH", "KH", "AC", "2C", "3C", "4C", "5C", "6C", "7C", "8C", "9C", "10C", "JC", "QC", "KC"]

# --- Constants for Layout Calculations ---
# Measured card dimensions from poker/utils/measure_card.py
DEFAULT_CARD_WIDTH: float = 1.0  # Base card width in Blender units
DEFAULT_CARD_HEIGHT: float = 1.4  # Base card height in Blender units

# Factors to determine gap based on card width for player hands
MIN_CENTER_GAP_FACTOR_X: float = 0.3  # Min gap between centers = 30% of card width (slight overlap)
MAX_CENTER_GAP_FACTOR_X: float = 1.5  # Max gap between centers = 150% of card width (clear space)

# Vertical gap factors (for stacked cards)
MIN_CENTER_GAP_FACTOR_Y: float = 0.1  # Min vertical gap = 10% of card height
MAX_CENTER_GAP_FACTOR_Y: float = 0.5  # Max vertical gap = 50% of card height

# Default card orientation
DEFAULT_Z_ROTATION: float = math.pi / 2  # 90 degrees, vertical cards
# ----------------------------------------

@dataclass
class CardModel:
    """Configuration for loading and placing a single poker card."""
    card_name: str  # e.g., "AS", "KC", "10H"
    location: tuple[float, float, float] = (0.0, 0.0, 0.91)
    scale: float | tuple[float, float, float] = 1.0
    rotation_euler: tuple[float, float, float] | None = None  # Optional rotation in radians (X, Y, Z)
    face_up: bool = True # True for recto (face up), False for verso (face down)

    def to_dict(self) -> dict[str, Any]:
        """Convert the CardModel instance to a dictionary."""
        return {
            'card_name': self.card_name,
            'location': self.location,
            'scale': self.scale,
            'rotation_euler': self.rotation_euler,
            'face_up': self.face_up,
        }

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> 'CardModel':
        """
        Create a CardModel instance from a dictionary.

        Args:
            config: A dictionary containing card configuration parameters.

        Returns:
            A CardModel instance.

        Raises:
            ValueError: If 'card_name' is missing in the config dictionary.
        """
        if not isinstance(config, dict):
            raise TypeError("config must be a dictionary")
        if 'card_name' not in config:
            raise ValueError("Missing required key 'card_name' in card configuration")

        # Provide defaults for optional fields if not present in the config
        default_instance = cls(card_name=config['card_name']) # Create a temporary instance to access defaults

        return cls(
            card_name=config['card_name'],
            location=config.get('location', default_instance.location),
            scale=config.get('scale', default_instance.scale),
            rotation_euler=config.get('rotation_euler', default_instance.rotation_euler),
            face_up=config.get('face_up', default_instance.face_up),
        )



@dataclass
class RiverModel:
    """Configuration for creating a horizontal layout of multiple cards."""
    n_cards: int # Moved before fields with defaults
    card_names: list[str] = field(default_factory=lambda: DEFAULT_CARD_NAMES) # Default value
    start_location: tuple[float, float, float] = (0.0, 0.0, 0.9) # Base location for the first card (leftmost)
    n_verso: int = 0 # Number of cards to place face down (verso)
    verso_loc: str = 'ordered' # How to place face-down cards ('ordered' or 'random')
    scale: float | tuple[float, float, float] = 0.1 # Uniform scale for all cards
    card_gap: dict[str, Any] = field(default_factory=lambda: {
        "base_gap_x": 0.15,         # Horizontal distance between card centers
        "base_gap_y": 0.005,        # Only used for random y-jitter magnitude
        "random_gap": False,        # Apply randomness to vertical position (y-jitter)
        "random_percentage": 0.2    # Max +/- percentage variation for y-jitter
    })
    random_seed: int | None = None # Optional seed for reproducibility of random elements

    def __post_init__(self):
        """Validate configuration after initialization."""
        if len(self.card_names) != self.n_cards:
            raise ValueError(f"Length of 'card_names' ({len(self.card_names)}) must equal 'n_cards' ({self.n_cards})")
        if self.n_verso > self.n_cards:
            raise ValueError(f"'n_verso' ({self.n_verso}) cannot be greater than 'n_cards' ({self.n_cards})")
        if self.n_verso < 0:
             raise ValueError(f"'n_verso' ({self.n_verso}) cannot be negative")
        if self.verso_loc not in ['ordered', 'random']:
            raise ValueError(f"'verso_loc' must be 'ordered' or 'random', got '{self.verso_loc}'")
        if 'base_gap_x' not in self.card_gap:
            raise ValueError("'card_gap' dictionary must contain 'base_gap_x'")
        if 'base_gap_y' not in self.card_gap:
            raise ValueError("'card_gap' dictionary must contain 'base_gap_y'")

    def to_dict(self) -> dict[str, Any]:
        """Convert the RiverModel instance to a dictionary."""
        return {
            'card_names': self.card_names,
            'n_cards': self.n_cards,
            'start_location': self.start_location,
            'n_verso': self.n_verso,
            'verso_loc': self.verso_loc,
            'scale': self.scale,
            'card_gap': self.card_gap,
            'random_seed': self.random_seed,
        }

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> 'RiverModel':
        """
        Create a RiverModel instance from a dictionary.
        [Docstring updated as needed]
        """
        print("config: ", config)
        if not isinstance(config, dict):
            raise TypeError("config must be a dictionary")
        if 'card_names' not in config:
            raise ValueError("Missing required key 'card_names' in river configuration")
        if 'n_cards' not in config:
            raise ValueError("Missing required key 'n_cards' in river configuration")

        temp_instance_for_defaults = cls(card_names=config['card_names'], n_cards=config['n_cards'])

        gap_config = config.get('card_gap', {})
        if not isinstance(gap_config, dict):
            gap_config = {}

        default_gap = temp_instance_for_defaults.card_gap
        final_gap_config = default_gap.copy()
        final_gap_config.update(gap_config)

        instance = cls(
            card_names=config['card_names'],
            n_cards=config['n_cards'],
            start_location=config.get('start_location', temp_instance_for_defaults.start_location),
            n_verso=config.get('n_verso', temp_instance_for_defaults.n_verso),
            verso_loc=config.get('verso_loc', temp_instance_for_defaults.verso_loc),
            scale=config.get('scale', temp_instance_for_defaults.scale),
            card_gap=final_gap_config,
            random_seed=config.get('random_seed', temp_instance_for_defaults.random_seed),
        )
        return instance

@dataclass
class PlayerHandModel:
    """Configuration for creating a player's hand of cards."""
    n_cards: int # Moved before fields with defaults
    location: tuple[float, float, float] # Moved before fields with defaults
    card_names: list[str] = field(default_factory=lambda: DEFAULT_CARD_NAMES) # Default value
    scale: float | tuple[float, float, float] = 0.1 # Uniform scale for all cards in hand
    spread_factor_h: float = 0.5 # Controls horizontal spread interpolation (0=stacked, 1=max spread)
    spread_factor_v: float = 0.0 # Controls vertical spread interpolation (0=flat, 1=max vertical)
    first_card_std_dev: float = 0.01 # Gaussian std dev for placing the first card around location
    rotation_std_dev: float = 0.0 # Gaussian std dev for rotating cards around center-facing direction (in radians)
    n_verso: int = 0 # Number of cards to place face down
    verso_loc: str = 'ordered' # How to place face-down cards ('ordered' or 'random')
    randomize_gap_h: bool = False # If True, randomize horizontal gap from 0 to calculated max
    randomize_gap_v: bool = False # If True, randomize vertical gap from 0 to calculated max
    random_seed: int | None = None # Optional seed for random elements

    def __post_init__(self):
        """Validate configuration after initialization."""
        if len(self.card_names) != self.n_cards:
            raise ValueError(f"Length of 'card_names' ({len(self.card_names)}) must equal 'n_cards' ({self.n_cards})")
        if not (self.n_cards >= 0):
            raise ValueError(f"'n_cards' ({self.n_cards}) must be non-negative")
        if self.n_verso > self.n_cards:
            raise ValueError(f"'n_verso' ({self.n_verso}) cannot be greater than 'n_cards' ({self.n_cards})")
        if self.n_verso < 0:
             raise ValueError(f"'n_verso' ({self.n_verso}) cannot be negative")
        if self.verso_loc not in ['ordered', 'random']:
            raise ValueError(f"'verso_loc' must be 'ordered' or 'random', got '{self.verso_loc}'")
        if not (0.0 <= self.spread_factor_h <= 1.0):
            raise ValueError(f"'spread_factor_h' ({self.spread_factor_h}) must be between 0.0 and 1.0")
        if not (0.0 <= self.spread_factor_v <= 1.0):
            raise ValueError(f"'spread_factor_v' ({self.spread_factor_v}) must be between 0.0 and 1.0")
        if self.first_card_std_dev < 0:
            raise ValueError(f"'first_card_std_dev' ({self.first_card_std_dev}) cannot be negative")
        if self.rotation_std_dev < 0:
            raise ValueError(f"'rotation_std_dev' ({self.rotation_std_dev}) cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        """Convert the PlayerHandModel instance to a dictionary."""
        return {
            'card_names': self.card_names,
            'n_cards': self.n_cards,
            'location': self.location,
            'scale': self.scale,
            'spread_factor_h': self.spread_factor_h,
            'spread_factor_v': self.spread_factor_v,
            'first_card_std_dev': self.first_card_std_dev,
            'rotation_std_dev': self.rotation_std_dev,
            'n_verso': self.n_verso,
            'verso_loc': self.verso_loc,
            'randomize_gap_h': self.randomize_gap_h,
            'randomize_gap_v': self.randomize_gap_v,
            'random_seed': self.random_seed,
        }

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> 'PlayerHandModel':
        """
        Create a PlayerHandModel instance from a dictionary.

        Args:
            config: A dictionary containing player hand configuration parameters.

        Returns:
            A PlayerHandModel instance.

        Raises:
            ValueError: If required keys are missing or validation fails.
            TypeError: If config is not a dictionary.
        """
        if not isinstance(config, dict):
            raise TypeError("config must be a dictionary")
        required_keys = ['card_names', 'n_cards', 'location']
        for key in required_keys:
            if key not in config:
                 raise ValueError(f"Missing required key '{key}' in player hand configuration")

        default_instance = cls(card_names=config['card_names'], n_cards=config['n_cards'], location=config['location'])

        # Handle legacy 'spread_factor' for backward compatibility
        spread_factor_h = config.get('spread_factor_h', config.get('spread_factor', default_instance.spread_factor_h))
        spread_factor_v = config.get('spread_factor_v', default_instance.spread_factor_v)

        instance = cls(
            card_names=config['card_names'],
            n_cards=config['n_cards'],
            location=config['location'],
            scale=config.get('scale', default_instance.scale),
            spread_factor_h=spread_factor_h,
            spread_factor_v=spread_factor_v,
            first_card_std_dev=config.get('first_card_std_dev', default_instance.first_card_std_dev),
            rotation_std_dev=config.get('rotation_std_dev', default_instance.rotation_std_dev),
            n_verso=config.get('n_verso', default_instance.n_verso),
            verso_loc=config.get('verso_loc', default_instance.verso_loc),
            randomize_gap_h=config.get('randomize_gap_h', default_instance.randomize_gap_h),
            randomize_gap_v=config.get('randomize_gap_v', default_instance.randomize_gap_v),
            random_seed=config.get('random_seed', default_instance.random_seed),
        )
        return instance

@dataclass
class CardDistributionInput:
    """Raw inputs potentially influencing card distribution."""
    n_players: int | None = None
    river_cards: int | None = None
    # Can be a single int (apply to all players) or a list matching n_players
    player_cards: int | list[int] | None = None
    # Allow specifying player cards via the variable map directly
    player_cards_map: dict[str, int] | None = None # e.g., {'players.0.hand_config.n_cards': 2}
    overall_cards: int | None = None # NEW: Specify total cards directly
    min_cards_per_player: int | None = None # NEW: Minimum cards each player should get
    n_verso: int | None = 0 # NEW: Number of cards to be dealt face down (verso)
    max_player_cards: int | None = 8 # NEW: Maximum cards any single player can receive
    max_river_cards: int | None = 12 # NEW: Maximum cards the river can receive
    max_piles_per_player: int | None = None # NEW: Maximum *piles* any single player can get (used with overall_piles)

# --- NEW: CardTypeModel ---

class CardTypeMode(Enum):
    FULL_DECK = "full_deck"       # Use all 52 standard cards
    SUBSET_N = "subset_n"         # Randomly select N cards from the full deck
    EXPLICIT_LIST = "explicit_list" # Use a specific list of card names
    SUIT_ONLY = "suit_only"         # Use only cards from a specific suit

@dataclass
class CardTypeModel:
    """Defines the pool of cards available for dealing."""
    mode: CardTypeMode = CardTypeMode.FULL_DECK
    subset_n: int | None = None # Required if mode is SUBSET_N
    card_list: list[str] | None = None # Required if mode is EXPLICIT_LIST
    suit: str | None = None # Required if mode is SUIT_ONLY (e.g., "Spades", "Hearts")
    allow_repetition: bool = True # If True, sample with replacement; if False, sample without

    def __post_init__(self):
        # Validation
        if self.mode == CardTypeMode.SUBSET_N and (not isinstance(self.subset_n, int) or self.subset_n <= 0):
            raise ValueError(f"Mode '{self.mode.value}' requires a positive integer for 'subset_n'.")
        if self.mode == CardTypeMode.EXPLICIT_LIST and (not self.card_list):
            raise ValueError(f"Mode '{self.mode.value}' requires a non-empty 'card_list'.")
        if self.mode == CardTypeMode.SUIT_ONLY and (not self.suit or self.suit not in ["Spades", "Hearts", "Diamonds", "Clubs"]):
            raise ValueError(f"Mode '{self.mode.value}' requires a valid 'suit' (Spades, Hearts, Diamonds, Clubs).")
        if not self.allow_repetition and self.mode == CardTypeMode.FULL_DECK and self.subset_n is None and not self.card_list and not self.suit:
             # This combination implies standard dealing from a full deck without repetition
             pass # Valid
        # Could add checks if subset_n > 52, or if card_list contains invalid cards

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> 'CardTypeModel':
        if not isinstance(config, dict):
            raise TypeError("CardTypeModel config must be a dictionary.")

        mode_str = config.get("mode", CardTypeMode.FULL_DECK.value)
        try:
            mode_enum = CardTypeMode(mode_str)
        except ValueError as e:
            raise ValueError(f"Invalid CardTypeMode: {mode_str}. Must be one of {[m.value for m in CardTypeMode]}") from e

        # Basic type checks for conditional required fields
        subset_n = config.get('subset_n')
        card_list = config.get('card_list')
        suit = config.get('suit')

        if mode_enum == CardTypeMode.SUBSET_N and not isinstance(subset_n, int):
             logger.warning(f"Mode is {mode_enum.value} but subset_n is not an int: {subset_n}. Validation may fail.")
        if mode_enum == CardTypeMode.EXPLICIT_LIST and not isinstance(card_list, list):
             logger.warning(f"Mode is {mode_enum.value} but card_list is not a list: {card_list}. Validation may fail.")
        if mode_enum == CardTypeMode.SUIT_ONLY and not isinstance(suit, str):
             logger.warning(f"Mode is {mode_enum.value} but suit is not a string: {suit}. Validation may fail.")

        instance = cls(
            mode=mode_enum,
            subset_n=subset_n,
            card_list=card_list,
            suit=suit,
            allow_repetition=config.get('allow_repetition', True) # Default to True
        )
        # __post_init__ handles detailed validation
        return instance

# --- END CardTypeModel ---

@dataclass
class ChipDistributionInput:
    """Raw inputs potentially influencing chip *pile* distribution and properties."""
    n_players: int | None = None # Provided by the context
    overall_piles: int | None = None # Specify total *piles* for all players
    piles_per_player: int | list[int] | None = None # Specify *piles* per player (int or list)
    min_piles_per_player: int | None = None # Minimum *piles* each player should get
    max_piles_per_player: int | None = None # NEW: Maximum *piles* any single player can get (used with overall_piles)
    # --- NEW: Options for randomizing pile properties ---
    n_chips_options: list[int] | None = None # List of possible chip counts per pile
    color_options: list[tuple[float, float, float, float] | None] | None = None # List of possible RGBA colors (or None for default)
    scale_options: list[float | tuple[float, float, float]] | None = None # List of possible scales (float or tuple)
    # Note: random_seed from PokerSceneModel will be used for selection

@dataclass
class ChipModel:
    """Configuration for loading and placing a single poker chip."""
    chip_object_name: str  # e.g., "Cylinder001"
    location: tuple[float, float, float] = (0.0, 0.0, 0.91) # Default location
    scale: float | tuple[float, float, float] = 0.06 # Default scale
    color: tuple[float, float, float, float] | None = None # Optional RGBA color override
    blend_file_path: str | None = None # Optional path to chip blend file

    def to_dict(self) -> dict[str, Any]:
        """Convert the ChipModel instance to a dictionary."""
        return {
            'chip_object_name': self.chip_object_name,
            'location': self.location,
            'scale': self.scale,
            'color': self.color,
            'blend_file_path': self.blend_file_path,
        }

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> 'ChipModel':
        """
        Create a ChipModel instance from a dictionary.

        Args:
            config: A dictionary containing chip configuration parameters.

        Returns:
            A ChipModel instance.

        Raises:
            ValueError: If 'chip_object_name' is missing.
            TypeError: If config is not a dictionary.
        """
        if not isinstance(config, dict):
            raise TypeError("config must be a dictionary")
        if 'chip_object_name' not in config:
            raise ValueError("Missing required key 'chip_object_name' in chip configuration")

        default_instance = cls(chip_object_name=config['chip_object_name'])

        return cls(
            chip_object_name=config['chip_object_name'],
            location=config.get('location', default_instance.location),
            scale=config.get('scale', default_instance.scale),
            color=config.get('color', default_instance.color),
            blend_file_path=config.get('blend_file_path', default_instance.blend_file_path),
        )

@dataclass
class ChipPileModel:
    """Configuration for creating a pile (stack) of poker chips."""
    n_chips: int
    base_chip_config: dict[str, Any] | ChipModel # Config for chip type (location ignored)
    location: tuple[float, float, float] # Center of the pile's base (x, y), Z of the bottom chip
    spread_factor: float = 0.0 # Horizontal randomization (0.0 = perfectly stacked, 1.0 = max spread)
    vertical_gap: float = 0.002 # Small gap between chips in the stack
    random_seed: int | None = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.n_chips <= 0:
            raise ValueError(f"'n_chips' ({self.n_chips}) must be positive.")
        if not (0.0 <= self.spread_factor <= 1.0):
            raise ValueError(f"'spread_factor' ({self.spread_factor}) must be between 0.0 and 1.0")
        if self.vertical_gap < 0:
             raise ValueError(f"'vertical_gap' ({self.vertical_gap}) cannot be negative.")
        if not isinstance(self.base_chip_config, dict | ChipModel):
            raise TypeError("'base_chip_config' must be a dictionary or ChipModel instance.")

    def to_dict(self) -> dict[str, Any]:
        """Convert the ChipPileModel instance to a dictionary."""
        base_config_dict = self.base_chip_config
        if isinstance(base_config_dict, ChipModel):
            base_config_dict = base_config_dict.to_dict()

        return {
            'n_chips': self.n_chips,
            'base_chip_config': base_config_dict,
            'location': self.location,
            'spread_factor': self.spread_factor,
            'vertical_gap': self.vertical_gap,
            'random_seed': self.random_seed,
        }

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> 'ChipPileModel':
        """
        Create a ChipPileModel instance from a dictionary.
        """
        if not isinstance(config, dict):
            raise TypeError("config must be a dictionary")
        required_keys = ['n_chips', 'base_chip_config']
        for key in required_keys:
            if key not in config:
                 raise ValueError(f"Missing required key '{key}' in chip pile configuration")

        # Ensure base_chip_config is valid early
        base_config = config['base_chip_config']
        if isinstance(base_config, dict):
             try:
                 _ = ChipModel.from_dict(base_config)
             except (TypeError, ValueError) as e:
                  raise ValueError(f"Invalid 'base_chip_config': {e}") from e
        elif not isinstance(base_config, ChipModel):
             raise TypeError("'base_chip_config' must be a dict or ChipModel instance.")

        # Create instance using defaults for optional fields
        # Provide a dummy location for the temporary instance, it's not used for defaults
        temp_instance = cls(n_chips=config['n_chips'], base_chip_config=base_config, location=(0,0,0))

        instance = cls(
            n_chips=config['n_chips'],
            base_chip_config=base_config, # Store as provided (dict or model)
            location=config.get('location', (0.0, 0.0, 0.0)), # Provide default if missing
            spread_factor=config.get('spread_factor', temp_instance.spread_factor),
            vertical_gap=config.get('vertical_gap', temp_instance.vertical_gap),
            random_seed=config.get('random_seed', temp_instance.random_seed),
        )
        # __post_init__ runs validation
        return instance

@dataclass
class ChipAreaConfig:
    """Configuration specifically for a chip area containing one or more piles."""
    base_pile_config: dict[str, Any] | ChipPileModel # Default pile settings (location ignored)
    n_piles: int = 1 # Number of piles()
    # --- Optional overrides per pile (lists must match n_piles if provided) ---
    n_chips_per_pile: list[int] | None = None
    pile_colors: list[tuple[float, float, float, float] | None] | None = None
    pile_spreads: list[float | None] | None = None
    # --- Layout relative to cards ---
    pile_gap_h: float = 0.02 # Base horizontal gap between pile centers
    pile_gap_random_factor: float = 0.5 # Randomization factor for horizontal gap (0=none, 1=full range)
    pile_gap_v_range: float = 0.02 # Max +/- vertical offset from baseline
    placement_offset_from_cards: float = 0.3 # How far towards table center from hand center
    random_seed: int | None = None # Seed for chip pile layout randomization

    def __post_init__(self):
        if self.n_piles < 0:
            raise ValueError(f"'n_piles' ({self.n_piles}) cannot be negative.")
        # Validate list lengths if provided
        if self.n_chips_per_pile is not None and len(self.n_chips_per_pile) != self.n_piles:
             raise ValueError(f"Length of 'n_chips_per_pile' ({len(self.n_chips_per_pile)}) must match 'n_piles' ({self.n_piles})")
        if self.pile_colors is not None and len(self.pile_colors) != self.n_piles:
             raise ValueError(f"Length of 'pile_colors' ({len(self.pile_colors)}) must match 'n_piles' ({self.n_piles})")
        if self.pile_spreads is not None and len(self.pile_spreads) != self.n_piles:
             raise ValueError(f"Length of 'pile_spreads' ({len(self.pile_spreads)}) must match 'n_piles' ({self.n_piles})")
        if not (0.0 <= self.pile_gap_random_factor <= 1.0):
             raise ValueError("'pile_gap_random_factor' must be between 0.0 and 1.0")

    def to_dict(self) -> dict[str, Any]:
        """Convert the ChipAreaConfig instance to a dictionary."""
        base_pile_conf_dict = self.base_pile_config
        if isinstance(base_pile_conf_dict, ChipPileModel):
            base_pile_conf_dict = base_pile_conf_dict.to_dict()

        return {
            'base_pile_config': base_pile_conf_dict,
            'n_piles': self.n_piles,
            'n_chips_per_pile': self.n_chips_per_pile,
            'pile_colors': self.pile_colors,
            'pile_spreads': self.pile_spreads,
            'pile_gap_h': self.pile_gap_h,
            'pile_gap_random_factor': self.pile_gap_random_factor,
            'pile_gap_v_range': self.pile_gap_v_range,
            'placement_offset_from_cards': self.placement_offset_from_cards,
            'random_seed': self.random_seed,
        }

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> 'ChipAreaConfig':
        """
        Create a ChipAreaConfig instance from a dictionary.
        """
        if not isinstance(config, dict):
            raise TypeError("config must be a dictionary")
        if 'base_pile_config' not in config:
            raise ValueError("Missing required key 'base_pile_config' in chip area configuration")

        # Validate base_pile_config structure early
        base_pile_conf = config['base_pile_config']
        if isinstance(base_pile_conf, dict):
            try:
                _ = ChipPileModel.from_dict(base_pile_conf)
            except (TypeError, ValueError) as e:
                raise ValueError(f"Invalid 'base_pile_config': {e}") from e
        elif not isinstance(base_pile_conf, ChipPileModel):
            raise TypeError("'base_pile_config' must be a dict or ChipPileModel")

        # Create a temporary default instance to get default values for optional fields
        # This requires providing valid defaults for required fields of the temp instance
        try:
             temp_default_base_pile = ChipPileModel.from_dict({'n_chips': 1, 'base_chip_config': {'chip_object_name': 'temp'}, 'location': (0,0,0)})
        except Exception as e:
             # Fallback if even default parsing fails (shouldn't happen)
             raise RuntimeError(f"Failed to create temporary default base pile model: {e}") from e
        temp_instance = cls(base_pile_config=temp_default_base_pile)

        return cls(
            base_pile_config=base_pile_conf, # Keep as provided (dict or model)
            n_piles=config.get('n_piles', temp_instance.n_piles),
            n_chips_per_pile=config.get('n_chips_per_pile'),
            pile_colors=config.get('pile_colors'),
            pile_spreads=config.get('pile_spreads'),
            pile_gap_h=config.get('pile_gap_h', temp_instance.pile_gap_h),
            pile_gap_random_factor=config.get('pile_gap_random_factor', temp_instance.pile_gap_random_factor),
            pile_gap_v_range=config.get('pile_gap_v_range', temp_instance.pile_gap_v_range),
            placement_offset_from_cards=config.get('placement_offset_from_cards', temp_instance.placement_offset_from_cards),
            random_seed=config.get('random_seed'),
        )

@dataclass
class PlayerModel:
    """Configuration for a single player, including hand and chip piles."""
    player_id: str | int = "Player"
    hand_config: dict[str, Any] | PlayerHandModel = field(default_factory=dict)
    chip_area_config: dict[str, Any] | ChipAreaConfig | None = None
    # Internal field to store the resolved number of piles from distribution
    # _resolved_pile_count: Optional[int] = field(init=False, repr=False, default=None) # REMOVED
    # NEW: Store resolved configs for each pile assigned to this player
    _resolved_pile_configs: list[dict[str, Any]] | None = field(init=False, repr=False, default=None)

    def to_dict(self) -> dict[str, Any]:
        hand_conf_dict = self.hand_config
        if isinstance(hand_conf_dict, PlayerHandModel):
             hand_conf_dict = hand_conf_dict.to_dict()

        chip_area_conf_dict = self.chip_area_config
        if isinstance(chip_area_conf_dict, ChipAreaConfig):
            # We need to call its to_dict method if it has one
            # Assuming ChipAreaConfig will have a to_dict
             try:
                 # Use getattr to safely call to_dict if it exists
                 to_dict_method = getattr(chip_area_conf_dict, "to_dict", None)
                 if callable(to_dict_method):
                     chip_area_conf_dict = to_dict_method()
                 # else: It might already be a dict if passed directly
             except Exception as e:
                 print(f"Error converting chip_area_config to dict: {e}")
                 # Decide how to handle - pass original dict or raise?
                 # Pass the original dict for now

        return {
            'player_id': self.player_id,
            'hand_config': hand_conf_dict,
            'chip_area_config': chip_area_conf_dict,
            # 'resolved_pile_count': self._resolved_pile_count, # REMOVED
        }

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> 'PlayerModel':
        if not isinstance(config, dict):
            raise TypeError("config must be a dictionary")

        hand_config = config.get('hand_config', {})
        if isinstance(hand_config, dict):
             hand_model = PlayerHandModel.from_dict(hand_config)
        elif isinstance(hand_config, PlayerHandModel):
             hand_model = hand_config
        else:
            raise TypeError("'hand_config' must be a dict or PlayerHandModel")

        chip_area_conf = config.get('chip_area_config')
        chip_area_model = None
        if isinstance(chip_area_conf, dict):
            chip_area_model = ChipAreaConfig.from_dict(chip_area_conf)
        elif isinstance(chip_area_conf, ChipAreaConfig):
            chip_area_model = chip_area_conf
        elif chip_area_conf is not None:
             raise TypeError("'chip_area_config' must be a dict, ChipAreaConfig, or None")

        return cls(
            player_id=config.get('player_id', "Player"),
            hand_config=hand_model,
            chip_area_config=chip_area_model
        )



# --- NEW: General Card Layout Model ---

class LayoutMode(Enum):
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"

@dataclass
class CardOverlapModel:
    """Configuration for creating a general layout of cards with overlap control."""
    # --- Core Settings ---
    overall_cards: int # Total number of cards in the layout
    layout_mode: LayoutMode # 'horizontal' or 'vertical'
    card_type_config: dict[str, Any] # Config defining the card pool (See CardTypeModel)

    # --- Layout Dimensions (Conditional) ---
    n_lines: int | None = None # Required if layout_mode is HORIZONTAL
    n_columns: int | None = None # Required if layout_mode is VERTICAL

    # --- Placement & Appearance ---
    center_location: tuple[float, float, float] = (0.0, 0.0, 0.91) # Center of the layout block
    scale: float | tuple[float, float, float] = 0.1 # Uniform scale for all cards
    horizontal_overlap_factor: float = 0.5 # 0.0=touching edges, <1.0 = overlap. Used in HORIZONTAL mode.
    vertical_overlap_factor: float = 0.5   # 0.0=touching edges, <1.0 = overlap. Used in VERTICAL mode.
    line_gap: float = 0.02 # Vertical distance between line centers (HORIZONTAL mode)
    column_gap: float = 0.02 # Horizontal distance between column centers (VERTICAL mode)

    # --- Card Facing ---
    n_verso: int = 0 # Number of cards face down
    verso_loc: str = 'random' # 'ordered' or 'random'

    # --- Misc ---
    layout_id: str | None = None # Optional identifier
    random_seed: int | None = None # Seed for randomness

    def __post_init__(self):
        """Validate the configuration."""
        # --- DEBUG PRINT ---
        print("[CardOverlapModel __post_init__] Validating instance:")
        print(f"  layout_mode: {self.layout_mode}")
        print(f"  n_lines: {self.n_lines} (type: {type(self.n_lines)}) ")
        print(f"  n_columns: {self.n_columns} (type: {type(self.n_columns)}) ")
        # --- END DEBUG ---
        if self.overall_cards < 0:
            raise ValueError("'overall_cards' cannot be negative.")
        if self.layout_mode == LayoutMode.HORIZONTAL and (self.n_lines is None or not isinstance(self.n_lines, int) or self.n_lines <= 0):
            raise ValueError("'n_lines' must be a positive integer for HORIZONTAL layout mode.")
        if self.layout_mode == LayoutMode.VERTICAL and (self.n_columns is None or not isinstance(self.n_columns, int) or self.n_columns <= 0):
            raise ValueError("'n_columns' must be a positive integer for VERTICAL layout mode.")
        if not (0.0 <= self.horizontal_overlap_factor < 1.0):
            raise ValueError("'horizontal_overlap_factor' must be between 0.0 (inclusive) and 1.0 (exclusive).")
        if not (0.0 <= self.vertical_overlap_factor < 1.0):
            raise ValueError("'vertical_overlap_factor' must be between 0.0 (inclusive) and 1.0 (exclusive).")
        if self.n_verso < 0:
            raise ValueError("'n_verso' cannot be negative.")
        if self.n_verso > self.overall_cards:
             raise ValueError(f"'n_verso' ({self.n_verso}) cannot exceed 'overall_cards' ({self.overall_cards})")
        if self.verso_loc not in ['ordered', 'random']:
            raise ValueError("'verso_loc' must be 'ordered' or 'random'.")
        if not isinstance(self.card_type_config, dict):
             raise TypeError("'card_type_config' must be a dictionary.")
        try:
            _ = CardTypeModel.from_dict(self.card_type_config)
        except (ValueError, TypeError) as e:
             # --- DEBUG PRINT ---
             print("--- TRACEBACK for CardTypeModel parsing error ---")
             traceback.print_exc()
             print("-------------------------------------------------")
             # --- END DEBUG ---
             raise ValueError(f"Invalid 'card_type_config': {e}") from e

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> 'CardOverlapModel':
        """Create a CardOverlapModel instance from a dictionary."""
        if not isinstance(config, dict):
            raise TypeError("Config must be a dictionary.")

        # Validate required fields
        required = ['overall_cards', 'layout_mode', 'card_type_config']
        for key in required:
            if key not in config:
                 raise ValueError(f"Missing required key: '{key}'")

        # Convert layout_mode string to Enum
        try:
            layout_mode_enum = LayoutMode(config['layout_mode'])
        except ValueError as e:
            raise ValueError(f"Invalid layout_mode: '{config['layout_mode']}'. Must be one of {[m.value for m in LayoutMode]}") from e

        # Create a temporary instance to access defaults
        # Need valid values for required fields AND conditional fields for the temp instance
        # because __post_init__ validation runs during its creation.
        temp_instance = cls(
            overall_cards=config['overall_cards'],
            layout_mode=layout_mode_enum,
            card_type_config=config['card_type_config'] ,
            n_lines=config.get('n_lines'), # Pass potential n_lines
            n_columns=config.get('n_columns') # Pass potential n_columns
        )

        instance = cls(
            overall_cards=config['overall_cards'],
            layout_mode=layout_mode_enum,
            card_type_config=config['card_type_config'],
            n_lines=config.get('n_lines'), # Validation in post_init checks if needed
            n_columns=config.get('n_columns'), # Validation in post_init checks if needed
            center_location=config.get('center_location', temp_instance.center_location),
            scale=config.get('scale', temp_instance.scale),
            horizontal_overlap_factor=config.get('horizontal_overlap_factor', temp_instance.horizontal_overlap_factor),
            vertical_overlap_factor=config.get('vertical_overlap_factor', temp_instance.vertical_overlap_factor),
            line_gap=config.get('line_gap', temp_instance.line_gap),
            column_gap=config.get('column_gap', temp_instance.column_gap),
            n_verso=config.get('n_verso', temp_instance.n_verso),
            verso_loc=config.get('verso_loc', temp_instance.verso_loc),
            layout_id=config.get('layout_id'),
            random_seed=config.get('random_seed'),
        )
        # --- DEBUG PRINT ---
        print("[CardOverlapModel from_dict] Created instance with:")
        print(f"  n_lines = {instance.n_lines}")
        print(f"  n_columns = {instance.n_columns}")
        # --- END DEBUG ---
        # __post_init__ handles detailed validation based on layout_mode
        return instance

    def to_dict(self) -> dict[str, Any]:
        """Serializes the CardOverlapModel instance to a dictionary."""
        data = {
            'layout_id': self.layout_id,
            'layout_mode': self.layout_mode.value if isinstance(self.layout_mode, LayoutMode) else self.layout_mode, # Serialize Enum
            'overall_cards': self.overall_cards,
            'n_lines': self.n_lines,
            'n_columns': self.n_columns,
            # Ensure card_type_config is serialized if it exists and has a to_dict method
            'card_type_config': self.card_type_config.to_dict() if hasattr(self.card_type_config, 'to_dict') else self.card_type_config,
            'center_location': list(self.center_location) if isinstance(self.center_location, tuple) else self.center_location,
            'scale': list(self.scale) if isinstance(self.scale, tuple) else self.scale,
            'horizontal_overlap_factor': self.horizontal_overlap_factor,
            'vertical_overlap_factor': self.vertical_overlap_factor,
            'line_gap': self.line_gap,
            'column_gap': self.column_gap,
            'n_verso': self.n_verso,
            'verso_loc': self.verso_loc,
            'random_seed': self.random_seed
        }
        # Remove keys with None values if desired (optional, depends on desired output)
        # return {k: v for k, v in data.items() if v is not None}
        return data
# --- END General Card Layout Model ---


@dataclass
class PlayerDistribution:
    """Configuration for distributing players around the table."""
    layout: str = "circular" # e.g., "circular", "rectangular" (future)
    n_players: int | None = None # Explicit number of players (can override PokerSceneModel.n_players for distribution logic)
    spacing_degrees: float | None = None # Override automatic spacing with a fixed angle
    start_angle_degrees: float = 0.0 # Angle of the first player (0 = positive X-axis)
    trigonometric_direction: bool = True # True = CCW (standard), False = CW

    def __post_init__(self):
        """Validate distribution parameters."""
        if self.layout not in ["circular"]:
            raise ValueError(f"Unsupported player distribution layout: {self.layout}")
        if self.spacing_degrees is not None and not (0 < self.spacing_degrees <= 360):
             raise ValueError(f"spacing_degrees ({self.spacing_degrees}) must be between 0 (exclusive) and 360 (inclusive).")
        if self.n_players is not None and self.n_players < 0:
             raise ValueError(f"n_players ({self.n_players}) in distribution cannot be negative.")
        if self.n_players == 0 and self.spacing_degrees is None:
             logger.warning("PlayerDistribution has n_players=0 and no explicit spacing_degrees. No locations will be generated.")

    @classmethod
    def from_dict(cls, config: dict[str, Any] | None) -> 'PlayerDistribution':
        """Creates a PlayerDistribution instance from a dictionary."""
        if config is None:
            return cls() # Return default if no config provided
        if not isinstance(config, dict):
            raise TypeError("PlayerDistribution config must be a dictionary or None")

        instance = cls(
            layout=config.get('layout', 'circular'),
            n_players=config.get('n_players'), # Allow explicit n_players here
            spacing_degrees=config.get('spacing_degrees'),
            start_angle_degrees=config.get('start_angle_degrees', 0.0),
            trigonometric_direction=config.get('trigonometric_direction', True)
        )
        # Validation happens in __post_init__
        return instance

    def to_dict(self) -> dict[str, Any]:
        """Convert the PlayerDistribution instance to a dictionary."""
        return {
            'layout': self.layout,
            'n_players': self.n_players,
            'spacing_degrees': self.spacing_degrees,
            'start_angle_degrees': self.start_angle_degrees,
            'trigonometric_direction': self.trigonometric_direction,
        }

@dataclass
class PokerSceneModel:
    """Configuration for an entire poker scene setup."""
    n_players: int # Expected number of players (for validation/info)

    # Explicit player list (takes precedence if provided)
    players: list[dict[str, Any] | PlayerModel] = field(default_factory=list)

    community_cards: dict[str, Any] | RiverModel | None = None # Config for board cards
    scene_setup: dict[str, Any] = field(default_factory=dict) # Config for general_setup
    noise_config: dict[str, Any] | None = None  # Optional noise effects configuration
    deck_blend_file: str | None = None # Optional override for deck file
    random_seed: int | None = None # Optional top-level random seed
    card_type_config: dict[str, Any] | None = None # NEW: Configuration for deck restrictions

    # --- Fields for automatic generation (used if 'players' is empty) ---
    default_player_config: dict[str, Any] | PlayerModel | None = None
    player_distribution_config: dict[str, Any] | None = field(default_factory=lambda: {"layout": "circular"})
    card_distribution_inputs: dict[str, Any] | None = None # Placeholder
    chip_distribution_inputs: dict[str, Any] | None = None # NEW: Inputs for chip pile distribution

    # --- Internal state ---
    # Stores the final list of players, either explicitly provided or generated.
    _resolved_players: list[PlayerModel] = field(init=False, repr=False, default_factory=list)
    # Stores the validated community cards model
    _resolved_community_cards: RiverModel | None = field(init=False, repr=False, default=None)

    # New field for the overlap builder
    card_overlap_config: CardOverlapModel | None = None

    def __post_init__(self):
        """Validate configuration and resolve players and community cards."""
        # Import dealer locally inside method to avoid circular import at module level
        from poker.card_dealer import deal_cards

        # Import chip dealer locally
        from poker.chip_dealer import deal_chip_piles

        if self.n_players < 0:
            raise ValueError(f"'n_players' ({self.n_players}) cannot be negative.")

        # --- Step 1: Parse explicit players or generate defaults ---
        explicit_players = []
        if self.players:
            try:
                explicit_players = [
                    PlayerModel.from_dict(p_conf) if isinstance(p_conf, dict) else p_conf
                    for p_conf in self.players
                    if isinstance(p_conf, dict | PlayerModel)
                ]
                if len(explicit_players) != len(self.players):
                    logger.warning("Some entries in the 'players' list were invalid and ignored.")
            except (ValueError, TypeError) as e:
                 logger.error(f"Error parsing explicit 'players' list: {e}. Player list may be incomplete.")
                 explicit_players = []

        if explicit_players:
            logger.debug(f"Using explicitly provided player list ({len(explicit_players)} players).")
            self._resolved_players = explicit_players
            if len(explicit_players) > self.n_players:
                logger.warning(f"Number of players provided ({len(explicit_players)}) exceeds n_players ({self.n_players}).")
        elif self.default_player_config is not None and self.n_players > 0:
             logger.debug(f"Attempting to automatically generate {self.n_players} players from default config.")
             self._generate_players_from_defaults()
        else:
             logger.debug("No explicit players provided and no default config for generation (or n_players is 0). Resolved player list will be empty.")
             self._resolved_players = []

        num_resolved_players = len(self._resolved_players)

        # --- Step 2: Temporarily parse community_cards config from YAML (if exists) ---
        yaml_community_config_dict = None
        if isinstance(self.community_cards, dict):
            yaml_community_config_dict = self.community_cards # Keep as dict for now
        elif isinstance(self.community_cards, RiverModel): # If passed as model initially
             yaml_community_config_dict = self.community_cards.to_dict()
        elif self.community_cards is not None:
             logger.warning(f"Ignoring invalid community_cards type: {type(self.community_cards)}")

        # --- Step 2.5: Construct Source Deck based on CardTypeModel ---
        source_deck = list(DEFAULT_CARD_NAMES) # Start with full deck by default
        allow_repetition = True # Default
        card_type_model = None
        if self.card_type_config is not None:
            try:
                card_type_model = CardTypeModel.from_dict(self.card_type_config)
                logger.info(f"Applying Card Type Config: mode={card_type_model.mode.value}, repetition={card_type_model.allow_repetition}")
                allow_repetition = card_type_model.allow_repetition
                rng_deck = random.Random(self.random_seed) # Use scene seed for deck construction if needed

                if card_type_model.mode == CardTypeMode.SUBSET_N:
                    n = card_type_model.subset_n
                    if n is not None and 0 < n <= len(source_deck):
                         source_deck = rng_deck.sample(source_deck, n)
                         logger.debug(f"Using SUBSET_N deck ({n} cards): {source_deck}")
                    else:
                         logger.warning(f"Invalid subset_n ({n}) for deck size {len(source_deck)}. Using full deck.")
                         # Fallback to full deck, repetition determined by allow_repetition flag
                elif card_type_model.mode == CardTypeMode.EXPLICIT_LIST:
                    if card_type_model.card_list:
                         # Validate cards? For now, just use the list.
                         source_deck = list(card_type_model.card_list)
                         logger.debug(f"Using EXPLICIT_LIST deck: {source_deck}")
                    else:
                         logger.warning("Mode is EXPLICIT_LIST but card_list is empty. Using full deck.")
                         # Fallback to full deck
                elif card_type_model.mode == CardTypeMode.SUIT_ONLY:
                    target_suit_char = card_type_model.suit[0].upper() if card_type_model.suit else ''
                    if target_suit_char in ['S', 'H', 'D', 'C']:
                         source_deck = [card for card in source_deck if card.endswith(target_suit_char)]
                         logger.debug(f"Using SUIT_ONLY deck ({card_type_model.suit}): {source_deck}")
                    else:
                         logger.warning(f"Invalid suit '{card_type_model.suit}'. Using full deck.")
                         # Fallback to full deck
                # FULL_DECK mode uses the default source_deck
            except (ValueError, TypeError) as e:
                 logger.error(f"Error parsing card_type_config: {e}. Using default full deck.")
                 source_deck = list(DEFAULT_CARD_NAMES)
                 allow_repetition = True # Reset to default on error
        else:
            logger.debug("No card_type_config provided. Using standard full deck.")
            # source_deck and allow_repetition already at defaults
        # --- End Deck Construction ---

        # --- Step 3: Deal Cards (if requested) ---
        dealt_cards_dict = None
        dealing_occurred = False
        if self.card_distribution_inputs is not None:
            dealing_occurred = True
            if not isinstance(self.card_distribution_inputs, dict):
                logger.error(f"Invalid type for card_distribution_inputs: {type(self.card_distribution_inputs)}. Expected dict. Skipping card dealing.")
                dealing_occurred = False # Override flag
            else:
                logger.info("Card distribution inputs provided. Attempting to deal cards...")
                try:
                    print("DEALING CARDS WITH: ", self.card_distribution_inputs)
                    dealt_cards_dict = deal_cards(
                        distribution_inputs=self.card_distribution_inputs,
                        num_players=num_resolved_players,
                        deck=source_deck, # Pass the constructed deck
                        random_seed=self.random_seed,
                        allow_repetition=allow_repetition # Pass the flag
                    )
                except (ValueError, TypeError) as e:
                     logger.error(f"Error during card dealing: {e}. Cards might not be assigned correctly.")
                     logger.error(traceback.format_exc())
                     dealt_cards_dict = None # Ensure it's None on error
                     dealing_occurred = False
                except Exception as e:
                     logger.error(f"Unexpected error during card dealing: {e}")
                     logger.error(traceback.format_exc())
                     dealt_cards_dict = None
                     dealing_occurred = False
        else:
            logger.debug("No card_distribution_inputs provided.")

        # --- Step 4: Handle River Cards from Distribution ---
        if dealing_occurred and dealt_cards_dict is not None:
            self._handle_river_cards_from_distribution(dealt_cards_dict)

        # --- Step 5: Assign Dealt Cards to Players (if dealing occurred) ---
        if dealing_occurred and dealt_cards_dict is not None:
            dealt_player_hands = dealt_cards_dict.get('players', [])
            if len(dealt_player_hands) == num_resolved_players:
                for i, player_model in enumerate(self._resolved_players):
                    dealt_hand = dealt_player_hands[i]
                    num_dealt_player = len(dealt_hand)

                    if player_model.hand_config.n_cards != num_dealt_player:
                         logger.debug(f"Updating n_cards for {player_model.player_id} from {player_model.hand_config.n_cards} to dealt count {num_dealt_player}")
                         player_model.hand_config.n_cards = num_dealt_player

                    player_model.hand_config.card_names = dealt_hand
                    logger.debug(f"Assigned dealt cards to {player_model.player_id}: {player_model.hand_config.card_names}")

            elif num_resolved_players > 0:
                 logger.error(f"Number of player hands dealt ({len(dealt_player_hands)}) does not match number of resolved players ({num_resolved_players}). Player cards not assigned.")
        elif not explicit_players: # No dealing and players were generated
             logger.warning("Card dealing did not occur, generated players will have placeholder cards ('XX') unless overridden elsewhere.")

        # --- Step 6: Resolve Chip Pile Counts & Properties (if inputs provided) ---
        if self.chip_distribution_inputs is not None:
            if not isinstance(self.chip_distribution_inputs, dict):
                logger.error(f"Invalid type for chip_distribution_inputs: {type(self.chip_distribution_inputs)}. Expected dict. Skipping chip pile dealing.")
            elif num_resolved_players > 0:
                logger.info("Chip distribution inputs provided. Attempting to resolve chip pile counts and properties...")
                try:
                    # Initialize RNG using scene seed if available
                    rng = random.Random(self.random_seed) if self.random_seed is not None else random.Random()

                    # a) Get the number of piles per player
                    player_pile_counts = deal_chip_piles(
                        chip_distribution_inputs=self.chip_distribution_inputs,
                        num_players=num_resolved_players,
                        random_seed=self.random_seed # Pass seed for count distribution
                    )

                    # b) Parse property options from inputs
                    dist_inputs = ChipDistributionInput(**self.chip_distribution_inputs) # Parse for options
                    n_chips_opts = dist_inputs.n_chips_options or []
                    color_opts = dist_inputs.color_options or []
                    scale_opts = dist_inputs.scale_options or []

                    if len(player_pile_counts) == num_resolved_players:
                        for i, player_model in enumerate(self._resolved_players):
                            pile_count = player_pile_counts[i]
                            resolved_configs_for_player: list[dict[str, Any]] = []

                            if pile_count > 0:
                                # c) Generate pile specs for this player
                                for _ in range(pile_count):
                                    pile_spec = {}
                                    if n_chips_opts:
                                        pile_spec['n_chips'] = rng.choice(n_chips_opts)
                                    if color_opts:
                                        pile_spec['color'] = rng.choice(color_opts)
                                    if scale_opts:
                                        pile_spec['scale'] = rng.choice(scale_opts)
                                    resolved_configs_for_player.append(pile_spec)

                                logger.debug(f"Assigned {pile_count} resolved pile configs to {player_model.player_id}: {resolved_configs_for_player}")
                            else:
                                logger.debug(f"{player_model.player_id} assigned 0 piles.")

                            # d) Store the list of resolved pile configs on the player model
                            player_model._resolved_pile_configs = resolved_configs_for_player
                    else:
                         logger.error(f"Number of pile counts returned ({len(player_pile_counts)}) does not match resolved players ({num_resolved_players}). Chip piles not assigned.")

                except (ValueError, TypeError) as e:
                     logger.error(f"Error during chip pile dealing: {e}. Chip piles might not be assigned correctly.")
                     logger.error(traceback.format_exc())
                except Exception as e:
                     logger.error(f"Unexpected error during chip pile dealing: {e}")
                     logger.error(traceback.format_exc())
            else: # chip_distribution_inputs provided, but num_resolved_players is 0
                logger.warning("Chip distribution inputs provided, but there are no resolved players to assign piles to.")
        else:
            logger.debug("No chip_distribution_inputs provided.")

        # --- Step 7: Resolve Card Overlap Config (if config exists)
        if self.card_overlap_config and not isinstance(self.card_overlap_config, CardOverlapModel):
            if isinstance(self.card_overlap_config, dict):
                try:
                    self.card_overlap_config = CardOverlapModel.from_dict(self.card_overlap_config)
                except Exception as e:
                    logger.error(f"Failed to parse card_overlap_config: {self.card_overlap_config}. Error: {e}", exc_info=True)
                    self.card_overlap_config = None # Set to None on failure
            else:
                logger.error(f"Invalid type for card_overlap_config: {type(self.card_overlap_config)}")
                self.card_overlap_config = None

        logger.debug("PokerSceneModel validation complete.")

        # 8. Ensure `players` is a list of PlayerModel instances
        if self.players:
            resolved_players = []
            for i, p_conf in enumerate(self.players):
                if isinstance(p_conf, PlayerModel):
                    resolved_players.append(p_conf)
                elif isinstance(p_conf, dict):
                    # If player_id is missing, generate one
                    if 'player_id' not in p_conf:
                        p_conf['player_id'] = f"Player_{i+1}"
                        logger.debug(f"Generated player_id '{p_conf['player_id']}' for player at index {i}.")
                    try:
                        resolved_players.append(PlayerModel.from_dict(p_conf))
                    except Exception as e:
                        logger.error(f"Failed to parse player config at index {i}: {p_conf}. Error: {e}", exc_info=True)
                        # Decide whether to raise or continue; continuing for robustness
                else:
                    logger.error(f"Invalid type for player config at index {i}: {type(p_conf)}")
            self.players = resolved_players # Update self.players with resolved models
            self._resolved_players = resolved_players # Also store in internal field
        else:
            self._resolved_players = []

        # 9. Resolve Community Cards (if config exists)
        if self.community_cards:
            if isinstance(self.community_cards, dict):
                try:
                    self._resolved_community_cards = RiverModel.from_dict(self.community_cards)
                except Exception as e:
                    logger.error(f"Failed to parse community_cards config: {self.community_cards}. Error: {e}", exc_info=True)
                    self._resolved_community_cards = None # Set to None on failure
            elif isinstance(self.community_cards, RiverModel):
                self._resolved_community_cards = self.community_cards
            else:
                logger.error(f"Invalid type for community_cards: {type(self.community_cards)}")
                self._resolved_community_cards = None
        else:
            self._resolved_community_cards = None

        # 10. Resolve Card Overlap Config (if config exists)
        # No complex resolution needed here, just type checking
        if self.card_overlap_config and not isinstance(self.card_overlap_config, CardOverlapModel):
            if isinstance(self.card_overlap_config, dict):
                try:
                    self.card_overlap_config = CardOverlapModel.from_dict(self.card_overlap_config)
                except Exception as e:
                    logger.error(f"Failed to parse card_overlap_config: {self.card_overlap_config}. Error: {e}", exc_info=True)
                    self.card_overlap_config = None # Set to None on failure
            else:
                logger.error(f"Invalid type for card_overlap_config: {type(self.card_overlap_config)}")
                self.card_overlap_config = None

    def _generate_players_from_defaults(self):
        """Generates the player list based on default_player_config and distribution settings. Populates self._resolved_players."""
        # Import locally to avoid circular dependency at module load time
        from poker.player_location_builder import calculate_player_locations

        if self.default_player_config is None:
             logger.error("Cannot generate players: default_player_config is None.")
             self._resolved_players = []
             return
        if self.n_players <= 0:
            logger.info("n_players is 0, skipping automatic player generation.")
            self._resolved_players = []
            return

        try:
            # 1. Keep Default Player Config as Dict For partial parsing
            if isinstance(self.default_player_config, PlayerModel):
                 # Convert back to dict if it was passed as a model instance
                 default_player_dict = self.default_player_config.to_dict()
            elif isinstance(self.default_player_config, dict):
                 default_player_dict = copy.deepcopy(self.default_player_config) # Use copy
            else:
                 raise TypeError("default_player_config must be a dict or PlayerModel")

            # Extract default hand config dict - DO NOT validate fully yet
            default_hand_config_dict = default_player_dict.get('hand_config', {})
            if not isinstance(default_hand_config_dict, dict):
                 logger.warning("Invalid 'hand_config' in default_player_config, using empty default.")
                 default_hand_config_dict = {}

            # Extract default chip config dict (if exists)
            default_chip_config_dict = default_player_dict.get('chip_area_config')
            if default_chip_config_dict is not None and not isinstance(default_chip_config_dict, dict):
                 logger.warning("Invalid 'chip_area_config' in default_player_config, ignoring.")
                 default_chip_config_dict = None

            # 2. Parse Player Distribution Config (remains the same)
            player_dist = PlayerDistribution.from_dict(self.player_distribution_config)

            # 3. Get Table Dimensions
            table_conf = self.scene_setup.get('table', {})
            table_height = table_conf.get('height', 0.9)

            # 4. Calculate Player Locations using table_conf directly
            player_locations = calculate_player_locations(
                distribution_config=player_dist,
                num_players_to_place=self.n_players,
                table_conf=table_conf, # Pass the conf dictionary
                table_height=table_height
            )

            if len(player_locations) != self.n_players:
                logger.error(f"Calculated locations ({len(player_locations)}) does not match n_players ({self.n_players}). Aborting generation.")
                self._resolved_players = []
                return

            # --- Player Generation Loop ---
            generated_players = []
            # Placeholder card counts/names - TODO: Replace with CardDistribution logic
            default_n_cards = default_hand_config_dict.get('n_cards', 0) # Get default N cards
            if not isinstance(default_n_cards, int) or default_n_cards < 0:
                 logger.warning(f"Invalid 'n_cards' in default hand config: {default_n_cards}. Defaulting to 0.")
                 default_n_cards = 0

            # We still need placeholder card names list of the correct length
            # for PlayerHandModel validation during initial creation here.
            # These names will be overwritten in __post_init__ if dealing occurs.
            placeholder_card_names = ["XX"] * default_n_cards

            for i in range(self.n_players):
                player_id = f"Player_{i+1}"
                loc_x, loc_y, loc_z = player_locations[i]
                # Use the default number of cards for initial model creation
                current_n_cards = default_n_cards

                # Create the Hand Config *within the loop* with overrides
                hand_config_for_player = default_hand_config_dict.copy()
                hand_config_for_player['location'] = (loc_x, loc_y, loc_z)
                hand_config_for_player['n_cards'] = current_n_cards
                # Provide placeholder names to satisfy validation for now
                hand_config_for_player['card_names'] = placeholder_card_names # List of correct length

                try:
                    # Now validate and create the PlayerHandModel instance
                    hand_model = PlayerHandModel.from_dict(hand_config_for_player)
                except (ValueError, TypeError) as e:
                     logger.error(f"Failed to create valid hand config for {player_id}: {e}")
                     # Skip this player or create with default hand? Skipping for now.
                     continue

                # Create chip area config (if default exists)
                chip_area_model = None
                if default_chip_config_dict:
                     try:
                         # TODO: Apply chip distribution logic here later if needed
                         chip_area_model = ChipAreaConfig.from_dict(default_chip_config_dict)
                     except (ValueError, TypeError) as e:
                         logger.error(f"Failed to create valid chip config for {player_id} from default: {e}")
                         # Continue without chips for this player

                # Create the final PlayerModel instance for this player
                player_model = PlayerModel(
                    player_id=player_id,
                    hand_config=hand_model,
                    chip_area_config=chip_area_model
                )
                generated_players.append(player_model)

            # These lines should align with the 'for' loop's indentation level, inside the 'try'
            self._resolved_players = generated_players
            logger.info(f"Successfully generated {len(self._resolved_players)} players.")

        except (ValueError, TypeError) as e:
             logger.error(f"Error generating players from defaults: {e}")
             logger.error(traceback.format_exc())
             self._resolved_players = []
        except Exception as e:
             logger.error(f"Unexpected error generating players from defaults: {e}")
             logger.error(traceback.format_exc())
             self._resolved_players = []

    def to_dict(self) -> dict[str, Any]:
        """Convert the PokerSceneModel instance to a dictionary, using resolved players/cards."""
        # Use the resolved list of players
        players_dicts = [p.to_dict() for p in self._resolved_players]

        # Use the resolved community cards model
        community_cards_dict = self._resolved_community_cards.to_dict() if self._resolved_community_cards else None

        # Include the original config elements that defined the scene
        return {
            'n_players': self.n_players,
            'players': players_dicts, # Use the resolved list
            'community_cards': community_cards_dict, # Use resolved
            'scene_setup': self.scene_setup,
            'noise_config': self.noise_config,
            'deck_blend_file': self.deck_blend_file,
            'random_seed': self.random_seed,
            # Optionally include original generation config if needed for metadata
            'card_type_config': self.card_type_config,
            'default_player_config': self.default_player_config.to_dict() if isinstance(self.default_player_config, PlayerModel) else self.default_player_config,
            'player_distribution_config': self.player_distribution_config,
            'card_distribution_inputs': self.card_distribution_inputs,
            'chip_distribution_inputs': self.chip_distribution_inputs,
            'card_overlap_config': self.card_overlap_config.to_dict() if isinstance(self.card_overlap_config, CardOverlapModel) else self.card_overlap_config,
        }

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> 'PokerSceneModel':
        """
        Create a PokerSceneModel instance from a dictionary.
        Passes raw config values; __post_init__ handles resolution/generation.
        """
        if not isinstance(config, dict):
            raise TypeError("config must be a dictionary")
        if 'n_players' not in config:
             raise ValueError("Missing required key 'n_players' in poker scene configuration")

        # Pass raw config values directly to the constructor.
        # __post_init__ will handle parsing 'players', 'community_cards',
        # and deciding whether to generate players based on 'default_player_config'.
        instance = cls(
            n_players=config['n_players'],
            players=config.get('players', []), # Pass the raw list/dicts or empty list
            community_cards=config.get('community_cards'), # Pass raw dict/model or None
            scene_setup=config.get('scene_setup', {}),
            noise_config=config.get('noise_config'),
            deck_blend_file=config.get('deck_blend_file'),
            random_seed=config.get('random_seed'),
            # Pass generation-related configs directly
            card_type_config=config.get('card_type_config'),
            default_player_config=config.get('default_player_config'),
            player_distribution_config=config.get('player_distribution_config'),
            card_distribution_inputs=config.get('card_distribution_inputs'),
            chip_distribution_inputs=config.get('chip_distribution_inputs'),
            card_overlap_config=config.get('card_overlap_config'),
        )
        # __post_init__ runs automatically after this, performing resolution/generation
        return instance

    def _handle_river_cards_from_distribution(self, dealt_cards_dict):
        """
        New Step 4: Handle the case where both card distribution and community cards
        are specified in the config.

        This ensures that dealt river cards from card_distribution_inputs are properly
        transferred to the community_cards configuration.
        """
        if not self.community_cards or not isinstance(self.community_cards, dict):
            return  # No community cards config to update

        if not dealt_cards_dict or 'river' not in dealt_cards_dict:
            return  # No dealt river cards

        dealt_river_cards = dealt_cards_dict.get('river', [])
        num_dealt_river = len(dealt_river_cards)

        if num_dealt_river == 0:
            return  # No river cards were dealt

        # Start with the existing community_cards dict
        community_config = self.community_cards.copy()

        # Update with the dealt river cards
        community_config['n_cards'] = num_dealt_river
        community_config['card_names'] = dealt_river_cards

        # Replace the original config with the updated one
        self.community_cards = community_config
        logger.debug(f"Updated community_cards config with {num_dealt_river} dealt cards: {dealt_river_cards}")

