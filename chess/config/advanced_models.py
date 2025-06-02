import enum
import math
import random
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar, runtime_checkable

# Type variable for generic methods
T = TypeVar('T')

# Enum for count specification type
class CountSpecificationType(enum.Enum):
    PRESET = "preset"
    FIXED = "fixed"
    RANGE = "range"
    EXPLICIT = "explicit"
    RANGE_BY_TYPE = "range_by_type"

# Enum for piece type specification
class PieceTypeSpecification(enum.Enum):
    PRESET = "preset"
    EXPLICIT = "explicit"
    RANDOM_N = "random_n"

# Enum for spread level
class SpreadLevel(enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

# Enum for preset starting points
class StartingPoint(enum.Enum):
    CENTER = "center"
    UPPER_LEFT = "upper_left"
    UPPER_RIGHT = "upper_right"
    LOWER_LEFT = "lower_left"
    LOWER_RIGHT = "lower_right"
    LEFT_SIDE = "left_side"
    RIGHT_SIDE = "right_side"
    TOP_SIDE = "top_side"
    BOTTOM_SIDE = "bottom_side"

# Abstract base classes for different configuration types
@runtime_checkable
class PieceCountModel(Protocol):
    """Protocol defining interface for piece count configuration."""
    def generate_counts(self, available_types: list[str]) -> dict[str, int]:
        """Generate a dictionary mapping piece types to counts."""
        ...

@runtime_checkable
class PieceTypeModel(Protocol):
    """Protocol defining interface for piece type configuration."""
    def generate_types(self, all_available_types: list[str]) -> list[str]:
        """Generate a list of piece types to use."""
        ...

@runtime_checkable
class PiecePositionModel(Protocol):
    """Protocol defining interface for piece position configuration."""
    def generate_positions(self, count: int, board_dims: tuple[int, int]) -> list[tuple[int, int]]:
        """Generate a list of positions for pieces."""
        ...

# Unified piece count config
@dataclass
class PieceCountModel:
    """
    Configuration for piece counts, supporting multiple input types.

    This class handles various ways to specify piece counts:
    1. Preset string: "low", "medium", "high"
    2. Fixed count: A specific integer
    3. Range: A min-max range of counts or a min-max range of strings
    4. Explicit: Dictionary mapping piece types to exact counts
    5. Range by type: Dictionary mapping piece types to min-max ranges

    Attributes:
        spec_type: Type of count specification
        preset: Preset name if using preset type ("low", "medium", "high")
        count: Fixed count value if using fixed type
        min_count: Minimum count if using range type
        max_count: Maximum count if using range type
        counts: Dictionary mapping piece types to counts if using explicit type
        count_ranges: Dictionary mapping piece types to min-max ranges if using range_by_type
        randomization: Whether to apply randomization to counts
        randomization_percentage: Percentage of variation for randomization (default 0.2)
    """
    spec_type: CountSpecificationType = CountSpecificationType.PRESET
    preset: str = "medium"
    count: int = 10
    min_count: int | str = 5
    max_count: int | str = 15
    counts: dict[str, int] = field(default_factory=dict)
    count_ranges: dict[str, tuple[int, int]] = field(default_factory=dict)
    randomization: bool = False
    randomization_percentage: float = 0.2

    def _apply_randomization(self, value: int) -> int:
        """
        Apply randomization to a value using uniform distribution.

        Args:
            value: Base value to randomize

        Returns:
            Randomized value
        """
        if not self.randomization:
            return value

        # Calculate range for uniform distribution
        variation = int(value * self.randomization_percentage)
        min_val = max(1, value - variation)  # Ensure at least 1 piece
        max_val = value + variation

        return random.randint(min_val, max_val)

    def generate_counts(self) -> dict[str, Any]:
        """
        Generate piece counts based on the configuration type.

        Returns:
            Dictionary with piece type counts or special keys like "_total_"
        """
        if self.spec_type == CountSpecificationType.PRESET:
            # Map presets to total piece counts
            preset_counts = {
                "low": 5,
                "medium": 10,
                "high": 18
            }
            count = preset_counts.get(self.preset.lower(), 10)
            if self.randomization:
                count = self._apply_randomization(count)
            return {"_total_": count}

        elif self.spec_type == CountSpecificationType.FIXED:
            count = self.count
            if self.randomization:
                count = self._apply_randomization(count)
            return {"_total_": count}

        elif self.spec_type == CountSpecificationType.RANGE:
            # For range, we already have min and max, just need to pick a value
            if isinstance(self.min_count, str) and isinstance(self.max_count, str):
                # map to preset counts
                preset_counts = {
                    "low": 5,
                    "medium": 10,
                    "high": 18
                }
                count = random.randint(preset_counts.get(self.min_count.lower(), 5), preset_counts.get(self.max_count.lower(), 18))
            else:
                count = random.randint(self.min_count, self.max_count)
            return {"_total_": count}

        elif self.spec_type == CountSpecificationType.EXPLICIT:
            # Return counts as is since types are already specified
            return self.counts

        elif self.spec_type == CountSpecificationType.RANGE_BY_TYPE:
            # Return count ranges as is since types are already specified
            return {k: {"min": v[0], "max": v[1]} for k, v in self.count_ranges.items()}

        # Default case, should not happen
        return {"_total_": 10}

    def to_dict(self) -> dict[str, Any]:
        """Convert the config to a dictionary."""
        result = {
            "type": self.spec_type.value,
            "randomization": self.randomization,
            "randomization_percentage": self.randomization_percentage
        }

        if self.spec_type == CountSpecificationType.PRESET:
            result["preset"] = self.preset
        elif self.spec_type == CountSpecificationType.FIXED:
            result["count"] = self.count
        elif self.spec_type == CountSpecificationType.RANGE:
            result["min_count"] = self.min_count
            result["max_count"] = self.max_count
        elif self.spec_type == CountSpecificationType.EXPLICIT:
            result["counts"] = self.counts
        elif self.spec_type == CountSpecificationType.RANGE_BY_TYPE:
            result["count_ranges"] = self.count_ranges

        return result

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> 'PieceCountModel':
        """Create a PieceCountModel from a dictionary."""
        if not isinstance(config, dict):
            # If input is not a dict, maybe try creating a default?
            # Or raise an error? Let's return default for now.
            # logger.warning(f"Invalid config type for PieceCountModel.from_dict: {type(config)}. Using default.")
            return cls()

        default_instance = cls()
        # Use lower case for type matching consistency
        config_type = config.get("type", default_instance.spec_type.value).lower()

        if config_type == "preset":
            return cls(
                spec_type=CountSpecificationType.PRESET,
                preset=config.get("preset", default_instance.preset),
                randomization=config.get("randomization", default_instance.randomization),
                randomization_percentage=config.get("randomization_percentage", default_instance.randomization_percentage)
            )
        elif config_type == "fixed":
            # <<< CORRECTED KEY LOOKUP: Use "value" for fixed type >>>
            return cls(
                spec_type=CountSpecificationType.FIXED,
                count=config.get("value", default_instance.count), # Look for "value", fallback to default count
                randomization=config.get("randomization", default_instance.randomization),
                randomization_percentage=config.get("randomization_percentage", default_instance.randomization_percentage)
            )
        elif config_type == "range":
            # Handle potential string presets in min/max for range
            min_c = config.get("min_count", default_instance.min_count)
            max_c = config.get("max_count", default_instance.max_count)
            return cls(
                spec_type=CountSpecificationType.RANGE,
                min_count=min_c,
                max_count=max_c,
                randomization=config.get("randomization", True), # Default randomization=True for range
                randomization_percentage=config.get("randomization_percentage", default_instance.randomization_percentage)
            )
        elif config_type == "explicit":
            return cls(
                spec_type=CountSpecificationType.EXPLICIT,
                counts=config.get("counts", default_instance.counts),
                randomization=config.get("randomization", default_instance.randomization),
                randomization_percentage=config.get("randomization_percentage", default_instance.randomization_percentage)
            )
        elif config_type == "range_by_type":
            return cls(
                spec_type=CountSpecificationType.RANGE_BY_TYPE,
                count_ranges=config.get("count_ranges", default_instance.count_ranges),
                randomization=config.get("randomization", True), # Default randomization=True for range_by_type
                randomization_percentage=config.get("randomization_percentage", default_instance.randomization_percentage)
            )
        else:
            # logger.warning(f"Unknown PieceCountModel type '{config_type}'. Using default preset.")
            return cls() # Default to preset if type is unknown

    @classmethod
    def from_preset(cls, preset: str, randomization: bool = False, randomization_percentage: float = 0.2) -> 'PieceCountModel':
        """Create from a preset string."""
        return cls(
            spec_type=CountSpecificationType.PRESET,
            preset=preset,
            randomization=randomization,
            randomization_percentage=randomization_percentage
        )

    @classmethod
    def from_fixed(cls, count: int, randomization: bool = False, randomization_percentage: float = 0.2) -> 'PieceCountModel':
        """Create from a fixed count."""
        return cls(
            spec_type=CountSpecificationType.FIXED,
            count=count,
            randomization=randomization,
            randomization_percentage=randomization_percentage
        )

    @classmethod
    def from_range(cls, min_count: int, max_count: int, randomization: bool = True, randomization_percentage: float = 0.2) -> 'PieceCountModel':
        """Create from a count range."""
        return cls(
            spec_type=CountSpecificationType.RANGE,
            min_count=min_count,
            max_count=max_count,
            randomization=randomization,
            randomization_percentage=randomization_percentage
        )

    @classmethod
    def from_explicit(cls, counts: dict[str, int], randomization: bool = False, randomization_percentage: float = 0.2) -> 'PieceCountModel':
        """Create from explicit counts by type."""
        return cls(
            spec_type=CountSpecificationType.EXPLICIT,
            counts=counts,
            randomization=randomization,
            randomization_percentage=randomization_percentage
        )

    @classmethod
    def from_range_by_type(cls, count_ranges: dict[str, tuple[int, int]], randomization: bool = True, randomization_percentage: float = 0.2) -> 'PieceCountModel':
        """Create from count ranges by type."""
        return cls(
            spec_type=CountSpecificationType.RANGE_BY_TYPE,
            count_ranges=count_ranges,
            randomization=randomization,
            randomization_percentage=randomization_percentage
        )

# Unified piece type config
@dataclass
class PieceTypeModel:
    """
    Unified configuration for piece types, supporting multiple input types.

    This class handles various ways to specify piece types:
    1. Preset string: "low", "medium", "high"
    2. Explicit list: Specific piece types to use
    3. Random N: Select N random piece types

    Attributes:
        spec_type: Type of piece type specification
        preset: Preset name if using preset type ("low", "medium", "high")
        types: List of specific piece types if using explicit type
        n_types: Number of random types to select if using random_n type
        randomization: Whether to apply randomization to selection
    """
    spec_type: PieceTypeSpecification = PieceTypeSpecification.PRESET
    preset: str = "medium"
    types: list[str] = field(default_factory=list)
    n_types: int = 3

    def generate_types(self) -> list[str]:
        """
        Generate list of piece types based on the configuration type.

        Args:
            all_available_types: List of all available piece types

        Returns:
            List of selected piece types
        """
        # Define the standard set of available chess piece types
        standard_available_types = ["pawn", "rook", "knight", "bishop", "queen", "king"]

        if self.spec_type == PieceTypeSpecification.PRESET:
            preset_lower = self.preset.lower()

            # Check if the preset value is actually one of the known piece types
            if preset_lower in standard_available_types:
                # If it's a specific type, return only that type
                return [preset_lower]

            # Otherwise, treat it as a preset count name (low, medium, high)
            preset_type_counts = {
                "none": 0,
                "low": 2,
                "medium": 4,
                "high": 6
            }
            # Default to medium count if preset name is unknown
            count = preset_type_counts.get(preset_lower, 4)
            # Sample from the standard types
            return random.sample(standard_available_types, min(count, len(standard_available_types)))

        elif self.spec_type == PieceTypeSpecification.EXPLICIT:
            # Return the explicitly provided types.
            # Consider adding validation against standard_available_types if necessary in the future.
            return [t.lower() for t in self.types if isinstance(t, str)] # Ensure lowercase and filter non-strings

        elif self.spec_type == PieceTypeSpecification.RANDOM_N:
            # Select N random types from the standard list
            # Ensure n_types is not larger than the available types
            num_to_sample = min(self.n_types, len(standard_available_types))
            # Ensure n_types is not negative
            num_to_sample = max(0, num_to_sample)
            return random.sample(standard_available_types, num_to_sample)

        # Default case (should ideally not be reached if spec_type is validated)
        # Return all types in a random order as a fallback
        return random.sample(standard_available_types, len(standard_available_types))

    def to_dict(self) -> dict[str, Any]:
        """Convert the config to a dictionary."""
        result = {
            "type": self.spec_type.value,
            "randomization": self.randomization
        }

        if self.spec_type == PieceTypeSpecification.PRESET:
            result["preset"] = self.preset
        elif self.spec_type == PieceTypeSpecification.EXPLICIT:
            result["types"] = self.types
        elif self.spec_type == PieceTypeSpecification.RANDOM_N:
            result["n_types"] = self.n_types

        return result

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> 'PieceTypeModel':
        """Create a PieceTypeModel from a dictionary."""
        if not isinstance(config, dict):
            config = {}

        default_instance = cls()
        config_type = config.get("type", default_instance.spec_type.value)

        try:
            spec_type = PieceTypeSpecification(config_type)
        except ValueError:
            spec_type = default_instance.spec_type

        if spec_type == PieceTypeSpecification.PRESET:
            return cls(
                spec_type=spec_type,
                preset=config.get("preset", default_instance.preset),
                            )
        elif spec_type == PieceTypeSpecification.EXPLICIT:
            return cls(
                spec_type=spec_type,
                types=config.get("types", default_instance.types),
            )
        elif spec_type == PieceTypeSpecification.RANDOM_N:
            return cls(
                spec_type=spec_type,
                n_types=config.get("n_types", default_instance.n_types),
            )
        else:
            # Default to preset
            return cls()

    @classmethod
    def from_preset(cls, preset: str) -> 'PieceTypeModel':
        """Create from a preset string."""
        return cls(
            spec_type=PieceTypeSpecification.PRESET,
            preset=preset,
        )

    @classmethod
    def from_explicit(cls, types: list[str]) -> 'PieceTypeModel':
        """Create from explicit types list."""
        return cls(
            spec_type=PieceTypeSpecification.EXPLICIT,
            types=types,
        )

    @classmethod
    def from_random_n(cls, n: int) -> 'PieceTypeModel':
        """Create from number of random types to select."""
        return cls(
            spec_type=PieceTypeSpecification.RANDOM_N,
            n_types=n,
        )

# Enhanced position configuration
@dataclass
class PiecePosition:
    """
    Enhanced configuration for piece positions with spread control.

    Attributes:
        allowed_positions: List of specific positions that are allowed, if empty all are allowed
        spread_level: How spread out the pieces should be (low, medium, high)
        start_point: The center point for piece distribution
        min_x: Minimum x coordinate
        max_x: Maximum x coordinate
        min_y: Minimum y coordinate
        max_y: Maximum y coordinate
        uniform: Whether to select positions uniformly at random (ignores spread_level if True)
    """
    allowed_positions: list[tuple[int, int]] = field(default_factory=list)
    spread_level: SpreadLevel = SpreadLevel.MEDIUM
    start_point: StartingPoint | tuple[int, int] = StartingPoint.CENTER
    min_x: int = 0
    max_x: int = 7
    min_y: int = 0
    max_y: int = 7
    uniform: bool = True

    def generate_positions(self, count: int, board_dims: tuple[int, int]) -> list[tuple[int, int]]:
        """
        Generate positions for pieces based on constraints.

        Args:
            count: Number of positions to generate
            board_dims: Board dimensions (rows, columns)

        Returns:
            List of (row, column) positions
        """
        # Define available positions
        rows, cols = board_dims
        if self.allowed_positions:
            # Use only allowed positions that are within bounds
            available_positions = [
                (x, y) for x, y in self.allowed_positions
                if 0 <= x < rows and 0 <= y < cols # Check against actual board dims
            ]
        else:
            # Generate all positions within bounds

            # Determine the effective bounds
            # Use board_dims if min/max are defaults, otherwise use specified min/max
            min_row = self.min_x if self.min_x != 0 else 0
            max_row = self.max_x if self.max_x != 7 else rows - 1
            min_col = self.min_y if self.min_y != 0 else 0
            max_col = self.max_y if self.max_y != 7 else cols - 1

            # Ensure bounds are still within the actual board dimensions
            min_row = max(0, min_row)
            max_row = min(rows - 1, max_row)
            min_col = max(0, min_col)
            max_col = min(cols - 1, max_col)

            available_positions = []
            if max_row >= min_row and max_col >= min_col:
                available_positions = [
                    (r, c)
                    for r in range(min_row, max_row + 1)
                    for c in range(min_col, max_col + 1)
                ]

        # If no positions available, return empty list
        if not available_positions:
            return []

        # If uniform selection is requested, just sample randomly
        if self.uniform:
            return random.sample(available_positions, min(count, len(available_positions)))

        # Determine starting point coordinates
        start_x, start_y = self._get_start_point_coords(board_dims)

        # Sort positions by distance from starting point
        # For lower spread, positions closer to start point are prioritized
        available_positions.sort(key=lambda pos: self._distance_from_point(pos, (start_x, start_y)))

        # Apply spread level
        spread_factor = self._get_spread_factor()

        # Select positions based on spread
        if spread_factor == 0:
            # Just take the closest positions
            return available_positions[:min(count, len(available_positions))]

        selected_positions = []
        while len(selected_positions) < count and available_positions:
            # Select a position considering the spread factor
            if random.random() < spread_factor or len(available_positions) == 1:
                # High spread factor means higher chance of selecting randomly
                position = available_positions.pop(random.randint(0, len(available_positions) - 1))
            else:
                # Take the next closest position
                position = available_positions.pop(0)

            selected_positions.append(position)

            # Reorder remaining positions by distance from the last selected position
            # This creates a spreading pattern where pieces tend to be placed near previously placed pieces
            if self.spread_level != SpreadLevel.HIGH and available_positions:
                available_positions.sort(key=lambda pos: self._distance_from_point(pos, position))

        return selected_positions

    def _get_start_point_coords(self, board_dims: tuple[int, int]) -> tuple[int, int]:
        """Get the coordinates of the starting point."""
        rows, cols = board_dims

        if isinstance(self.start_point, tuple):
            # Use explicit coordinates
            return self.start_point

        # Map preset starting points to coordinates
        if self.start_point == StartingPoint.CENTER:
            return (rows // 2, cols // 2)
        elif self.start_point == StartingPoint.UPPER_LEFT:
            return (rows // 4, cols // 4)
        elif self.start_point == StartingPoint.UPPER_RIGHT:
            return (rows // 4, 3 * cols // 4)
        elif self.start_point == StartingPoint.LOWER_LEFT:
            return (3 * rows // 4, cols // 4)
        elif self.start_point == StartingPoint.LOWER_RIGHT:
            return (3 * rows // 4, 3 * cols // 4)
        elif self.start_point == StartingPoint.LEFT_SIDE:
            return (rows // 2, cols // 4)
        elif self.start_point == StartingPoint.RIGHT_SIDE:
            return (rows // 2, 3 * cols // 4)
        elif self.start_point == StartingPoint.TOP_SIDE:
            return (rows // 4, cols // 2)
        elif self.start_point == StartingPoint.BOTTOM_SIDE:
            return (3 * rows // 4, cols // 2)
        else:
            # Default to center
            return (rows // 2, cols // 2)

    def _get_spread_factor(self) -> float:
        """Get the spread factor based on spread level."""
        if self.spread_level == SpreadLevel.LOW:
            return 0.2  # Low randomness, pieces clustered together
        elif self.spread_level == SpreadLevel.MEDIUM:
            return 0.5  # Medium spread
        elif self.spread_level == SpreadLevel.HIGH:
            return 0.8  # High randomness, pieces spread out
        return 0.5  # Default

    def _distance_from_point(self, pos1: tuple[int, int], pos2: tuple[int, int]) -> float:
        """Calculate Euclidean distance between two positions."""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def to_dict(self) -> dict[str, Any]:
        """Convert the config to a dictionary."""
        result = {
            "min_x": self.min_x,
            "max_x": self.max_x,
            "min_y": self.min_y,
            "max_y": self.max_y,
            "spread_level": self.spread_level.value if isinstance(self.spread_level, SpreadLevel) else self.spread_level,
            "uniform": self.uniform
        }

        # Handle allowed positions
        if self.allowed_positions:
            result["allowed_positions"] = self.allowed_positions

        # Handle start point
        if isinstance(self.start_point, tuple):
            result["start_point_coords"] = self.start_point
        else:
            result["start_point"] = self.start_point.value if isinstance(self.start_point, StartingPoint) else self.start_point

        return result

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> 'PiecePosition':
        """Create a PiecePosition from a dictionary."""
        if not isinstance(config, dict):
            config = {}

        default_instance = cls()

        # Handle spread level
        spread_level_str = config.get("spread_level", default_instance.spread_level.value)
        try:
            spread_level = SpreadLevel(spread_level_str)
        except ValueError:
            spread_level = default_instance.spread_level

        # Handle start point
        if "start_point_coords" in config:
            start_point = tuple(config["start_point_coords"])
        else:
            start_point_str = config.get("start_point", default_instance.start_point.value)
            try:
                start_point = StartingPoint(start_point_str)
            except ValueError:
                start_point = default_instance.start_point

        return cls(
            allowed_positions=config.get("allowed_positions", default_instance.allowed_positions),
            spread_level=spread_level,
            start_point=start_point,
            min_x=config.get("min_x", default_instance.min_x),
            max_x=config.get("max_x", default_instance.max_x),
            min_y=config.get("min_y", default_instance.min_y),
            max_y=config.get("max_y", default_instance.max_y),
            uniform=config.get("uniform", default_instance.uniform)
        )

    @classmethod
    def from_bounds(cls, min_x: int = 0, max_x: int = 7, min_y: int = 0, max_y: int = 7) -> 'PiecePosition':
        """Create from bounds only."""
        return cls(
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y
        )

    @classmethod
    def from_allowed_positions(cls, positions: list[tuple[int, int]]) -> 'PiecePosition':
        """Create from a list of allowed positions."""
        return cls(
            allowed_positions=positions
        )

    @classmethod
    def from_spread(cls, level: str | SpreadLevel,
                  start: str | tuple[int, int] | StartingPoint = StartingPoint.CENTER) -> 'PiecePosition':
        """Create from spread level and starting point."""
        # Convert string to enum if needed
        if isinstance(level, str):
            try:
                spread_level = SpreadLevel(level)
            except ValueError:
                spread_level = SpreadLevel.MEDIUM
        else:
            spread_level = level

        # Convert string to enum if needed
        if isinstance(start, str):
            try:
                start_point = StartingPoint(start)
            except ValueError:
                start_point = StartingPoint.CENTER
        else:
            start_point = start

        return cls(
            spread_level=spread_level,
            start_point=start_point
        )

# Alias for backward compatibility
PositionBounds = PiecePosition
