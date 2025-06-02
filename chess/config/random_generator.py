import random
from typing import Any

from chess.config.difficulty_presets import get_difficulty_preset
from chess.config.models import (
    BoardModel,
    BoardRandomizationModel,
    GeometryModel,
    MaterialModel,
    PieceModel,
    PieceRandomizationModel,
    RandomizationRange,
)


class RandomConfigGenerator:
    """Generator for random chess configurations."""

    def __init__(
        self,
        difficulty: str = "medium",
        board_config: dict[str, Any] | None = None,
        piece_config: dict[str, Any] | None = None,
        seed: int | None = None
    ):
        """
        Initialize the random config generator.

        Args:
            difficulty: Difficulty level ('easy', 'medium', 'hard')
            board_config: Custom board randomization config dictionary
            piece_config: Custom piece randomization config dictionary
            seed: Random seed for reproducibility
        """
        # Set random seed if provided
        if seed is not None:
            random.seed(seed)

        # Get preset or use custom configs
        preset = get_difficulty_preset(difficulty)

        # Convert configs to models
        self.board_config = (BoardRandomizationModel.from_dict(board_config)
                           if board_config is not None
                           else preset.board_config)
        self.piece_config = (PieceRandomizationModel.from_dict(piece_config)
                           if piece_config is not None
                           else preset.piece_config)

    def _random_from_range(self, range_config: RandomizationRange) -> float:
        """Generate a random value within the given range."""
        if range_config.step:
            steps = int((range_config.max_value - range_config.min_value) / range_config.step)
            return range_config.min_value + random.randint(0, steps) * range_config.step
        return random.uniform(range_config.min_value, range_config.max_value)

    def _generate_board_location(self) -> tuple[float, float, float]:
        """Generate random board location."""
        x = self._random_from_range(self.board_config.location_bounds[0])
        y = self._random_from_range(self.board_config.location_bounds[1])
        return (x, y, 0.9)  # Fixed z coordinate at 0.9

    def generate_board_config(self) -> dict[str, Any]:
        """Generate random board configuration."""
        location = self._generate_board_location()
        if self.board_config is not None:
            rows = self.board_config.rows if self.board_config.rows is not None else 8
            columns = self.board_config.columns if self.board_config.columns is not None else 8
        else:
            rows = 8
            columns = 8

        board_config = BoardModel(
            length=0.7,  # Fixed length
            width=0.7,   # Fixed width
            location=location,
            random_pattern=self.board_config.pattern_randomization,
            pattern_seed=self.board_config.pattern_seed,
            rows=rows,
            columns=columns
        )

        return board_config.to_dict()

    def _generate_piece_counts(self) -> dict[str, int]:
        """Generate random piece counts based on configuration."""
        counts = {}
        total_pieces = 0

        # First pass: generate counts within individual ranges
        for piece_type, count_range in self.piece_config.piece_counts.items():
            count = random.randint(count_range.min_count, count_range.max_count)
            counts[piece_type] = count
            total_pieces += count

        # Second pass: adjust if total constraints are specified
        if any(getattr(range_config, attr) is not None
               for range_config in self.piece_config.piece_counts.values()
               for attr in ['total_min', 'total_max']):

            # Find global min/max
            total_min = max(
                (range_config.total_min for range_config in self.piece_config.piece_counts.values()
                 if range_config.total_min is not None),
                default=0
            )
            total_max = min(
                (range_config.total_max for range_config in self.piece_config.piece_counts.values()
                 if range_config.total_max is not None),
                default=float('inf')
            )

            # Adjust counts if needed
            while total_pieces < total_min:
                # Add pieces
                piece_type = random.choice(list(counts.keys()))
                if counts[piece_type] < self.piece_config.piece_counts[piece_type].max_count:
                    counts[piece_type] += 1
                    total_pieces += 1

            while total_pieces > total_max:
                # Remove pieces
                piece_type = random.choice(list(counts.keys()))
                if counts[piece_type] > self.piece_config.piece_counts[piece_type].min_count:
                    counts[piece_type] -= 1
                    total_pieces -= 1

        return counts

    def _generate_piece_scale(self, piece_type: str) -> float:
        """Generate scale for a piece based on configuration."""
        base_scale = self._random_from_range(self.piece_config.scale_range)

        # Apply variations if enabled
        if self.piece_config.intra_class_variation:
            # Add small random variation (±10% of the scale range)
            scale_range = self.piece_config.scale_range.max_value - self.piece_config.scale_range.min_value
            variation = random.uniform(-0.1, 0.1) * scale_range
            base_scale = max(self.piece_config.scale_range.min_value,
                           min(self.piece_config.scale_range.max_value,
                               base_scale + variation))

        if self.piece_config.extra_class_variation:
            # Different base scales for different piece types
            type_scales = {
                "pawn": 0.9,     # Slightly smaller
                "knight": 1.0,
                "bishop": 1.0,
                "rook": 1.05,    # Slightly larger
                "queen": 1.1,    # Larger
                "king": 1.1      # Larger
            }
            # Apply relative scaling while keeping within bounds
            multiplier = type_scales.get(piece_type, 1.0)
            base_scale = max(self.piece_config.scale_range.min_value,
                           min(self.piece_config.scale_range.max_value,
                               base_scale * multiplier))

        return base_scale

    def _generate_piece_location(self, used_positions: list[tuple[int, int]], board_config: dict[str, Any]) -> tuple[int, int]:
        """Generate a random board position that hasn't been used."""
        # Get board dimensions from config
        rows = board_config.get("rows", 8)
        columns = board_config.get("columns", 8)

        # Generate available positions within board dimensions
        available_positions = [
            (row, col)
            for row in range(rows)
            for col in range(columns)
            if (row, col) not in used_positions
        ]

        if not available_positions:
            print("Warning: No available positions left on the board")
            # Return center position as fallback
            return (rows // 2, columns // 2)

        return random.choice(available_positions)

    def generate_pieces_config(self, board_config: dict[str, Any]) -> dict[str, dict[str, Any]]:
        """Generate random piece configurations."""
        piece_counts = self._generate_piece_counts()
        used_positions = []
        pieces_config = {}

        # Get board dimensions
        rows = board_config.get("rows", 8)
        columns = board_config.get("columns", 8)

        # Calculate total pieces
        total_pieces = sum(piece_counts.values())
        current_piece_index = 0

        # Validate piece counts against available positions
        total_positions = rows * columns

        if total_pieces > total_positions:
            print(f"Warning: More pieces ({total_pieces}) than available positions ({total_positions})")
            # Adjust piece counts proportionally
            scale_factor = total_positions / total_pieces
            for piece_type in piece_counts:
                piece_counts[piece_type] = int(piece_counts[piece_type] * scale_factor)
            total_pieces = sum(piece_counts.values())

        for piece_type, count in piece_counts.items():
            for i in range(count):
                current_piece_index += 1

                # Generate position
                position = self._generate_piece_location(used_positions, board_config)
                used_positions.append(position)

                # Generate scale
                scale = self._generate_piece_scale(piece_type)

                # Generate color
                color = random.choice(self.piece_config.allowed_colors)

                # Create piece config
                piece_id = f"{piece_type}_{i+1}"
                piece_config = PieceModel(
                    piece_type=piece_type,
                    location=position,
                    material=MaterialModel(color=color),
                    geometry=GeometryModel(scale=scale)
                )
                pieces_config[piece_id] = piece_config.to_dict()

        return pieces_config

    def generate_all_configs(self) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        """Generate both board and pieces configurations."""
        board_config = self.generate_board_config()
        pieces_config = self.generate_pieces_config(board_config)

        return board_config, pieces_config
