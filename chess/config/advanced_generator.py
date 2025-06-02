import random
import os
from typing import Dict, List, Tuple, Union, Optional, Any

from .advanced_models import (
    PieceTypeModel,
    PieceCountModel, CountSpecificationType,
    PiecePosition
)

from .models import (
    BoardModel, PieceModel, MaterialModel, GeometryModel
)


class PieceCountGenerator:
    """Generator for piece counts based on various configuration types."""
    
    def __init__(
        self, 
        config: Union[str, int, Tuple[int, int], List[Tuple[str, int]], List[Tuple[str, Tuple[int, int]]], Dict[str, Any], PieceCountModel],
        randomization: bool = False,
        seed: Optional[int] = None
    ):
        """
        Initialize the piece count generator.
        
        Args:
            config: Configuration for piece counts, can be:
                   - str: preset like "low", "medium", "high"
                   - int: fixed count
                   - Tuple[int, int]: min-max range
                   - List[Tuple[str, int]]: explicit counts by piece type
                   - List[Tuple[str, Tuple[int, int]]]: min-max ranges by piece type
                   - Dict[str, Any]: Configuration dictionary
                   - PieceCountModel: Configuration object
            randomization: Whether to apply randomization
            seed: Random seed for reproducibility
        """
        self.randomization = randomization
        
        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
        
        # Convert config to appropriate object based on type
        if isinstance(config, str):
            self.config = PieceCountModel.from_preset(config, randomization)
        elif isinstance(config, int):
            self.config = PieceCountModel.from_fixed(config, randomization)
        elif isinstance(config, tuple) and len(config) == 2 and all(isinstance(x, int) for x in config):
            self.config = PieceCountModel.from_range(config[0], config[1], randomization)
        elif isinstance(config, list):
            if all(isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str) for item in config):
                # Check if it's a list of (str, int) or (str, (int, int))
                if all(isinstance(item[1], int) for item in config):
                    # List of (str, int)
                    counts = {item[0]: item[1] for item in config}
                    self.config = PieceCountModel.from_explicit(counts, randomization)
                elif all(isinstance(item[1], tuple) and len(item[1]) == 2 for item in config):
                    # List of (str, (int, int))
                    count_ranges = {item[0]: item[1] for item in config}
                    self.config = PieceCountModel.from_range_by_type(count_ranges, randomization)
                else:
                    raise ValueError(f"Invalid list format for piece count config: {config}")
            else:
                raise ValueError(f"Invalid list format for piece count config: {config}")
        elif isinstance(config, dict):
            self.config = PieceCountModel.from_dict(config)
        elif isinstance(config, PieceCountModel):
            self.config = config
        else:
            raise ValueError(f"Invalid config type: {type(config)}")
    
    def _get_allowed_types(self) -> List[str]:
        """
        Determine which piece types are allowed based on the configuration.
        
        Returns:
            List of allowed piece types
        """
        if self.config.spec_type == CountSpecificationType.EXPLICIT:
            return list(self.config.counts.keys())
        elif self.config.spec_type == CountSpecificationType.RANGE_BY_TYPE:
            return list(self.config.count_ranges.keys())
        else:
            # For PRESET, FIXED, and RANGE types, we need to get types from PieceTypeGenerator
            # This will be handled in generate() method
            return []
    
    def generate(self, board_dims: Tuple[int, int] = (8, 8), available_types: Optional[List[str]] = None) -> Dict[str, int]:
        """
        Generate piece counts.
        
        Args:
            board_dims: Board dimensions (rows, columns)
            
        Returns:
            Dictionary mapping piece types to counts
        """
        # Get allowed types based on configuration
        allowed_types = self._get_allowed_types()
        
        # If no types are specified in the configuration, use provided available_types
        if not allowed_types:
            if available_types is None:
                raise ValueError("available_types must be provided for PRESET, FIXED, or RANGE configurations")
            allowed_types = available_types
        
        # Get raw counts from config
        raw_counts = self.config.generate_counts()
        
        # Handle special cases
        if "_total_" in raw_counts:
            total_count = raw_counts["_total_"]
            # Cap total count to board size
            max_pieces = board_dims[0] * board_dims[1]
            total_count = min(total_count, max_pieces)
            
            # Distribute total count among allowed types
            return self._distribute_total_count(total_count, allowed_types)
        
        # Handle explicit counts
        result = {}
        for piece_type, count in raw_counts.items():
            if piece_type not in allowed_types:
                continue
                
            if isinstance(count, dict) and "min" in count and "max" in count:
                # Range by type
                min_val = count["min"]
                max_val = count["max"]
                count = random.randint(min_val, max_val)
            
            result[piece_type] = max(0, count)  # Ensure no negative counts
        
        # Ensure we don't exceed board capacity
        max_pieces = board_dims[0] * board_dims[1]
        total = sum(result.values())
        if total > max_pieces:
            # Scale down proportionally
            scale_factor = max_pieces / total
            for piece_type in result:
                result[piece_type] = max(0, int(result[piece_type] * scale_factor))
        
        return result
    
    def _distribute_total_count(self, total_count: int, available_types: List[str]) -> Dict[str, int]:
        """
        Randomly distribute a total count among available piece types.
        
        Args:
            total_count: Total number of pieces
            available_types: Available piece types from PieceTypeGenerator
            
        Returns:
            Dictionary mapping piece types to counts
        """
        if not available_types:
            return {}
            
        # Initialize counts to 0 for all available types
        result = {piece_type: 0 for piece_type in available_types}
        
        # Randomly assign each piece to a type
        for _ in range(total_count):
            piece_type = random.choice(available_types)
            result[piece_type] += 1
        
        return result


class PieceTypeGenerator:
    """Generator for piece types."""
    
    def __init__(
        self, 
        config: Union[str, List[str], int, Dict[str, Any], PieceTypeModel],
        seed: Optional[int] = None
    ):
        """
        Initialize the piece type generator.
        
        Args:
            config: Configuration for piece types, can be:
                   - str: preset like "low", "medium", "high"
                   - List[str]: explicit list of piece types
                   - int: number of random piece types to select
                   - Dict[str, Any]: Configuration dictionary
                   - PieceTypeModel: Configuration object
            seed: Random seed for reproducibility
        """
        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
        
        # Convert config to appropriate object based on type
        if isinstance(config, str):
            self.config = PieceTypeModel.from_preset(config)
        elif isinstance(config, list):
            self.config = PieceTypeModel.from_explicit(config)
        elif isinstance(config, int):
            self.config = PieceTypeModel.from_random_n(config)
        elif isinstance(config, dict):
            self.config = PieceTypeModel.from_dict(config)
        elif isinstance(config, PieceTypeModel):
            self.config = config
        else:
            raise ValueError(f"Invalid config type: {type(config)}")
    
    def generate(self) -> List[str]:
        """
        Generate list of piece types based on configuration.
        
        Returns:
            List of selected piece types
        """
        return self.config.generate_types()


class PiecePositionGenerator:
    """Generator for piece positions based on configuration."""
    
    def __init__(
        self, 
        config: Union[str, Dict[str, Any], PiecePosition, List[Tuple[int, int]], Tuple[str, Union[str, Tuple[int, int]]], None] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize the piece position generator.
        
        Args:
            config: Configuration for positions, can be:
                   - str: "low", "medium", "high" spread level
                   - Dict[str, Any]: Configuration dictionary
                   - PiecePosition: Configuration object
                   - List[Tuple[int, int]]: List of allowed positions
                   - Tuple[str, Union[str, Tuple[int, int]]]: (spread_level, start_point)
                   - None: Use default configuration
            seed: Random seed for reproducibility
        """
        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
        
        # Convert config to appropriate object based on type
        if config is None:
            self.config = PiecePosition()
        elif isinstance(config, str):
            # Assuming it's a spread level
            self.config = PiecePosition.from_spread(config)
        elif isinstance(config, dict):
            self.config = PiecePosition.from_dict(config)
        elif isinstance(config, PiecePosition):
            self.config = config
        elif isinstance(config, list) and all(isinstance(pos, tuple) and len(pos) == 2 for pos in config):
            # List of allowed positions
            self.config = PiecePosition.from_allowed_positions(config)
        elif isinstance(config, tuple) and len(config) == 2 and isinstance(config[0], str):
            # (spread_level, start_point)
            self.config = PiecePosition.from_spread(config[0], config[1])
        else:
            raise ValueError(f"Invalid config type: {type(config)}")
    
    def generate(self, count: int, board_dims: Tuple[int, int] = (8, 8)) -> List[Tuple[int, int]]:
        """
        Generate positions for pieces.
        
        Args:
            count: Number of positions to generate
            board_dims: Board dimensions (rows, columns)
            
        Returns:
            List of (row, column) positions
        """
        # Delegate to the PiecePosition class's generate_positions method
        positions = self.config.generate_positions(count, board_dims)
        
        # Ensure we don't return more positions than requested
        return positions[:min(count, len(positions))]


class ChessConfigGenerator:
    """Generator for complete chess configurations."""
    
    def __init__(
        self,
        count_config: Union[str, int, Tuple[int, int], List[Tuple[str, int]], List[Tuple[str, Tuple[int, int]]], Dict[str, Any], PieceCountModel] = "medium",
        type_config: Union[str, List[str], int, Dict[str, Any], PieceTypeModel] = "medium",
        position_config: Union[str, Dict[str, Any], PiecePosition, List[Tuple[int, int]], Tuple[str, Union[str, Tuple[int, int]]], None] = None,
        board_config: Optional[Dict[str, Any]] = None,
        randomization: bool = False,
        seed: Optional[int] = None
    ):
        """
        Initialize the chess configuration generator.
        
        Args:
            count_config: Configuration for piece counts, can be:
                       - str: preset like "low", "medium", "high"
                       - int: fixed count
                       - Tuple[int, int]: min-max range
                       - List[Tuple[str, int]]: explicit counts by piece type
                       - List[Tuple[str, Tuple[int, int]]]: min-max ranges by piece type
                       - Dict[str, Any]: Configuration dictionary
                       - PieceCountModel: Configuration object
            type_config: Configuration for piece types, can be:
                      - str: preset like "low", "medium", "high"
                      - List[str]: explicit list of piece types
                      - int: number of random piece types to select
                      - Dict[str, Any]: Configuration dictionary
                      - PieceTypeModel: Configuration object
            position_config: Configuration for piece positions
            board_config: Board configuration dictionary
            randomization: Whether to apply randomization to counts
            seed: Random seed for reproducibility
        """
        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
            local_seed = lambda: random.randint(0, 1000000)
        else:
            local_seed = lambda: None
        
        # Initialize generators
        self.count_generator = PieceCountGenerator(count_config, randomization, local_seed())
        self.type_generator = PieceTypeGenerator(type_config, local_seed())
        self.position_generator = PiecePositionGenerator(position_config, local_seed())
        
        # Store board configuration
        self.board_config = board_config
        self.initial_board_config = board_config # Store optional initial board config
    
    def generate_board_config(self, rows: Optional[int] = None, columns: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate board configuration, allowing overrides for rows and columns.

        Args:
            rows: Optional number of rows to override default.
            columns: Optional number of columns to override default.

        Returns:
            Board configuration dictionary.
        """
        # Use initial board config if provided, otherwise create default
        if self.initial_board_config:
            # Ensure it's a dictionary
            if not isinstance(self.initial_board_config, dict):
                raise ValueError("board_config must be a dictionary")
            board_model = BoardModel.from_dict(self.initial_board_config)
        else:
            # Create default BoardModel and convert to dictionary
            board_model = BoardModel()

        # Override rows/columns if provided
        if rows is not None: board_model.rows = rows
        if columns is not None: board_model.columns = columns

        return board_model.to_dict()
    
    def generate_pieces_config(self, board_config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Generate piece configurations.
        
        Args:
            board_config: Board configuration dictionary
            
        Returns:
            Dictionary mapping piece IDs to piece configuration dictionaries
        """
        # Get board dimensions
        rows = board_config.get("rows", 8)
        columns = board_config.get("columns", 8)
        board_dims = (rows, columns)
        
        # Generate piece types to use
        selected_types = self.type_generator.generate()
        
        # Generate piece counts with available types
        piece_counts = self.count_generator.generate(board_dims, available_types=selected_types)
        
        # Generate positions for all pieces
        positions = self.position_generator.generate(sum(piece_counts.values()), board_dims)
        
        # Create piece configurations
        pieces_config = {}
        position_index = 0
        
        for piece_type, count in piece_counts.items():
            for i in range(count):
                # Get position
                if position_index < len(positions):
                    position = positions[position_index]
                    position_index += 1
                else:
                    # Fallback if not enough positions (shouldn't happen with proper scaling)
                    print("Warning: Not enough positions available")
                    break
                
                # Generate color
                color = random.choice(["white", "black"])
                
                # Create piece config
                piece_id = f"{piece_type}_{i+1}"
                piece_config = PieceModel(
                    piece_type=piece_type,
                    location=position,
                    material=MaterialModel(color=color),
                    geometry=GeometryModel()
                )
                pieces_config[piece_id] = piece_config.to_dict()
        
        return pieces_config
    
    def generate_all_configs(self, rows: Optional[int] = None, columns: Optional[int] = None) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        """
        Generate both board and pieces configurations, accepting row/column overrides.
        
        Returns:
            Tuple containing board config dictionary and pieces config dictionary
        """
        # Generate board config using potential overrides
        board_config = self.generate_board_config(rows=rows, columns=columns)

        # Extract the final dimensions used in the generated board config
        final_rows = board_config.get("rows", 8)
        final_columns = board_config.get("columns", 8)

        # Generate pieces using the final board config (which has correct dimensions)
        pieces_config = self.generate_pieces_config(board_config)
        # Note: generate_pieces_config internally extracts dims from the passed board_config
        return board_config, pieces_config 