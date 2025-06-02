"""
Chess configuration module for generating random chess scenes.

This package provides tools for generating chess boards and pieces with various
randomization options.
"""

# Import from original modules
from chess.config.advanced_generator import (
    ChessConfigGenerator,
    PieceCountGenerator,
    PiecePositionGenerator,
    PieceTypeGenerator,
)

# Import from advanced modules
from chess.config.advanced_models import (
    CountSpecificationType,
    PieceCountModel,
    PiecePositionModel,
    PieceTypeModel,
    PieceTypeSpecification,
    SpreadLevel,
    StartingPoint,
)
from chess.config.models import (
    BoardModel,
    BoardRandomizationModel,
    DifficultyPreset,
    GeometryModel,
    MaterialModel,
    PieceCountRange,
    PieceModel,
    PieceRandomizationModel,
    RandomizationRange,
)
from chess.config.random_generator import RandomConfigGenerator

__all__ = [
    # Original classes
    'MaterialModel', 'GeometryModel', 'PieceModel', 'BoardModel',
    'RandomizationRange', 'PieceCountRange', 'BoardRandomizationModel',
    'PieceRandomizationModel', 'DifficultyPreset', 'RandomConfigGenerator',

    # Advanced model classes
    'PieceCountModel',
    'CountSpecificationType',
    'PieceTypeModel',
    'PieceTypeSpecification',
    'PiecePositionModel',
    'SpreadLevel',
    'StartingPoint',

    # Advanced generator classes
    'PieceCountGenerator',
    'PieceTypeGenerator',
    'PiecePositionGenerator',
    'ChessConfigGenerator'
]
