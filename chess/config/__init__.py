"""
Chess configuration module for generating random chess scenes.

This package provides tools for generating chess boards and pieces with various
randomization options.
"""

# Import from original modules
from .advanced_generator import (
    ChessConfigGenerator,
    PieceCountGenerator,
    PiecePositionGenerator,
    PieceTypeGenerator,
)

# Import from advanced modules
from .advanced_models import (
    CountSpecificationType,
    PieceCountModel,
    PiecePositionModel,
    PieceTypeModel,
    PieceTypeSpecification,
    SpreadLevel,
    StartingPoint,
)
from .models import (
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
from .random_generator import RandomConfigGenerator

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
