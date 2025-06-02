"""
Chess configuration module for generating random chess scenes.

This package provides tools for generating chess boards and pieces with various
randomization options.
"""

# Import from original modules
from .models import (
    MaterialModel, GeometryModel, PieceModel, BoardModel,
    RandomizationRange, PieceCountRange, BoardRandomizationModel,
    PieceRandomizationModel, DifficultyPreset
)
from .random_generator import RandomConfigGenerator

# Import from advanced modules
from .advanced_models import (
    PieceCountModel,
    CountSpecificationType,
    PieceTypeModel,
    PieceTypeSpecification,
    PiecePositionModel,
    SpreadLevel,
    StartingPoint
)
from .advanced_generator import (
    PieceCountGenerator,
    PieceTypeGenerator,
    PiecePositionGenerator,
    ChessConfigGenerator
)

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