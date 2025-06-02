"""Difficulty presets for chess scene generation."""

from typing import Dict, Any

from .models import (
    DifficultyPreset,
    BoardRandomizationModel,
    PieceRandomizationModel,
    PieceCountRange,
    RandomizationRange
)

# Default presets dictionary
DEFAULT_PRESETS = {
    "easy": DifficultyPreset(
        name="easy",
        board_config=BoardRandomizationModel(
            location_bounds=(
                RandomizationRange(-0.0, 0.0),  # x
                RandomizationRange(-0.0, 0.0)   # y
            ),
            pattern_randomization=False
        ),
        piece_config=PieceRandomizationModel(
            piece_counts={
                "pawn": PieceCountRange(1, 3),
                "rook": PieceCountRange(0, 1),
                "knight": PieceCountRange(0, 1),
                "bishop": PieceCountRange(0, 1),
                "queen": PieceCountRange(0, 1),
                "king": PieceCountRange(0, 1)
            },
            scale_range=RandomizationRange(0.07, 0.09),
            intra_class_variation=False,
            extra_class_variation=False
        )
    ),
    "medium": DifficultyPreset(
        name="medium",
        board_config=BoardRandomizationModel(
            location_bounds=(
                RandomizationRange(-0.3, 0.3),  # x
                RandomizationRange(-0.3, 0.3)   # y
            ),
            pattern_randomization=True
        ),
        piece_config=PieceRandomizationModel(
            piece_counts={
                "pawn": PieceCountRange(3, 6),
                "rook": PieceCountRange(1, 2),
                "knight": PieceCountRange(1, 2),
                "bishop": PieceCountRange(1, 2),
                "queen": PieceCountRange(1, 1),
                "king": PieceCountRange(1, 1)
            },
            scale_range=RandomizationRange(0.06, 0.10),
            intra_class_variation=True,
            extra_class_variation=False
        )
    ),
    "hard": DifficultyPreset(
        name="hard",
        board_config=BoardRandomizationModel(
            location_bounds=(
                RandomizationRange(-0.7, 0.7),  # x
                RandomizationRange(-0.7, 0.7)   # y
            ),
            pattern_randomization=True
        ),
        piece_config=PieceRandomizationModel(
            piece_counts={
                "pawn": PieceCountRange(4, 8),
                "rook": PieceCountRange(1, 2),
                "knight": PieceCountRange(1, 2),
                "bishop": PieceCountRange(1, 2),
                "queen": PieceCountRange(1, 2),
                "king": PieceCountRange(1, 2)
            },
            scale_range=RandomizationRange(0.05, 0.12),
            intra_class_variation=True,
            extra_class_variation=True
        )
    )
}

def get_difficulty_preset(difficulty: str = "medium") -> DifficultyPreset:
    """
    Get a predefined difficulty preset
    
    Args:
        difficulty: Difficulty level ('easy', 'medium', 'hard')
        
    Returns:
        DifficultyPreset configuration for the specified difficulty
    """
    return DEFAULT_PRESETS.get(difficulty.lower(), DEFAULT_PRESETS["medium"])

def get_preset_as_dict(difficulty: str = "medium") -> Dict[str, Any]:
    """
    Get a predefined difficulty preset as a dictionary
    
    Args:
        difficulty: Difficulty level ('easy', 'medium', 'hard')
        
    Returns:
        Dictionary representation of the difficulty preset
    """
    preset = get_difficulty_preset(difficulty)
    return preset.to_dict()

def create_preset_from_dict(config: Dict[str, Any]) -> DifficultyPreset:
    """
    Create a difficulty preset from a dictionary configuration
    
    Args:
        config: Dictionary containing the preset configuration
        
    Returns:
        DifficultyPreset created from the dictionary
    """
    return DifficultyPreset.from_dict(config)

def export_all_presets() -> Dict[str, Dict[str, Any]]:
    """
    Export all default presets as dictionaries
    
    Returns:
        Dictionary containing all presets in dictionary format
    """
    return {name: preset.to_dict() for name, preset in DEFAULT_PRESETS.items()} 