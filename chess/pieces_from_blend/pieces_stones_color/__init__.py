"""Chess pieces loaded from blend files."""

from .pieces import (
    create_pawn, create_rook, create_queen, create_king,
    create_bishop, create_knight
)

__all__ = [
    'create_pawn', 'create_rook', 'create_queen', 'create_king',
    'create_bishop', 'create_knight'
] 