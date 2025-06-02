"""Chess pieces loaded from blend files."""

from chess.pieces_from_blend.pieces_stones_color.pieces import (
    create_bishop,
    create_king,
    create_knight,
    create_pawn,
    create_queen,
    create_rook,
)

__all__ = [
    'create_pawn', 'create_rook', 'create_queen', 'create_king',
    'create_bishop', 'create_knight'
]
