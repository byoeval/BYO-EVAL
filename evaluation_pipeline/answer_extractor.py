from typing import Any


class AnswerExtractor:
    '''Extracts answers from game data based on a question key.'''

    def __init__(self):
        '''Initializes the AnswerExtractor.'''

    def extract_answer(self, data: dict[str, Any], question_key: str, game_type: str, question:str=None) -> str | list[str] | dict[str, Any] | None:
        '''
        Extracts an answer from the provided data based on the question key and game type.

        Args:
            data (Dict[str, Any]): The loaded JSON data from an image annotation file.
            question_key (str): The specific key from all_questions.json.
            game_type (str): "chess" or "poker".
            question (str): The actual question text, which may contain substituted values for placeholders like "X".

        Returns:
            Union[str, List[str], Dict[str, Any], None]: The extracted answer.
        '''
        pieces = data.get('pieces', {})
        num_pieces = len(pieces)

        if game_type == "chess":
            first_piece = None

            if num_pieces == 1:
                first_piece = next(iter(pieces.values()))
            elif num_pieces >= 2:
                p_vals = list(pieces.values())
                piece_positions = [p.get('board_position', [0,0]) for p in p_vals]

            if question_key == "count_pieces":
                return str(num_pieces)

            elif question_key == "count_squares":
                board = data.get('board', {})
                rows = board.get('dimensions', {}).get('rows', 0)
                columns = board.get('dimensions', {}).get('columns', 0)
                return str(rows * columns)

            elif question_key == "localize_column_one_piece":
                if num_pieces == 1 and first_piece:
                    return str(first_piece.get('board_position', [0, 0])[0])
                return None

            elif question_key == "localize_row_one_piece":
                if num_pieces == 1 and first_piece:
                    return str(first_piece.get('board_position', [0, 0])[1])
                return None

            elif question_key == "localize_rows_between_two_pieces":
                if num_pieces == 2 and piece_positions[0] and piece_positions[1]:
                    return str(abs(piece_positions[0][1] - piece_positions[1][1]))
                return None

            elif question_key == "localize_columns_between_two_pieces":
                if num_pieces == 2 and piece_positions[0] and piece_positions[1]:
                    return str(abs(piece_positions[0][0] - piece_positions[1][0]))
                return None

            elif question_key == "localize_row_closest_piece":
                if num_pieces > 1:
                    closest_rows: list[str] = []
                    piece_values = list(pieces.values())
                    for i, piece in enumerate(piece_values):
                        current_pos = piece.get('board_position')
                        if not current_pos: continue

                        other_pieces_list = piece_values[:i] + piece_values[i+1:]
                        if not other_pieces_list: continue

                        closest_p = min(other_pieces_list, key=lambda p:
                            abs(p.get('board_position', [float('inf'), float('inf')])[0] - current_pos[0]) +
                            abs(p.get('board_position', [float('inf'), float('inf')])[1] - current_pos[1]))
                        closest_rows.append(str(closest_p.get('board_position', [0,0])[1]))
                    return closest_rows if closest_rows else None
                return None

            elif question_key == "localize_column_closest_piece":
                if num_pieces > 1:
                    closest_cols: list[str] = []
                    piece_values = list(pieces.values())
                    for i, piece in enumerate(piece_values):
                        current_pos = piece.get('board_position')
                        if not current_pos: continue

                        other_pieces_list = piece_values[:i] + piece_values[i+1:]
                        if not other_pieces_list: continue

                        closest_p = min(other_pieces_list, key=lambda p:
                            abs(p.get('board_position', [float('inf'), float('inf')])[0] - current_pos[0]) +
                            abs(p.get('board_position', [float('inf'), float('inf')])[1] - current_pos[1]))
                        closest_cols.append(str(closest_p.get('board_position', [0,0])[0]))
                    return closest_cols if closest_cols else None
                return None

            elif question_key == "identify_color_one_piece":
                if num_pieces == 1 and first_piece:
                    return first_piece.get('color', '').lower()
                return None

            elif question_key == "identify_type_one_piece":
                if num_pieces == 1 and first_piece:
                    return first_piece.get('type', '').lower()
                return None

            elif question_key == "identify_type_several_pieces":
                if num_pieces > 0:
                    return [p.get('type', '').lower() for p in pieces.values() if p.get('type')]
                return None

            elif question_key == "count_localization_pieces_on_row":
                board = data.get('board', {})
                rows = board.get('dimensions', {}).get('rows', 0)
                counts_per_row: dict[str, int] = {}
                for row_id in range(rows):
                    count = sum(1 for piece in pieces.values()
                                if piece.get('board_position', [-1, -1])[1] == row_id)
                    if count > 0:
                        counts_per_row[str(row_id)] = count
                return counts_per_row if counts_per_row else None

            elif question_key == "count_localization_pieces_on_column":
                board = data.get('board', {})
                columns = board.get('dimensions', {}).get('columns', 0)
                counts_per_col: dict[str, int] = {}
                for col_id in range(columns):
                    count = sum(1 for piece in pieces.values()
                                if piece.get('board_position', [-1, -1])[0] == col_id)
                    if count > 0:
                        counts_per_col[str(col_id)] = count
                return counts_per_col if counts_per_col else None

            elif question_key == "count_identification_white_pieces":
                return str(sum(1 for piece in pieces.values()
                               if piece.get('color', '').lower() == 'white'))

            elif question_key == "count_identification_black_pieces":
                return str(sum(1 for piece in pieces.values()
                               if piece.get('color', '').lower() == 'black'))

            elif question_key == "count_identification_piece_type":
                # If the original question is provided and contains a specific piece type
                if question:
                    import re
                    # Extract the specific piece type from the question
                    # Format would be like "How many pawn pieces are there on the board?"
                    match = re.search(r"how many (\w+) pieces", question.lower())
                    if match:
                        piece_type = match.group(1).lower()
                        return str(sum(1 for piece in pieces.values()
                                    if piece.get('type', '').lower() == piece_type))

                # Fall back to returning counts per piece type if no specific type in question
                counts: dict[str, int] = {}
                for piece in pieces.values():
                    p_type = piece.get('type', '').lower()
                    if p_type:
                        counts[p_type] = counts.get(p_type, 0) + 1
                return counts if counts else None

            elif question_key == "identify_localization_piece_color":
                colors_at_pos: dict[str, str] = {}
                for _piece_key, piece_value in pieces.items():
                    pos = piece_value.get('board_position')
                    color = piece_value.get('color', '').lower()
                    if pos and color:
                        colors_at_pos[f"{pos[0]},{pos[1]}"] = color
                return colors_at_pos if colors_at_pos else None

            elif question_key == "identify_localization_piece_type":
                types_at_pos: dict[str, str] = {}
                for _piece_key, piece_value in pieces.items():
                    pos = piece_value.get('board_position')
                    p_type = piece_value.get('type', '').lower()
                    if pos and p_type:
                        types_at_pos[f"{pos[0]},{pos[1]}"] = p_type
                return types_at_pos if types_at_pos else None

            elif question_key == "count_localization_pieces_row":
                board = data.get('board', {})
                rows = board.get('dimensions', {}).get('rows', 0)
                counts: dict[str, dict[str, int]] = {}
                for row_id in range(rows):
                    types_in_row: dict[str, int] = {}
                    for piece in pieces.values():
                        if piece.get('board_position', [-1,-1])[1] == row_id:
                            p_type = piece.get('type', '').lower()
                            if p_type:
                                types_in_row[p_type] = types_in_row.get(p_type, 0) + 1
                    if types_in_row:
                        counts[str(row_id)] = types_in_row
                return counts if counts else None

            elif question_key == "count_localization_pieces_column":
                board = data.get('board', {})
                columns = board.get('dimensions', {}).get('columns', 0)
                counts: dict[str, dict[str, int]] = {}
                for col_id in range(columns):
                    types_in_col: dict[str, int] = {}
                    for piece in pieces.values():
                        if piece.get('board_position', [-1,-1])[0] == col_id:
                            p_type = piece.get('type', '').lower()
                            if p_type:
                                types_in_col[p_type] = types_in_col.get(p_type, 0) + 1
                    if types_in_col:
                        counts[str(col_id)] = types_in_col
                return counts if counts else None

            elif question_key == "count_identification_localization_white_pieces_row":
                board = data.get('board', {})
                rows = board.get('dimensions', {}).get('rows', 0)
                counts_per_row: dict[str, int] = {}
                for row_id in range(rows):
                    count = sum(1 for piece in pieces.values()
                                if piece.get('color', '').lower() == 'white' and \
                                   piece.get('board_position', [-1, -1])[1] == row_id)
                    if count > 0:
                        counts_per_row[str(row_id)] = count
                return counts_per_row if counts_per_row else None

            elif question_key == "count_identification_localization_black_pieces_column":
                board = data.get('board', {})
                columns = board.get('dimensions', {}).get('columns', 0)
                counts_per_col: dict[str, int] = {}
                for col_id in range(columns):
                    count = sum(1 for piece in pieces.values()
                                if piece.get('color', '').lower() == 'white' and \
                                   piece.get('board_position', [-1, -1])[0] == col_id)
                    if count > 0:
                        counts_per_col[str(col_id)] = count
                return counts_per_col if counts_per_col else None

            else:
                return None

        elif game_type == "poker":
            players = data.get("players", [])
            community = data.get("community_cards", {})
            card_overlap = data.get("card_overlap_layout")

            if question_key == "count_total_cards" or question_key == "count_cards_overlap":
                n_community = community.get("n_cards", 0) if community else 0
                n_player_cards = sum(player.get("hand_config", {}).get("n_cards", 0) for player in players)
                n_overlap_cards = card_overlap.get("overall_cards", 0) if card_overlap else 0
                return str(n_community + n_player_cards + n_overlap_cards)

            elif question_key == "count_table_chip_piles" or question_key == "count_chip_piles_per_player":
                return None

            elif question_key == "count_players":
                return str(len(players))

            elif question_key == "count_cards_per_player":
                if not players: return None
                return [str(player.get("hand_config", {}).get("n_cards", 0)) for player in players]

            elif question_key == "count_community_cards":
                return str(community.get("n_cards", 0))

            elif question_key == "identify_cards" or question_key == "identify_community_cards":
                return community.get("card_names", [])

            elif question_key == "count_identify_face_up_cards":
                n_faceup = community.get("n_cards", 0) - community.get("n_verso", 0)
                return str(max(n_faceup, 0))

            elif question_key == "count_identify_face_down_cards":
                return str(community.get("n_verso", 0))

            elif question_key == "identify_most_cards_player":
                if not players: return None
                max_cards = -1
                max_player_id = None
                for player in players:
                    n_cards = player.get("hand_config", {}).get("n_cards", 0)
                    if n_cards > max_cards:
                        max_cards = n_cards
                        max_player_id = player.get("player_id", "Unknown")
                return str(max_player_id) if max_player_id is not None else None

            elif question_key == "count_identification_color_cards":
                # Extract the suit/color from the question
                if question:
                    import re
                    match = re.search(r"how many (hearts|diamonds|clubs|spades) cards", question.lower())
                    if match:
                        target_suit = match.group(1).capitalize()
                        # Map full suit name to its single letter code
                        suit_map = {"Hearts": "H", "Diamonds": "D", "Clubs": "C", "Spades": "S"}
                        suit_code = suit_map.get(target_suit, "")

                        # Count cards with this suit in community cards
                        if suit_code:
                            card_names = community.get("card_names", [])
                            count = sum(1 for card in card_names if card and card[-1] == suit_code)
                            return str(count)
                return "0"  # Default to 0 if no match

            elif question_key == "count_identification_rank_cards":
                # Extract the rank from the question
                if question:
                    import re
                    match = re.search(r"how many (\w+|\d+) cards", question.lower())
                    if match:
                        target_rank = match.group(1).upper()

                        # Count cards with this rank in community cards
                        card_names = community.get("card_names", [])
                        count = sum(1 for card in card_names if card and card[:-1] == target_rank)
                        return str(count)
                return "0"  # Default to 0 if no match

            elif question_key == "count_identification_type_cards":
                # Extract the card type from the question (Ace, King, etc.)
                if question:
                    import re
                    match = re.search(r"how many (ace|king|queen|jack|\d+) cards", question.lower())
                    if match:
                        target_type = match.group(1).lower()

                        # Map card type names to their codes
                        type_map = {"ace": "A", "king": "K", "queen": "Q", "jack": "J", "10": "T"}
                        type_code = type_map.get(target_type, target_type)

                        # Count cards with this type in community cards
                        card_names = community.get("card_names", [])
                        count = sum(1 for card in card_names if card and card[0] == type_code)
                        return str(count)
                return "0"  # Default to 0 if no match

            elif question_key in ["localize_card_on_grid_row", "localize_card_on_grid_column",
                                 "localize_card_on_grid_number", "localize_card_on_grid_3x3"]:
                # These questions apply to images with a single card

                # Look for card in card_grid_locations
                card_grid_locations = data.get("card_grid_locations", {})

                # Find the card's grid location - for images with a single card
                grid_info = None
                card_id = None

                # First try card_grid_locations
                if card_grid_locations:
                    if len(card_grid_locations) == 1:
                        # There's exactly one card, use it
                        card_id = next(iter(card_grid_locations.keys()))
                        grid_info = card_grid_locations[card_id]
                    else:
                        # Multiple cards in grid_locations, use the first one
                        card_id = next(iter(card_grid_locations.keys()))
                        grid_info = card_grid_locations[card_id]

                # If not found in card_grid_locations, check players
                if not grid_info:
                    for player in players:
                        player_grid_locations = player.get("grid_locations", {})
                        if player_grid_locations:
                            if len(player_grid_locations) == 1:
                                # Player has exactly one card
                                card_id = next(iter(player_grid_locations.keys()))
                                grid_info = player_grid_locations[card_id]
                                break
                            else:
                                # Player has multiple cards, use the first one
                                card_id = next(iter(player_grid_locations.keys()))
                                grid_info = player_grid_locations[card_id]
                                break

                if not grid_info:
                    return None

                if "center_grid_cell" not in grid_info:
                    return None

                center_cell = grid_info["center_grid_cell"]

                # Extract row and column from center_cell - maintaining [row, col] format
                row = int(center_cell[0]) if isinstance(center_cell[0], int | float) else 0
                col = int(center_cell[1]) if isinstance(center_cell[1], int | float) else 0

                if question_key == "localize_card_on_grid_row":
                    # Return row (0-indexed as specified)
                    result = str(row)
                    return result

                elif question_key == "localize_card_on_grid_column":
                    # Return column (0-indexed as specified)
                    result = str(col)
                    return result

                elif question_key == "localize_card_on_grid_number":
                    # For grid cell number, compute based on row and column
                    # Numbering from left to right and bottom to top as specified in the question

                    # Get grid dimensions
                    grid_dims = data.get("scene_setup", {}).get("grid", {}).get("granularity", 3)

                    # Calculate cell number (0-indexed)
                    # In the question: "numbering from left to right and bottom to top"
                    # With bottom left as (0,0), this is already correct orientation
                    cell_number = row * grid_dims + col
                    return str(cell_number)

                elif question_key == "localize_card_on_grid_3x3":
                    # Map grid coordinates to position words
                    # Using the coordinate system where bottom left is (0,0) and upper right is (3,3)

                    # Grid positions matching the orientation where bottom left is (0,0)
                    # and rows increase upward, columns increase rightward
                    position_map = {
                        # Bottom row (row=0)
                        (0, 0): "bottom left",
                        (0, 1): "bottom middle",
                        (0, 2): "bottom right",

                        # Middle row (row=1)
                        (1, 0): "middle left",
                        (1, 1): "middle",
                        (1, 2): "middle right",

                        # Upper row (row=2)
                        (2, 0): "upper left",
                        (2, 1): "upper middle",
                        (2, 2): "upper right"
                    }

                    # Get result from map, with bounds checking to avoid out-of-range indices
                    if 0 <= row <= 2 and 0 <= col <= 2:
                        result = position_map.get((row, col), "unknown position")
                    else:
                        # Handle the case where indices are out of the 3x3 grid range
                        # Map to the closest valid position
                        valid_row = max(0, min(row, 2))
                        valid_col = max(0, min(col, 2))
                        result = position_map.get((valid_row, valid_col), "unknown position")

                    return result

            else:
                return None
        else:
            return None

if __name__ == '__main__':
    extractor = AnswerExtractor()
    sample_chess_data = {
        "pieces": {
            "piece1": {"type": "rook", "color": "white", "board_position": [0, 0]},
            "piece2": {"type": "pawn", "color": "black", "board_position": [1, 1]},
            "piece3": {"type": "pawn", "color": "white", "board_position": [0, 2]},
        },
        "board": {"dimensions": {"rows": 8, "columns": 8}}
    }
    print("--- CHESS ---")
    chess_keys = [
        "count_pieces", "identify_type_several_pieces",
        "count_localization_pieces_on_row", "identify_localization_piece_color",
        "count_identification_localization_black_pieces_column"
    ]
    for key in chess_keys:
        ans = extractor.extract_answer(sample_chess_data, key, "chess")
        print(f"Q: {key} -> A: {ans}")

    sample_poker_data = {
        "community_cards": {
            "n_cards": 3,
            "n_verso": 1,
            "card_names": ["Ace of Spades", "King of Hearts", "Queen of Diamonds"]
        },
        "players": [
            {"player_id": "player1", "hand_config": {"n_cards": 2}},
            {"player_id": "player2", "hand_config": {"n_cards": 5}}
        ],
        "card_overlap_layout": {
            "layout_id": None,
            "layout_mode": "horizontal",
            "overall_cards": 6,
            "n_lines": 3,
            "n_columns": 1,
            "card_type_config": {
                "mode": "full_deck",
                "allow_repetition": False
            },
            "center_location": [0.0, 0.0, 0.91],
            "horizontal_overlap_factor": 0.5,
            "vertical_overlap_factor": 0.5,
            "line_gap": 0.15,
            "column_gap": 0.1,
            "n_verso": 0,
            "verso_loc": "random"
        }
    }
    print("\n--- POKER ---") # Corrected newline escape
    poker_keys = [
        "count_total_cards", "identify_community_cards",
        "identify_most_cards_player", "count_cards_per_player",
        "count_overlap_cards", "identify_overlap_layout_mode",
        "count_overlap_lines", "count_overlap_columns",
        "get_overlap_factors"
    ]
    for key in poker_keys:
        ans = extractor.extract_answer(sample_poker_data, key, "poker")
        print(f"Q: {key} -> A: {ans}")
