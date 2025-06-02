import json
import re  # For more complex string manipulations if needed
import sys

preprompts = {
    "chess":{
        "debiased": "This is not a real chess game. The number of each piece and their position can vary arbitrary. Just focus on answering the following question based on the visual content.",
        "cot": "First, think carefully, step by step, about the question being asked and the relevant elements of the image to which the question refers. End you message with {answer : <answer>}",
        "debiased_cot": "This is not a real chess game. The number of each piece and their position can vary arbitrary. Just focus on answering the following question based on the visual content. First, think carefully, step by step, about the question being asked and the relevant elements of the image to which the question refers. End you message with {answer : <answer>}",
},
    "poker":{
        "debiased":"This is not a real poker chess game. The number of each card and their position can vary arbitrary. Just focus on answering the following question based on the visual content. ",
        "cot":"First, think carefully, step by step, about the question being asked and the relevant elements of the image to which the question refers. End you message with {answer : <answer>}",
        "debiased_cot": "This is not a real poker chess game. The number of each card and their position can vary arbitrary. Just focus on answering the following question based on the visual content. First, think carefully, step by step, about the question being asked and the relevant elements of the image to which the question refers. End you message with {answer : <answer>}",
    }
}


class QuestionHandler:
    """
    A class to handle identification, conversion, and reformulation of questions
    based on a predefined JSON structure of questions.
    """

    def __init__(self, all_questions_path: str = "diagnostic/questions/all_questions.json"):
        """
        Initializes the QuestionConverter.

        Args:
            all_questions_path: Path to the JSON file containing all questions.
        """
        self.all_questions_path: str = all_questions_path
        self.all_questions_data: dict | None = self._load_json_data(self.all_questions_path)
        if self.all_questions_data is None:
            print(f"Warning: Could not load questions from {self.all_questions_path}. "
                  "Converter may not function correctly.", file=sys.stderr)

        # Direct mapping of question keys to their declarative forms
        self.declarative_instructions: dict[str, dict[str, str]] = {
            "chess": {
                "count_pieces": "The number of pieces in the image is:",
                "count_squares": "The number of squares the board has is:",
                "localize_column_one_piece": "The column on which the piece is on the board is:",
                "localize_row_one_piece": "The row on which the piece is on the board is:",
                "localize_rows_between_two_pieces": "The distance in rows between the two pieces on the board is:",
                "localize_columns_between_two_pieces": "The distance in columns between the two pieces on the board is:",
                "localize_row_closest_piece": "The row of the closest piece to the piece at position X,Y is:",
                "localize_column_closest_piece": "The column of the closest piece to the piece at position X,Y is:",
                "identify_color_one_piece": "The color of the piece on the board is:",
                "identify_type_one_piece": "The piece on the board is:",
                "identify_type_several_pieces": "The pieces on the board are:",
                "count_localization_pieces_on_row": "The number of pieces on the row X is:",
                "count_localization_pieces_on_column": "The number of pieces on the column X is:",
                "count_identification_white_pieces": "The number of white pieces on the board is:",
                "count_identification_black_pieces": "The number of black pieces on the board is:",
                "count_identification_piece_type": "The number of X pieces on the board is:",
                "identify_localization_piece_color": "The color of the piece on the board is:",
                "identify_localization_piece_type": "The value of the piece on the board is:",
                "count_localization_pieces_row": "The number of X pieces on the board at row Y is:",
                "count_localization_pieces_column": "The number of X pieces on the board at column Y is:",
                "count_identification_localization_white_pieces_row": "The number of white pieces on the board at row X is:",
                "count_identification_localization_black_pieces_column": "The number of white pieces on the board at column X is:"
            },
            "poker": {
                "count_total_cards": "The number of cards present in the entire scene (including all hands and community cards) is:",
                "count_cards_overlap": "The number of cards on the table is:",
                "count_table_chip_piles": "The number of chip piles present on the table is:",
                "count_chip_piles_per_player": "The number of chip piles each player has is:",
                "count_players": "The number of players present at the table is:",
                "count_cards_per_player": "The number of cards each player has in their hand is:",
                "count_community_cards": "The number of community cards visible on the table is:",
                "identify_cards": "The cards on the table are:",
                "identify_community_cards": "The names of the community cards shown are:",
                "count_identify_face_up_cards": "The number of cards that are face up on the table is:",
                "count_identify_face_down_cards": "The number of cards that are face down on the table is:",
                "identify_most_cards_player": "The player who has the most cards in their hand is:",
                "count_identification_color_cards": "The number of X cards on the table is:",
                "count_identification_rank_cards": "The number of X cards on the table is:",
                "count_identification_type_cards": "The number of X cards on the table is:",
                "localize_card_on_grid_number": "The cell number of the grid where card X is located is:",
                "localize_card_on_grid_row": "The row number of the grid where card X is located is:",
                "localize_card_on_grid_column": "The column number of the grid where card X is located is:",
                "localize_card_on_grid_3x3": "The position in the 3x3 grid where card X is located is:",
            }
        }

        # Direct mapping of question keys to their fill-in-the-blank forms
        self.fill_in_blank_instructions: dict[str, dict[str, str]] = {
            "chess": {
                "count_pieces": "There are ____ pieces in the image.",
                "count_squares": "The board has ____ squares.",
                "localize_column_one_piece": "The piece is on column ____.",
                "localize_row_one_piece": "The piece is on row ____.",
                "localize_rows_between_two_pieces": "There is a distance of ____ rows between the two pieces on the board.",
                "localize_columns_between_two_pieces": "There is a distance ____ columns separating the two pieces on the board.",
                "localize_row_closest_piece": "The closest piece to the one at position X,Y is on row ____.",
                "localize_column_closest_piece": "The closest piece to the one at position X,Y is on column ____.",
                "identify_color_one_piece": "The piece on the board is ____ in color.",
                "identify_type_one_piece": "The piece on the board is a ____.",
                "identify_type_several_pieces": "The board contains these pieces: ____.",
                "count_localization_pieces_on_row": "There are ____ pieces on row X.",
                "count_localization_pieces_on_column": "There are ____ pieces on column X.",
                "count_identification_white_pieces": "There are ____ white pieces on the board.",
                "count_identification_black_pieces": "There are ____ black pieces on the board.",
                "count_identification_piece_type": "There are ____ X pieces on the board.",
                "identify_localization_piece_color": "The piece at position X,Y is ____ in color.",
                "identify_localization_piece_type": "The piece at position X,Y has value ____.",
                "count_localization_pieces_row": "There are ____ X pieces on row Y.",
                "count_localization_pieces_column": "There are ____ X pieces on column Y.",
                "count_identification_localization_white_pieces_row": "There are ____ white pieces on row X.",
                "count_identification_localization_black_pieces_column": "There are ____ white pieces on column X."
            },
            "poker": {
                "count_total_cards": "There are ____ cards in the entire scene (including all hands and community cards).",
                "count_table_chip_piles": "There are ____ chip piles on the table.",
                "count_chip_piles_per_player": "Each player has ____ chip piles.",
                "count_players": "There are ____ players at the table.",
                "count_cards_per_player": "Each player has ____ cards in their hand.",
                "count_community_cards": "There are ____ community cards visible on the table.",
                "identify_cards": "The table has these cards: ____.",
                "identify_community_cards": "The community cards shown are: ____.",
                "count_identify_face_up_cards": "There are ____ cards face up on the table.",
                "count_identify_face_down_cards": "There are ____ cards face down on the table.",
                "identify_most_cards_player": "The player with the most cards is ____.",
                "count_identification_color_cards": "There are ____ X cards on the table.",
                "count_identification_rank_cards": "There are ____ X cards on the table.",
                "count_identification_type_cards": "There are ____ X cards on the table.",
                "localize_card_on_grid_number": "Card X is on cell number ____ of the grid.",
                "localize_card_on_grid_row": "Card X is on row ____ of the grid.",
                "localize_card_on_grid_column": "Card X is on column ____ of the grid.",
                "localize_card_on_grid_3x3": "Card X is in the ____ position of the 3x3 grid.",

            }
        }

    def _load_json_data(self, file_path: str) -> dict | None:
        """
        Loads data from a JSON file. (Internal method)

        Args:
            file_path: The path to the JSON file.

        Returns:
            A dictionary containing the JSON data, or None if an error occurs.
        """
        try:
            with open(file_path) as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Error: The file {file_path} was not found.", file=sys.stderr)
            return None
        except json.JSONDecodeError as e:
            print(f"Error: Could not decode JSON from {file_path}. Details: {e}", file=sys.stderr)
            return None

    def add_preprompt(self, questions: list[str], game: str, preprompt_types: list[str]) -> list[str]:
        """
        Adds preprompts to a list of questions based on the game and preprompt types.

        Args:
            questions: List of questions to add preprompts to.
            game: The game context for the questions ('chess' or 'poker').
            preprompt_types: List of preprompt types to add. Can include 'debiased',
                            'cot', 'debiased_cot', '' (empty string), or a custom prompt.

        Returns:
            A list of questions with preprompts added.
        """
        result = []

        for question in questions:
            modified_question = question

            for preprompt_type in preprompt_types:
                # Skip if preprompt_type is empty
                if not preprompt_type:
                    continue

                # Check if preprompt_type is a standard type
                if preprompt_type in ["debiased", "cot", "debiased_cot"]:
                    # Get the preprompt from the preprompts dictionary
                    if game in preprompts and preprompt_type in preprompts[game]:
                        preprompt = preprompts[game][preprompt_type]
                        modified_question = f"{preprompt} {modified_question}"
                else:
                    # If it's a custom preprompt, add it directly
                    modified_question = f"{preprompt_type} {modified_question}"

            result.append(modified_question)

        return result

    def _get_original_question_text(self, question_key: str, game: str, legend: str = None) -> str | None:
        """
        Retrieves the original question text for a given key and game.

        Args:
            question_key: The key of the question.
            game: The game associated with the question.
            legend: Optional JSON string containing game configuration details.

        Returns:
            The original question text, or None if not found or if data isn't loaded.
        """
        if self.all_questions_data is None:
            # This path should ideally be caught by the __init__ warning,
            # but good to have a check here too.
            # print(f"Error: Question data not loaded. Cannot get original text for '{question_key}' in game '{game}'.", file=sys.stderr)
            return None
        try:
            game_questions: dict[str, str] = self.all_questions_data[game]
            original_question: str = game_questions[question_key]

            # Special case for chess and poker games when legend is provided
            if legend and "X" in original_question:
                import json
                import random

                if game == "chess" and question_key == "count_identification_piece_type":
                    try:
                        legend_data = json.loads(legend) if isinstance(legend, str) else legend
                        pieces = legend_data.get("pieces", {})
                        # Extract all unique piece types from the legend
                        piece_types = {piece.get("type") for piece in pieces.values() if "type" in piece}
                        if piece_types:
                            # Replace X with a random piece type from the legend
                            random_type = random.choice(list(piece_types))
                            original_question = original_question.replace("X", random_type)
                    except (json.JSONDecodeError, AttributeError, TypeError) as e:
                        print(f"Error processing chess legend for question '{question_key}': {e}", file=sys.stderr)

                elif game == "poker" and any(key in question_key for key in ["count_identification_color_cards", "count_identification_rank_cards", "count_identification_type_cards"]):
                    try:
                        legend_data = json.loads(legend) if isinstance(legend, str) else legend
                        community_cards = legend_data.get("community_cards", {})
                        card_names = community_cards.get("card_names", [])

                        if "count_identification_color_cards" in question_key and card_names:
                            # For poker suits (colors): Hearts (H), Diamonds (D), Clubs (C), Spades (S)
                            suits = {card[-1] for card in card_names if card and len(card) >= 1}
                            if suits:
                                # Map single letter to full suit name
                                suit_names = {"H": "Hearts", "D": "Diamonds", "C": "Clubs", "S": "Spades"}
                                random_suit = random.choice(list(suits))
                                suit_name = suit_names.get(random_suit, random_suit)
                                original_question = original_question.replace("X", suit_name)

                        elif "count_identification_rank_cards" in question_key and card_names:
                            # Extract ranks from card names (e.g., "2S" -> "2", "10H" -> "10", "QC" -> "Q")
                            ranks = {card[:-1] for card in card_names if card and len(card) >= 2}
                            if ranks:
                                random_rank = random.choice(list(ranks))
                                original_question = original_question.replace("X", random_rank)

                        elif "count_identification_type_cards" in question_key and card_names:
                            # In poker context, "type" might refer to face cards or numerical cards
                            card_types = {"A": "Ace", "K": "King", "Q": "Queen", "J": "Jack", "T": "10"}
                            # Extract first character of each card name
                            types = {card[0] for card in card_names if card}
                            if types:
                                random_type = random.choice(list(types))
                                type_name = card_types.get(random_type, random_type)
                                original_question = original_question.replace("X", type_name)
                    except (json.JSONDecodeError, AttributeError, TypeError) as e:
                        print(f"Error processing poker legend for question '{question_key}': {e}", file=sys.stderr)

            return original_question
        except KeyError:
            # print(f"Error: Game '{game}' or question key '{question_key}' not found in loaded data.", file=sys.stderr)
            return None

    def identify_question_type(self, question_key: str, game: str) -> str | None:
        """
        Identifies the type of a question based on its key.
        Types can be 'count', 'identification', 'localization', or combinations.

        Args:
            question_key: The key of the question.
            game: The game of the question.

        Returns:
            A string representing the identified question type (e.g., "count",
            "identification_localization", "unknown"), or None if data isn't loaded
            or game/key is not found.
        """
        if self.all_questions_data is None:
            return None # Data loading failed

        try:
            game_questions: dict[str, str] = self.all_questions_data[game]
        except KeyError:
            # print(f"Error: Game '{game}' not found in {self.all_questions_path}.", file=sys.stderr)
            return None # Game not found

        if question_key not in game_questions:
            # This print matches the original behavior for a missing key.
            print(f"Error: Question key '{question_key}' not found in game '{game}' in {self.all_questions_path}.", file=sys.stderr)
            return None # Question key not found in the specified game

        type_prefixes: dict[str, str] = {
            "count_identification_localization": "count_identification_localization",
            "count_identification": "count_identification",
            "count_localization": "count_localization",
            "identify_localization": "identify_localization",
            "count": "count",
            "identify": "identification",
            "localize": "localization",
        }

        for prefix, q_type in type_prefixes.items():
            if question_key.startswith(prefix):
                return q_type

        return "unknown" # Key exists, but no prefix matches

    def question_to_declarative(self, question_key: str, game: str, original_question: str = None, legend: str = None) -> str | None:
        """
        Returns a declarative form of the question using direct mapping.

        Args:
            question_key: The key of the question.
            game: The game of the question.
            original_question: The original question text with any substitutions already applied.
            legend: Optional JSON string containing game configuration details.

        Returns:
            The declarative form of the question, or None if an error occurs.
        """
        if self.all_questions_data is None:
            return None

        # Check if the game exists
        if game not in self.declarative_instructions:
            print(f"Error: Game '{game}' not found in declarative_instructions mapping.", file=sys.stderr)
            return None

        # Get the declarative instruction for this key
        if question_key in self.declarative_instructions[game]:
            declarative_template = self.declarative_instructions[game][question_key]

            # If we have both an original question with substitutions and the template contains "X"
            if original_question and "X" in declarative_template:
                # For chess: extract the piece type from the original question
                if game == "chess" and question_key == "count_identification_piece_type":
                    import re
                    match = re.search(r"how many (\w+) pieces", original_question.lower())
                    if match:
                        piece_type = match.group(1).lower()
                        declarative_template = declarative_template.replace("X", piece_type)

                # For poker: extract the card attribute from the original question
                elif game == "poker" and any(key in question_key for key in ["count_identification_color_cards", "count_identification_rank_cards", "count_identification_type_cards"]):
                    import re
                    if "color_cards" in question_key:
                        match = re.search(r"how many (hearts|diamonds|clubs|spades) cards", original_question.lower())
                        if match:
                            declarative_template = declarative_template.replace("X", match.group(1))
                    elif "rank_cards" in question_key:
                        match = re.search(r"how many (\w+|\d+) cards", original_question.lower())
                        if match:
                            declarative_template = declarative_template.replace("X", match.group(1))
                    elif "type_cards" in question_key:
                        match = re.search(r"how many (ace|king|queen|jack|\d+) cards", original_question.lower())
                        if match:
                            declarative_template = declarative_template.replace("X", match.group(1))

            return declarative_template

        # Fall back to old method if key doesn't exist in mapping
        if not original_question:
            original_question = self._get_original_question_text(question_key, game, legend)
        if original_question is None:
            return None

        identified_type = self.identify_question_type(question_key, game)
        if identified_type is None:
            return None

        # SHOULD NOT BE USED. Use declarative_instructions instead.
        return self._generate_declarative_statement(original_question, identified_type)

    # DEPRECATED: Use declarative_instructions instead.
    def _generate_declarative_statement(self, original_question: str, question_type: str) -> str:
        """
        DEPRECATED: Use declarative_instructions instead.
        """
        # Remove trailing question mark if present
        if original_question.endswith("?"):
            statement_base = original_question[:-1].strip()
        else:
            statement_base = original_question.strip()

        # General transformations (can be refined based on patterns)
        if question_type == "count":
            if statement_base.lower().startswith("how many"):
                transformed = re.sub(r"^how many\\s+(.*?)\\s+(?:are there|are|has)\\b",
                                     r"The number of \\1",
                                     statement_base, flags=re.IGNORECASE)
                if transformed != statement_base: # Check if substitution happened
                    if " has" in statement_base.lower() and " the board has" not in transformed.lower():
                        transformed = transformed.replace(" has", "")
                        transformed = transformed.replace("the board", "the board has")
                    return f"{transformed.strip()} is:"
            return f"The count for '{statement_base}' is:"
        elif question_type == "identification":
            if statement_base.lower().startswith("what piece is on the board"):
                return "The piece on the board (among pawn, rook, knight, bishop, king and queen) is:"
            if statement_base.lower().startswith("what pieces are on the board"):
                return "The pieces on the board (among pawn, rook, knight, bishop, king and queen) are:"
            if statement_base.lower().startswith("what color is the piece on the board"):
                return "The color of the piece on the board is:"
            if statement_base.lower().startswith("what are the cards on the table"):
                 return "The cards on the table are:"
            if statement_base.lower().startswith("which player has the most cards"):
                return "The player with the most cards is:"
            return f"The identification for '{statement_base}' is:"
        elif question_type == "localization":
            if statement_base.lower().startswith("how many"):
                if " separate " in statement_base.lower():
                    transformed = re.sub(r"^how many\\s+(.*?)\\s+separate\\s+(.*?)$",
                                         r"The number of \\1 separating \\2",
                                         statement_base, flags=re.IGNORECASE)
                    if transformed != statement_base:
                        return f"{transformed.strip()} is:"
                else:
                    transformed = re.sub(r"^how many\\s+(.*?)\\s+(?:are there|are|has)\\b",
                                         r"The number of \\1",
                                         statement_base, flags=re.IGNORECASE)
                    if transformed != statement_base:
                        return f"{transformed.strip()} is:"
            match = re.match(r"^(.*?):COMMA:\\s*on which (column|row) is (.*?)$", statement_base.replace(",", ":COMMA:"), flags=re.IGNORECASE) # Temporarily replace comma
            if match:
                leading_text, col_row, subject = match.groups()
                leading_text = leading_text.replace(":COMMA:", ",")
                return f"{leading_text.strip()} the {col_row} on which {subject} is is:"
            return f"The localization for '{statement_base}' is:"
        elif question_type == "count_localization":
            if statement_base.lower().startswith("numbering") and "how many" in statement_base.lower():
                transformed = re.sub(r"how many\\s+(.+?)\\s+(?:is|are)\\s+(on the row X|on the column X|on row Y|at column Y|on the board at row Y|on the board at column Y)",
                                     r"the number of \\1 \\2",
                                     statement_base, flags=re.IGNORECASE)
                if transformed != statement_base:
                     return f"{transformed.strip()} is:"
            return f"The count and localization for '{statement_base}' is:"
        elif question_type == "count_identification":
            if statement_base.lower().startswith("how many") and ("white pieces" in statement_base.lower() or "black pieces" in statement_base.lower() or "x pieces" in statement_base.lower()):
                transformed = re.sub(r"^how many\\s+(.*?)\\s+(?:are there|are)\\b",
                                     r"The number of \\1",
                                     statement_base, flags=re.IGNORECASE)
                if transformed != statement_base:
                    return f"{transformed.strip()} is:"
            return f"The count and identification for '{statement_base}' is:"
        elif question_type == "identify_localization":
            if statement_base.lower().startswith("numbering") and "what color is the piece" in statement_base.lower():
                transformed = statement_base.replace("what color is the piece", "the color of the piece")
                return f"{transformed.strip()} is:"
            if statement_base.lower().startswith("numbering") and "what value corresponds to the piece" in statement_base.lower():
                transformed = statement_base.replace("what value corresponds to the piece", "the value that corresponds to the piece")
                return f"{transformed.strip()} is:"
            return f"The identification and localization for '{statement_base}' is:"
        if question_type == "unknown":
            return f"The statement for '{statement_base}' is:"
        type_name_formatted = question_type.replace("_", " ")
        return f"The {type_name_formatted} for '{statement_base}' is:"

    def question_to_fill_in_the_blanks(self, question_key: str, game: str, original_question: str = None, legend: str = None) -> str | None:
        """
        Returns a fill-in-the-blank form of the question using direct mapping.

        Args:
            question_key: The key of the question.
            game: The game of the question.
            original_question: The original question text with any substitutions already applied.
            legend: Optional JSON string containing game configuration details.

        Returns:
            The fill-in-the-blank form of the question, or None if an error occurs.
        """
        if self.all_questions_data is None:
            return None

        # Check if the game exists
        if game not in self.fill_in_blank_instructions:
            print(f"Error: Game '{game}' not found in fill_in_blank_instructions mapping.", file=sys.stderr)
            return None

        # Check if the key exists in the game
        if question_key in self.fill_in_blank_instructions[game]:
            fill_in_template = self.fill_in_blank_instructions[game][question_key]

            # If we have both an original question with substitutions and the template contains "X"
            if original_question and "X" in fill_in_template:
                # For chess: extract the piece type from the original question
                if game == "chess" and question_key == "count_identification_piece_type":
                    import re
                    match = re.search(r"how many (\w+) pieces", original_question.lower())
                    if match:
                        piece_type = match.group(1).lower()
                        fill_in_template = fill_in_template.replace("X", piece_type)

                # For poker: extract the card attribute from the original question
                elif game == "poker" and any(key in question_key for key in ["count_identification_color_cards", "count_identification_rank_cards", "count_identification_type_cards"]):
                    import re
                    if "color_cards" in question_key:
                        match = re.search(r"how many (hearts|diamonds|clubs|spades) cards", original_question.lower())
                        if match:
                            fill_in_template = fill_in_template.replace("X", match.group(1))
                    elif "rank_cards" in question_key:
                        match = re.search(r"how many (\w+|\d+) cards", original_question.lower())
                        if match:
                            fill_in_template = fill_in_template.replace("X", match.group(1))
                    elif "type_cards" in question_key:
                        match = re.search(r"how many (ace|king|queen|jack|\d+) cards", original_question.lower())
                        if match:
                            fill_in_template = fill_in_template.replace("X", match.group(1))

            # Return the fill-in-the-blank form with instruction to only fill in the blank
            return f"{fill_in_template} Write nothing more than the content to fill-in the blank"

        # Fall back to using the declarative form with a fill-in-the-blank suffix
        declarative_form = self.question_to_declarative(question_key, game, original_question, legend)
        if declarative_form is None:
            return None

        # Add the fill-in-the-blank suffix
        return f"{declarative_form} ____. Write nothing more than the content to fill-in the blank"

    # DEPRECATED: Use fill_in_blank_instructions instead.
    def _generate_fill_in_the_blank_statement(self, original_question: str, question_type: str) -> str:
        """
        DEPRECATED: Use fill_in_blank_instructions instead.
        """
        fill_in_suffix = "____. Write nothing more than the content to fill-in the blank"
        if original_question.endswith("?"):
            statement_base = original_question[:-1].strip()
        else:
            statement_base = original_question.strip()

        if question_type == "count":
            if statement_base.lower().startswith("how many"):
                transformed = re.sub(r"^how many\\s+(.*?)\\s+(?:are there|are|has)\\b",
                                     r"The number of \\1",
                                     statement_base, flags=re.IGNORECASE)
                if transformed != statement_base:
                    if " has" in statement_base.lower() and " the board has" not in transformed.lower():
                        transformed = transformed.replace(" has", "")
                        transformed = transformed.replace("the board", "the board has")
                    return f"{transformed.strip()} is {fill_in_suffix}"
            return f"The count for '{statement_base}' is {fill_in_suffix}"
        elif question_type == "identification":
            if statement_base.lower().startswith("what piece is on the board"):
                return f"The piece on the board (among pawn, rook, knight, bishop, king and queen) is {fill_in_suffix}"
            if statement_base.lower().startswith("what pieces are on the board"):
                return f"The pieces on the board (among pawn, rook, knight, bishop, king and queen) are {fill_in_suffix}"
            if statement_base.lower().startswith("what color is the piece on the board"):
                return f"The color of the piece on the board is {fill_in_suffix}"
            if statement_base.lower().startswith("what are the cards on the table"):
                return f"The cards on the table are {fill_in_suffix}"
            if statement_base.lower().startswith("which player has the most cards"):
                return f"The player with the most cards is {fill_in_suffix}"
            return f"The identification for '{statement_base}' is {fill_in_suffix}"
        elif question_type == "localization":
            if statement_base.lower().startswith("how many"):
                if " separate " in statement_base.lower():
                    transformed = re.sub(r"^how many\\s+(.*?)\\s+separate\\s+(.*?)$",
                                         r"The number of \\1 separating \\2",
                                         statement_base, flags=re.IGNORECASE)
                    if transformed != statement_base:
                        return f"{transformed.strip()} is {fill_in_suffix}"
                else:
                    transformed = re.sub(r"^how many\\s+(.*?)\\s+(?:are there|are|has)\\b",
                                         r"The number of \\1",
                                         statement_base, flags=re.IGNORECASE)
                    if transformed != statement_base:
                        return f"{transformed.strip()} is {fill_in_suffix}"
            match = re.match(r"^(.*?):COMMA:\\s*on which (column|row) is (.*?)$", statement_base.replace(",", ":COMMA:"), flags=re.IGNORECASE)
            if match:
                leading_text, col_row, subject = match.groups()
                leading_text = leading_text.replace(":COMMA:", ",")
                return f"{leading_text.strip()} the {col_row} on which {subject} is is {fill_in_suffix}"
            return f"The localization for '{statement_base}' is {fill_in_suffix}"
        elif question_type == "count_localization":
            if statement_base.lower().startswith("numbering") and "how many" in statement_base.lower():
                if "rows from top to bottom" in statement_base.lower() and "pieces on the row" in statement_base.lower():
                    return f"Numbering the rows from top to bottom, starting with 0, the number of pieces on the row X is {fill_in_suffix}"
                if "columns from left to right" in statement_base.lower() and "pieces on the column" in statement_base.lower():
                    return f"Numbering the columns from left to right, starting with 0, the number of pieces on the column X is {fill_in_suffix}"
                transformed = re.sub(r"how many\\s+(.+?)\\s+(?:is|are)\\s+(on the row X|on the column X|on row Y|at column Y|on the board at row Y|on the board at column Y)",
                                     r"the number of \\1 \\2",
                                     statement_base, flags=re.IGNORECASE)
                if transformed != statement_base:
                    return f"{transformed.strip()} is {fill_in_suffix}"
            return f"The count and localization for '{statement_base}' is {fill_in_suffix}"
        elif question_type == "count_identification":
            if statement_base.lower() == "how many white pieces are there on the board":
                return f"The number of white pieces on the board is {fill_in_suffix}"
            if statement_base.lower() == "how many black pieces are there on the board":
                return f"The number of black pieces on the board is {fill_in_suffix}"
            if statement_base.lower().startswith("how many") and ("white pieces" in statement_base.lower() or "black pieces" in statement_base.lower() or "x pieces" in statement_base.lower()):
                transformed = re.sub(r"^how many\\s+(.*?)\\s+(?:are there|are)\\b",
                                     r"The number of \\1",
                                     statement_base, flags=re.IGNORECASE)
                if transformed != statement_base:
                    return f"{transformed.strip()} is {fill_in_suffix}"
            return f"The count and identification for '{statement_base}' is {fill_in_suffix}"
        elif question_type == "identify_localization":
            if statement_base.lower().startswith("numbering") and "what color is the piece" in statement_base.lower():
                transformed = statement_base.replace("what color is the piece", "the color of the piece")
                return f"{transformed.strip()} is {fill_in_suffix}"
            if statement_base.lower().startswith("numbering") and "what value corresponds to the piece" in statement_base.lower():
                transformed = statement_base.replace("what value corresponds to the piece", "the value that corresponds to the piece")
                return f"{transformed.strip()} is {fill_in_suffix}"
            return f"The identification and localization for '{statement_base}' is {fill_in_suffix}"
        if question_type == "unknown":
            return f"The statement for '{statement_base}' is {fill_in_suffix}"
        type_name_formatted = question_type.replace("_", " ")
        return f"The {type_name_formatted} for '{statement_base}' is {fill_in_suffix}"

    def generate_questions(self, game: str, keys: list[str], instruction_specs: list[str], tasks: list[str] = None, preprompt_types: list[str] = None, img_legend: str = None) -> list[str]:
        """
        Generates questions based on keys or tasks, with instructions and preprompts.

        Args:
            game: The game context for the questions.
            keys: A list of question keys. If empty, tasks parameter will be used.
            instruction_specs: A list of instruction specifications to add at the end of questions.
                              Possible values:
                              - "": Default type-based instruction
                              - "neutral": Same as ""
                              - "declarative": Convert to declarative form
                              - "fill_in": Convert to fill-in-the-blank form
                              - Any custom text: Add as custom instruction
            tasks: A list of task types to generate questions for when keys is empty.
                  Possible values: "counting", "localization", "identification", or
                  cross-tasks with "_" separators.
            preprompt_types: A list of preprompt types to add. Can include 'debiased',
                            'cot', 'debiased_cot', '' (empty string), or a custom prompt.

        Returns:
            A list of final questions with preprompts and instructions added.
            For each key, generates len(instruction_specs) different versions.
        """
        # If both keys and preprompt_types are empty, make them empty lists
        if keys is None:
            keys = []
        if preprompt_types is None:
            preprompt_types = []
        if instruction_specs is None or len(instruction_specs) == 0:
            instruction_specs = [""]  # Default to empty spec

        # If keys is empty and tasks is provided, generate keys based on tasks
        if not keys and tasks:
            generated_keys = []

            for task in tasks:
                if task == "counting":
                    if game == "chess":
                        generated_keys.append("count_pieces")
                    elif game == "poker":
                        generated_keys.append("count_total_cards")
                    else:
                        print(f"Warning: Task '{task}' not implemented for game '{game}'", file=sys.stderr)

                elif task == "localization":
                    if game == "chess":
                        # Add both row and column localization questions
                        generated_keys.append("localize_row_one_piece")
                        generated_keys.append("localize_column_one_piece")
                    elif game == "poker":
                        # Add card localization questions
                        generated_keys.append("localize_card_on_grid_row")
                        generated_keys.append("localize_card_on_grid_column")
                        generated_keys.append("localize_card_on_grid_number")
                        generated_keys.append("localize_card_on_grid_3x3")
                    else:
                        print(f"Warning: Task '{task}' not implemented for game '{game}'", file=sys.stderr)

                elif task == "identification":
                    if game == "chess":
                        generated_keys.append("identify_type_one_piece")
                    elif game == "poker":
                        generated_keys.append("identify_cards")
                    else:
                        print(f"Warning: Task '{task}' not implemented for game '{game}'", file=sys.stderr)

                elif "_" in task:
                    # Cross-task (e.g., "counting_localization")
                    print(f"Warning: Cross-task '{task}' not fully implemented yet", file=sys.stderr)

                else:
                    print(f"Warning: Unknown task '{task}'", file=sys.stderr)

            keys = generated_keys

        # Step 1: Generate questions with instructions for all keys
        questions_with_instructions = []

        for key in keys:
            original_question_text = self._get_original_question_text(key, game, img_legend)
            if original_question_text is None:
                print(f"Warning: Question key '{key}' not found in game '{game}'")
                continue  # Skip this question if the key is not found

            # Get the base question text without trailing question mark
            base_question = original_question_text.rstrip("?").strip()

            # For each key, apply all instruction specs
            for spec in instruction_specs:
                if spec == "" or spec == "neutral":
                    # Add type-based instruction
                    question_type = self.identify_question_type(key, game)
                    if question_type is None:
                        questions_with_instructions.append(original_question_text)  # Use original as fallback
                        continue

                    instruction = ""
                    if question_type == "count":
                        instruction = "? Answer with a single number."
                    elif question_type == "identification":
                        instruction = "? Answer with a single word."
                    elif question_type == "localization":
                        # Special handling for the 3x3 grid localization in poker
                        if game == "poker" and key == "localize_card_on_grid_3x3":
                            instruction = "? Answer with a position word (e.g., upper left, middle right, etc.)."
                        else:
                            instruction = "? Answer with a single number."
                    elif question_type == "count_identification":
                        instruction = "? Answer with a single number."
                    elif question_type == "count_localization":
                        instruction = "? Answer with a single number"
                    elif question_type == "identify_localization":
                        instruction = "? Answer with either a single word or a single number or tuple."
                    elif question_type == "count_identification_localization":
                        instruction = "? Answer with a single number."
                    elif question_type == "unknown":
                        instruction = "? Concise answer :"  # Default for unknown
                    else:  # Should not be reached if identify_question_type is comprehensive
                        instruction = "? Concise answer :"

                    questions_with_instructions.append(f"{base_question}{instruction}")

                elif spec == "declarative":
                    # Add declarative format instruction instead of replacing the question
                    declarative_form = self.question_to_declarative(key, game, original_question_text, img_legend)
                    if declarative_form:
                        questions_with_instructions.append(f"{base_question}? Respond in a declarative format: '{declarative_form}'")
                    else:
                        # Fall back to original question with generic instruction if declarative form not available
                        questions_with_instructions.append(f"{base_question}? Respond with a declarative statement.")

                elif spec == "fill_in":
                    # Add fill-in-the-blank instruction instead of replacing the question
                    fill_in_form = self.question_to_fill_in_the_blanks(key, game, original_question_text, img_legend)
                    if fill_in_form:
                        # Extract just the fill-in template without the "Write nothing more" part
                        fill_in_template = fill_in_form.split("Write nothing")[0].strip()
                        questions_with_instructions.append(f"{base_question}? Complete this blank: {fill_in_template}")
                    else:
                        # Fall back to original question with generic instruction if fill-in form not available
                        questions_with_instructions.append(f"{base_question}? Fill in the blank with your answer.")

                else:
                    # If spec is a custom instruction, append it to the question
                    questions_with_instructions.append(f"{base_question}? {spec}")

        # Step 2: Add preprompts to questions with instructions
        if preprompt_types:
            final_questions = self.add_preprompt(questions_with_instructions, game, preprompt_types)
        else:
            final_questions = questions_with_instructions

        return final_questions


if __name__ == "__main__":
    # Create a formatted test output function
    def print_test_section(title):
        print(f"\n{'=' * 50}")
        print(f"  {title}")
        print(f"{'=' * 50}")

    # Initialize the QuestionHandler
    print_test_section("INITIALIZING QuestionHandler")
    handler = QuestionHandler("diagnostic/questions/all_questions.json")

    if handler.all_questions_data is None:
        print("❌ ERROR: Could not load questions from diagnostic/questions/all_questions.json")
        print("    Please make sure the file exists and contains valid JSON.")
        exit(1)
    else:
        print("✅ Successfully loaded questions data")
        print(f"    Found {len(handler.all_questions_data)} game(s)")

        # Print a sample of the games and questions
        print("\nGames available:")
        for game, questions in handler.all_questions_data.items():
            print(f"  - {game}: {len(questions)} questions")

        # Choose a game for testing (use the first one available)
        test_game = list(handler.all_questions_data.keys())[0]
        sample_questions = list(handler.all_questions_data[test_game].keys())[:3]

        print(f"\nSample questions from '{test_game}':")
        for q_key in sample_questions:
            q_text = handler._get_original_question_text(q_key, test_game)
            print(f"  - {q_key}: {q_text}")

    # Test question type identification
    print_test_section("TESTING identify_question_type()")
    for q_key in sample_questions:
        q_type = handler.identify_question_type(q_key, test_game)
        print(f"Question key: {q_key}")
        print(f"Type identified: {q_type}")
        print("---")

    # Test question to declarative conversion
    print_test_section("TESTING question_to_declarative()")
    for q_key in sample_questions:
        original = handler._get_original_question_text(q_key, test_game)
        declarative = handler.question_to_declarative(q_key, test_game)
        print(f"Original: {original}")
        print(f"Declarative: {declarative}")
        print("---")

    # Test question to fill-in-the-blank conversion
    print_test_section("TESTING question_to_fill_in_the_blanks()")
    for q_key in sample_questions:
        original = handler._get_original_question_text(q_key, test_game)
        blank = handler.question_to_fill_in_the_blanks(q_key, test_game)
        print(f"Original: {original}")
        print(f"Fill-in-blank: {blank}")
        print("---")

    # Test reformulation with multiple specs
    print_test_section("TESTING generate_questions with multiple instruction specs")

    # Use different types of instruction specs
    multiple_specs = ["", "declarative", "fill_in", "Answer concisely with a single word."]

    # Get a single question key for clarity
    test_key = sample_questions[0] if sample_questions else None
    if test_key:
        print(f"\nGenerating multiple versions of question key: {test_key}")
        original = handler._get_original_question_text(test_key, test_game)
        print(f"Original question: {original}")

        # Generate multiple versions with different instruction specs
        multiple_versions = handler.generate_questions(test_game, [test_key], multiple_specs)

        print(f"\nGenerated {len(multiple_versions)} versions:")
        for i, version in enumerate(multiple_versions):
            print(f"  Version {i+1} ({multiple_specs[i]}): {version}")

    # Test with preprompts
    print("\nAdding preprompts to multiple instruction specs:")
    if test_key:
        versions_with_preprompt = handler.generate_questions(
            test_game, [test_key], multiple_specs, preprompt_types=["debiased"]
        )

        for i, version in enumerate(versions_with_preprompt):
            print(f"  Version {i+1} ({multiple_specs[i]} + debiased): {version}")

    # Test task-based question generation with multiple instruction types
    print_test_section("TESTING tasks-based generation with multiple instruction specs")

    # Generate questions for the "counting" task with multiple instruction types
    for game in ["chess", "poker"]:
        if game not in handler.all_questions_data:
            continue

        print(f"\nTask-based generation for {game} with multiple instruction types:")
        task_questions = handler.generate_questions(
            game, [], ["", "declarative", "fill_in"], ["counting"]
        )

        print(f"Generated {len(task_questions)} questions:")
        for i, question in enumerate(task_questions):
            print(f"  Question {i+1}: {question}")

    # Test the full combination of tasks, instruction specs, and preprompts
    print_test_section("TESTING full combination with multiple tasks, specs, and preprompts")

    # Define test parameters
    test_tasks = ["counting", "identification"]
    test_specs = ["", "Answer in one word only."]
    test_preprompts = ["debiased", "cot"]

    print("\nGenerating questions with:")
    print(f"  Tasks: {test_tasks}")
    print(f"  Instruction specs: {test_specs}")
    print(f"  Preprompt types: {test_preprompts}")

    # For chess game only (for brevity)
    if "chess" in handler.all_questions_data:
        # Generate the comprehensive set of questions
        full_questions = handler.generate_questions(
            "chess", [], test_specs, test_tasks, test_preprompts
        )

        print(f"\nGenerated {len(full_questions)} total questions:")
        for i, question in enumerate(full_questions):
            print(f"  Question {i+1}: {question}")

        # Expected number calculation
        expected_count = len(test_tasks) * len(test_specs)
        print(f"\nExpected question count: {len(test_tasks)} tasks × {len(test_specs)} specs = {expected_count} variants")
        print(f"Actual question count: {len(full_questions)}")

        # Note: for localization in chess, we generate 2 questions (row and column)
        if "localization" in test_tasks:
            print("  Note: 'localization' task in chess generates 2 questions (row and column)")

    print("\nTest script complete!")
