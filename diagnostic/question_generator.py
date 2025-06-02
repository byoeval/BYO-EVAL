import json


def reformulate_question(question: str, dynamic_values: list[str] = None, reformulation_type: str="declarative_form") -> str:
    """
    Transform a question into a reformulated sentence using the template question.

    Args:
        question (str): The question to transform
        dynamic_values (List[str]): List of values for X, Y, Z in order
        reformulation_type (str): Type of reformulation (e.g., "declarative_form", "imperative_form")
    """
    try:
        file = "diagnostic/questions/" + reformulation_type + "_form.json"
        with open(file) as f:
            data = json.load(f)

        # Get the reformulated form from the JSON file
        reformulated_form = data.get(question)
        if not reformulated_form:
            print(f"Warning: No reformulation found for question: {question}")
            return question

        # Replace placeholders with actual values in the reformulated form
        if dynamic_values:
            for i, value in enumerate(dynamic_values):
                if i == 0:
                    reformulated_form = reformulated_form.replace("X", value)
                elif i == 1:
                    reformulated_form = reformulated_form.replace("Y", value)
                elif i == 2:
                    reformulated_form = reformulated_form.replace("Z", value)

        return reformulated_form

    except Exception as e:
        print(f"Error reformulating question: {str(e)}")
        return question

def generate_reformulated_questions(questions: list[str], dynamic_values: list[str], template_questions: list[str], reformulation_type: str="declarative_form") -> list[str]:
    """
    Generate reformulated versions of all questions.

    Args:
        questions (List[str]): List of original questions
        dynamic_values (List[str]): List of dynamic values for each question
        template_questions (List[str]): List of template questions (with X, Y, Z)
        reformulation_type (str): Type of reformulation to apply

    Returns:
        List[str]: List of reformulated questions
    """
    reformulated_sentences = []
    for _q, dv, tq in zip(questions, dynamic_values, template_questions, strict=False):
        reformulated_sentence = reformulate_question(tq, dv, reformulation_type)
        reformulated_sentences.append(reformulated_sentence)

    return reformulated_sentences


def generate_questions_and_answers(json_file: str, question_numbers: list[int], game_type: str = "chess") -> tuple[list[str], list[str], list[list[str]], list[str]]:
    """
    Generate questions and their corresponding answers based on the JSON file, question numbers, and game type.

    Args:
        json_file (str): Path to the JSON file in annotation/legend_json
        question_numbers (List[int]): List of question numbers from all_questions.json
        game_type (str): 'chess' or 'poker' (default: 'chess')

    Returns:
        Tuple[List[str], List[str], List[List[str]], List[str]]:
            - List of questions
            - List of answers
            - List of dynamic values for each question (values for X, Y, Z in order)
            - List of template questions (with X, Y, Z placeholders)
    """
    # Load the JSON file
    with open(json_file) as f:
        data = json.load(f)

    # Load questions from all_questions.json
    with open("diagnostic/questions/all_questions.json") as f:
        all_questions_json = json.load(f)
    if game_type not in all_questions_json:
        raise ValueError(f"Game type '{game_type}' not found in all_questions.json")
    all_questions = all_questions_json[game_type]

    questions = []
    answers = []
    dynamic_values_list = []
    template_questions = []

    chess_type_mapping = {'pawn': '0',
                        'rook': '1',
                        'knight': '2',
                        'bishop': '3',
                        'king': '4',
                        'queen': '5'}

    for question_number in question_numbers:
        question = all_questions[str(question_number)]

        pieces = data.get('pieces', {})
        num_pieces = len(pieces)
        add_question = True

        if num_pieces == 1:
            first_piece = next(iter(pieces.values()))
            col, row = first_piece.get('board_position', [0, 0])
        elif num_pieces == 2:
            piece1, piece2 = list(pieces.values())[:2]
            col1, row1 = piece1.get('board_position', [0, 0])
            col2, row2 = piece2.get('board_position', [0, 0])

        ## Handle counting questions ##
        if question == "How many pieces are there in the image ?": #1
            answers.append(str(num_pieces))
            dynamic_values_list.append([])
            template_questions.append(question)
        elif question == "How many squares has the board ?": #2
            board = data.get('board', {})
            rows = board.get('dimensions', {}).get('rows', 0)
            columns = board.get('dimensions', {}).get('columns', 0)
            total_squares = rows * columns
            answers.append(str(total_squares))
            dynamic_values_list.append([])
            template_questions.append(question)
        elif question == "How many total pieces are there in the image ?":  # 3,4
            answers.append(str(num_pieces))
            dynamic_values_list.append([])
            template_questions.append(question)

        ## Handle localization questions ##
        elif question == "Numbering the columns from left to right, starting with 0, on which column is the piece on the board ?" : #5
            if num_pieces == 1:
                answers.append(str(col))
                dynamic_values_list.append([])
                template_questions.append(question)
            else:
                add_question = False
                print(f"0 or several pieces, question {question_number} skipped")
        elif question == "Numbering the rows from top to bottom, starting with 0, on which row is the piece on the board ?": #6
            if num_pieces == 1:
                answers.append(str(row))
                dynamic_values_list.append([])
                template_questions.append(question)
            else:
                add_question = False
                print(f"Not exactly 1 piece, question {question_number} skipped")
        elif question == "How many rows separate the two pieces on the board ?": #7
            if num_pieces == 2:
                row_separation = abs(row1 - row2)
                answers.append(str(row_separation))
                dynamic_values_list.append([])
                template_questions.append(question)
            else:
                add_question = False
                print(f"Not exactly 2 pieces, question {question_number} skipped")
        elif question == "How many columns separate the two pieces on the board ?": #8
            if num_pieces == 2:
                col_separation = abs(col1 - col2)
                answers.append(str(col_separation))
                dynamic_values_list.append([])
                template_questions.append(question)
            else:
                add_question = False
                print(f"Not exactly 2 pieces, question {question_number} skipped")
        elif "closest" in question: #9, 10
            if num_pieces > 1:
                # Find closest piece
                for i, piece in enumerate(pieces.values()):
                    other_pieces = list(pieces.values())
                    other_pieces.pop(i)
                    closest_piece = min(other_pieces, key=lambda p:
                        abs(p['board_position'][0] - piece['board_position'][0]) +
                        abs(p['board_position'][1] - piece['board_position'][1]))

                    closest_row = closest_piece['board_position'][1]
                    closest_col = closest_piece['board_position'][0]
                    current_row = piece['board_position'][1]
                    current_col = piece['board_position'][0]

                    if question == "Numbering the rows from top to bottom, starting with 0, and the columns from left to right, starting also with 0, on which row is the closest piece to the piece at position X,Y ?": #9
                        questions.append(f"Numbering the rows from top to bottom, starting with 0, and the columns from left to right, starting also with 0, on which row is the closest piece to the piece at position {str(current_col)},{str(current_row)} ?")
                        answers.append(str(closest_row))
                        dynamic_values_list.append([str(current_col), str(current_row)])
                        template_questions.append(question)
                    if question == "Numbering the rows from top to bottom, starting with 0, and the columns from left to right, starting also with 0, on which column is the closest piece to the piece at position X,Y ?": #10
                        questions.append(f"Numbering the rows from top to bottom, starting with 0, and the columns from left to right, starting also with 0, on which column is the closest piece to the piece at position {str(current_col)},{str(current_row)} ?")
                        answers.append(str(closest_col))
                        dynamic_values_list.append([str(current_col), str(current_row)])
                        template_questions.append(question)
            else:
                print(f"Less than 2 pieces, question {question_number} skipped")
            add_question = False

        # Handle identification questions
        elif question == "By associating white with 0 and black with 1, what color is the piece on the board ?":
            if num_pieces == 1:
                color = first_piece.get('color', '')
                color_code = '0' if color.lower() == 'white' else '1'
                answers.append(color_code)
                dynamic_values_list.append([])
                template_questions.append(question)
            else:
                add_question = False
                print(f"Not exactly 1 piece, question {question_number} skipped")
        elif question == "Assigning the numbers 0, 1, 2, 3, 4 and 5 respectively to the pieces pawn, rook, knight, bishop, king and queen, what value corresponds to the piece on the board?":
            if num_pieces == 1:
                piece_type = first_piece.get('type', '')
                type_code = chess_type_mapping.get(piece_type.lower(), '')
                if type_code:
                    answers.append(type_code)
                    dynamic_values_list.append([])
                    template_questions.append(question)
            else:
                add_question = False
                print(f"Not exactly 1 piece, question {question_number} skipped")

        # Handle counting with localization questions
        elif question == "Numbering the rows from top to bottom, starting with 0, how many pieces are on the row X ?": #13
            board = data.get('board', {})
            rows = board.get('dimensions', {}).get('rows', 0)
            for row_id in range(rows):
                pieces_in_row = sum(1 for piece in pieces.values()
                                    if piece.get('board_position', [0, 0])[1] == row_id)
                if pieces_in_row > 0:
                    questions.append(f"Numbering the rows from top to bottom, starting with 0, how many pieces are on the row {str(row_id)} ?")
                    answers.append(str(pieces_in_row))
                    dynamic_values_list.append([str(row_id)])
                    template_questions.append(question)
            add_question = False
        elif question == "Numbering the columns from left to right, starting with 0, how many pieces are on the column X ?":  # 14
            board = data.get('board', {})
            columns = board.get('dimensions', {}).get('columns', 0)
            for col_id in range(columns):
                pieces_in_col = sum(1 for piece in pieces.values()
                                    if piece.get('board_position', [0, 0])[0] == col_id)
                if pieces_in_col > 0:
                    questions.append(f"Numbering the columns from left to right, starting with 0, how many pieces are on the column {str(col_id)} ?")
                    answers.append(str(pieces_in_col))
                    dynamic_values_list.append([str(col_id)])
                    template_questions.append(question)
            add_question = False

        # Handle counting with identification questions
        elif question == "How many white pieces are there on the board ?": # 15
            white_pieces = sum(1 for piece in pieces.values()
                                if piece.get('color', '').lower() == 'white')
            answers.append(str(white_pieces))
            dynamic_values_list.append([])
            template_questions.append(question)
        elif question == "How many black pieces are there on the board ?": # 16
            black_pieces = sum(1 for piece in pieces.values()
                                if piece.get('color', '').lower() == 'black')
            answers.append(str(black_pieces))
            dynamic_values_list.append([])
            template_questions.append(question)
        elif question == "How many X pieces are there on the board ?": # 17
            piece_types = {}
            for piece in pieces.values():
                piece_type = piece.get('type', '')
                piece_types[piece_type] = piece_types.get(piece_type, 0) + 1

            for piece_type, count in piece_types.items():
                questions.append(f"How many {piece_type} pieces are there on the board ?")
                answers.append(str(count))
                dynamic_values_list.append([piece_type])
                template_questions.append(question)
            add_question = False

        # Handle localization and identification questions
        elif question == "Numbering the rows from top to bottom, starting with 0, and the columns from left to right, starting also with 0, by associating white with 0 and black with 1, what color is the piece on the board at position X,Y ?": # 18
            for piece in pieces.values():
                col_curr, row_curr = piece.get('board_position', [0, 0])
                color = piece.get('color', '')
                color_code = '0' if color.lower() == 'white' else '1'
                questions.append(f"Numbering the rows from top to bottom, starting with 0, and the columns from left to right, starting also with 0, by associating white with 0 and black with 1, what color is the piece on the board at position {str(col_curr)},{str(row_curr)} ?")
                answers.append(color_code)
                dynamic_values_list.append([str(col_curr), str(row_curr)])
                template_questions.append(question)
            add_question = False
        elif question == "Numbering the rows from top to bottom, starting with 0, and the columns from left to right, starting also with 0, by assigning the numbers 0, 1, 2, 3, 4 and 5 respectively to the pieces pawn, rook, knight, bishop, king and queen, what value corresponds to the piece on the board at position X,Y ?":  # 19
            for piece in pieces.values():
                col_curr, row_curr = piece.get('board_position', [0, 0])
                piece_type = piece.get('type', '')
                type_code = chess_type_mapping.get(piece_type.lower(), '')
                if type_code:
                    questions.append(f"Numbering the rows from top to bottom, starting with 0, and the columns from left to right, starting also with 0, by assigning the numbers 0, 1, 2, 3, 4 and 5 respectively to the pieces pawn, rook, knight, bishop, king and queen, what value corresponds to the piece on the board at position {str(col_curr)},{str(row_curr)} ?")
                    answers.append(type_code)
                    dynamic_values_list.append([str(col_curr), str(row_curr)])
                    template_questions.append(question)
            add_question = False

        # Handle counting and localization and identification questions
        elif question == "Numbering the rows from top to bottom, starting with 0, how many X pieces are there on the board at row Y ?": # 20
            board = data.get('board', {})
            rows = board.get('dimensions', {}).get('rows', 0)
            for row_id in range(rows):
                piece_types_in_row = {}
                for piece in pieces.values():
                    if piece.get('board_position', [0, 0])[1] == row_id:
                        piece_type = piece.get('type', '')
                        piece_types_in_row[piece_type] = piece_types_in_row.get(piece_type, 0) + 1

                for piece_type, count in piece_types_in_row.items():
                    questions.append(f"Numbering the rows from top to bottom, starting with 0, how many {piece_type} pieces are there on the board at row {row_id} ?")
                    answers.append(str(count))
                    dynamic_values_list.append([piece_type, str(row_id)])
                    template_questions.append(question)
            add_question = False

        elif question == "Numbering the columns from left to right, starting with 0, how many X pieces are there on the board at column Y ?": # 21
            board = data.get('board', {})
            columns = board.get('dimensions', {}).get('columns', 0)
            for col_id in range(columns):
                piece_types_in_col = {}
                for piece in pieces.values():
                    if piece.get('board_position', [0, 0])[0] == col_id:
                        piece_type = piece.get('type', '')
                        piece_types_in_col[piece_type] = piece_types_in_col.get(piece_type, 0) + 1

                for piece_type, count in piece_types_in_col.items():
                    questions.append(f"Numbering the columns from left to right, starting with 0, how many {piece_type} pieces are there on the board at column {col_id} ?")
                    answers.append(str(count))
                    dynamic_values_list.append([piece_type, str(col_id)])
                    template_questions.append(question)
            add_question = False

        elif question == "Numbering the rows from top to bottom, starting with 0, how many white pieces are there on the board at row X ?": #22
            board = data.get('board', {})
            rows = board.get('dimensions', {}).get('rows', 0)
            for row_id in range(rows):
                white_pieces = sum(1 for piece in pieces.values()
                                if piece.get('color', '').lower() == 'white' and piece.get('board_position', [0, 0])[1] == row_id)
                if white_pieces > 0:
                    questions.append(f"Numbering the rows from top to bottom, starting with 0, how many white pieces are there on the board at row {row_id} ?")
                    answers.append(str(white_pieces))
                    dynamic_values_list.append([str(row_id)])
                    template_questions.append(question)
            add_question = False

        elif question == "Numbering the columns from left to right, starting with 0, how many white pieces are there on the board at column X ?": #23
            board = data.get('board', {})
            columns = board.get('dimensions', {}).get('columns', 0)
            for col_id in range(columns):
                white_pieces = sum(1 for piece in pieces.values()
                                if piece.get('color', '').lower() == 'white' and piece.get('board_position', [0, 0])[0] == col_id)
                if white_pieces > 0:
                    questions.append(f"Numbering the rows from top to bottom, starting with 0, how many white pieces are there on the board at row {col_id} ?")
                    answers.append(str(white_pieces))
                    dynamic_values_list.append([str(col_id)])
                    template_questions.append(question)
            add_question = False
        # add current question
        if add_question:
            questions.append(question)

    if game_type == "poker":
        for question_number in question_numbers:
            question = all_questions[str(question_number)]
            add_question = True
            # 1. How many community cards are visible on the table?
            if question == "How many community cards are visible on the table?":
                community = data.get("community_cards") or {}
                n_community = community.get("n_cards", 0)
                answers.append(str(n_community))
                dynamic_values_list.append([str(n_community)])
                template_questions.append(question)
            # 2. What are the names of the community cards shown?
            elif question == "What are the names of the community cards shown?":
                community = data.get("community_cards") or {}
                card_names = community.get("card_names", [])
                answers.append(", ".join(card_names))
                dynamic_values_list.append([", ".join(card_names)])
                template_questions.append(question)
            # 3. How many players are present at the table?
            elif question == "How many players are present at the table?":
                num_players = len(data.get("players", []))
                answers.append(str(num_players))
                dynamic_values_list.append([str(num_players)])
                template_questions.append(question)
            # 4. How many cards does each player have in their hand?
            elif question == "How many cards does each player have in their hand?":
                players = data.get("players", [])
                n_cards_list = [str(player.get("hand_config", {}).get("n_cards", 0)) for player in players]
                answers.append(", ".join(n_cards_list))
                dynamic_values_list.append([", ".join(n_cards_list)])
                template_questions.append(question)
            # Example: Dynamic value poker question
            elif question == "How many cards does Player X have?":
                players = data.get("players", [])
                for player in players:
                    player_id = player.get("player_id", "?")
                    n_cards = player.get("hand_config", {}).get("n_cards", 0)
                    # Replace X in the question with the player id
                    q_instance = question.replace("X", player_id)
                    questions.append(q_instance)
                    answers.append(str(n_cards))
                    dynamic_values_list.append([player_id, str(n_cards)])
                    template_questions.append(question)
                add_question = False
            # 11. How many cards are present in the entire scene (including all hands and community cards)?
            elif question == "How many cards are present in the entire scene (including all hands and community cards)?":
                community = data.get("community_cards") or {}
                n_community = community.get("n_cards", 0)
                players = data.get("players", [])
                n_player_cards = sum(player.get("hand_config", {}).get("n_cards", 0) for player in players)
                total_cards = n_community + n_player_cards
                answers.append(str(total_cards))
                dynamic_values_list.append([str(total_cards)])
                template_questions.append(question)
            # 14. How many cards are face up on the table?
            elif question == "How many cards are face up on the table?":
                community = data.get("community_cards") or {}
                n_faceup = community.get("n_cards", 0) - community.get("n_verso", 0)
                answers.append(str(max(n_faceup, 0)))
                dynamic_values_list.append([str(max(n_faceup, 0))])
                template_questions.append(question)
            # 15. How many cards are face down on the table?
            elif question == "How many cards are face down on the table?":
                community = data.get("community_cards") or {}
                n_verso = community.get("n_verso", 0)
                answers.append(str(n_verso))
                dynamic_values_list.append([str(n_verso)])
                template_questions.append(question)
            # 17. Which player has the most cards in their hand?
            elif question == "Which player has the most cards in their hand?":
                players = data.get("players", [])
                max_cards = -1
                max_player = None
                for player in players:
                    n_cards = player.get("hand_config", {}).get("n_cards", 0)
                    if n_cards > max_cards:
                        max_cards = n_cards
                        max_player = player.get("player_id", "Unknown")
                answers.append(str(max_player) if max_player is not None else "None")
                dynamic_values_list.append([str(max_player) if max_player is not None else "None", str(max_cards)])
                template_questions.append(question)
            # 19. What is the width and length of the poker table?
            elif question == "What is the width and length of the poker table?":
                table = data.get("scene_setup", {}).get("table", {})
                width = table.get("width", "?")
                length = table.get("length", "?")
                answers.append(f"width: {width}, length: {length}")
                dynamic_values_list.append([str(width), str(length)])
                template_questions.append(question)
            # 20. What is the vertical angle of the camera?
            elif question == "What is the vertical angle of the camera?":
                camera = data.get("scene_setup", {}).get("camera", {})
                angle = camera.get("angle", "?")
                answers.append(str(angle))
                dynamic_values_list.append([str(angle)])
                template_questions.append(question)
            else:
                add_question = False
                print(f"[Poker] Unimplemented or unrecognized question: {question} (number {question_number}) - skipping.")
            if add_question:
                questions.append(question)

    return questions, answers, dynamic_values_list, template_questions

if __name__ == "__main__":
    # example
    json_file = "annotation/legend_json/image_000001.json"
    question_numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

    questions, answers, dynamic_values, template_questions = generate_questions_and_answers(json_file, question_numbers)
    for q, a, dv, tq in zip(questions, answers, dynamic_values, template_questions, strict=False):
        print(f"\nQ: {q}")
        print(f"D: {reformulate_question(tq, dv)}")
        print(f"D: {reformulate_question(tq, dv, 'missing_word_form')}")
        print(f"A: {a}")
