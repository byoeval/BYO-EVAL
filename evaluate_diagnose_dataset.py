"""
Evaluate VLM capabilities on the diagnostic dataset for different tasks:
- Counting
- Identification
- Localization

This script tests both Azure OpenAI and Groq models on the dataset
and provides detailed evaluation metrics and visualizations.
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# Best thing in terminal :D
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.prompt import Confirm, Prompt
from rich.table import Table

from evaluation_pipeline.answer_extractor import AnswerExtractor

# Local import
from evaluation_pipeline.get_vlm import VLMProvider, get_vlm
from evaluation_pipeline.question_handler import QuestionHandler

console = Console()

# Load environment variables # TODO: Add Claude API and HuggingFace API
load_dotenv()

# Config
DATASET_PATH = Path("/home/test_count")
IMAGE_DIR = DATASET_PATH / "img"
ANNOTATION_DIR = DATASET_PATH / "legend_json"
OUTPUT_DIR = Path("results/diagnostic_evaluation")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Task types to evaluate
TASKS = ["counting", "identification", "localization"]

# VLM providers to test
PROVIDERS = {
    "azure_openai_1": {
        "provider": VLMProvider.AZURE_OPENAI,
        "model_name": os.environ.get("AZURE_OPENAI_MODEL_NAME", "gpt-4.1"),
        "api_version": os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        "credentials": {
            "api_key": os.environ.get("AZURE_OPENAI_API_KEY"),
            "endpoint": os.environ.get("AZURE_OPENAI_ENDPOINT")
        }
    },
    "azure_openai_2": {
        "provider": VLMProvider.AZURE_OPENAI,
        "model_name": os.environ.get("AZURE_OPENAI_MODEL_NAME_2", "gpt-4.1-mini"),
        "api_version": os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        "credentials": {
            "api_key": os.environ.get("AZURE_OPENAI_API_KEY"),
            "endpoint": os.environ.get("AZURE_OPENAI_ENDPOINT")
        }
    },
    "groq_1": {
        "provider": VLMProvider.GROQ,
        "model_name": os.environ.get("GROQ_MODEL_NAME", "meta-llama/llama-4-scout-17b-16e-instruct"),
        "credentials": {
            "api_key": os.environ.get("GROQ_API_KEY")
        }
    },
    "groq_2": {
        "provider": VLMProvider.GROQ,
        "model_name": os.environ.get("GROQ_MODEL_NAME_2", "meta-llama/llama-4-maverick-17b-128e-instruct"),
        "credentials": {
            "api_key": os.environ.get("GROQ_API_KEY")
        }
    },
    "ollama_1": {
        "provider": VLMProvider.OLLAMA,
        "model_name": os.environ.get("OLLAMA_MODEL_NAME_1", "gemma3:4b"),
        "credentials": {
            "base_url": os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        }
    },
    "ollama_2": {
        "provider": VLMProvider.OLLAMA,
        "model_name": os.environ.get("OLLAMA_MODEL_NAME_2", "gemma3:12b"),
        "credentials": {
            "base_url": os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        }
    },
    "ollama_3": {
        "provider": VLMProvider.OLLAMA,
        "model_name": os.environ.get("OLLAMA_MODEL_NAME_3", "llama3.2-vision:11b"),
        "credentials": {
            "base_url": os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        }
    },
    "ollama_4": {
        "provider": VLMProvider.OLLAMA,
        "model_name": os.environ.get("OLLAMA_MODEL_NAME_4", "mistral-small3.1"),
        "credentials": {
            "base_url": os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        }
    },
    "huggingface": {
        "provider": VLMProvider.HUGGINGFACE,
        "model_name": os.environ.get("HUGGINGFACE_MODEL_NAME", "Salesforce/blip-image-captioning-base"),
        "credentials": {
            "api_key": os.environ.get("HUGGINGFACE_API_KEY")
        }
    }
}


GAME_TYPE = None  # Will be set by prompt

def find_images_and_annotations() -> list[tuple[Path, Path]]:
    """Find all matching image and annotation files in the dataset.

    Returns:
        List of tuples containing (image_path, annotation_path)
    """
    matches = []

    # Get all annotation files
    annotation_files = list(ANNOTATION_DIR.glob("*.json"))
    # For each annotation file, try to find the matching image
    for ann_file in annotation_files:
        image_name = ann_file.stem
        image_candidates = [
            IMAGE_DIR / f"{image_name}.png",
            IMAGE_DIR / f"{image_name}.jpg",
            IMAGE_DIR / f"{image_name}.jpeg"
        ]

        # Use the first image that exists
        for img_path in image_candidates:
            if img_path.exists():
                matches.append((img_path, ann_file))
                break

    return matches

def extract_annotation_info(annotation_path: Path) -> dict:
    """Extract key information from an annotation file.

    Args:
        annotation_path: Path to the annotation JSON file

    Returns:
        Dictionary with extracted information
    """
    with open(annotation_path) as f:
        data = json.load(f)

    # Extract board information --> only for chess for now
    board_info = data.get('board', {})

    # Extract pieces information
    pieces = data.get('pieces', {})
    piece_count = len(pieces)
    piece_types = {}

    for _piece_id, piece_data in pieces.items():
        piece_type = piece_data.get('type')
        if piece_type in piece_types:
            piece_types[piece_type] += 1
        else:
            piece_types[piece_type] = 1

    # Summarized information
    return {
        "board_pattern": board_info.get("pattern", "unknown"),
        "board_dimensions": f"{board_info.get('dimensions', {}).get('rows', 0)}x{board_info.get('dimensions', {}).get('columns', 0)}",
        "piece_count": piece_count,
        "piece_types": piece_types,
        "all_pieces": pieces
    }

def generate_questions_for_image(annotation_path: Path, task: str, reformulate: str="", pre_prompt: str="") -> tuple[list[str], list[str], list]:
    global GAME_TYPE
    """Generate questions and answers for an image based on the task.

    Args:
        annotation_path: Path to the annotation JSON file
        task: The task type (counting, identification, localization)
        reformulate: The type of format to chose for reformulation. Default is "" for None.
        pre_prompt: The pre_prompt to add before each question. Default is "".

    Returns:
        Tuple containing lists of questions, answers, and dynamic values
    """
    # Load annotation to check piece count before generating questions
    with open(annotation_path) as f:
        annotation_data = json.load(f)

    pieces = annotation_data.get('pieces', {})
    piece_count = len(pieces)

    # if GAME_TYPE is None, look in the legend json file to find the game type
    if GAME_TYPE is None:
        if annotation_data.get('board', None) is not None:
            GAME_TYPE = "chess"
        else:
            GAME_TYPE = "poker"

    # Define question keys corresponding to each task with fallback options
    if GAME_TYPE == "chess":
        if task == "counting":
            question_keys = ["count_pieces"]
        elif task == "identification":
            if piece_count == 0:
                custom_questions = ["Is the chess board empty?"]
                custom_answers = ["Yes"]
                return custom_questions, custom_answers, []
            elif piece_count == 1:
                question_keys = ["identify_color_one_piece", "identify_type_one_piece"]
            else:
                question_keys = ["identify_type_several_pieces"]
                if piece_count >= 2:
                    piece_positions = list(pieces.values())
                    if len(piece_positions) >= 2 and all('position' in p for p in piece_positions[:2]):
                        question_keys.extend(["identify_localization_piece_color", "identify_localization_piece_type"])
        elif task == "localization":
            if piece_count == 0:
                custom_questions = ["Are there any pieces on the board?"]
                custom_answers = ["No"]
                return custom_questions, custom_answers, []
            elif piece_count == 1:
                question_keys = ["localize_column_one_piece", "localize_row_one_piece"]
            elif piece_count >= 2:
                question_keys = ["localize_rows_between_two_pieces", "localize_columns_between_two_pieces"]
                piece_positions = list(pieces.values())
                if all('position' in p for p in piece_positions[:2]):
                    question_keys.extend(["localize_row_closest_piece", "localize_column_closest_piece"])
        else:
            return [], [], []
    elif GAME_TYPE == "poker":
        if task == "counting":
            question_keys = ["count_total_cards", "count_players"]
        elif task == "identification":
            question_keys = ["identify_cards", "identify_community_cards"]
        elif task == "localization":
            #question_keys = ["count_identify_face_up_cards", "count_identify_face_down_cards"]
            question_keys = []
            # Add new card grid localization questions
            cards_data = annotation_data.get("card_grid_locations", {})
            players = annotation_data.get("players", [])
            # Check if any player has grid_locations data
            has_grid_data = cards_data or any(player.get("grid_locations") for player in players)
            if has_grid_data:
                question_keys.extend([
                    "localize_card_on_grid_row",
                    "localize_card_on_grid_column",
                    "localize_card_on_grid_number",
                    "localize_card_on_grid_3x3"
                ])
        else:
            return [], [], []
    else:
        return [], [], []

    # If we defined custom questions directly, return them
    if 'custom_questions' in locals():
        return custom_questions, custom_answers, []

    # Generate fallback questions if no suitable questions found
    if not question_keys:
        if task == "identification":
            return ["How many different types of pieces are on the board?"], ["0" if piece_count == 0 else str(len({p.get('type') for p in pieces.values() if 'type' in p}))], []
        elif task == "localization":
            return ["Is the board set up with pieces?"], ["No" if piece_count == 0 else "Yes"], []
        else:
            return [], [], []

    try:
        # Use the question keys directly with the QuestionHandler and AnswerExtractor
        question_handler = QuestionHandler()

        # Handle different reformulation formats
        instruction_specs = []
        if reformulate == "declarative":
            instruction_specs = ["declarative"]
        elif reformulate == "missing_word":
            instruction_specs = ["fill_in"]
        elif reformulate == "both":
            instruction_specs = ["declarative", "fill_in"]
        elif not reformulate:
            instruction_specs = [""]
        else:
            # Split by comma in case multiple reformulation types are specified
            reformulate_types = [r.strip() for r in reformulate.split(",")]
            for r_type in reformulate_types:
                if r_type == "declarative":
                    instruction_specs.append("declarative")
                elif r_type == "missing_word":
                    instruction_specs.append("fill_in")
                else:
                    instruction_specs.append(r_type)

        # print inputs
        print("Game type: ", GAME_TYPE)
        print("Question keys: ", question_keys)
        print("Instruction specs: ", instruction_specs)
        print("Preprompt types: ", [pre_prompt] if pre_prompt else [])
        questions = question_handler.generate_questions(
            game=GAME_TYPE,
            keys=question_keys,
            instruction_specs=instruction_specs,
            preprompt_types=[pre_prompt] if pre_prompt else []
        )
        answer_extractor = AnswerExtractor()
        answers = [answer_extractor.extract_answer(annotation_data, key, GAME_TYPE) for key in question_keys]
        dynamic_values_list = [[] for _ in question_keys]
        return question_keys, answers, dynamic_values_list
    except Exception as e:
        console.print(f"[bold red]Error generating questions: {e}[/bold red]")
        if task == "counting":
            return ["How many total pieces are on the board?"], [str(piece_count)], []
        elif task == "identification":
            return ["Are there any chess pieces on the board?"], ["No" if piece_count == 0 else "Yes"], []
        elif task == "localization":
            return ["Is the chess board empty?"], ["Yes" if piece_count == 0 else "No"], []
        else:
            return [], [], []

def modify_counting_questions(questions: list[str]) -> list[str]:
    """Modify counting-related questions to explicitly request numeric answers."""
    modified_questions = []

    for question in questions:
        # Always ensure we're asking for a numeric answer with counting questions
        if question.endswith("?"):
            if question.startswith("First"): # cot case
                modified_question = question[:-1] + "? Please respond with a single number only and end your answer with the number."
            else:
                modified_question = question[:-1] + "? Please respond with a single number only."
        else:
            if question.startswith("First"): # cot case
                modified_question = question + " Please respond with a single number only and end your answer with the number."
            else:
                modified_question = question + " Please respond with a single number only."

        modified_questions.append(modified_question)

    return modified_questions

def modify_identification_questions(questions: list[str]) -> list[str]:
    """Modify identification questions to request more specific answers."""
    modified_questions = []

    for question in questions:
        # For yes/no questions, request clear yes/no answer
        if any(word in question.lower() for word in ["is there", "are there", "is the", "are the", "does", "do"]):
            if question.endswith("?"):
                modified_question = question[:-1] + "? Please answer with 'Yes' or 'No' only."
            else:
                modified_question = question + " Please answer with 'Yes' or 'No' only."
        # For counting-related questions in identification task
        elif any(keyword in question.lower() for keyword in ["how many", "count", "number", "there are", "squares"]):
            if question.endswith("?"):
                modified_question = question[:-1] + "? Please respond with a single number only."
            else:
                modified_question = question + " Please respond with a single number only."
        # For other identification questions
        else:
            if question.endswith("?"):
                modified_question = question[:-1] + "? Please be specific and concise in your answer."
            else:
                modified_question = question + " Please be specific and concise in your answer."

        modified_questions.append(modified_question)

    return modified_questions

def modify_localization_questions(questions: list[str]) -> list[str]:
    """Modify localization questions to request more precise location descriptions."""
    modified_questions = []

    for question in questions:
        # For grid localization in poker with 3x3 grid
        if "on which cell of the 3x3 grid" in question.lower():
            if question.endswith("?"):
                modified_question = question[:-1] + "? Please answer with the position (e.g., upper left, middle, lower right)."
            else:
                modified_question = question + " Please answer with the position (e.g., upper left, middle, lower right)."
        # For other grid localization in poker
        elif any(keyword in question.lower() for keyword in ["on which cell of the grid", "on which row of the grid", "on which column of the grid"]):
            if question.endswith("?"):
                modified_question = question[:-1] + "? Please answer with a single number only."
            else:
                modified_question = question + " Please answer with a single number only."
        # For questions asking about positions or coordinates
        elif any(keyword in question.lower() for keyword in ["where", "position", "locate", "coordinate", "square"]):
            if question.endswith("?"):
                modified_question = question[:-1] + "? Please be precise with locations or coordinates."
            else:
                modified_question = question + " Please be precise with locations or coordinates."
        # For counting questions that may appear in localization tasks
        elif any(keyword in question.lower() for keyword in ["how many", "count", "number", "there are"]):
            if question.endswith("?"):
                modified_question = question[:-1] + "? Please respond with a single number only."
            else:
                modified_question = question + " Please respond with a single number only."
        # For yes/no questions
        elif any(word in question.lower() for word in ["is there", "are there", "is the", "are the", "does", "do"]):
            if question.endswith("?"):
                modified_question = question[:-1] + "? Please answer with 'Yes' or 'No' only."
            else:
                modified_question = question + " Please answer with 'Yes' or 'No' only."
        # For other localization questions
        else:
            if question.endswith("?"):
                modified_question = question[:-1] + "? Please be specific about the location in your answer."
            else:
                modified_question = question + " Please be specific about the location in your answer."

        modified_questions.append(modified_question)

    return modified_questions

def extract_numeric_value(answer) -> str:
    """
    Extract the numeric value from an answer string
    Args:
        answer: The answer string or dictionary that may contain additional text
    Returns:
        The extracted numeric value as a string
    """
    import re

    # Convert answer to string if it's a dictionary or other non-string type
    if not isinstance(answer, str):
        answer = str(answer)

    # First try to find standalone numbers (surrounded by spaces, punctuation or at start/end)
    standalone_numbers = re.findall(r'(?:^|[\s.,;:!?()])(\d+)(?:$|[\s.,;:!?()])', answer)
    if standalone_numbers:
        return standalone_numbers[0]  # Return the first standalone number

    # If no standalone numbers, try to find any sequence of digits
    numbers = re.findall(r'\d+', answer)
    if numbers:
        return numbers[0]  # Return the first number found

    # Try to convert word numbers to digits (only for common numbers)
    word_to_number = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10', 'eleven': '11', 'twelve': '12', 'thirteen': '13', 'fourteen': '14',
        'fifteen': '15', 'sixteen': '16', 'seventeen': '17', 'eighteen': '18', 'nineteen': '19',
        'twenty': '20', 'thirty': '30', 'forty': '40', 'fifty': '50',
        'sixty': '60', 'seventy': '70', 'eighty': '80', 'ninety': '90'
    }

    words = answer.lower().split()
    for word in words:
        # Clean up the word (remove punctuation)
        clean_word = re.sub(r'[^\w\s]', '', word)
        if clean_word in word_to_number:
            return word_to_number[clean_word]

    # If no numbers found, return original string
    return answer.strip()

def evaluate_vlm_on_task(vlm, image_path: Path, annotation_path: Path, question_keys: list[str], reformulate:str, pre_prompt:str) -> dict:
    """Evaluate a VLM on a specific task for one image."""
    global GAME_TYPE # Make sure GAME_TYPE is accessible

    console.print(f"[bold magenta][DEBUG] Entered evaluate_vlm_on_task for image: {image_path}, annotation: {annotation_path}, question_keys: {question_keys}, reformulate: {reformulate}, pre_prompt: {pre_prompt}")
    # Load annotation data
    with open(annotation_path) as f:
        annotation_data = json.load(f)

    if not question_keys:
        console.print("[bold red][DEBUG] No question keys provided for evaluation.")
        return {
            "error": "No question keys provided for evaluation.",
            "accuracy": 0,
            "normalized_mae": 0,
            "details": [],
            "numeric_count": 0,
            "total_questions": 0,
            "correct_count": 0
        }

    # Instantiate QuestionHandler and AnswerExtractor
    question_handler = QuestionHandler()
    answer_extractor = AnswerExtractor()

    # Prepare instruction_specs for QuestionHandler
    instruction_specs: list[str]
    if reformulate == "declarative":
        instruction_specs = ["declarative"]
    elif reformulate == "missing_word": # Assuming 'missing_word' maps to 'fill_in'
        instruction_specs = ["fill_in"]
    elif reformulate == "both":
        instruction_specs = ["declarative", "fill_in"]
    elif not reformulate: # Empty string means default
        instruction_specs = [""]
    else: # If reformulate contains a custom instruction string or comma-separated values
        # Split by comma in case multiple reformulation types are specified
        reformulate_types = [r.strip() for r in reformulate.split(",")]
        instruction_specs = []
        for r_type in reformulate_types:
            if r_type == "declarative":
                instruction_specs.append("declarative")
            elif r_type == "missing_word":
                instruction_specs.append("fill_in")
            else:
                instruction_specs.append(r_type)

    # Prepare preprompt_types for QuestionHandler
    # pre_prompt is the actual string for the preprompt or empty
    preprompt_types = [pre_prompt] if pre_prompt else []

    # if GAME_TYPE is None, look in the legend json file to find the game type
    if GAME_TYPE is None:
        if annotation_data.get('board', None) is not None:
            GAME_TYPE = "chess"
        else:
            GAME_TYPE = "poker"

    # Generate questions using QuestionHandler
    questions = question_handler.generate_questions(
        game=GAME_TYPE,
        keys=question_keys,
        instruction_specs=instruction_specs,
        preprompt_types=preprompt_types,
        img_legend=annotation_data,
    )
    console.print(f"[bold magenta][DEBUG] Generated questions: {questions}")
    if not questions:
        console.print(f"[bold red][DEBUG] No questions generated by QuestionHandler for keys: {question_keys}")
        return {
            "error": f"No questions generated by QuestionHandler for keys: {question_keys}",
            "accuracy": 0,
            "normalized_mae": 0,
            "details": [],
            "numeric_count": 0,
            "total_questions": 0,
            "correct_count": 0
        }

    # Generate expected answers using AnswerExtractor
    # Ensure the order of answers matches the order of generated questions (which is based on question_keys)
    expected_answers = [answer_extractor.extract_answer(annotation_data, key, GAME_TYPE, question) for key, question in zip(question_keys, questions, strict=False)]
    console.print(f"[bold magenta][DEBUG] Expected answers: {expected_answers}")

    # Get VLM answers for each question
    vlm_answers = []
    call_metrics = []
    correct_count = 0
    error_sum = 0
    numeric_count = 0

    for i, (question, expected) in enumerate(zip(questions, expected_answers, strict=False)):
        question_key = question_keys[i] if i < len(question_keys) else None
        console.print(f"[bold yellow][DEBUG] Question key:[/bold yellow] {question_key}")
        console.print(f"[bold yellow][DEBUG] Question sent to VLM:[/bold yellow] {question!r}")
        console.print(f"[bold yellow][DEBUG] Expected answer:[/bold yellow] {expected!r}")
        try:
            result = vlm.answer_question(str(image_path), question)
            if isinstance(result, dict) and "result" in result:
                answer= vlm._extract_answer(result.get("result", ""))
            else:
                answer = result
            console.print(f"[bold yellow][DEBUG] VLM answer:[/bold yellow] {answer!r}")
            vlm_answers.append(answer)
            call_metrics.append(result.get("call_metrics", {}))

            # Use Regex for numeric extraction
            if any(keyword in question.lower() for keyword in ["how many", "count","there are","number", "squares"]):
                try:
                    extracted_answer = extract_numeric_value(answer)
                    extracted_expected = extract_numeric_value(expected)

                    # Check if correct based on extracted numbers
                    is_correct = extracted_answer == extracted_expected
                    if is_correct:
                        correct_count += 1

                    # Try to calculate error for numeric answers
                    try:
                        vlm_val = float(extracted_answer)
                        expected_val = float(extracted_expected)
                        error = abs(vlm_val - expected_val)

                        # Normalize error if true answer is non-zero
                        if expected_val != 0:
                            error_sum += error / expected_val
                        else:
                            error_sum += 1.0 if error > 0 else 0.0

                        numeric_count += 1
                    except (ValueError, TypeError):
                        # Non-numeric answers
                        pass
                except Exception as e:
                    console.print(f"[bold red][DEBUG] Error extracting numeric value: {e}, answer={answer}, expected={expected}")
                    continue
            # For yes/no questions (often used as fallbacks)
            elif any(word in question.lower() for word in ["is there", "are there", "is the", "are the", "does", "do"]):
                # Process yes/no answers
                answer_normalized = answer.lower().strip()
                expected_normalized = expected.lower().strip()

                # Extract yes/no from the beginning of the response
                if answer_normalized.startswith("yes"):
                    answer_normalized = "yes"
                elif answer_normalized.startswith("no"):
                    answer_normalized = "no"

                # Check for match
                is_correct = answer_normalized == expected_normalized or (
                    answer_normalized.startswith(expected_normalized) or
                    expected_normalized in answer_normalized
                )

                if is_correct:
                    correct_count += 1
            # Handle 3x3 grid position answers using position words
            elif question_key == "localize_card_on_grid_3x3" or "on which cell of the 3x3 grid" in question.lower():
                answer_lower = answer.lower().strip()
                expected_lower = expected.lower().strip() if expected else ""

                # Check if the expected position word appears in the answer
                is_correct = expected_lower in answer_lower

                if is_correct:
                    correct_count += 1
            else:
                # flexible string comparison
                answer_lower = answer.lower().strip() if isinstance(answer, str) else str(answer).lower().strip()
                expected_lower = expected.lower().strip() if isinstance(expected, str) else str(expected).lower().strip()

                # Consider correct if the expected answer is contained in the model's response
                is_correct = expected_lower in answer_lower or answer_lower == expected_lower

                if is_correct:
                    correct_count += 1

        except Exception as e:
            console.print(f"[bold red][DEBUG] Exception in VLM answer: {e}")
            vlm_answers.append(f"ERROR: {str(e)}")

    # Calculate metrics
    num_questions = len(questions)
    accuracy = correct_count / num_questions if num_questions > 0 else 0
    nmae = error_sum / numeric_count if numeric_count > 0 else 0

    details = []
    for i, (q, answer) in enumerate(zip(questions, vlm_answers, strict=False)):
        expected = expected_answers[i] # Get corresponding expected answer
        # Ensure expected is a string, as downstream code expects it
        expected_str = str(expected) if expected is not None else ""
        console.print(f"[bold cyan][DEBUG] Checking answer: question={q}, expected={expected_str}, answer={answer}")
        if any(keyword in q.lower() for keyword in ["how many", "count", "there are", "number", "squares"]):
            try:
                extracted_answer = extract_numeric_value(answer)
                extracted_expected = extract_numeric_value(expected_str)
                console.print(f"[bold cyan][DEBUG] Extracted numeric: expected={extracted_expected}, answer={extracted_answer}")
                is_correct = extracted_answer == extracted_expected
            except Exception as e:
                console.print(f"[bold red][DEBUG] Error extracting numeric value in details: {e}, answer={answer}, expected={expected_str}")
                is_correct = False
        elif any(word in q.lower() for word in ["is there", "are there", "is the", "are the", "does", "do"]):
            # For yes/no questions
            answer_norm = answer.lower().strip() if isinstance(answer, str) else str(answer).lower().strip()
            expected_norm = expected_str.lower().strip()

            if answer_norm.startswith("yes"):
                answer_norm = "yes"
            elif answer_norm.startswith("no"):
                answer_norm = "no"

            is_correct = answer_norm == expected_norm or expected_norm in answer_norm
            console.print(f"[bold cyan][DEBUG] Yes/No check: expected={expected_norm}, answer={answer_norm}, is_correct={is_correct}")
        elif question_keys[i] == "localize_card_on_grid_3x3" or "on which cell of the 3x3 grid" in q.lower():
            # Special handling for 3x3 grid position answers
            answer_lower = answer.lower().strip() if isinstance(answer, str) else str(answer).lower().strip()
            expected_lower = expected_str.lower().strip()

            # Check if the expected position word appears in the answer
            is_correct = expected_lower in answer_lower
            console.print(f"[bold cyan][DEBUG] 3x3 grid position check: expected={expected_lower}, answer={answer_lower}, is_correct={is_correct}")
        else:
            # More flexible matching for other questions
            answer_lower_str = answer.lower().strip() if isinstance(answer, str) else str(answer).lower().strip()
            expected_lower_str = expected_str.lower().strip()
            is_correct = expected_lower_str in answer_lower_str or answer_lower_str == expected_lower_str
            console.print(f"[bold cyan][DEBUG] Flexible check: expected={expected_lower_str}, answer={answer_lower_str}, is_correct={is_correct}")

        # Safely extract numeric values if needed
        extracted_expected = None
        extracted_answer = None
        if any(keyword in q.lower() for keyword in ["how many", "count", "there are", "number", "squares"]):
            try:
                extracted_expected = extract_numeric_value(expected_str)
                extracted_answer = extract_numeric_value(answer)
            except Exception as e:
                console.print(f"[bold red][DEBUG] Error extracting numeric value for details dict: {e}")

        details.append({
            "question": q,
            "expected": expected_str,
            "answer": answer,
            "extracted_expected": extracted_expected,
            "extracted_answer": extracted_answer,
            "correct": is_correct
        })

    return {
        "question_keys": question_keys,
        "vlm_answers": vlm_answers,
        "accuracy": accuracy,
        "normalized_mae": nmae,
        "details": details,
        "numeric_count": numeric_count,
        "total_questions": num_questions,
        "correct_count": correct_count,
        "questions": questions,
        "expected_answers": expected_answers,
        "call_metrics" : call_metrics
    }

def calculate_additional_metrics(results: dict) -> dict:
    """Calculate additional metrics beyond basic accuracy and NMAE.

    Args:
        results: The evaluation results dictionary

    Returns:
        Dictionary with additional metrics
    """
    metrics = {}
    providers = list(results["providers"].keys())
    all_tasks = []

    for provider in providers:
        for task in results["providers"][provider]["tasks"]:
            if task not in all_tasks:
                all_tasks.append(task)

    for provider in providers:
        if provider not in results["providers"]:
            continue

        provider_name = results["providers"][provider]["model"]
        metrics[provider] = {"model": provider_name, "tasks": {}}

        for task in all_tasks:
            if task not in results["providers"][provider]["tasks"]:
                continue

            task_metrics = {
                "accuracy": 0.0,
                "f1_score": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "nmae": 0.0,
                "rmse": 0.0,
                "mse": 0.0,   # Added Mean Square Error
                "mae": 0.0,   # Added Mean Absolute Error
                "reliability": 0.0,
                "consistency": 0.0,
                # Standard deviation metrics
                "accuracy_std": 0.0,
                "f1_score_std": 0.0,
                "precision_std": 0.0,
                "recall_std": 0.0,
                "nmae_std": 0.0,
                "rmse_std": 0.0,
                "mse_std": 0.0,
                "mae_std": 0.0,
                "reliability_std": 0.0,
                "consistency_std": 0.0
            }

            correct_count = 0
            total_questions = 0
            true_positive = 0
            false_positive = 0
            false_negative = 0

            # For numeric questions
            squared_errors = []
            abs_errors = []
            expected_values = []

            # For consistency and reliability
            response_consistency = {}  # Track responses to identical questions

            # Lists to store per-image metrics for std calculations
            image_accuracies = []
            image_f1_scores = []
            image_precisions = []
            image_recalls = []
            image_maes = []
            image_mses = []
            image_rmses = []
            image_nmaes = []

            # Process each image
            for _image_name, image_data in results["providers"][provider]["images"].items():
                if task not in image_data["tasks"]:
                    continue

                task_data = image_data["tasks"][task]
                image_correct = 0
                image_total = 0
                image_true_positive = 0
                image_false_positive = 0
                image_false_negative = 0
                image_abs_errors = []
                image_squared_errors = []
                image_normalized_errors = []
                image_expected_values = []

                # Process each question's details
                for detail in task_data.get("details", []):
                    total_questions += 1
                    image_total += 1

                    # Calculate basic metrics
                    if detail["correct"]:
                        correct_count += 1
                        image_correct += 1
                        true_positive += 1
                        image_true_positive += 1
                    else:
                        # For binary classification metrics
                        if detail["expected"].lower() in ["yes", "true", "1"]:
                            false_negative += 1
                            image_false_negative += 1
                        else:
                            false_positive += 1
                            image_false_positive += 1

                    # Track consistency
                    q_key = detail["question"]
                    if q_key not in response_consistency:
                        response_consistency[q_key] = []

                    # Convert answer to string if it's a dictionary or other non-hashable type
                    answer = detail["answer"]
                    if isinstance(answer, dict) or not isinstance(answer, str | int | float | bool):
                        answer = str(answer)

                    response_consistency[q_key].append(answer)

                    # Calculate metrics for numeric questions
                    if any(keyword in detail["question"].lower() for keyword in ["how many", "count","there are", "number","squares"]):
                        try:
                            expected_val = float(detail.get("extracted_expected", "0"))
                            actual_val = float(detail.get("extracted_answer", "0"))
                            error = abs(expected_val - actual_val)
                            squared_error = error ** 2

                            abs_errors.append(error)
                            squared_errors.append(squared_error)
                            expected_values.append(expected_val)

                            # Also track per-image errors for std calculation
                            image_abs_errors.append(error)
                            image_squared_errors.append(squared_error)
                            image_expected_values.append(expected_val)
                        except (ValueError, TypeError):
                            pass

                # Calculate per-image metrics
                if image_total > 0:
                    # Accuracy
                    image_accuracy = image_correct / image_total
                    image_accuracies.append(image_accuracy)

                    # Precision, recall, F1
                    if (image_true_positive + image_false_positive) > 0:
                        image_precision = image_true_positive / (image_true_positive + image_false_positive)
                        image_precisions.append(image_precision)

                    if (image_true_positive + image_false_negative) > 0:
                        image_recall = image_true_positive / (image_true_positive + image_false_negative)
                        image_recalls.append(image_recall)

                    if len(image_precisions) > 0 and len(image_recalls) > 0:
                        last_precision = image_precisions[-1]
                        last_recall = image_recalls[-1]
                        if (last_precision + last_recall) > 0:
                            image_f1 = 2 * last_precision * last_recall / (last_precision + last_recall)
                            image_f1_scores.append(image_f1)

                # Calculate per-image error metrics for numeric questions
                if image_abs_errors:
                    # MAE
                    image_mae = sum(image_abs_errors) / len(image_abs_errors)
                    image_maes.append(image_mae)

                    # MSE
                    image_mse = sum(image_squared_errors) / len(image_squared_errors)
                    image_mses.append(image_mse)

                    # RMSE
                    image_rmse = image_mse ** 0.5
                    image_rmses.append(image_rmse)

                    # NMAE
                    image_normalized_errors = []
                    for error, expected in zip(image_abs_errors, image_expected_values, strict=False):
                        if expected != 0:
                            image_normalized_errors.append(error / expected)
                        else:
                            image_normalized_errors.append(1.0 if error > 0 else 0.0)

                    if image_normalized_errors:
                        image_nmae = sum(image_normalized_errors) / len(image_normalized_errors)
                        image_nmaes.append(image_nmae)

            # Calculate global accuracy
            task_metrics["accuracy"] = correct_count / total_questions if total_questions > 0 else 0

            # F1, precision, recall
            precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
            recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
            task_metrics["precision"] = precision
            task_metrics["recall"] = recall
            task_metrics["f1_score"] = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            # Calculate error metrics for numeric questions
            if abs_errors and expected_values:
                # Calculate MAE (Mean Absolute Error)
                task_metrics["mae"] = sum(abs_errors) / len(abs_errors)

                # Calculate MSE (Mean Squared Error)
                task_metrics["mse"] = sum(squared_errors) / len(squared_errors)

                # Calculate RMSE (Root Mean Squared Error)
                task_metrics["rmse"] = task_metrics["mse"] ** 0.5

                # Calculate NMAE (Normalized Mean Absolute Error)
                normalized_errors = []
                for error, expected in zip(abs_errors, expected_values, strict=False):
                    if expected != 0:
                        normalized_errors.append(error / expected)
                    else:
                        normalized_errors.append(1.0 if error > 0 else 0.0)

                task_metrics["nmae"] = sum(normalized_errors) / len(normalized_errors)

            # Calculate reliability (consistency across images)
            if image_accuracies:
                task_metrics["reliability"] = sum(image_accuracies) / len(image_accuracies)
                # Standard deviation of accuracies indicates reliability variation
                mean_acc = task_metrics["reliability"]
                variance = sum((acc - mean_acc) ** 2 for acc in image_accuracies) / len(image_accuracies)
                task_metrics["reliability_std"] = variance ** 0.5

            # Calculate consistency (same answer to same question)
            consistency_scores = []
            for _question, answers in response_consistency.items():
                if len(answers) > 1:
                    # Calculate how often the most common answer appears
                    answer_counts = {}
                    for answer in answers:
                        # Ensure the answer is a hashable type (str, int, float, bool, tuple)
                        if not isinstance(answer, str | int | float | bool | tuple):
                            answer = str(answer)
                        answer_counts[answer] = answer_counts.get(answer, 0) + 1
                    most_common = max(answer_counts.values())
                    consistency_scores.append(most_common / len(answers))

            if consistency_scores:
                task_metrics["consistency"] = sum(consistency_scores) / len(consistency_scores)
                # Calculate std for consistency
                mean_consistency = task_metrics["consistency"]
                if len(consistency_scores) > 1:
                    variance = sum((score - mean_consistency) ** 2 for score in consistency_scores) / len(consistency_scores)
                    task_metrics["consistency_std"] = variance ** 0.5

            # Calculate standard deviations for other metrics
            if len(image_accuracies) > 1:
                task_metrics["accuracy_std"] = calculate_std(image_accuracies)

            if len(image_f1_scores) > 1:
                task_metrics["f1_score_std"] = calculate_std(image_f1_scores)

            if len(image_precisions) > 1:
                task_metrics["precision_std"] = calculate_std(image_precisions)

            if len(image_recalls) > 1:
                task_metrics["recall_std"] = calculate_std(image_recalls)

            if len(image_maes) > 1:
                task_metrics["mae_std"] = calculate_std(image_maes)

            if len(image_mses) > 1:
                task_metrics["mse_std"] = calculate_std(image_mses)

            if len(image_rmses) > 1:
                task_metrics["rmse_std"] = calculate_std(image_rmses)

            if len(image_nmaes) > 1:
                task_metrics["nmae_std"] = calculate_std(image_nmaes)

            # Store calculated metrics
            metrics[provider]["tasks"][task] = task_metrics

    return metrics

def calculate_std(values):
    """Helper function to calculate standard deviation.

    Args:
        values: List of values

    Returns:
        Standard deviation
    """
    if len(values) <= 1:
        return 0.0

    mean = sum(values) / len(values)
    variance = sum((val - mean) ** 2 for val in values) / len(values)
    return variance ** 0.5

def save_predictions_to_csv(results: dict, output_dir: Path, pre_prompt_key: str="default") -> None:
    """Save all predictions and targets to CSV files for post-experiment analysis.

    Args:
        results: The evaluation results dictionary
        output_dir: Directory to save the CSV files
        pre_prompt_key: The key/name of the pre-prompt used
    """
    csv_dir = output_dir / "csv_results"
    csv_dir.mkdir(exist_ok=True, parents=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Get the reformulation type from the results
    reformulation_type = results.get("reformulation_type", "")
    reformulate_suffix = f"_{reformulation_type}" if reformulation_type else ""

    # Extract providers and tasks
    providers = list(results["providers"].keys())
    all_tasks = []
    for provider in providers:
        for task in results["providers"][provider]["tasks"]:
            if task not in all_tasks:
                all_tasks.append(task)

    # Create a CSV file for each task, organized in task-specific folders
    for task in all_tasks:
        # Create a task-specific directory
        task_dir = csv_dir / task
        task_dir.mkdir(exist_ok=True, parents=True)

        # Create CSV file in the task directory
        csv_file = task_dir / f"predictions_{pre_prompt_key}{reformulate_suffix}_{timestamp}.csv"

        # Collect all data for this task across providers and images
        csv_data = []
        for provider in providers:
            if provider not in results["providers"] or task not in results["providers"][provider]["tasks"]:
                continue

            provider_name = results["providers"][provider]["model"]
            for image_name, image_data in results["providers"][provider]["images"].items():
                if task in image_data["tasks"]:
                    task_data = image_data["tasks"][task]

                    # Add each question, expected answer, and model answer to the data
                    for detail in task_data.get("details", []):
                        is_numeric = any(keyword in detail["question"].lower() for keyword in ["how many", "count"])

                        row = {
                            "provider": provider,
                            "model": provider_name,
                            "image": image_name,
                            "task": task,
                            "pre_prompt": pre_prompt_key,  # Add pre-prompt key to the CSV
                            "reformulation": reformulation_type,  # Add reformulation type to the CSV
                            "question": detail["question"],
                            "target": detail["expected"],
                            "prediction": detail["answer"],
                            "is_correct": detail["correct"],
                            "is_numeric": is_numeric
                        }

                        # Add extracted values for numeric questions
                        if is_numeric:
                            row["extracted_target"] = detail.get("extracted_expected")
                            row["extracted_prediction"] = detail.get("extracted_answer")

                            # Calculate error metrics for numeric questions
                            try:
                                expected_val = float(detail.get("extracted_expected", "0"))
                                actual_val = float(detail.get("extracted_answer", "0"))

                                # Basic error metrics
                                row["absolute_error"] = abs(expected_val - actual_val)
                                row["squared_error"] = row["absolute_error"] ** 2

                                # Normalized error
                                if expected_val != 0:
                                    row["normalized_error"] = row["absolute_error"] / expected_val
                                else:
                                    row["normalized_error"] = 1.0 if row["absolute_error"] > 0 else 0.0

                                # Percentage error
                                row["percentage_error"] = row["normalized_error"] * 100
                            except (ValueError, TypeError):
                                # Couldn't convert to float
                                row["absolute_error"] = None
                                row["squared_error"] = None
                                row["normalized_error"] = None
                                row["percentage_error"] = None

                        csv_data.append(row)

        # Save to CSV
        if csv_data:
            df = pd.DataFrame(csv_data)
            df.to_csv(csv_file, index=False)
            console.print(f"[green]Predictions for {task} task saved to {csv_file}[/green]")

    # Calculate and save advanced metrics
    advanced_metrics = calculate_additional_metrics(results)

    # Create a detailed metrics CSV with all the advanced metrics
    detailed_metrics_file = csv_dir / f"detailed_metrics_{pre_prompt_key}{reformulate_suffix}_{timestamp}.csv"
    detailed_metrics_rows = []

    for provider, provider_data in advanced_metrics.items():
        model_name = provider_data["model"]

        for task, metrics in provider_data["tasks"].items():
            row = {
                "provider": provider,
                "model": model_name,
                "task": task,
                "pre_prompt": pre_prompt_key,
                "reformulation": reformulation_type
            }
            # Add all calculated metrics
            row.update(metrics)
            detailed_metrics_rows.append(row)

    if detailed_metrics_rows:
        df = pd.DataFrame(detailed_metrics_rows)
        df.to_csv(detailed_metrics_file, index=False)
        console.print(f"[green]Detailed metrics saved to {detailed_metrics_file}[/green]")

    # Create a summary CSV with aggregated metrics - also organized by task folders
    for task in all_tasks:
        task_dir = csv_dir / task
        task_summary_file = task_dir / f"summary_metrics_{pre_prompt_key}{reformulate_suffix}_{timestamp}.csv"
        task_summary_data = []

        for provider in providers:
            if provider not in results["providers"]:
                continue

            provider_name = results["providers"][provider]["model"]

            if task not in results["providers"][provider]["tasks"]:
                continue

            task_metrics = results["providers"][provider]["tasks"][task]

            # Get the advanced metrics as well
            advanced_task_metrics = {}
            if provider in advanced_metrics and task in advanced_metrics[provider]["tasks"]:
                advanced_task_metrics = advanced_metrics[provider]["tasks"][task]

            row = {
                "provider": provider,
                "model": provider_name,
                "task": task,
                "pre_prompt": pre_prompt_key,
                "reformulation": reformulation_type,
                "accuracy": task_metrics.get("accuracy", 0),
                "normalized_mae": task_metrics.get("normalized_mae", float('nan')),
                "total_questions": task_metrics.get("count", 0)
            }

            # Add the calculated advanced metrics
            if advanced_task_metrics:
                row.update({
                    "f1_score": advanced_task_metrics.get("f1_score", float('nan')),
                    "precision": advanced_task_metrics.get("precision", float('nan')),
                    "recall": advanced_task_metrics.get("recall", float('nan')),
                    "mae": advanced_task_metrics.get("mae", float('nan')),  # Added MAE
                    "mse": advanced_task_metrics.get("mse", float('nan')),  # Added MSE
                    "rmse": advanced_task_metrics.get("rmse", float('nan')),
                    "reliability": advanced_task_metrics.get("reliability", float('nan')),
                    "reliability_std": advanced_task_metrics.get("reliability_std", float('nan')),
                    "consistency": advanced_task_metrics.get("consistency", float('nan')),
                    # Add standard deviation metrics
                    "accuracy_std": advanced_task_metrics.get("accuracy_std", float('nan')),
                    "f1_score_std": advanced_task_metrics.get("f1_score_std", float('nan')),
                    "precision_std": advanced_task_metrics.get("precision_std", float('nan')),
                    "recall_std": advanced_task_metrics.get("recall_std", float('nan')),
                    "mae_std": advanced_task_metrics.get("mae_std", float('nan')),
                    "mse_std": advanced_task_metrics.get("mse_std", float('nan')),
                    "rmse_std": advanced_task_metrics.get("rmse_std", float('nan')),
                    "nmae_std": advanced_task_metrics.get("nmae_std", float('nan')),
                    "consistency_std": advanced_task_metrics.get("consistency_std", float('nan'))
                })

            task_summary_data.append(row)

        if task_summary_data:
            df = pd.DataFrame(task_summary_data)
            df.to_csv(task_summary_file, index=False)
            console.print(f"[green]Summary metrics for {task} task saved to {task_summary_file}[/green]")

    # Also create a global summary file with all tasks for comparison
    summary_file = csv_dir / f"all_tasks_summary_{pre_prompt_key}{reformulate_suffix}_{timestamp}.csv"
    summary_data = []

    for provider in providers:
        if provider not in results["providers"]:
            continue

        provider_name = results["providers"][provider]["model"]

        for task in all_tasks:
            if task not in results["providers"][provider]["tasks"]:
                continue

            task_metrics = results["providers"][provider]["tasks"][task]

            # Get the advanced metrics as well
            advanced_task_metrics = {}
            if provider in advanced_metrics and task in advanced_metrics[provider]["tasks"]:
                advanced_task_metrics = advanced_metrics[provider]["tasks"][task]

            row = {
                "provider": provider,
                "model": provider_name,
                "task": task,
                "pre_prompt": pre_prompt_key,
                "reformulation": reformulation_type,
                "accuracy": task_metrics.get("accuracy", 0),
                "normalized_mae": task_metrics.get("normalized_mae", float('nan')),
                "total_questions": task_metrics.get("count", 0)
            }

            # Add the calculated advanced metrics
            if advanced_task_metrics:
                row.update({
                    "f1_score": advanced_task_metrics.get("f1_score", float('nan')),
                    "precision": advanced_task_metrics.get("precision", float('nan')),
                    "recall": advanced_task_metrics.get("recall", float('nan')),
                    "mae": advanced_task_metrics.get("mae", float('nan')),  # Added MAE
                    "mse": advanced_task_metrics.get("mse", float('nan')),  # Added MSE
                    "rmse": advanced_task_metrics.get("rmse", float('nan')),
                    "reliability": advanced_task_metrics.get("reliability", float('nan')),
                    "reliability_std": advanced_task_metrics.get("reliability_std", float('nan')),
                    "consistency": advanced_task_metrics.get("consistency", float('nan')),
                    # Add standard deviation metrics
                    "accuracy_std": advanced_task_metrics.get("accuracy_std", float('nan')),
                    "f1_score_std": advanced_task_metrics.get("f1_score_std", float('nan')),
                    "precision_std": advanced_task_metrics.get("precision_std", float('nan')),
                    "recall_std": advanced_task_metrics.get("recall_std", float('nan')),
                    "mae_std": advanced_task_metrics.get("mae_std", float('nan')),
                    "mse_std": advanced_task_metrics.get("mse_std", float('nan')),
                    "rmse_std": advanced_task_metrics.get("rmse_std", float('nan')),
                    "nmae_std": advanced_task_metrics.get("nmae_std", float('nan')),
                    "consistency_std": advanced_task_metrics.get("consistency_std", float('nan'))
                })

            summary_data.append(row)

    if summary_data:
        df = pd.DataFrame(summary_data)
        df.to_csv(summary_file, index=False)
        console.print(f"[green]Global summary metrics saved to {summary_file}[/green]")

def generate_latex_tables(results: dict, output_dir: Path, pre_prompt_key: str="default") -> None:
    """Generate LaTeX tables for the evaluation results in NeurIPS style.

    Args:
        results: The evaluation results dictionary
        output_dir: Directory to save the LaTeX files
        pre_prompt_key: The key/name of the pre-prompt used
    """
    latex_dir = output_dir / "latex_tables"
    latex_dir.mkdir(exist_ok=True, parents=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Get the reformulation type from the results
    reformulation_type = results.get("reformulation_type", "")
    reformulation_text = f" with {reformulation_type} reformulation" if reformulation_type else ""
    reformulate_suffix = f"_{reformulation_type}" if reformulation_type else ""

    # Create a preamble file with required packages and styles for NeurIPS format
    preamble_file = latex_dir / "neurips_table_preamble.tex"
    with open(preamble_file, "w") as f:
        f.write("""% NeurIPS style preamble for tables
% Include this in your LaTeX document
\\usepackage{booktabs}
\\usepackage{xcolor}
\\usepackage{colortbl}
""")

    # Extract relevant data for tables
    providers = list(results["providers"].keys())
    tasks = []
    for provider in providers:
        for task in results["providers"][provider]["tasks"]:
            if task not in tasks:
                tasks.append(task)

    # Create task-specific latex directories
    for task in tasks:
        task_latex_dir = latex_dir / task
        task_latex_dir.mkdir(exist_ok=True, parents=True)

    # ----- Generate detailed question tables with improved style -----
    for provider in providers:
        model_name = results["providers"][provider]["model"]
        provider_short = model_name.replace("/", "-").replace(":", "-")
        short_name = model_name.split('/')[-1] if '/' in model_name else model_name

        # Create task-specific LaTeX files for each provider
        for task in tasks:
            if task in results["providers"][provider]["tasks"]:
                task_latex_dir = latex_dir / task
                latex_file = task_latex_dir / f"detailed_results_{provider_short}_{pre_prompt_key}{reformulate_suffix}_{timestamp}.tex"

                with open(latex_file, "w") as f:
                    f.write("""% Detailed evaluation results - NeurIPS format
% Generated on: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """

% Include the preamble file for styling
% \\input{../neurips_table_preamble.tex}

""")

                    f.write(f"\\section{{Evaluation of {short_name} on {task.capitalize()} Task}}\n\n")

                    # Create one consolidated table per task instead of per image for cleaner presentation
                    f.write("\\begin{table}[htbp]\n")
                    f.write("\\centering\n")
                    f.write("\\caption{" + f"Detailed evaluation of {short_name} on {task.capitalize()} Task{reformulation_text}" + "}\n")
                    f.write("\\label{tab:" + f"details_{provider_short}_{task}_{pre_prompt_key}" + "}\n")
                    f.write("\\renewcommand{\\arraystretch}{1.2}\n") # Better row spacing
                    f.write("\\begin{tabular}{p{0.27\\textwidth}p{0.15\\textwidth}p{0.27\\textwidth}c}\n")
                    f.write("\\toprule\n")
                    f.write("\\headrow \\textbf{Question} & \\textbf{Expected} & \\textbf{Model Response} & \\textbf{Correct} \\\\\n")
                    f.write("\\midrule\n")

                    # Group by image
                    image_counter = 0
                    for image_name, image_data in results["providers"][provider]["images"].items():
                        if task in image_data["tasks"]:
                            task_data = image_data["tasks"][task]
                            image_counter += 1

                            # Add image header row
                            f.write("\\midrule\n")
                            f.write(f"\\multicolumn{{4}}{{l}}{{\\subheadrow \\textbf{{Image {image_counter}: {image_name}}}}} \\\\\n")
                            f.write("\\midrule\n")

                            # Add each question and answer with improved formatting
                            for detail in task_data.get("details", []):
                                question = detail["question"].replace("_", "\\_").replace("#", "\\#").replace("%", "\\%").replace("&", "\\&").replace("~", "\\~{}")
                                expected = str(detail["expected"]).replace("_", "\\_").replace("#", "\\#").replace("%", "\\%").replace("&", "\\&").replace("~", "\\~{}")
                                answer = str(detail["answer"]).replace("_", "\\_").replace("#", "\\#").replace("%", "\\%").replace("&", "\\&").replace("~", "\\~{}")

                                # Truncate long answers but in a smarter way
                                if len(answer) > 50:
                                    # Try to truncate at a word boundary
                                    cutoff = answer[:47].rfind(' ')
                                    if cutoff > 30:  # Ensure we don't cut too early
                                        answer = answer[:cutoff] + "..."
                                    else:
                                        answer = answer[:47] + "..."

                                # Use checkmarks and x-marks with colors
                                correct_symbol = "\\textcolor{green}{\\checkmark}" if detail["correct"] else "\\textcolor{red}{\\times}"

                                f.write(f"{question} & {expected} & {answer} & {correct_symbol} \\\\\n")

                    f.write("\\bottomrule\n")
                    f.write("\\end{tabular}\n")
                    f.write("\\end{table}\n\n")

                    # Add a concise metrics summary table for this task
                    f.write("\\begin{table}[htbp]\n")
                    f.write("\\centering\n")
                    f.write("\\caption{" + f"Performance metrics for {short_name} on {task.capitalize()} Task{reformulation_text}" + "}\n")
                    f.write("\\label{tab:" + f"metrics_{provider_short}_{task}_{pre_prompt_key}" + "}\n")
                    f.write("\\renewcommand{\\arraystretch}{1.2}\n")
                    f.write("\\begin{tabular}{lc}\n")
                    f.write("\\toprule\n")
                    f.write("\\headrow \\textbf{Metric} & \\textbf{Value} \\\\\n")
                    f.write("\\midrule\n")

                    # Aggregate metrics across all images for this task
                    total_questions = 0
                    correct_count = 0
                    total_accuracy = 0
                    total_nmae = 0
                    image_count = 0

                    for image_name, image_data in results["providers"][provider]["images"].items():
                        if task in image_data["tasks"]:
                            task_data = image_data["tasks"][task]
                            total_questions += task_data.get("total_questions", 0)
                            correct_count += task_data.get("correct_count", 0)
                            total_accuracy += task_data.get("accuracy", 0)
                            total_nmae += task_data.get("normalized_mae", 0)
                            image_count += 1

                    avg_accuracy = total_accuracy / image_count if image_count > 0 else 0
                    avg_nmae = total_nmae / image_count if image_count > 0 else 0
                    overall_accuracy = correct_count / total_questions if total_questions > 0 else 0

                    f.write(f"Overall Accuracy & {overall_accuracy:.2f} \\\\\n")
                    f.write(f"Average Accuracy per Image & {avg_accuracy:.2f} \\\\\n")
                    if "counting" in task.lower():
                        f.write(f"Average Normalized MAE & {avg_nmae:.2f} \\\\\n")
                    f.write(f"Total Questions & {total_questions} \\\\\n")
                    f.write(f"Total Correct Answers & {correct_count} \\\\\n")
                    f.write(f"Number of Images & {image_count} \\\\\n")

                    f.write("\\bottomrule\n")
                    f.write("\\end{tabular}\n")
                    f.write("\\end{table}\n\n")

    # ----- Generate comprehensive summary table in NeurIPS style -----
    # Create task-specific summary tables as well as an overall summary
    for task in tasks:
        task_latex_dir = latex_dir / task
        task_summary = task_latex_dir / f"task_summary_{pre_prompt_key}{reformulate_suffix}_{timestamp}.tex"

        with open(task_summary, "w") as f:
            f.write(f"""% Model Performance Summary for {task.capitalize()} Task - NeurIPS format
% Generated on: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """

\\begin{{table*}}[ht]
\\centering
\\caption{{Comparative Performance of Vision-Language Models on {task.capitalize()} Task}}
\\label{{tab:model_performance_{task}_{pre_prompt_key}}}
\\vspace{{0.8mm}}
\\renewcommand{{\\arraystretch}}{{1.2}}
\\setlength{{\\tabcolsep}}{{4pt}} % Reduce column spacing
""")

            # Create header row with model names
            num_providers = len(providers)
            f.write(f"\\begin{{tabular}}{{l{' c' * num_providers}}}\n")
            f.write("\\toprule\n")
            f.write("\\textbf{Metric} ")

            for provider in providers:
                if provider in results["providers"]:
                    model_name = results["providers"][provider]["model"]
                    # Get a shorter name
                    short_name = ""
                    if "gpt-4.1-mini" in model_name:
                        short_name = "G4.1m"
                    elif "gpt-4.1" in model_name:
                        short_name = "G4.1"
                    elif "llama-3-inst" in model_name or "llama-3-vist" in model_name:
                        short_name = "L3-ins"
                    elif "llama-3.2-vision" in model_name:
                        short_name = "L3-vis"
                    elif "gemma3:12b" in model_name or "gemma3-2" in model_name:
                        short_name = "G3-v2"
                    elif "gemma3" in model_name:
                        short_name = "G3"
                    elif "mistral" in model_name:
                        short_name = "M3.1"
                    elif "blip-large" in model_name:
                        short_name = "BLIP-l"
                    elif "blip" in model_name:
                        short_name = "BLIP"
                    else:
                        # Generic shortening
                        if '/' in model_name:
                            parts = model_name.split('/')
                            short_name = parts[-1][:5]
                        elif ':' in model_name:
                            short_name = model_name.split(':')[0][:5]
                        else:
                            short_name = model_name[:5]

                    f.write(f"& \\textbf{{{short_name}}} ")
            f.write("\\\\\n")
            f.write("\\midrule\n")

            # Write metrics for this task
            metrics = ["Accuracy", "NMAE", "F1", "Precision", "Recall"]
            metric_keys = ["accuracy", "normalized_mae", "f1_score", "precision", "recall"]

            for metric, key in zip(metrics, metric_keys, strict=False):
                f.write(f"\\textbf{{{metric}}} ")
                for provider in providers:
                    if (provider in results["providers"] and
                        task in results["providers"][provider]["tasks"]):

                        if key == "normalized_mae" and "counting" not in task.lower():
                            f.write("& - ")
                            continue

                        task_data = results["providers"][provider]["tasks"][task]
                        value = task_data.get(key, float('nan'))
                        f.write(f"& {value:.2f} ")
                    else:
                        f.write("& - ")
                f.write("\\\\\n")

            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table*}\n")

    # Create the overall summary table
    latex_summary = latex_dir / f"all_tasks_summary_{pre_prompt_key}{reformulate_suffix}_{timestamp}.tex"

    with open(latex_summary, "w") as f:
        f.write("""% Model Performance Summary - NeurIPS format
% Generated on: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """

\\begin{table*}[ht]
\\centering
\\caption{Comparative Performance of Vision-Language Models on Diagnostic Tasks""" + reformulation_text + """}
\\label{tab:model_performance_summary_{pre_prompt_key}}
\\vspace{0.8mm}
\\renewcommand{\\arraystretch}{1.2}
\\setlength{\\tabcolsep}{4pt} % Reduce column spacing
""")

        # Create header row with model names, using shortened versions
        num_providers = len(providers)
        f.write(f"\\begin{{tabular}}{{l{' c' * num_providers}}}\n")
        f.write("\\toprule\n")
        f.write("\\textbf{Task} ")

        for provider in providers:
            model_name = results["providers"][provider]["model"]
            # Get a shorter, concise name for the model
            short_name = ""
            if "gpt-4.1-mini" in model_name:
                short_name = "G4.1m"
            elif "gpt-4.1" in model_name:
                short_name = "G4.1"
            elif "llama-3-inst" in model_name or "llama-3-vist" in model_name:
                short_name = "L3-ins"
            elif "llama-3.2-vision" in model_name:
                short_name = "L3-vis"
            elif "gemma3:12b" in model_name or "gemma3-2" in model_name:
                short_name = "G3-v2"
            elif "gemma3" in model_name:
                short_name = "G3"
            elif "mistral" in model_name:
                short_name = "M3.1"
            elif "blip-large" in model_name:
                short_name = "BLIP-l"
            elif "blip" in model_name:
                short_name = "BLIP"
            else:
                # Generic shortening if none of the above match
                if '/' in model_name:
                    parts = model_name.split('/')
                    short_name = parts[-1][:5]
                elif ':' in model_name:
                    short_name = model_name.split(':')[0][:5]
                else:
                    short_name = model_name[:5]

            f.write(f"& \\textbf{{{short_name}}} ")
        f.write("\\\\\n")
        f.write("\\midrule\n")

        # Write each task's metrics
        for task in tasks:
            f.write(f"\\textbf{{{task.capitalize()}}} ")
            for provider in providers:
                if (provider in results["providers"] and
                    task in results["providers"][provider]["tasks"]):

                    metrics = results["providers"][provider]["tasks"][task]
                    accuracy = metrics.get("accuracy", 0)

                    f.write(f"& {accuracy:.2f} ")
                else:
                    f.write("& - ")
            f.write("\\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table*}\n")

        # Add a table note
        f.write("""
\\vspace{1mm}
\\begin{minipage}{\\textwidth}
\\small
\\textit{Note:} Values shown are accuracy scores. Best performance on each task is highlighted in \\textbf{bold}.
All values are averaged across """ + str(len(list(results["providers"][providers[0]]["images"].keys()))) + """ diagnostic images.
""" + (f"Reformulation: {reformulation_type}. " if reformulation_type else "") + """Pre-prompt: {pre_prompt_key}.
\\end{minipage}
""")

    console.print(f"[green]NeurIPS-style LaTeX tables generated in {latex_dir}[/green]")
    console.print(f"[blue]1. Overall summary table: {latex_summary}[/blue]")
    console.print("[blue]2. Task-specific summaries in task folders[/blue]")
    console.print(f"[blue]3. Style preamble: {preamble_file}[/blue]")

def generate_extended_latex_tables(results: dict, metrics: dict, output_dir: Path, pre_prompt_key: str="default", include_std: bool = True) -> None:
    """Generate enhanced LaTeX tables with additional metrics for paper.

    Args:
        results: The evaluation results dictionary
        metrics: Additional metrics dictionary
        output_dir: Directory to save the LaTeX files
        pre_prompt_key: The key/name of the pre-prompt used
        include_std: Whether to include standard deviation values in parentheses
    """
    latex_dir = output_dir / "latex_tables"
    latex_dir.mkdir(exist_ok=True, parents=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Get the reformulation type from the results
    reformulation_type = results.get("reformulation_type", "")
    reformulation_text = f" with {reformulation_type} reformulation" if reformulation_type else ""
    reformulate_suffix = f"_{reformulation_type}" if reformulation_type else ""

    # Extract providers and tasks
    providers = list(metrics.keys())
    tasks = set()
    for provider in providers:
        for task in metrics[provider]["tasks"]:
            tasks.add(task)
    tasks = sorted(tasks)

    # Create task-specific directories if they don't exist
    for task in tasks:
        task_dir = latex_dir / task
        task_dir.mkdir(exist_ok=True, parents=True)

    # Define the metrics to display
    metric_columns = [
        ("Accuracy", "accuracy", "↑"),
        ("F1", "f1_score", "↑"),
        ("MAE", "mae", "↓"),     # Added MAE
        ("MSE", "mse", "↓"),     # Added MSE
        ("NMAE", "nmae", "↓"),
        ("RMSE", "rmse", "↓"),
        ("Consistency", "consistency", "↑")
    ]

    # Create a task-specific metrics comparison file for each task
    for task in tasks:
        task_dir = latex_dir / task
        task_metrics_file = task_dir / f"extended_metrics_{pre_prompt_key}{reformulate_suffix}_{timestamp}.tex"

        with open(task_metrics_file, "w") as f:
            f.write(f"""% Comprehensive Metrics for {task.capitalize()} Task - NeurIPS format
% Generated on: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """

\\begin{{table*}}[ht]
\\centering
\\caption{{Multi-dimensional Performance Analysis of Vision-Language Models on {task.capitalize()} Task{reformulation_text}}}
\\label{{tab:metrics_comparison_{task}_{pre_prompt_key}}}
\\vspace{{0.8mm}}
\\renewcommand{{\\arraystretch}}{{1.2}}
\\setlength{{\\tabcolsep}}{{4pt}} % Reduce column spacing
""")

            # Create the table
            metric_count = len(metric_columns)
            col_spec = "l" + " c" * len(providers)
            f.write(f"\\begin{{tabular}}{{{col_spec}}}\n")
            f.write("\\toprule\n")

            # Create header with model names
            f.write("\\textbf{Metric} ")
            for provider in providers:
                if provider in metrics:
                    model_name = metrics[provider]["model"]
                    # Get a shorter name
                    short_name = ""
                    if "gpt-4.1-mini" in model_name:
                        short_name = "GPT4.1m"
                    elif "gpt-4.1" in model_name:
                        short_name = "GPT4.1"
                    elif "llama-3-inst" in model_name:
                        short_name = "L3-ins"
                    elif "llama-3.2-vision" in model_name:
                        short_name = "L3-vis"
                    elif "gemma3:12b" in model_name:
                        short_name = "G3:12b"
                    elif "gemma3" in model_name:
                        short_name = "G3"
                    elif "mistral" in model_name:
                        short_name = "M3.1"
                    elif "blip" in model_name:
                        short_name = "BLIP"
                    else:
                        short_name = model_name[:5]

                    f.write(f"& \\textbf{{{short_name}}} ")
            f.write("\\\\\n")
            f.write("\\midrule\n")

            # Write each model's metrics for this task
            for _metric_idx, (name, key, direction) in enumerate(metric_columns):
                # Skip NMAE and MSE for non-counting tasks
                if (key in ["nmae", "mse", "mae", "rmse"] and
                    "counting" not in task.lower()):
                    continue

                f.write(f"{name} {direction} ")

                # Find best value for this metric
                best_value = None
                for provider in providers:
                    if (provider in metrics and
                        "tasks" in metrics[provider] and
                        task in metrics[provider]["tasks"] and
                        key in metrics[provider]["tasks"][task]):

                        value = metrics[provider]["tasks"][task][key]
                        if best_value is None or direction == "↑" and value > best_value or direction == "↓" and value < best_value:
                            best_value = value

                # Write values for each model, highlighting the best one
                for provider in providers:
                    if (provider in metrics and
                        "tasks" in metrics[provider] and
                        task in metrics[provider]["tasks"] and
                        key in metrics[provider]["tasks"][task]):

                        value = metrics[provider]["tasks"][task][key]
                        std_key = f"{key}_std"
                        std_value = metrics[provider]["tasks"][task].get(std_key, 0) if include_std else 0

                        # Highlight best value
                        if abs(value - best_value) < 0.001:
                            if include_std and std_value > 0:
                                f.write(f"& \\textbf{{{value:.3f} $\\pm$ {std_value:.3f}}} ")
                            else:
                                f.write(f"& \\textbf{{{value:.3f}}} ")
                        else:
                            if include_std and std_value > 0:
                                f.write(f"& {value:.3f} $\\pm$ {std_value:.3f} ")
                            else:
                                f.write(f"& {value:.3f} ")
                    else:
                        f.write("& - ")

                f.write("\\\\\n")

            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")

            # Add a table note
            f.write("""
\\vspace{1mm}
\\begin{minipage}{\\textwidth}
\\small
\\textit{Note:} Best performance on each metric is highlighted in \\textbf{bold}.
Arrows indicate whether higher (↑) or lower (↓) values are better.
NMAE = Normalized Mean Absolute Error; RMSE = Root Mean Squared Error.""" +
(f" Reformulation: {reformulation_type}." if reformulation_type else "") +
(" Values shown with $\\pm$ standard deviation." if include_std else "") + """
\\end{minipage}
""")

            f.write("\\end{table*}\n")

    # Create a global metrics comparison table for all tasks
    metrics_comparison_file = latex_dir / f"all_tasks_metrics_comparison_{pre_prompt_key}{reformulate_suffix}_{timestamp}.tex"

    with open(metrics_comparison_file, "w") as f:
        f.write("""% Comprehensive Metrics Comparison - NeurIPS format
% Generated on: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """

\\begin{table*}[ht]
\\centering
\\caption{Multi-dimensional Performance Analysis of Vision-Language Models""" + reformulation_text + """}
\\label{tab:metrics_comparison_{pre_prompt_key}}
\\vspace{0.8mm}
\\renewcommand{\\arraystretch}{1.2}
\\setlength{\\tabcolsep}{4pt} % Reduce column spacing
""")

        # Create a subtable for each task
        for task in tasks:
            f.write(f"\\subsection*{{{task.capitalize()}}}\n\n")

            # Create the subtable
            metric_count = len(metric_columns)
            col_spec = "l" + " c" * (len(providers) * metric_count)
            f.write(f"\\begin{{tabular}}{{{col_spec}}}\n")
            f.write("\\toprule\n")

            # Create header with model names spanning multiple columns
            f.write("\\multirow{2}{*}{\\textbf{Metric}} ")
            for provider in providers:
                if provider in metrics:
                    model_name = metrics[provider]["model"]
                    # Get a shorter name
                    short_name = ""
                    if "gpt-4.1-mini" in model_name:
                        short_name = "GPT4.1m"
                    elif "gpt-4.1" in model_name:
                        short_name = "GPT4.1"
                    elif "llama-3-inst" in model_name:
                        short_name = "L3-ins"
                    elif "llama-3.2-vision" in model_name:
                        short_name = "L3-vis"
                    elif "gemma3:12b" in model_name:
                        short_name = "G3:12b"
                    elif "gemma3" in model_name:
                        short_name = "G3"
                    elif "mistral" in model_name:
                        short_name = "M3.1"
                    elif "blip" in model_name:
                        short_name = "BLIP"
                    else:
                        short_name = model_name[:5]

                    f.write(f"& \\multicolumn{{{metric_count}}}{{c}}{{\\textbf{{{short_name}}}}} ")
            f.write("\\\\\n")

            # Create the metric type row
            f.write("\\cmidrule{2-" + str(len(providers) * metric_count + 1) + "}\n")

            for provider in providers:
                for name, key, _ in metric_columns:
                    f.write(f"& {name} ")
            f.write("\\\\\n")
            f.write("\\midrule\n")

            # Write each model's metrics for this task
            # We're going to re-organize by metric rather than by model
            for _metric_idx, (name, key, direction) in enumerate(metric_columns):
                f.write(f"{name} {direction} ")

                # Find best value for this metric
                best_value = None
                for provider in providers:
                    if (provider in metrics and
                        "tasks" in metrics[provider] and
                        task in metrics[provider]["tasks"] and
                        key in metrics[provider]["tasks"][task]):

                        value = metrics[provider]["tasks"][task][key]
                        if best_value is None or direction == "↑" and value > best_value or direction == "↓" and value < best_value:
                            best_value = value

                # Write values for each model, highlighting the best one
                for provider in providers:
                    for _col_idx, (_, metric_key, _) in enumerate(metric_columns):
                        if metric_key == key:
                            if (provider in metrics and
                                "tasks" in metrics[provider] and
                                task in metrics[provider]["tasks"] and
                                key in metrics[provider]["tasks"][task]):

                                value = metrics[provider]["tasks"][task][key]
                                std_key = f"{key}_std"
                                std_value = metrics[provider]["tasks"][task].get(std_key, 0) if include_std else 0

                                # Highlight best value
                                if abs(value - best_value) < 0.001:
                                    if include_std and std_value > 0:
                                        f.write(f"& \\textbf{{{value:.3f} $\\pm$ {std_value:.3f}}} ")
                                    else:
                                        f.write(f"& \\textbf{{{value:.3f}}} ")
                                else:
                                    if include_std and std_value > 0:
                                        f.write(f"& {value:.3f} $\\pm$ {std_value:.3f} ")
                                    else:
                                        f.write(f"& {value:.3f} ")
                            else:
                                f.write("& - ")
                        else:
                            # This cell doesn't belong to the current metric row
                            pass

                f.write("\\\\\n")

            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")

            # Add a table note
            f.write("""
\\vspace{1mm}
\\begin{minipage}{\\textwidth}
\\small
\\textit{Note:} Best performance on each metric is highlighted in \\textbf{bold}.
Arrows indicate whether higher (↑) or lower (↓) values are better.
NMAE = Normalized Mean Absolute Error; RMSE = Root Mean Squared Error.""" +
(f" Reformulation: {reformulation_type}." if reformulation_type else "") + """
\\end{minipage}
""")

            f.write("\\end{table*}\n")

    console.print("[green]Enhanced metrics tables generated:[/green]")
    console.print(f"[blue]1. Global comparison: {metrics_comparison_file}[/blue]")
    console.print("[blue]2. Task-specific metrics in task folders[/blue]")

def run_evaluation(selected_images: list[tuple[Path, Path]], selected_providers: list[str],
                  selected_tasks: list[str], reformulate: str="", pre_prompt: str="",
                  pre_prompt_key: str="default") -> dict:
    """Run the evaluation on selected images, providers, and tasks.

    Args:
        selected_images: List of (image_path, annotation_path) tuples
        selected_providers: List of provider keys to use
        selected_tasks: List of tasks to evaluate
        reformulate: The format for reformulating questions
        pre_prompt: The pre-prompt to add to each question
        pre_prompt_key: The key/name of the pre-prompt being used

    Returns:
        Dictionary with all evaluation results
    """
    results = {
        "timestamp": datetime.now().isoformat(),
        "providers": {},
        "summary": {},
        "pre_prompt_key": pre_prompt_key,  # Store the pre-prompt key in the results
        "reformulation_type": reformulate  # Store the reformulation type in the results
    }

    # For each provider
    for provider_key in selected_providers:
        provider_config = PROVIDERS[provider_key]
        provider_results = {
            "model": provider_config["model_name"],
            "images": {},
            "tasks": {task: {"accuracy": 0, "normalized_mae": 0, "count": 0} for task in selected_tasks}
        }

        # Initialize VLM
        with console.status(f"[bold blue]Initializing {provider_key} provider...", spinner="dots"):
            try:
                credentials = provider_config["credentials"]
                vlm = get_vlm(
                    provider=provider_config["provider"],
                    model_name=provider_config["model_name"],
                    **credentials
                )
                console.print(f"[green]Successfully initialized {provider_key} with model {provider_config['model_name']}[/green]")
            except Exception as e:
                console.print(f"[bold red]Error initializing {provider_key}: {e}[/bold red]")
                continue  # Skip to next provider if initialization fails

        # Process each image
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            image_task = progress.add_task(f"[cyan]Processing images with {provider_key}", total=len(selected_images))

            for i, (image_path, annotation_path) in enumerate(selected_images):
                image_name = image_path.stem
                progress.update(image_task, description=f"[cyan]Processing image {i+1}/{len(selected_images)}: {image_name}")
                console.print(f"[bold magenta][DEBUG] About to evaluate image: {image_name}, provider: {provider_key}")

                # Get annotation info
                annotation_info = extract_annotation_info(annotation_path)

                # Initialize results for this image
                image_results = {
                    "annotation_info": annotation_info,
                    "tasks": {}
                }

                # Process each task
                task_progress = progress.add_task(f"[magenta]Running tasks for {image_name}", total=len(selected_tasks))

                for task in selected_tasks:
                    progress.update(task_progress, description=f"[magenta]Running {task} task for {image_name}")

                    # Generate question keys for this image and task
                    question_keys, answers, dynamic_values_list = generate_questions_for_image(annotation_path, task, reformulate, pre_prompt)
                    if not question_keys:
                        console.print(f"[bold red][DEBUG] No question keys generated for image {image_name}, task {task}. Skipping.")
                        continue

                    # Evaluate VLM on this task
                    evaluation = evaluate_vlm_on_task(vlm, image_path, annotation_path, question_keys, reformulate, pre_prompt)
                    image_results["tasks"][task] = evaluation

                    # Update task statistics
                    provider_results["tasks"][task]["accuracy"] += evaluation["accuracy"]
                    provider_results["tasks"][task]["normalized_mae"] += evaluation.get("normalized_mae", 0)
                    provider_results["tasks"][task]["count"] += 1

                    # Update progress
                    progress.update(task_progress, advance=1)

                    # Add a small delay to avoid rate limits
                    time.sleep(0.5)

                # Store results for this image
                provider_results["images"][image_name] = image_results

                # Clean up task progress bar
                # progress.remove_task(task_progress)
                progress.update(task_progress, visible=False)


                # Advance the image progress
                progress.update(image_task, advance=1)

                # Save intermediate results
                results["providers"][provider_key] = provider_results
                with open(OUTPUT_DIR / f"intermediate_results_{provider_key}.json", "w") as f:
                    json.dump(results, f, indent=2)

        # Calculate average metrics for each task
        for task in selected_tasks:
            task_count = provider_results["tasks"][task]["count"]
            if task_count > 0:
                provider_results["tasks"][task]["accuracy"] /= task_count
                provider_results["tasks"][task]["normalized_mae"] /= task_count

        # Display provider summary
        table = Table(title=f"{provider_key.upper()} Results Summary")
        table.add_column("Task", style="cyan")
        table.add_column("Accuracy", style="green")
        table.add_column("Normalized MAE", style="yellow")
        table.add_column("Reformulation", style="magenta")

        # Format reformulation display
        displayed_reformulation = reformulate if reformulate else "None"

        for task, metrics in provider_results["tasks"].items():
            if metrics["count"] > 0:
                table.add_row(
                    task,
                    f"{metrics['accuracy']:.2f}",
                    f"{metrics['normalized_mae']:.2f}",
                    displayed_reformulation
                )

        console.print(table)

    # Calculate cross-provider comparison
    if len(selected_providers) > 1:
        summary_table = Table(title="Cross-Provider Comparison")
        summary_table.add_column("Task", style="cyan")

        for provider in selected_providers:
            if provider in results["providers"]:
                summary_table.add_column(f"{PROVIDERS[provider]['model_name'].upper()}", style="green")

        summary_table.add_column("Reformulation", style="magenta")

        for task in selected_tasks:
            row_data = [task]
            for provider in selected_providers:
                if provider in results["providers"] and task in results["providers"][provider]["tasks"]:
                    metrics = results["providers"][provider]["tasks"][task]
                    row_data.append(f"{metrics['accuracy']:.2f}")

            # Add reformulation as the last column
            row_data.append(displayed_reformulation)

            summary_table.add_row(*row_data)

        console.print(summary_table)

    # Save final results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    reformulate_suffix = f"_{reformulate}" if reformulate else ""
    results_file = OUTPUT_DIR / f"diagnostic_evaluation_results_{pre_prompt_key}{reformulate_suffix}_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    console.print(f"[green]Results for pre-prompt '{pre_prompt_key}'{' with reformulation ' + reformulate if reformulate else ''} saved to {results_file}[/green]")

    # Save predictions and targets to CSV files
    save_predictions_to_csv(results, OUTPUT_DIR, pre_prompt_key)

    # Calculate additional metrics
    enhanced_metrics = calculate_additional_metrics(results)

    # Generate LaTeX tables for paper publication
    generate_latex_tables(results, OUTPUT_DIR, pre_prompt_key)
    generate_extended_latex_tables(results, enhanced_metrics, OUTPUT_DIR, pre_prompt_key, include_std=True)

    # Display summary of enhanced metrics
    display_enhanced_metrics_summary(enhanced_metrics, reformulate, include_std=True)

    return results

def display_enhanced_metrics_summary(metrics: dict, reformulation_type: str = "", include_std: bool = True) -> None:
    """Display a summary table of the enhanced metrics in the terminal.

    Args:
        metrics: The enhanced metrics dictionary
        reformulation_type: The reformulation type used (if any)
        include_std: Whether to include standard deviation values in parentheses
    """
    # Create a summary table for each task
    providers = list(metrics.keys())
    tasks = set()
    for provider in providers:
        for task in metrics[provider]["tasks"]:
            tasks.add(task)

    for task in sorted(tasks):
        table = Table(title=f"Enhanced Metrics for {task.capitalize()} Task")
        if reformulation_type:
            table.title = f"{table.title} with {reformulation_type} Reformulation"

        # Add columns
        table.add_column("Metric", style="cyan")
        for provider in providers:
            if provider in metrics:
                model_name = metrics[provider]["model"]
                # Get a shorter name for display
                if '/' in model_name:
                    model_name = model_name.split('/')[-1]
                elif ':' in model_name:
                    model_name = model_name.split(':')[0]
                table.add_column(f"{model_name}", style="green")

        # Add rows for each metric
        metric_keys = ["accuracy", "f1_score", "precision", "recall", "mae", "mse", "nmae", "rmse", "reliability", "consistency"]
        metric_names = ["Accuracy", "F1 Score", "Precision", "Recall", "MAE", "MSE", "NMAE", "RMSE", "Reliability", "Consistency"]

        for metric_key, metric_name in zip(metric_keys, metric_names, strict=False):
            row = [metric_name]

            for provider in providers:
                if (provider in metrics and
                    "tasks" in metrics[provider] and
                    task in metrics[provider]["tasks"] and
                    metric_key in metrics[provider]["tasks"][task]):

                    value = metrics[provider]["tasks"][task][metric_key]
                    std_key = f"{metric_key}_std"

                    if include_std and std_key in metrics[provider]["tasks"][task]:
                        std_value = metrics[provider]["tasks"][task][std_key]
                        row.append(f"{value:.3f} (±{std_value:.3f})")
                    else:
                        row.append(f"{value:.3f}")
                else:
                    row.append("-")

            table.add_row(*row)

        console.print(table)

def save_combined_results_to_csv(all_results: dict[str, dict], output_dir: Path) -> None:
    """Save combined results from all pre-prompts to CSV files for cross-analysis.

    Args:
        all_results: Dictionary mapping pre-prompt keys to their respective results
        output_dir: Directory to save the combined CSV files
    """
    csv_dir = output_dir / "csv_results"
    csv_dir.mkdir(exist_ok=True, parents=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create a combined predictions CSV file
    combined_predictions_file = csv_dir / f"combined_predictions_{timestamp}.csv"
    combined_predictions_data = []

    # Create a combined metrics summary CSV file
    combined_metrics_file = csv_dir / f"combined_metrics_summary_{timestamp}.csv"
    combined_metrics_data = []

    # Process each pre-prompt's results
    for combined_key, results in all_results.items():
        # Extract providers and tasks
        providers = list(results["providers"].keys())
        reformulation_type = results.get("reformulation_type", "")

        # Extract the actual pre-prompt key from the combined key
        # For keys like "declarative_helpful", extract "helpful"
        pre_prompt_key = combined_key
        if reformulation_type and "_" in combined_key and combined_key.startswith(reformulation_type + "_"):
            pre_prompt_key = combined_key[(len(reformulation_type) + 1):]  # +1 for the underscore

        # Collect prediction data
        for provider in providers:
            if provider not in results["providers"]:
                continue

            provider_name = results["providers"][provider]["model"]

            for image_name, image_data in results["providers"][provider]["images"].items():
                for task_name, task_data in image_data["tasks"].items():
                    # Add each question, expected answer, and model answer to the data
                    for detail in task_data.get("details", []):
                        is_numeric = any(keyword in detail["question"].lower() for keyword in ["how many", "count"])

                        row = {
                            "provider": provider,
                            "model": provider_name,
                            "image": image_name,
                            "task": task_name,
                            "pre_prompt": pre_prompt_key,
                            "reformulation": reformulation_type,
                            "question": detail["question"],
                            "target": detail["expected"],
                            "prediction": detail["answer"],
                            "is_correct": detail["correct"],
                            "is_numeric": is_numeric
                        }

                        # Add extracted values for numeric questions
                        if is_numeric:
                            row["extracted_target"] = detail.get("extracted_expected")
                            row["extracted_prediction"] = detail.get("extracted_answer")

                            # Add error metrics if available
                            for metric in ["absolute_error", "squared_error", "normalized_error", "percentage_error"]:
                                if metric in detail:
                                    row[metric] = detail[metric]

                        combined_predictions_data.append(row)

            # Collect summary metrics data
            enhanced_metrics = calculate_additional_metrics(results)

            for task, task_metrics in results["providers"][provider]["tasks"].items():
                # Get the task metrics
                metrics_row = {
                    "provider": provider,
                    "model": provider_name,
                    "task": task,
                    "pre_prompt": pre_prompt_key,
                    "reformulation": reformulation_type,
                    "accuracy": task_metrics.get("accuracy", 0),
                    "normalized_mae": task_metrics.get("normalized_mae", float('nan')),
                    "total_questions": task_metrics.get("count", 0)
                }

                # Add enhanced metrics if available
                if provider in enhanced_metrics and task in enhanced_metrics[provider]["tasks"]:
                    enhanced_task_metrics = enhanced_metrics[provider]["tasks"][task]
                    metrics_row.update({
                        "f1_score": enhanced_task_metrics.get("f1_score", float('nan')),
                        "precision": enhanced_task_metrics.get("precision", float('nan')),
                        "recall": enhanced_task_metrics.get("recall", float('nan')),
                        "mae": enhanced_task_metrics.get("mae", float('nan')),
                        "mse": enhanced_task_metrics.get("mse", float('nan')),
                        "rmse": enhanced_task_metrics.get("rmse", float('nan')),
                        "reliability": enhanced_task_metrics.get("reliability", float('nan')),
                        "reliability_std": enhanced_task_metrics.get("reliability_std", float('nan')),
                        "consistency": enhanced_task_metrics.get("consistency", float('nan')),
                        # Add standard deviation metrics
                        "accuracy_std": enhanced_task_metrics.get("accuracy_std", float('nan')),
                        "f1_score_std": enhanced_task_metrics.get("f1_score_std", float('nan')),
                        "precision_std": enhanced_task_metrics.get("precision_std", float('nan')),
                        "recall_std": enhanced_task_metrics.get("recall_std", float('nan')),
                        "mae_std": enhanced_task_metrics.get("mae_std", float('nan')),
                        "mse_std": enhanced_task_metrics.get("mse_std", float('nan')),
                        "rmse_std": enhanced_task_metrics.get("rmse_std", float('nan')),
                        "nmae_std": enhanced_task_metrics.get("nmae_std", float('nan')),
                        "consistency_std": enhanced_task_metrics.get("consistency_std", float('nan'))
                    })

                combined_metrics_data.append(metrics_row)

    # Save to CSV files
    if combined_predictions_data:
        df = pd.DataFrame(combined_predictions_data)
        df.to_csv(combined_predictions_file, index=False)
        console.print(f"[green]Combined predictions across all pre-prompts saved to {combined_predictions_file}[/green]")

    if combined_metrics_data:
        df = pd.DataFrame(combined_metrics_data)
        df.to_csv(combined_metrics_file, index=False)
        console.print(f"[green]Combined metrics across all pre-prompts saved to {combined_metrics_file}[/green]")

    # Also create task-specific combined files for easier analysis
    task_metrics = {}
    for row in combined_metrics_data:
        task = row["task"]
        if task not in task_metrics:
            task_metrics[task] = []
        task_metrics[task].append(row)

    for task, metrics in task_metrics.items():
        task_file = csv_dir / f"combined_{task}_metrics_{timestamp}.csv"
        df = pd.DataFrame(metrics)
        df.to_csv(task_file, index=False)
        console.print(f"[green]Combined metrics for {task} task saved to {task_file}[/green]")

def display_combined_metrics_summary(all_results: dict[str, dict], include_std: bool = True) -> None:
    """Display a combined metrics summary table comparing different reformulation types.

    Args:
        all_results: Dictionary mapping result keys to their respective results
        include_std: Whether to include standard deviation values in parentheses
    """
    # Extract all unique reformulation types, providers, and tasks
    reformulation_types = set()
    providers_by_reformulation = {}
    tasks_by_provider = {}

    # First pass to extract metadata
    for _combined_key, results in all_results.items():
        reformulation_type = results.get("reformulation_type", "")
        reformulation_display = reformulation_type if reformulation_type else "None"
        reformulation_types.add(reformulation_display)

        # Track which providers are used with each reformulation type
        if reformulation_display not in providers_by_reformulation:
            providers_by_reformulation[reformulation_display] = set()

        for provider in results["providers"]:
            providers_by_reformulation[reformulation_display].add(provider)

            # Track which tasks are used with each provider
            if provider not in tasks_by_provider:
                tasks_by_provider[provider] = set()

            for task in results["providers"][provider]["tasks"]:
                tasks_by_provider[provider].add(task)

    # Convert to sorted lists for consistent display
    reformulation_types = sorted(reformulation_types)

    # Define metrics to display
    metric_keys = ["accuracy", "f1_score", "precision", "recall",
                  "mae", "mse", "nmae", "rmse",
                  "reliability", "consistency"]
    metric_names = ["Accuracy", "F1 Score", "Precision", "Recall",
                   "MAE", "MSE", "NMAE", "RMSE",
                   "Reliability", "Consistency"]

    # Display tables for each provider and task
    for provider in sorted(tasks_by_provider.keys()):
        for task in sorted(tasks_by_provider[provider]):
            # Create a table for this provider and task
            table = Table(title=f"[bold]{provider} - {task.capitalize()} Metrics by Reformulation Type[/bold]")

            # Add metric column and reformulation type columns
            table.add_column("Metric", style="cyan")
            for reformulation_type in reformulation_types:
                if provider in providers_by_reformulation.get(reformulation_type, set()):
                    table.add_column(reformulation_type, style="green")

            # Calculate enhanced metrics for each reformulation type
            metrics_by_reformulation = {}
            for _combined_key, results in all_results.items():
                reformulation_type = results.get("reformulation_type", "")
                reformulation_display = reformulation_type if reformulation_type else "None"

                if provider in results["providers"]:
                    # Calculate enhanced metrics for this result
                    enhanced_metrics = calculate_additional_metrics(results)
                    if provider in enhanced_metrics and task in enhanced_metrics[provider]["tasks"]:
                        metrics_by_reformulation[reformulation_display] = enhanced_metrics[provider]["tasks"][task]

            # Add rows for each metric
            for metric_key, metric_name in zip(metric_keys, metric_names, strict=False):
                row_data = [metric_name]

                for reformulation_type in reformulation_types:
                    if reformulation_type in metrics_by_reformulation:
                        metrics = metrics_by_reformulation[reformulation_type]
                        if metric_key in metrics:
                            value = metrics[metric_key]
                            std_key = f"{metric_key}_std"

                            if include_std and std_key in metrics:
                                std_value = metrics[std_key]
                                row_data.append(f"{value:.3f} (±{std_value:.3f})")
                            else:
                                row_data.append(f"{value:.3f}")
                        else:
                            row_data.append("-")
                    elif provider in providers_by_reformulation.get(reformulation_type, set()):
                        row_data.append("-")

                table.add_row(*row_data)

            # Display the table
            console.print(table)
            console.print("\n")

def main():
    """Main function to run the diagnostic evaluation."""
    global GAME_TYPE, DATASET_PATH, IMAGE_DIR, ANNOTATION_DIR
    parser = argparse.ArgumentParser(description="Evaluate VLMs on diagnostic datasets.")
    parser.add_argument('--folder', '-f', type=str, required=True, help='Path to the dataset folder (should contain img/ and legend_json/).')
    args = parser.parse_args()
    DATASET_PATH = Path(args.folder)
    IMAGE_DIR = DATASET_PATH / "img"
    ANNOTATION_DIR = DATASET_PATH / "legend_json"
    console.print(Panel.fit(f"[bold green]Selected dataset folder: {DATASET_PATH}[/bold green]"))

    # Infer game type from folder path
    folder_str = str(DATASET_PATH).lower()
    if "chess" in folder_str:
        GAME_TYPE = "chess"
    elif "poker" in folder_str:
        GAME_TYPE = "poker"
    else:
        console.print("[bold red]Could not infer game type from folder path. Please include 'chess' or 'poker' in the folder name.[/bold red]")
        exit(1)
    # Normalize GAME_TYPE to ensure correct preprompt lookup
    GAME_TYPE = GAME_TYPE.strip().lower()

    console.print(Panel.fit(f"[bold green]Inferred game: {GAME_TYPE}[/bold green]"))

    # Find all images with matching annotations
    print("Annotation directory: ", ANNOTATION_DIR)
    print("Image directory: ", IMAGE_DIR)
    with console.status("[bold blue]Looking for image-annotation pairs...", spinner="dots"):
        image_pairs = find_images_and_annotations()

    if not image_pairs:
        console.print("[bold red]No matching image-annotation pairs found in the dataset![/bold red]")
        return

    console.print(f"[green]Found {len(image_pairs)} image-annotation pairs in the dataset.[/green]")

    # Display dataset statistics
    with console.status("[bold blue]Analyzing dataset...", spinner="dots"):
        annotation_infos = [extract_annotation_info(ann_path) for _, ann_path in image_pairs]

    stats_table = Table(title="Dataset Statistics")
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="green")

    stats_table.add_row("Total images", str(len(image_pairs)))

    if GAME_TYPE == "chess":
        # Piece count statistics
        piece_counts = [info["piece_count"] for info in annotation_infos]
        stats_table.add_row("Minimum piece count", str(min(piece_counts)))
        stats_table.add_row("Maximum piece count", str(max(piece_counts)))
        stats_table.add_row("Average piece count", f"{sum(piece_counts) / len(piece_counts):.1f}")
        # Count piece types
        all_piece_types = {}
        for info in annotation_infos:
            for piece_type, count in info["piece_types"].items():
                if piece_type in all_piece_types:
                    all_piece_types[piece_type] += count
                else:
                    all_piece_types[piece_type] = count
        piece_type_str = ", ".join([f"{k}: {v}" for k, v in all_piece_types.items()])
        stats_table.add_row("Piece types", piece_type_str)
    elif GAME_TYPE == "poker":
        # Poker-specific statistics
        num_players_list = []
        num_community_cards_list = []
        cards_per_player = []
        for _, ann_path in image_pairs:
            with open(ann_path) as f:
                data = json.load(f)
            players = data.get("players", [])
            num_players = len(players)
            num_players_list.append(num_players)
            # Community cards
            community = data.get("community_cards") or {}
            num_community_cards = community.get("n_cards", 0)
            num_community_cards_list.append(num_community_cards)
            # Cards per player
            for player in players:
                hand = player.get("hand_config", {})
                n_cards = hand.get("n_cards", 0)
                cards_per_player.append(n_cards)
        # Aggregate stats
        if num_players_list:
            stats_table.add_row("Min players per image", str(min(num_players_list)))
            stats_table.add_row("Max players per image", str(max(num_players_list)))
            stats_table.add_row("Avg players per image", f"{sum(num_players_list)/len(num_players_list):.2f}")
        if num_community_cards_list:
            stats_table.add_row("Min community cards", str(min(num_community_cards_list)))
            stats_table.add_row("Max community cards", str(max(num_community_cards_list)))
            stats_table.add_row("Avg community cards", f"{sum(num_community_cards_list)/len(num_community_cards_list):.2f}")
        if cards_per_player:
            stats_table.add_row("Min cards per player", str(min(cards_per_player)))
            stats_table.add_row("Max cards per player", str(max(cards_per_player)))
            stats_table.add_row("Avg cards per player", f"{sum(cards_per_player)/len(cards_per_player):.2f}")
    else:
        stats_table.add_row("[red]Unknown game type for statistics!", "")

    console.print(stats_table)

    # Select VLM providers to test
    available_providers = []
    for provider_key, config in PROVIDERS.items():
        credentials = config["credentials"]
        if all(credentials.values()):
            available_providers.append(provider_key)

    if not available_providers:
        console.print("[bold red]No VLM providers with valid credentials found![/bold red]")
        console.print("Please set up the API keys in your .env file.")
        return

    console.print("[bold]Available VLM providers:[/bold]")
    for i, provider in enumerate(available_providers):
        console.print(f"[{i}] {PROVIDERS[provider]['model_name']}")

    # Get user selections
    selected_providers = []
    provider_input = Prompt.ask(
        "Enter provider numbers to test (comma separated, or 'all')",
        default="all"
    )

    if provider_input.lower() == "all":
        selected_providers = available_providers
    else:
        for idx in provider_input.split(","):
            try:
                provider_idx = int(idx.strip())
                if 0 <= provider_idx < len(available_providers):
                    selected_providers.append(available_providers[provider_idx])
            except ValueError:
                pass

    if not selected_providers:
        console.print("[bold red]No valid providers selected. Using the first available provider.[/bold red]")
        selected_providers = [available_providers[0]]

    # Select tasks to evaluate
    console.print("[bold]Available tasks:[/bold]")
    for i, task in enumerate(TASKS):
        console.print(f"[{i}] {task}")

    task_input = Prompt.ask(
        "Enter task numbers to evaluate (comma separated, or 'all')",
        default="all"
    )

    selected_tasks = []
    if task_input.lower() == "all":
        selected_tasks = TASKS
    else:
        for idx in task_input.split(","):
            try:
                task_idx = int(idx.strip())
                if 0 <= task_idx < len(TASKS):
                    selected_tasks.append(TASKS[task_idx])
            except ValueError:
                pass

    if not selected_tasks:
        console.print("[bold red]No valid tasks selected. Using all tasks.[/bold red]")
        selected_tasks = TASKS

    # Select number of images to process
    max_images = len(image_pairs)
    num_images = Prompt.ask(
        f"Enter number of images to process (1-{max_images}, or 'all')",
        default="5"
    )

    if num_images.lower() == "all":
        selected_images = image_pairs
    else:
        try:
            num = int(num_images)
            selected_images = image_pairs[:num] if 1 <= num <= max_images else image_pairs[:5]
        except ValueError:
            console.print("[bold red]Invalid input. Using 5 images.[/bold red]")
            selected_images = image_pairs[:5]

    # Add reformulation option
    reformulation = Prompt.ask(
        "Enter reformulation format if needed: 'declarative', 'missing_word', or 'both' (for both formats)",
        default=""
    )
    reformulate=""
    if reformulation.lower() in ["declarative", "missing_word", "both"]:
        reformulate = reformulation.lower()
    elif "," in reformulation:
        # Handle comma-separated values like "declarative,missing_word"
        reformulation_types = [r.strip().lower() for r in reformulation.split(",")]
        valid_types = ["declarative", "missing_word"]
        if all(r_type in valid_types for r_type in reformulation_types):
            reformulate = reformulation.lower()
        else:
            console.print("[bold red]Invalid input. No reformulation used.[/bold red]")
    else:
        console.print("[bold red]Invalid or no input. No reformulation used.[/bold red]")

    # Load available pre-prompt options
    preprompt_json_path = "diagnostic/questions/preprompt.json"
    with open(preprompt_json_path) as f:
        preprompts = json.load(f)
    game_preprompts = preprompts.get(GAME_TYPE, {})

    # Display available pre-prompt options
    console.print("[bold]Available pre-prompt options:[/bold]")
    for i, option in enumerate(game_preprompts.keys()):
        console.print(f"[{i}] {option}")

    # Get user selection for pre-prompt
    pre_prompt_input = Prompt.ask(
        f"Enter pre-prompt option numbers (comma separated, or 'all' for all {len(game_preprompts.keys())} options)",
        default="0"  # Default to first option (likely 'helpful')
    )

    selected_pre_prompt_keys = []
    if pre_prompt_input.lower() == "all":
        selected_pre_prompt_keys = list(game_preprompts.keys())
    else:
        for idx in pre_prompt_input.split(","):
            try:
                option_idx = int(idx.strip())
                pre_prompt_options = list(game_preprompts.keys())
                if 0 <= option_idx < len(pre_prompt_options):
                    selected_pre_prompt_keys.append(pre_prompt_options[option_idx])
            except ValueError:
                pass

    if not selected_pre_prompt_keys:
        console.print("[bold red]Invalid input. Using 'helpful' pre-prompt.[/bold red]")
        if "helpful" in game_preprompts:
            selected_pre_prompt_keys = ["helpful"]
        else:
            selected_pre_prompt_keys = [list(game_preprompts.keys())[0]]  # First available option

    # Get the pre-prompt strings for each selected key
    selected_pre_prompts = {key: game_preprompts.get(key, "") for key in selected_pre_prompt_keys}

    # Check for any empty pre-prompts and just notify about them, don't treat as errors
    for key, prompt in selected_pre_prompts.items():
        if not prompt:
            console.print(f"[bold yellow]Note: Pre-prompt '{key}' for game '{GAME_TYPE}' is empty and will be used as a no-prompt option.[/bold yellow]")

    # Don't filter out empty pre-prompts
    # selected_pre_prompts = {k: v for k, v in selected_pre_prompts.items() if v}

    if not selected_pre_prompts:
        console.print("[bold red]No valid pre-prompts found. Proceeding without pre-prompts.[/bold red]")

    # Summary of selections
    pre_prompt_summary = ", ".join(selected_pre_prompts.keys())
    console.print(Panel.fit(
        f"[bold]Evaluation Configuration[/bold]\n"
        f"Providers: {', '.join(selected_providers)}\n"
        f"Tasks: {', '.join(selected_tasks)}\n"
        f"Reformulation: {reformulate}\n"
        f"Pre-prompts: {pre_prompt_summary}\n"
        f"Images: {len(selected_images)} / {len(image_pairs)}"
    ))

    if Confirm.ask("Proceed with evaluation?"):
        # Create a dictionary to store all results
        all_results = {}

        # Process reformulation types
        reformulation_types = []
        if reformulate == "both":
            reformulation_types = ["declarative", "missing_word"]
        elif "," in reformulate:
            # Split comma-separated values
            reformulation_types = [r.strip().lower() for r in reformulate.split(",")]
        elif reformulate:
            reformulation_types = [reformulate]
        else:
            reformulation_types = [""]  # No reformulation

        # Run evaluation for each reformulation type and pre-prompt combination
        for reformulation_type in reformulation_types:
            for pre_prompt_key, pre_prompt in selected_pre_prompts.items():
                # Create a combined key for storage
                combined_key = f"{pre_prompt_key}"
                if reformulation_type:
                    combined_key = f"{reformulation_type}_{pre_prompt_key}"

                if reformulation_type:
                    console.print(f"\n[bold]Running evaluation with reformulation: {reformulation_type}, pre-prompt: {pre_prompt_key}[/bold]")
                else:
                    console.print(f"\n[bold]Running evaluation with pre-prompt: {pre_prompt_key} (no reformulation)[/bold]")

                result = run_evaluation(selected_images, selected_providers, selected_tasks, reformulation_type, pre_prompt, pre_prompt_key)
                all_results[combined_key] = result

        # Display combined metrics summary comparing different reformulation types
        console.print("\n[bold]Combined Metrics Summary (comparing reformulation types):[/bold]")
        display_combined_metrics_summary(all_results, include_std=True)

        # Save combined results to a single CSV file across all reformulations and pre-prompts
        save_combined_results_to_csv(all_results, OUTPUT_DIR)

    else:
        console.print("[yellow]Evaluation cancelled.[/yellow]")

if __name__ == "__main__":
    main()
