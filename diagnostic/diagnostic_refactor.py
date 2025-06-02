import argparse  # Added for CLI argument parsing
import os
import sys  # Add sys import
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd  # Added for DataFrame handling
import yaml  # For YAML config loading
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


from evaluate_diagnose_dataset import evaluate_vlm_on_task
from evaluation_pipeline.get_vlm import (  # Import necessary components
    VLM,
    VLMProvider,
    get_vlm,
)
from src.create_dataframe import AnnotationDataset
from utils.utils import replace_predictions_with_words

# TODO:
# - ALLOW A LIST OF REFOMULATE
# - ALLOW A LIST OF PRE_PROMPT
# - ALLOW MANUAL QUESTIONS


console = Console()

class Diagnostic:
    """
    A class to perform diagnostic tests on VLM anwers.
    """
    # Maps criteria keys to DataFrame column names and potential lambda functions for extraction
    _criteria_to_df_map: dict[str, tuple[str, Any]] = {
        # Common criteria for all games
        "camera_distance": ("camera", lambda x: x.get("distance") if isinstance(x, dict) else None),
        "camera_angle": ("camera", lambda x: x.get("angle") if isinstance(x, dict) else None),
        "camera_horizontal_angle": ("camera", lambda x: x.get("horizontal_angle") if isinstance(x, dict) else None),
        "blur": ("noise", lambda x: x.get("blur") if isinstance(x, dict) else None),
        "table_texture": ("noise", lambda x: x.get("table_texture") if isinstance(x, dict) else None),

        # Chess-specific criteria
        "number": ("piece_number", None),
        "background": ("board", lambda x: x.get("board_pattern") if isinstance(x, dict) else None),
        "types": ("pieces", lambda piece_list: ";".join(sorted({str(p.get('piece_type')) for p in piece_list if p and p.get('piece_type') is not None})) if isinstance(piece_list, list) else ""),
        "piece_types": ("piece_types", lambda types_set: ";".join(sorted(types_set)) if isinstance(types_set, set) else ""),
        "piece_colors": ("piece_colors", lambda colors_set: ";".join(sorted(colors_set)) if isinstance(colors_set, set) else ""),
        "piece_type_count": ("piece_types", lambda types_set: len(types_set) if isinstance(types_set, set) else 0),
        "board_rows": ("board", lambda x: x.get('board_rows') if isinstance(x, dict) else None),
        "board_columns": ("board", lambda x: x.get('board_columns') if isinstance(x, dict) else None),
        "board_pattern": ("board", lambda x: x.get('board_pattern') if isinstance(x, dict) else None),
        "board_location": ("board", lambda x: x.get('board_location') if isinstance(x, dict) else None),

        # Poker-specific criteria
        "card_number": ("n_cards", None),  # Direct column access
        "pile_number": ("n_piles", None),  # Direct column access
        "player_number": ("n_players", None),  # Direct column access
        "card_colors": ("card_types", lambda types_set: ";".join([t[0] if len(t) > 0 else "" for t in types_set]) if isinstance(types_set, set) else ""),  # Extract first character (color) from card types
        "player_horizontal_spread": ("player_horizontal_spread", None),  # Direct column access
        "player_vertical_spread": ("player_vertical_spread", None),  # Direct column access
        "community_horizontal_spread": ("community_horizontal_spread", None),  # Direct column access
        "community_vertical_spread": ("community_vertical_spread", None),  # Direct column access
        "card_types": ("card_types", lambda types_set: ";".join(sorted(types_set)) if isinstance(types_set, set) else ""),
        "lighting": ("setup", lambda x: x.get('lighting') if isinstance(x, dict) else None),
        "table_shape": ("setup", lambda x: x.get('table', {}).get('shape') if isinstance(x, dict) else None),
        "table_width": ("setup", lambda x: x.get('table', {}).get('width') if isinstance(x, dict) else None),
        "table_length": ("setup", lambda x: x.get('table', {}).get('length') if isinstance(x, dict) else None),
        "resolution_width": ("setup", lambda x: x.get('resolution', {}).get('width') if isinstance(x, dict) else None),
        "resolution_height": ("setup", lambda x: x.get('resolution', {}).get('height') if isinstance(x, dict) else None),
        "community_card_count": ("community_info", lambda x: x.get('n_cards') if isinstance(x, dict) else None),
        "community_cards": ("community_info", lambda x: ";".join(sorted(x.get('cards', []))) if isinstance(x, dict) else ""),

        # Card overlap layout specific criteria
        "has_overlap_layout": ("has_overlap_layout", None),  # Direct column access
        "card_overlap_count": ("card_overlap_count", None),  # Direct column access
        "overlap_layout_mode": ("card_overlap_layout", lambda x: x.get('layout_mode') if isinstance(x, dict) else None),
        "overlap_n_lines": ("card_overlap_layout", lambda x: x.get('n_lines') if isinstance(x, dict) else None),
        "overlap_n_columns": ("card_overlap_layout", lambda x: x.get('n_columns') if isinstance(x, dict) else None),
        "overlap_horizontal_factor": ("card_overlap_layout", lambda x: x.get('horizontal_overlap_factor') if isinstance(x, dict) else None),
        "overlap_vertical_factor": ("card_overlap_layout", lambda x: x.get('vertical_overlap_factor') if isinstance(x, dict) else None),
        "overlap_card_type_mode": ("card_overlap_layout", lambda x: x.get('card_type_config', {}).get('mode') if isinstance(x, dict) else None)
    }

    def __init__(self):
        # Potentially initialize with a full config or other parameters later
        self.config: dict[str, Any] = {}
        self.filtered_df: pd.DataFrame = pd.DataFrame()

    def filter_existing_content(
        self,
        dataset_path: Path,
        filtering_config: dict[str, Any],
        max_images_per_level: int = None
    ) -> tuple[list[Path], list[Path], dict[str, list[Any]]]:
        """
        Filters the DataFrame based on fixed variables and the current value of the variable under test.
        Also extracts the values of the 'variables_to_test' for each filtered image.
        Optionally, limits the number of images per unique combination of variables_to_test ("per level").

        Args:
            dataset_path (Path): The path to the dataset folder.
            filtering_config (Dict[str, Any]): Dictionary containing 'fixed_variables_exact',
                                             'fixed_variables_range', 'variables_to_test', and 'game'.
            max_images_per_level (int, optional): Maximum number of images per unique combination of variables_to_test. Default is None (no limit).

        Returns:
            Tuple[List[Path], List[Path], Dict[str, List[Any]]]:
                - A list of image Paths for the filtered images.
                - A list of legend Paths for the filtered images.
                - A dictionary where keys are from 'variables_to_test' and values are lists
                  of corresponding data for each filtered image.
        """
        # Extract variables from filtering_config
        fixed_variables_exact = filtering_config.get('fixed_variables_exact', {})
        fixed_variables_range = filtering_config.get('fixed_variables_range', {})
        variables_to_test_config = filtering_config.get('variables_to_test', {}) # These are the criteria for filtering images
        game = filtering_config.get('game', 'chess')  # Default to 'chess' if not specified
        print(f"Filtering over the following variables for game '{game}': ", variables_to_test_config)

        # First, ensure the path to the dataset is valid
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path {dataset_path} does not exist.")

        # Load DataFrame
        print("Loading dataset from path: ", dataset_path)
        annotation_path = dataset_path / "legend_json"
        dataset = AnnotationDataset(annotation_path, game=game)

        print("Building dataset from path: ", annotation_path)
        print("From a total of : ", len(os.listdir(dataset_path / "legend_json")), " legends")
        df = dataset.get_dataframe()
        total_images = len(df)
        print("Total images: ", total_images)

        filtered_df = df.copy()

        # DEBUG: Print types of pieces across all images
        if 'piece_types' in filtered_df.columns:
            print("\n===== DEBUG: PIECE TYPE INFORMATION =====")
            # Print the type of the piece_types column for the first row
            if not filtered_df.empty:
                first_row_piece_types = filtered_df.iloc[0]['piece_types']
                print(f"Type of piece_types column: {type(first_row_piece_types)}")
                print(f"Content of first row piece_types: {first_row_piece_types}")

                # Count images by number of piece types
                piece_type_counts = {}
                for _, row in filtered_df.iterrows():
                    piece_types = row['piece_types']
                    count = len(piece_types) if isinstance(piece_types, set) else 0
                    if count not in piece_type_counts:
                        piece_type_counts[count] = 0
                    piece_type_counts[count] += 1

                print("Distribution of images by number of piece types:")
                for count, num_images in sorted(piece_type_counts.items()):
                    print(f"  {count} piece types: {num_images} images")

                # Test our filter function
                accessor_fn = self._criteria_to_df_map["piece_type_count"][1]
                print("\nTesting piece_type_count accessor function on first 5 rows:")
                for i in range(min(5, len(filtered_df))):
                    piece_types = filtered_df.iloc[i]['piece_types']
                    count = accessor_fn(piece_types)
                    print(f"  Row {i}: piece_types={piece_types}, count={count}")

                # Count images that would pass our filter
                min_count = 3
                passing_images = 0
                for _, row in filtered_df.iterrows():
                    piece_types = row['piece_types']
                    count = accessor_fn(piece_types)
                    if count >= min_count:
                        passing_images += 1

                print(f"\nImages with {min_count}+ piece types (should match filter): {passing_images}")
            else:
                print("DataFrame is empty, cannot check piece_types.")
        else:
            print("\n===== WARNING: piece_types column not found in DataFrame =====")

        # Apply exact fixed variable filters
        for variable_key, exact_value in fixed_variables_exact.items():
            if variable_key in self._criteria_to_df_map:
                col_name, accessor_fn = self._criteria_to_df_map[variable_key]
                if col_name not in filtered_df.columns:
                    print(f"Warning: Column '{col_name}' for exact fixed variable '{variable_key}' not found. Skipping.")
                    continue

                series_to_filter = filtered_df[col_name]
                if accessor_fn:
                    series_to_filter = series_to_filter.apply(accessor_fn)

                # Save number of rows before filtering for reporting
                before_count = len(filtered_df)
                filtered_df = filtered_df[series_to_filter == exact_value]
                after_count = len(filtered_df)
                print(f"Applied filter for '{variable_key}' == {exact_value}: {before_count} → {after_count} images")
            else:
                print(f"Warning: Exact fixed variable criterion '{variable_key}' not defined in _criteria_to_df_map. Skipping.")

        # Apply range fixed variable filters
        for variable_key, value_range in fixed_variables_range.items():
            if variable_key in self._criteria_to_df_map:
                col_name, accessor_fn = self._criteria_to_df_map[variable_key]
                if col_name not in filtered_df.columns:
                    print(f"Warning: Column '{col_name}' for range fixed variable '{variable_key}' not found. Skipping.")
                    continue

                series_to_filter = filtered_df[col_name]
                if accessor_fn:
                    print(f"\nApplying accessor function for '{variable_key}' on column '{col_name}'")
                    # Debug: Print before applying accessor
                    print(f"  Column type before accessor: {series_to_filter.dtype}")
                    if not filtered_df.empty:
                        print(f"  First few values before accessor: {series_to_filter.head(3).tolist()}")

                    # Apply accessor and capture result separately for debugging
                    series_to_filter = series_to_filter.apply(accessor_fn)

                    # Debug: Print after applying accessor
                    print(f"  Column type after accessor: {series_to_filter.dtype}")
                    if not filtered_df.empty:
                        print(f"  First few values after accessor: {series_to_filter.head(3).tolist()}")

                # Ensure series is numeric for range comparison, attempt conversion if not
                try:
                    print(f"\nConverting series for '{variable_key}' to numeric for range filtering")
                    numeric_series = pd.to_numeric(series_to_filter, errors='coerce')
                    print(f"  Series after numeric conversion: {numeric_series.head(3).tolist() if not filtered_df.empty else 'empty'}")
                    print(f"  NaN values after conversion: {numeric_series.isna().sum()}")

                    if isinstance(value_range, list) and len(value_range) == 2:
                        min_val, max_val = value_range
                        print(f"  Filtering for range: {min_val} to {max_val}")

                        # Save number of rows before filtering for reporting
                        before_count = len(filtered_df)
                        filtered_df = filtered_df[numeric_series.between(min_val, max_val, inclusive="both")]
                        after_count = len(filtered_df)
                        print(f"Applied filter for '{variable_key}' in range {value_range}: {before_count} → {after_count} images")
                    else:
                        print(f"Warning: Invalid range format for '{variable_key}'. Expected [min, max]. Skipping.")
                except Exception as e:
                    print(f"Warning: Could not convert series for '{variable_key}' to numeric for range filtering: {e}. Skipping.")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"Warning: Range fixed variable criterion '{variable_key}' not defined in _criteria_to_df_map. Skipping.")

        # Filter by variables_to_test if needed
        for variable_key, variable_value in variables_to_test_config.items():
            if variable_key in self._criteria_to_df_map:
                col_name, accessor_fn = self._criteria_to_df_map[variable_key]
                if col_name not in filtered_df.columns:
                    print(f"Warning: Column '{col_name}' not found in DataFrame. Skipping criterion '{variable_key}'.")
                    continue

                # Determine the actual series of values from the DataFrame to filter on
                series_to_filter = filtered_df[col_name]
                if accessor_fn:
                    series_to_filter = series_to_filter.apply(accessor_fn)

                # Apply filter based on the type and content of variable_value
                if variable_value is None or (isinstance(variable_value, list) and not variable_value):
                    # If value is None or an empty list, skip filtering for this criterion
                    continue
                elif isinstance(variable_value, list):
                    # If value is a non-empty list, filter rows where the series value is in the list
                    filtered_df = filtered_df[series_to_filter.isin(variable_value)]
                else:
                    # Otherwise, perform a direct equality comparison (for single scalar values)
                    filtered_df = filtered_df[series_to_filter == variable_value]
            else:
                print(f"Warning: Criterion '{variable_key}' not defined in _criteria_to_df_map. Skipping.")

        # Store the fully filtered DataFrame
        # If max_number_of_images is set and variables_to_test is not empty, sample per group
        if max_images_per_level is not None and len(variables_to_test_config) > 0:
            group_cols = []
            for var_key in variables_to_test_config:
                if var_key in self._criteria_to_df_map:
                    col_name, accessor_fn = self._criteria_to_df_map[var_key]
                    if accessor_fn:
                        filtered_df[col_name + "_for_grouping"] = filtered_df[col_name].apply(accessor_fn)
                        group_cols.append(col_name + "_for_grouping")
                    else:
                        group_cols.append(col_name)

            # Sample per group
            before_count = len(filtered_df)
            sampled_df = filtered_df.groupby(group_cols, dropna=False, group_keys=False).apply(lambda x: x.sample(n=min(len(x), max_images_per_level), random_state=42))
            sampled_df = sampled_df.reset_index(drop=True)
            self.filtered_df = sampled_df.copy()
            after_count = len(self.filtered_df)

            if before_count != after_count:
                print(f"Applied max_images_per_level={max_images_per_level}: {before_count} → {after_count} images")
        else:
            self.filtered_df = filtered_df.copy() # Make a copy to be safe

        image_paths: list[Path] = []
        legend_paths: list[Path] = []

        # Initialize dict to store actual values of variables_to_test for the filtered images
        # The keys for this dictionary are the actual variable names we are testing (e.g., "number", "blur")
        # These keys come from the `variables_to_test_config` passed in filtering_config.
        test_variable_data_for_df: dict[str, list[Any]] = {
            key: [] for key in variables_to_test_config if key in self._criteria_to_df_map
        }

        for _, row in self.filtered_df.iterrows():
            image_name = row['image']
            # Remove "_legend" suffix from image name if present
            clean_image_name = image_name.replace("_legend", "") if "_legend" in image_name else image_name
            image_paths.append(dataset_path / "img" / f"{clean_image_name}.png")

            base_name = Path(image_name).stem
            legend_paths.append(dataset_path / "legend_json" / f"{base_name}.json")

            # Extract and store the values for each variable under test for the current image
            for var_key in test_variable_data_for_df: # Iterate only over keys we initialized
                col_name, accessor_fn = self._criteria_to_df_map[var_key]
                if col_name in row:
                    raw_value = row[col_name]
                    processed_value = accessor_fn(raw_value) if accessor_fn else raw_value
                    test_variable_data_for_df[var_key].append(processed_value)
                else:
                    # This case should ideally not be hit if df construction is robust
                    # and _criteria_to_df_map refers to valid columns.
                    test_variable_data_for_df[var_key].append(pd.NA)

        print(f"Found {len(image_paths)} images (from {total_images} total)")

        return image_paths, legend_paths, test_variable_data_for_df

    def run_evaluation(
        self,
        selected_images: list[tuple[Path, Path]],
        selected_legends: list[str],
        test_variable_values_for_df: dict[str, list[Any]],
        vlm: VLM,
        model_name: str,
        question_keys: list[str],  # always a single-item list in this refactor
        task: str = "",
        reformulate: str = "",
        pre_prompt: str = "",
        output_dir: Path = Path("diagnostic_results")
    ) -> None:
        """Run the evaluation on selected images for a single VLM provider and a single question key.

        Args:
            selected_images: List of (image_path, annotation_path) tuples
            selected_legends: List of legend paths (as strings)
            test_variable_values_for_df: Dict of variable values for each image
            vlm: The VLM model instance (already initialized)
            model_name: The model name (for logging/results)
            question_keys: List with a single question key to evaluate
            task: The task type (for special handling, e.g., counting)
            reformulate: The format for reformulating questions
            pre_prompt: The pre_prompt to add to each question
            output_dir: Directory to save results

        Returns:
            None. Results are saved to CSV.
        """

        assert len(question_keys) == 1, "run_evaluation expects a single question key per call."
        question_key = question_keys[0]

        col_data = {
            "model": [],
            "task_name": [],
            "image": [],
            "question": [],
            "target": [],
            "pred": [],
            "pre_prompt": [],
            "reformulate": [],
        }

        actual_variables_to_add_as_columns = list(test_variable_values_for_df.keys())
        for var_key in actual_variables_to_add_as_columns:
            col_data[var_key] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            image_task = progress.add_task("[cyan]Processing images", total=len(selected_images))

            # List to collect call_metrics with context
            call_metrics_records = []

            for i, (image_path, annotation_path) in enumerate(zip(selected_images, selected_legends, strict=False)):
                image_name = image_path.stem
                progress.update(image_task, description=f"[cyan]Processing image {i+1}/{len(selected_images)}: {image_name}")

                image_specific_test_values = {}
                image_row_in_filtered_df = self.filtered_df[self.filtered_df['image'] == image_name]

                if not image_row_in_filtered_df.empty:
                    for var_key in actual_variables_to_add_as_columns:
                        map_col_name, accessor_fn = self._criteria_to_df_map[var_key]
                        if map_col_name in image_row_in_filtered_df.columns:
                            raw_val = image_row_in_filtered_df.iloc[0][map_col_name]
                            image_specific_test_values[var_key] = accessor_fn(raw_val) if accessor_fn else raw_val
                        else:
                            image_specific_test_values[var_key] = pd.NA
                else:
                    for var_key in actual_variables_to_add_as_columns:
                        image_specific_test_values[var_key] = pd.NA

                # Run the evaluation for the single question key
                evaluation = evaluate_vlm_on_task(vlm, image_path, annotation_path, [question_key], reformulate, pre_prompt)
                print("evaluation: ", evaluation)

                col_data["model"].append(model_name)
                col_data["task_name"].append(question_key)
                col_data["image"].append(image_name)
                col_data["question"].append(evaluation["questions"])

                if task == "counting":
                    col_data["target"].append(int(evaluation["expected_answers"][0]) if evaluation["expected_answers"] and isinstance(evaluation["expected_answers"], list) else pd.NA)
                else:
                    col_data["target"].append(evaluation["expected_answers"][0] if evaluation["expected_answers"] and isinstance(evaluation["expected_answers"], list) else pd.NA)

                col_data["pred"].append(evaluation["vlm_answers"][0] if evaluation["vlm_answers"] and isinstance(evaluation["vlm_answers"], list) else pd.NA)

                col_data["pre_prompt"].append(pre_prompt)
                col_data["reformulate"].append(reformulate)

                for var_key in actual_variables_to_add_as_columns:
                    col_data[var_key].append(test_variable_values_for_df[var_key][i])

                # --- Collect call_metrics with context ---
                for idx, call_metric in enumerate(evaluation.get("call_metrics", [])):
                    record = {
                        "model": model_name,
                        "task_name": question_key,
                        "image": image_name,
                        "question": evaluation["questions"][idx] if idx < len(evaluation["questions"]) else None,
                        "pre_prompt": pre_prompt,
                        "reformulate": reformulate,
                    }
                    if isinstance(call_metric, dict):
                        record.update(call_metric)
                    call_metrics_records.append(record)
                time.sleep(0.5)
                progress.update(image_task, advance=1)

        # Save final results to CSV using the collected column data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df = pd.DataFrame(col_data)
        # ensure the output directory exists or create it
        output_dir.mkdir(exist_ok=True, parents=True)

        # Sanitize model_name and question_key for filename safety
        safe_model_name = str(model_name).replace('/', '_').replace(' ', '_')
        safe_question_key = str(question_key).replace('/', '_').replace(' ', '_')

        csv_filename = f"diagnostic_evaluation_results_{safe_model_name}_{safe_question_key}_{timestamp}.csv"
        df = replace_predictions_with_words(df, save=False)

        print("SAVING TO: ", output_dir / csv_filename)
        print("DF FILENAME: ", csv_filename)
        df.to_csv(output_dir / csv_filename, index=False)
        print(f"Diagnostic complete for {question_key}. Results saved in {output_dir}")

        # --- Save call_metrics summary DataFrame ---
        if call_metrics_records:
            call_metrics_df = pd.DataFrame(call_metrics_records)
            # Only keep numeric columns for aggregation
            numeric_cols = call_metrics_df.select_dtypes(include=['number']).columns
            summary = []
            # Compute mean per image for each metric
            if not call_metrics_df.empty and 'image' in call_metrics_df.columns:
                per_image_means = call_metrics_df.groupby('image')[numeric_cols].mean()
                mean_per_image_dict = per_image_means.mean().to_dict()
            else:
                mean_per_image_dict = {col: None for col in numeric_cols}
            for col in numeric_cols:
                col_sum = call_metrics_df[col].dropna().sum()
                col_mean = call_metrics_df[col].dropna().mean()
                mean_per_image = mean_per_image_dict.get(col, None)
                summary.append({
                    'metric': col,
                    'sum': col_sum,
                    'mean': col_mean,
                    'mean_per_image': mean_per_image,
                    'model': model_name,
                    'question': question_key
                })
            summary_df = pd.DataFrame(summary)
            call_metrics_summary_csv_filename = f"diagnostic_evaluation_call_metrics_summary_{model_name}_{question_key}_{timestamp}.csv"
            call_metrics_summary_csv_filename = str(call_metrics_summary_csv_filename).replace('/', '_').replace(' ', '_')
            summary_df.to_csv(output_dir / call_metrics_summary_csv_filename, index=False)
            print(f"Call metrics summary saved in {output_dir / call_metrics_summary_csv_filename}")

        return None

    def run_diagnostic(self, config: dict[str, Any], vlm: VLM, dataset_path: Path, output_dir: Path = Path("diagnostic_results")):
        """
        Runs the diagnostic on the provided dataset and saves the results to the output directory.
        For each question key, a separate result CSV is generated.
        """
        self.config = config
        max_images_per_level = config.get("max_images_per_level")
        selected_images, selected_legends, test_variable_data_for_df = self.filter_existing_content(dataset_path, config, max_images_per_level=max_images_per_level)
        if not selected_images:
            print("No images found after filtering. Aborting diagnostic run.")
            return None
        print(config["model_name"])

        # Handle reformulation type
        reformulation_type_str = config["reformulate"]
        if isinstance(reformulation_type_str, list) and reformulation_type_str:
            reformulation_type_str = reformulation_type_str[0]
        elif isinstance(reformulation_type_str, list) and not reformulation_type_str:
            reformulation_type_str = ""
        task_key = config.get("task", "")

        # Handle preprompt type
        preprompt_type_str = config["pre_prompt"]
        if isinstance(preprompt_type_str, list) and preprompt_type_str:
            preprompt_type_str = preprompt_type_str[0]
        elif isinstance(preprompt_type_str, list) and not preprompt_type_str:
            preprompt_type_str = ""

        question_keys = config["question_keys"]

        for question_key in question_keys:
            self.run_evaluation(
                selected_images,
                selected_legends,
                test_variable_data_for_df,
                vlm,
                model_name=config["model_name"],
                question_keys=[question_key],
                task=task_key,
                reformulate=reformulation_type_str,
                pre_prompt=preprompt_type_str,
                output_dir=output_dir
            )
        return None


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run diagnostic evaluation on a dataset.")
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Path to the dataset directory."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="diagnostic_results",
        help="Directory to save results."
    )
    parser.add_argument(
        "--model_provider",
        type=str,
        choices=["azure_openai", "groq", "ollama", "huggingface"],
        help="Model provider to use."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Model name to use."
    )

    parser.add_argument(
        "--pre_prompt",
        type=str,
        default="",
        help="Pre-prompt to use (default: empty)."
    )
    parser.add_argument(
        "--reformulate",
        type=str,
        default="declarative",
        help="Reformulation type (default: declarative)."
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Path to YAML config file for game and variable settings."
    )
    args = parser.parse_args()

    # Map string to VLMProvider enum
    provider_map = {
        "azure_openai": VLMProvider.AZURE_OPENAI,
        "groq": VLMProvider.GROQ,
        "ollama": VLMProvider.OLLAMA,
        "huggingface": VLMProvider.HUGGINGFACE,
    }

    selected_provider = provider_map[args.model_provider.lower()]
    provider_kwargs = {}

    if selected_provider == VLMProvider.AZURE_OPENAI:
        provider_kwargs['api_key'] = os.getenv("AZURE_OPENAI_API_KEY")
        provider_kwargs['endpoint'] = os.getenv("AZURE_OPENAI_ENDPOINT")
        if args.model_name == "gpt-4.1-mini":
            provider_kwargs['api_version'] = os.getenv("AZURE_OPENAI_API_VERSION_GPT41mini")
            provider_kwargs['model_name'] = os.getenv("AZURE_OPENAI_MODEL_NAME_GPT41mini", "gpt-4.1-mini")
        elif args.model_name == "gpt-4.1":
            provider_kwargs['api_version'] = os.getenv("AZURE_OPENAI_API_VERSION_GPT41")
            provider_kwargs['model_name'] = os.getenv("AZURE_OPENAI_MODEL_NAME_GPT41", "gpt-4.1")
        if not provider_kwargs['api_key'] or not provider_kwargs['endpoint']:
            raise ValueError("Missing AZURE_OPENAI_API_KEY or AZURE_OPENAI_ENDPOINT in environment variables.")

    elif selected_provider == VLMProvider.GROQ:
        provider_kwargs['api_key'] = os.getenv("GROQ_API_KEY")
        provider_kwargs['model_name'] = args.model_name # meta-llama/llama-4-scout-17b-16e-instruct
    elif selected_provider == VLMProvider.OLLAMA:
        provider_kwargs['host'] = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        provider_kwargs['model_name'] = args.model_name
    elif selected_provider == VLMProvider.HUGGINGFACE:
        provider_kwargs['api_key'] = os.getenv("HUGGINGFACE_API_KEY")
        provider_kwargs['model_name'] = args.model_name

    print("provider_kwargs: ", provider_kwargs)
    vlm = get_vlm(selected_provider, **provider_kwargs)

    DATASET_PATH = Path(args.dataset_path)
    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    # Example question key groups for reference (not used directly)
    # count_list_key=["count_pieces"]
    # identification_list_key=["identify_type_one_piece"]
    # absolute_localization_list_key=["localize_row_one_piece", "localize_column_one_piece"]
    # relative_localization_list_key=["localize_rows_between_two_pieces", "localize_columns_between_two_pieces"]

    config_yaml = {}
    if args.config_path:
        try:
            with open(args.config_path) as f:
                config_yaml = yaml.safe_load(f)
            # Ensure the keys exist, else set to default
            for key, default in [
                ("game", "chess"),
                ("fixed_variables_exact", {}),
                ("fixed_variables_range", {}),
                ("variables_to_test", {}),
                ("max_images_per_level", None),
                ("question_keys", None)
            ]:
                if key not in config_yaml:
                    config_yaml[key] = default
            if config_yaml["question_keys"] is None:
                raise ValueError("The YAML config must contain a 'question_keys' field (list of question keys to evaluate).")
        except Exception as e:
            print(f"Error loading YAML config from {args.config_path}: {e}")
            exit(1)
    else:
        config_yaml = {
            "game": "chess",
            "fixed_variables_exact": {},
            "fixed_variables_range": {},
            "variables_to_test": {},
            "max_images_per_level": None,
            "question_keys": None,
        }
        raise ValueError("You must provide a YAML config file with a 'question_keys' field.")

    config = {
        "game": config_yaml["game"],
        "fixed_variables_exact": config_yaml["fixed_variables_exact"],
        "fixed_variables_range": config_yaml["fixed_variables_range"],
        "variables_to_test": config_yaml["variables_to_test"],
        "max_images_per_level": config_yaml["max_images_per_level"],
        "question_keys": config_yaml["question_keys"],
        "model_name": provider_kwargs.get("model_name", args.model_provider),
        "reformulate": [args.reformulate],
        "pre_prompt": [args.pre_prompt],
    }

    diagnostic = Diagnostic()
    results = diagnostic.run_diagnostic(config, vlm, DATASET_PATH, output_dir=OUTPUT_DIR)
    # The user had run_diagnostic return None from run_evaluation. If so, results here will be None.
    # print(f"Diagnostic complete. Results saved in {OUTPUT_DIR}") # This print is now inside run_evaluation


#################### TO RUN THE CODE ####################

"""
python diagnostic/diagnostic_refactor.py \
    --dataset_path /path/to/dataset \
    --config_path diagnostic/diagnostic_configs/chess_count_config.yaml \
    --output_dir diagnostic_results/gpt-4.1-mini \
    --model_provider azure_openai \
    --model_name gpt-4.1-mini \
    --pre_prompt debiased \
    --reformulate declarative
"""
