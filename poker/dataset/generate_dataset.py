import argparse
import itertools
import json
import logging
import random
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from poker.config.models import PokerSceneModel
from poker.scene_generator import generate_poker_scene_from_config

logger = logging.getLogger(__name__)

@dataclass
class VariableConfig:
    """Configuration for a variable in the dataset.

    Attributes:
        variate_type (str): Type of variable ('fixed', 'varying_random', 'varying_all', 'varying_among_range', 'varying_all_range')
        variate_levels (Any): Value(s) for the variable.
        n_images (int): Number of times to use each level (deterministic) or number of random samples per base combo (random). Defaults to 1.
        randomize (bool): Whether to randomize around the value (semantic meaning may need clarification). DEPRECATED for count_config.
        randomize_percentage (float): Percentage for randomization. DEPRECATED for count_config.
    """
    variate_type: str
    variate_levels: Any | list[Any] | tuple[Any, Any]
    n_images: int = 1 # Default to 1 image per level/sample
    randomize: bool = False
    randomize_percentage: float = 0.2

    def __post_init__(self):
        """Validate the configuration after initialization."""
        valid_types = {'fixed', 'varying_random', 'varying_all', 'varying_among_range', 'varying_all_range'}
        if self.variate_type not in valid_types:
            raise ValueError(f"Invalid type: {self.variate_type}. Must be one of {valid_types}")

        if self.variate_type == "fixed" and self.variate_levels is None:
            raise ValueError("Fixed type must have a value")

        if self.n_images < 1:
             logger.warning(f"n_images is {self.n_images}, setting to 1.")
             self.n_images = 1

        if self.variate_type in {"varying_random", "varying_all", "varying_among_range", "varying_all_range"}:
            if self.variate_levels is None:
                raise ValueError(f"Varying type '{self.variate_type}' must have variate_levels")


@dataclass
class DatasetConfig:
    """Complete dataset configuration.

    Attributes:
        name: Name of the dataset
        output_dir: Directory to save the dataset
        seed: Random seed for reproducibility
        piece_set: Name of the piece set to use (e.g., 'default', 'old_school')
        variables: Dictionary mapping variable names to their configurations
    """
    name: str
    output_dir: str
    seed: int | None = None
    piece_set: str = "default" # Default to standard pieces
    variables: dict[str, VariableConfig] = field(default_factory=dict)

    def __post_init__(self):
        """Validate the configuration."""
        if self.variables is None:
            self.variables = {}

        # Ensure output directory exists
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


class VariableCombinationGenerator:
    def __init__(self, config: DatasetConfig):
        self.config = config
        if config.seed is not None:
            random.seed(config.seed)

    def _is_location_var(self, var_name: str) -> bool:
        """Checks if a variable name corresponds to the board location."""
        return var_name == 'chess.board_config.location'

    def _add_z_coordinate(self, pos: list[float]) -> list[float]:
        """Adds the z=0.9 coordinate if it's a 2D position."""
        if isinstance(pos, list) and len(pos) == 2:
            return [pos[0], pos[1], 0.9]
        return pos # Return as is if already 3D or not a list

    def _get_deterministic_values(self, var_name: str, var_config: VariableConfig) -> list[Any]:
        """Generates the complete list of values for deterministic variable types, repeating based on n_images."""
        values = []
        n_images_per_level = var_config.n_images

        if var_config.variate_type == "varying_all":
            # Directly use the list provided
            raw_values = list(var_config.variate_levels)
        elif var_config.variate_type == "varying_all_range":
            if isinstance(var_config.variate_levels, tuple) and len(var_config.variate_levels) == 2:
                start, end = var_config.variate_levels
                if isinstance(start, int) and isinstance(end, int):
                     raw_values = list(range(start, end + 1))
                else: # Assume float range - sample min, mid, max
                     raw_values = [start, (start+end)/2.0, end]
            else:
                 raise ValueError(f"Invalid variate_levels for varying_all_range: {var_config.variate_levels}")
        else:
            raise ValueError(f"Unsupported deterministic type: {var_config.variate_type}")

        # Ensure raw_values is defined before proceeding (should always be by logic above)
        if 'raw_values' not in locals():
             raise RuntimeError("Internal logic error: raw_values not assigned in _get_deterministic_values")

        # Add z-coordinate if location before repetition
        if self._is_location_var(var_name):
             processed_values = [self._add_z_coordinate(val) for val in raw_values]
        else:
             processed_values = raw_values

        # Repeat values based on n_images
        if n_images_per_level > 1:
            values = [val for val in processed_values for _ in range(n_images_per_level)]
            print(f"  Deterministic values for {var_name} (repeated {n_images_per_level} times per level): {values}")
        else:
             values = processed_values
             print(f"  Deterministic values for {var_name}: {values}")

        return values

    def _get_single_random_value(self, var_name: str, var_config: VariableConfig) -> Any:
        """Generates a single random value for random variable types."""
        value = None
        if var_config.variate_type == "varying_random":
             # Sample from the provided list
             if isinstance(var_config.variate_levels, list):
                 value = random.choice(var_config.variate_levels)
             else:
                 raise ValueError(f"varying_random requires a list for variate_levels, got: {var_config.variate_levels}")
        elif var_config.variate_type == "varying_among_range":
            if isinstance(var_config.variate_levels, tuple) and len(var_config.variate_levels) == 2:
                 start, end = var_config.variate_levels
                 if isinstance(start, int) and isinstance(end, int):
                     value = random.randint(start, end)
                 elif isinstance(start, float) and isinstance(end, float):
                     value = random.uniform(start, end)
                 else:
                      raise ValueError(f"Unsupported range types for varying_among_range: {type(start)}, {type(end)}")
            # Handle location range format [[x_min, x_max], [y_min, y_max]]
            elif (self._is_location_var(var_name) and
                  isinstance(var_config.variate_levels, list) and len(var_config.variate_levels) == 2 and
                  isinstance(var_config.variate_levels[0], list) and len(var_config.variate_levels[0]) == 2 and
                  isinstance(var_config.variate_levels[1], list) and len(var_config.variate_levels[1]) == 2):
                 x_min, x_max = var_config.variate_levels[0]
                 y_min, y_max = var_config.variate_levels[1]
                 x = random.uniform(x_min, x_max)
                 y = random.uniform(y_min, y_max)
                 value = [x, y] # z-coordinate added later
            else:
                 raise ValueError(f"Invalid variate_levels for varying_among_range: {var_config.variate_levels}")
        else:
            raise ValueError(f"Unsupported random type: {var_config.variate_type}")

        # Special handling for location (apply z-coordinate)
        if self._is_location_var(var_name):
            value = self._add_z_coordinate(value)

        # print(f"  Generated random value for {var_name}: {value}") # Too verbose for every step
        return value

    def generate_variable_combinations(self) -> list[dict[str, Any]]:
        """
        Generates dataset configurations based on variable types.
        - Fixed variables have constant values.
        - Deterministic variables ('varying_all', 'varying_all_range') define the base combinations.
        - Random variables ('varying_random', 'varying_among_range') are re-sampled for each base combination.
        The total number of combinations is determined by the product of deterministic variable levels
        multiplied by the product of n_images for each random variable.
        """
        print("\n=== Generating Variable Combinations (New Strategy with n_images) ===")

        fixed_vars: dict[str, Any] = {}
        deterministic_vars: dict[str, VariableConfig] = {}
        random_vars: dict[str, VariableConfig] = {}

        # 1. Categorize variables
        print("Categorizing variables...")
        for var_name, var_config in self.config.variables.items():
            if var_config.variate_type == 'fixed':
                # Handle location's z-coordinate for fixed type
                val = var_config.variate_levels
                if self._is_location_var(var_name):
                     val = self._add_z_coordinate(val)
                fixed_vars[var_name] = val
                print(f"- Fixed: {var_name} = {val}")
            elif var_config.variate_type in {'varying_all', 'varying_all_range'}:
                deterministic_vars[var_name] = var_config
                print(f"- Deterministic: {var_name} ({var_config.variate_type}, n_images={var_config.n_images})")
            elif var_config.variate_type in {'varying_random', 'varying_among_range'}:
                random_vars[var_name] = var_config
                print(f"- Random: {var_name} ({var_config.variate_type}, n_images={var_config.n_images})")
            else:
                 print(f"Warning: Unknown variate_type '{var_config.variate_type}' for variable '{var_name}' - Skipping.")


        # 2. Generate deterministic values and base combinations
        print("Generating deterministic values...")
        deterministic_values_lists = []
        deterministic_var_names = list(deterministic_vars.keys())

        if not deterministic_var_names:
             print("No deterministic variables found. Generating 1 configuration.")
             base_combinations = [{}] # Start with one empty base if no deterministic vars
        else:
            for var_name in deterministic_var_names:
                var_config = deterministic_vars[var_name]
                # Pass n_images to deterministic value generation
                values = self._get_deterministic_values(var_name, var_config)
                if not values:
                     print(f"Warning: No values generated for deterministic variable '{var_name}' - Skipping.")
                     # Or raise error? For now, skip.
                else:
                    deterministic_values_lists.append(values)

            print("Generating base combinations from deterministic variables...")
            # Check if we actually have lists to combine
            if not deterministic_values_lists:
                 print("No values generated from any deterministic variable. Generating 1 configuration.")
                 base_combinations = [{}]
            else:
                 base_combinations_tuples = list(itertools.product(*deterministic_values_lists))
                 # Convert tuples back to dictionaries
                 base_combinations = [
                     dict(zip(deterministic_var_names, combo_tuple, strict=False))
                     for combo_tuple in base_combinations_tuples
                 ]
            print(f"Generated {len(base_combinations)} base combinations (after deterministic n_images expansion).")
            # print("Base combinations:", base_combinations) # Can be very long


        # 3. Generate final configurations iteratively, expanding for random variables
        print("Generating final configurations by adding fixed and random variables (with n_images expansion)...")
        current_stage_combinations = []
        # Start with base combinations, adding fixed values
        for base_combo in base_combinations:
            initial_config = {}
            initial_config.update(base_combo)
            initial_config.update(fixed_vars)
            current_stage_combinations.append(initial_config)

        print(f"  Starting with {len(current_stage_combinations)} combinations after deterministic+fixed.")

        # Iteratively expand for each random variable
        for var_name, var_config in random_vars.items():
            n_images_for_random = var_config.n_images
            print(f"  Expanding for random variable '{var_name}' (n_images={n_images_for_random})...")
            next_stage_combinations = []
            for config_so_far in current_stage_combinations:
                for _ in range(n_images_for_random):
                    # Generate a new random value for this specific image instance
                    random_value = self._get_single_random_value(var_name, var_config)
                    # Create a copy and add the new random value
                    new_config = config_so_far.copy()
                    new_config[var_name] = random_value
                    next_stage_combinations.append(new_config)
            current_stage_combinations = next_stage_combinations # Update for the next random variable
            print(f"    Combinations after '{var_name}': {len(current_stage_combinations)}")

        final_combinations = current_stage_combinations

        # 4. Apply multiplier for fixed variables with n_images > 1
        fixed_multiplier = 1
        fixed_vars_to_multiply = []
        for var_name, var_config in self.config.variables.items():
            if var_config.variate_type == 'fixed' and var_config.n_images > 1:
                fixed_multiplier *= var_config.n_images
                fixed_vars_to_multiply.append(f"{var_name} (n_images={var_config.n_images})")

        if fixed_multiplier > 1:
            print(f"Applying fixed variable multiplier: {fixed_multiplier} (from: {', '.join(fixed_vars_to_multiply)})")
            original_len = len(final_combinations)
            final_combinations = final_combinations * fixed_multiplier
            print(f"  Expanded combinations from {original_len} to {len(final_combinations)}")

        print(f"Total final combinations generated: {len(final_combinations)}")
        # print("FINAL COMBINATIONS:") # Can be very long
        # print(final_combinations)
        return final_combinations

class ConfigConverter:
     def __init__(self, default_config: dict[str, Any], dataset_config: DatasetConfig):
         self.default_config = default_config
         self.dataset_config = dataset_config

     def _update_config_with_variable(self, config: dict[str, Any], variable_name: str, value: Any):
         """Updates config dict assuming base structure exists."""
         # Special handling for noise configuration to map to noise_config field
         if variable_name.startswith('noise.'):
             # Split the variable name to get the noise parameter
             parts = variable_name.split('.')
             if len(parts) >= 2:
                 # Get just the parameter name (blur, table_texture, etc.)
                 noise_param = parts[1]
                 # Make sure noise_config exists
                 if 'noise_config' not in config:
                     config['noise_config'] = {}
                 # Set the parameter in noise_config
                 config['noise_config'][noise_param] = value
                 logger.debug(f"Mapped noise variable '{variable_name}' to noise_config.{noise_param}={value}")
                 return

         # Standard handling for all other variables
         keys = variable_name.split('.')
         current_level = config
         try:
             for i, key in enumerate(keys[:-1]):
                 if key.isdigit() and isinstance(current_level, list):
                      index = int(key)
                      if not (0 <= index < len(current_level)):
                          logger.error(f"Index {index} out of bounds (len {len(current_level)}) at path {'.'.join(keys[:i+1])} for '{variable_name}'")
                          return
                      current_level = current_level[index]
                 elif isinstance(current_level, dict):
                      if key not in current_level:
                          logger.warning(f"Creating missing dict key '{key}' at path {'.'.join(keys[:i])} for '{variable_name}'")
                          current_level[key] = {}
                      current_level = current_level[key]
                 else:
                      logger.error(f"Cannot traverse path. Non-container found at path {'.'.join(keys[:i+1])} for '{variable_name}'")
                      return
             final_key = keys[-1]
             if final_key.isdigit() and isinstance(current_level, list):
                 index = int(final_key)
                 if index >= len(current_level):
                     current_level.extend([None] * (index - len(current_level) + 1))
                 current_level[index] = value
             elif isinstance(current_level, dict):
                 current_level[final_key] = value
             else:
                 logger.error(f"Cannot set final value. Path termination point is not list/dict at path {'.'.join(keys[:-1])} for '{variable_name}'")
         except Exception as e:
             logger.error(f"Error in _update_config for '{variable_name}' = {value}: {e}", exc_info=True)


     def create_image_config(self, variable_values: dict[str, Any]) -> dict[str, Any]:
         """Creates the final image config by merging variable values into the default config.

         Relies on PokerSceneModel.__post_init__ to handle card/chip distribution based
         on the high-level inputs provided in variable_values.
         """
         logger.debug(f"Input variable values for config creation: {variable_values}")

         # 1. Start with deep copy of default config
         # Using json load/dump for a potentially safer deep copy
         try:
             config = json.loads(json.dumps(self.default_config))
         except TypeError as e:
             logger.error(f"Default config is not JSON serializable, falling back to basic copy: {e}")
             import copy
             config = copy.deepcopy(self.default_config)

         # 2. Apply Variables from the current combination
         for variable_name, value in variable_values.items():
             # Skip internal metadata key if present
             if variable_name == '_variable_values_used':
                 continue

             logger.debug(f"Applying variable: {variable_name} = {value}")
             # Use the helper to update the nested dictionary
             self._update_config_with_variable(config, variable_name, value)

         # 3. Return the merged config
         # PokerSceneModel.from_dict(config) will handle the rest in its __post_init__
         logger.debug("Finished merging variables into config.")
         # logger.debug(f"Final config passed to PokerSceneModel.from_dict: {json.dumps(config, indent=2)}") # Potentially very verbose
         return config


class DatasetGenerator:
    def __init__(self, config_path: str, default_config_path: str):
        self.config_path = config_path
        self.default_config_path = default_config_path
        self.config = self._load_config()
        self.default_config = self._load_default_config()
        self._setup_logging()

    def _load_config(self) -> DatasetConfig:
        """Load and validate YAML configuration."""
        # ... (loading logic) ...
        try:
            with open(self.config_path) as f:
                raw_config = yaml.safe_load(f)
                print("\n=== Raw YAML Configuration ===")
                print(yaml.dump(raw_config, default_flow_style=False))
        except yaml.YAMLError as e:
            logger.error(f"Error loading config: {e}\n{traceback.format_exc()}")
            raise

        if 'dataset' not in raw_config:
            raise ValueError("Missing required 'dataset' section in configuration")

        dataset_config_data = raw_config['dataset']

        variables = {}
        if 'variables' in raw_config:
            print("\n=== Processing Variables ===")
            for key, value in raw_config['variables'].items():
                print(f"\nProcessing variable: {key}")
                print(f"Raw value: {value}")
                if isinstance(value, dict):
                    variate_levels = value.get('variate_levels')
                    variate_type = value.get('variate_type')
                    # Convert list to tuple for range types if appropriate
                    if variate_type in ['varying_all_range', 'varying_among_range'] and isinstance(variate_levels, list):
                         # Check if it's a range like [min, max] or location range [[xmin,xmax],[ymin,ymax]]
                         if not (len(variate_levels) == 2 and isinstance(variate_levels[0], list) and isinstance(variate_levels[1], list)):
                              variate_levels = tuple(variate_levels)

                    variables[key] = VariableConfig(
                        variate_type=variate_type,
                        variate_levels=variate_levels,
                        n_images=value.get('n_images', 1), # Read n_images, default to 1
                        randomize=value.get('randomize', False),
                        randomize_percentage=value.get('randomize_percentage', 0.2),
                    )
                    print(f"Created VariableConfig: {variables[key]}")
                else: # Handle simple fixed values directly under variables:
                     variables[key] = VariableConfig(
                          variate_type='fixed',
                          variate_levels=value,
                          n_images=1 # n_images doesn't apply to fixed
                     )
                     print(f"Created Fixed VariableConfig: {variables[key]}")

        # Get piece_set, default to 'default' if not specified
        piece_set_name = dataset_config_data.get('piece_set', 'default').lower()
        allowed_piece_sets = {'default', 'old_school', 'stones_color'}
        if piece_set_name not in allowed_piece_sets:
            logger.warning(f"Invalid piece_set '{piece_set_name}' specified. Falling back to 'default'. Allowed: {allowed_piece_sets}")
            piece_set_name = 'default'

        return DatasetConfig(
            name=dataset_config_data.get('name', 'chess_dataset'),
            output_dir=dataset_config_data.get('output_dir', 'output'),
            seed=dataset_config_data.get('seed'),
            piece_set=piece_set_name,
            variables=variables
        )

    def _load_default_config(self) -> dict[str, Any]:
        """Load the default configuration."""
        with open(self.default_config_path) as f:
            return yaml.safe_load(f)

    def _setup_logging(self):
        """Setup logging configuration."""
        log_dir = Path(self.config.output_dir) / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s\n%(exc_info)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        logger.handlers = []
        log_file = log_dir / f'{self.config.name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        logger.addHandler(console_handler)
        logger.setLevel(logging.DEBUG)
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        # Add root handler only if it doesn't have one, to avoid duplicates if run multiple times
        if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
             root_handler = logging.StreamHandler()
             root_handler.setFormatter(formatter)
             root_handler.setLevel(logging.INFO)
             root_logger.addHandler(root_handler)


    def _generate_image(self, image_config: dict[str, Any], output_path_hint: Path):
        """Generate a single image using the poker scene generator."""
        # Determine final output paths based on the hint
        output_dir = output_path_hint.parent
        base_filename = output_path_hint.stem
        img_output_path = output_dir / f"{base_filename}.png"

        try:
            logger.info(f"Starting image generation for config derived from: {image_config.get('_variable_values_used', 'N/A')}")
            # Remove internal metadata key before passing to model
            variable_values_used = image_config.pop('_variable_values_used', None)

            # Create the PokerSceneModel instance. Its __post_init__ does the heavy lifting.
            scene_model = PokerSceneModel.from_dict(image_config)

            # Call the actual scene generation function
            generation_result_data = generate_poker_scene_from_config(
                 scene_model=scene_model, # MODIFIED: Use correct argument name 'scene_model'
                 output_dir=str(output_dir),
                 base_filename=base_filename,
                 # deck_blend_file=image_config.get('deck_blend_file') # scene_model now handles this
             )

            # Check if generation succeeded
            if generation_result_data and generation_result_data.get("image_path"):
                logger.info(f"Generated image: {generation_result_data['image_path']}")

                # Prepare the final config dict for metadata
                final_configs = {
                    "image_path": str(generation_result_data["image_path"]),
                    "final_scene_config": generation_result_data.get("final_scene_config"),
                    # Add the driving variable values back for metadata
                    "_variable_values_used": variable_values_used
                }
                return final_configs
            else:
                 logger.error(f"Image generation failed for {base_filename}. Result: {generation_result_data}")
                 return None # Indicate failure

        except Exception as e:
            logger.error(f"Error processing config or generating scene: {e}\n{traceback.format_exc()}")
            # print(f"Skipping image {output_path_hint.name} due to scene generation error: {e}")
            return None # Indicate failure

    def generate(self):
        """Generate the dataset based on the configuration."""
        logger.info(f"Starting dataset generation: {self.config.name}")
        output_dir_path = Path(self.config.output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        generator = VariableCombinationGenerator(self.config)
        combinations = generator.generate_variable_combinations()
        actual_num_images_planned = len(combinations)
        logger.info(f"Planned number of images based on config: {actual_num_images_planned}")

        converter = ConfigConverter(self.default_config, self.config)
        generation_results = []
        successful_generations = 0

        for i, variable_values in enumerate(combinations):
            logger.info(f"--- Generating image {i+1}/{actual_num_images_planned} ---")
            # Add original variables to dict for metadata tracking before passing to converter
            variable_values['_variable_values_used'] = variable_values.copy()
            logger.debug(f"Base variable values for image {i+1}: {variable_values['_variable_values_used']}")

            try:
                 image_config = converter.create_image_config(variable_values)

                 output_path_hint = output_dir_path / f"{self.config.name}_img_{i:05d}"
                 result = self._generate_image(image_config, output_path_hint)
                 if result:
                     generation_results.append(result)
                     successful_generations += 1
                 else:
                     logger.warning(f"Image {i+1} generation failed or returned None.")
            except Exception as e:
                logger.error(f"Failed to create config or generate image {i+1}: {e}", exc_info=True)
                # print(f"Skipping image {i+1} due to error: {e}") # Already logged
                continue

        logger.info(f"Finished generation loop. Successfully generated {successful_generations}/{actual_num_images_planned} images.")
        self._generate_metadata(generation_results)
        logger.info(f"Dataset generation complete: {self.config.name}")

    def _make_serializable(self, obj: Any) -> Any:
        """Recursively convert non-JSON serializable objects to strings."""
        if isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, str | int | float | bool) or obj is None:
            return obj
        else:
            # Attempt to convert other types to string
            try:
                # Test if it's directly serializable first (might catch edge cases)
                json.dumps(obj)
                return obj
            except TypeError:
                # If not, convert to string as fallback
                logger.debug(f"Converting non-serializable type {type(obj)} to string.")
                return str(obj)

    def _generate_metadata(self, generation_results: list[dict[str, Any]]):
        """Generate detailed metadata file (JSON Lines) for the dataset."""
        if not generation_results:
            logger.warning("No images were generated successfully, skipping metadata generation.")
            return

        num_images_generated = len(generation_results)
        # Use .jsonl for line-delimited JSON
        metadata_path = Path(self.config.output_dir) / 'metadata.jsonl'
        basic_metadata_path = Path(self.config.output_dir) / 'metadata_summary.json'

        # --- Prepare Serializable Variable Config for Summary ---
        serializable_var_config = {}
        if self.config.variables:
            for name, var in self.config.variables.items():
                levels = var.variate_levels
                safe_levels = levels # Default to original
                try:
                    # Test if levels are directly JSON serializable
                    json.dumps(levels)
                except TypeError:
                    # If not serializable, convert to string as a fallback
                    safe_levels = str(levels)
                    logger.debug(f"Converted non-serializable variate_levels for '{name}' to string in summary.")

                serializable_var_config[name] = {
                    'variate_type': var.variate_type,
                    'variate_levels': safe_levels, # Use the safe version
                    'n_images': var.n_images
                }
        else:
             logger.warning("No variables found in config for metadata summary.")

        # --- Create Summary Metadata Dictionary ---
        summary_metadata = {
            'name': self.config.name,
            'generation_date': datetime.now().isoformat(),
            'num_images_generated': num_images_generated,
            'seed': self.config.seed,
            'original_variable_config': serializable_var_config # Use the pre-processed dict
        }

        # --- Write Summary Metadata ---
        try:
            with open(basic_metadata_path, 'w') as f_sum:
                 json.dump(summary_metadata, f_sum, indent=2)
            logger.info(f"Generated summary metadata: {basic_metadata_path}")
        except TypeError as e:
             logger.error(f"Could not serialize summary metadata to JSON: {e}", exc_info=True)
             # Log the problematic dict for debugging
             logger.debug(f"Problematic summary_metadata: {summary_metadata}")
        except Exception as e:
             logger.error(f"Error writing summary metadata file: {e}", exc_info=True)

        # --- Write Detailed Line-by-Line Metadata ---
        try:
            with open(metadata_path, 'w') as f_meta:
                for result in generation_results:
                    # Make the config serializable before dumping
                    serializable_config = self._make_serializable(result.get('used_config'))
                    meta_entry = {
                        'file_path': result.get('image_path', 'UNKNOWN'),
                        'config': serializable_config # Use the cleaned config
                    }
                    json.dump(meta_entry, f_meta)
                    f_meta.write('\n')
            logger.info(f"Generated detailed metadata: {metadata_path}")
        except Exception as e:
             logger.error(f"Error writing detailed metadata: {e}", exc_info=True)

# ... (main function remains the same) ...
def main():
    """Main entry point for dataset generation."""
    parser = argparse.ArgumentParser(description="Generate chess image dataset from YAML configuration")
    parser.add_argument("config_path", type=str, help="Path to the main YAML configuration file.")
    parser.add_argument("--default-config", type=str, default=None, help="Path to the default YAML configuration file. If omitted, assumes 'default_config.yaml' in the same directory as config_path.")

    args = parser.parse_args()

    try:
        # --- Determine paths ---
        config_path = Path(args.config_path)
        if not config_path.is_file():
            logger.critical(f"Input configuration file not found: {config_path}")
            print(f"FATAL ERROR: Input configuration file not found: {config_path}")
            return

        if args.default_config:
            default_config_path = Path(args.default_config)
            logger.info(f"Using provided default config path: {default_config_path}")
        else:
            default_config_path = config_path.parent / "default_config.yml"
            logger.info(f"Default config path not provided, assuming: {default_config_path}")

        if not default_config_path.is_file():
            logger.critical(f"Default configuration file not found: {default_config_path}")
            print(f"FATAL ERROR: Default configuration file not found: {default_config_path}")
            return
        # ---------------------

        dataset_generator = DatasetGenerator(
            config_path=str(config_path),              # Pass as string
            default_config_path=str(default_config_path) # Pass as string
        )
        dataset_generator.generate()
        logger.info("Dataset generation completed successfully.")
    except Exception as e:
        # Log the final error trace before exiting
        logger.critical(f"An unhandled error occurred during dataset generation: {e}", exc_info=True)
        print(f"FATAL ERROR: {e}") # Also print to console

if __name__ == "__main__":
    # Setup basic logging JUST for argument parsing errors if main script execution fails early
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    main()

