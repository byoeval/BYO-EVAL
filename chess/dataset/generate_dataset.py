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

# ... (ConfigConverter class should ideally not need changes for this part) ...
# ... (DatasetGenerator class needs minor changes to stop using num_images/n_images) ...

class ConfigConverter:
     def __init__(self, default_config: dict[str, Any], dataset_config: DatasetConfig):
         self.default_config = default_config
         self.dataset_config = dataset_config # Keep reference for variable configs if needed

     def _update_config_with_variable(self, config: dict[str, Any], variable_name: str, value: Any):
         # This method should now correctly receive the final generated value
         # for each variable in the combination dictionary.
         # The logic to add z=0.9 for location should have happened during value generation.
         keys = variable_name.split('.')
         current_dict = config

         # Special handling for complex config structures
         if variable_name == 'chess.count_config':
            # Ensure nested structure exists
            current_dict = current_dict.setdefault('chess', {}).setdefault('count_config', {})
            # Update with converted value
            current_dict.update(self._convert_count_config(value))
         elif variable_name == 'chess.type_config':
            # Ensure nested structure exists
            current_dict = current_dict.setdefault('chess', {}).setdefault('type_config', {})
            # Update with converted value
            current_dict.update(self._convert_type_config(value))
         # Handle position config components individually
         elif variable_name == 'chess.position_config.spread_level':
            # Ensure nested structure exists and set type
            pos_config = current_dict.setdefault('chess', {}).setdefault('position_config', {})
            pos_config['type'] = 'spread' # Assume spread type
            pos_config['value'] = str(value).lower() # Set spread level value
         elif variable_name == 'chess.position_config.start_point':
            # Ensure nested structure exists and set type
            pos_config = current_dict.setdefault('chess', {}).setdefault('position_config', {})
            pos_config['type'] = 'spread' # Assume spread type
            pos_config['start_point'] = str(value).lower() # Set start point value
         # Handle board config components individually
         elif variable_name.startswith('chess.board_config.'):
             board_key = keys[-1]
             board_config = current_dict.setdefault('chess', {}).setdefault('board_config', {})
             board_config[board_key] = value # Value should be processed already (e.g., z=0.9 for location)
         else:
             # Default handling for simple dot-notation paths
             for key in keys[:-1]:
                 current_dict = current_dict.setdefault(key, {})
             current_dict[keys[-1]] = value


     # _convert_count_config, _convert_type_config remain mostly the same
     def _convert_count_config(self, value: Any) -> dict[str, Any]:
         """Convert a count value into the proper count_config dictionary format."""
         # Need to access original VariableConfig for randomization flags
         var_config = self.dataset_config.variables.get('chess.count_config')
         randomize = var_config.randomize if var_config else False
         randomize_percentage = var_config.randomize_percentage if var_config else 0.2

         if isinstance(value, int):
             return {
             'type': 'fixed', # Assuming int count is fixed type
             'value': value,
             'randomize': randomize,
             'randomize_percentage': randomize_percentage
             }
         elif isinstance(value, str):
             return {
                 'type': 'preset', # Assuming str count is preset type
                 'value': str(value),
                 'randomize': randomize,
                 'randomize_percentage': randomize_percentage
             }
         elif isinstance(value, dict) and 'value' in value:
              return value
         raise ValueError(f"Invalid count value type: {type(value)}, value: {value}")

     def _convert_type_config(self, value: Any) -> dict[str, Any]:
         """Convert a type value into the proper type_config dictionary format."""
         if isinstance(value, str):
             return {
                 'spec_type': 'preset',
                 'value': str(value).lower()
             }
         elif isinstance(value, list):
             return {
                 'spec_type': 'explicit',
                 'value': value
             }
         elif isinstance(value, int):
             return {
                 'spec_type': 'random_n',
                 'value': value
             }
         elif isinstance(value, dict) and 'spec_type' in value:
              return value
         raise ValueError(f"Invalid type_config value type: {type(value)}, value: {value}")

     def create_image_config(self, variable_values: dict[str, Any]) -> dict[str, Any]:
        """Create a full image configuration from variable values."""
        print("\n=== Creating Image Configuration ===")
        print(f"Input variable values: {variable_values}")

        # Start with deep copy of default config to avoid modification issues
        config = json.loads(json.dumps(self.default_config))

        # Update the default config with variable values
        for variable_name, value in variable_values.items():
             print(f"Processing variable: {variable_name} = {value}")
             self._update_config_with_variable(config, variable_name, value)

        # Ensure position_config has default start_point if only spread_level was provided
        if 'chess' in config and 'position_config' in config['chess']:
             if 'start_point' not in config['chess']['position_config']:
                  config['chess']['position_config']['start_point'] = 'center'


        print("Final image config created.")
        return config


class DatasetGenerator:
    def __init__(self, config_path: str, default_config_path: str):
        self.config_path = config_path
        self.default_config_path = default_config_path
        self.config = self._load_config()
        self.default_config = self._load_default_config()
        self._setup_logging()

    # ... (_load_config, _load_default_config, _setup_logging remain largely the same) ...
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


    def _generate_image(self, image_config: dict[str, Any], output_path: Path):
         """Generate a single image using the appropriate chess image generator."""
         # ... (image generation logic remains the same) ...
         try:
             temp_config_path = output_path.parent / f'temp_config_{output_path.stem}.yaml'
             # print as yaml
             print("IMAGE CONFIG : " + str(yaml.dump(image_config, default_flow_style=None, sort_keys=False)))
             try:
                 with open(temp_config_path, 'w') as f:
                     yaml.dump(image_config, f, default_flow_style=None, sort_keys=False) # Nicer formatting

                 try:
                     from chess.generate_img_from_yaml import ChessImageGenerator
                     logger.info(f"Generating image from config: {temp_config_path}")
                     # Pass the piece_set from the main config directly
                     generator = ChessImageGenerator(
                         config_path=str(temp_config_path),
                         piece_set=self.config.piece_set
                     )
                     generator.load_config()
                     # Now capture all return values, including final configs
                     (
                         image_path,
                         txt_legend_path,
                         json_legend_path,
                         final_scene_config,
                         final_noise_config
                     ) = generator.generate_image(
                          output_dir=str(output_path.parent),
                          base_filename=output_path.stem
                      )
                     logger.info(f"Generated image: {image_path}")

                     # If image generation was successful, prepare the final config dict
                     final_configs = {
                         "image_path": str(image_path),
                         "initial_config": image_config, # Keep initial for reference?
                         "final_scene_config": final_scene_config,
                         "final_noise_config": final_noise_config,
                         # Add final board/piece configs if needed later?
                     }
                     # Return success with data (still inside the inner 'try' block)
                     return final_configs
                 except ImportError as e:
                     logger.error(f"Could not import ChessImageGenerator: {e}")
                     raise
                 except Exception as e:
                     logger.error(f"Error generating image for {temp_config_path}: {e}")
                     print(f"Skipping image {output_path.name} due to generation error.")
                     return None # Indicate failure (still inside the inner 'try' block)
             finally:
                 # This 'finally' block executes after the 'try' block, regardless of success/failure/return
                 if temp_config_path.exists():
                     temp_config_path.unlink()
         except Exception as e:
             # This catches errors outside the inner 'try' (e.g., temp file creation)
             logger.error(f"Error during image generation process setup: {e}")
             raise # Raise outer errors

    def generate(self):
        """Generate the dataset based on the new combination strategy."""
        logger.info(f"Starting dataset generation: {self.config.name} (using deterministic strategy)")

        # Initialize variable combination generator
        generator = VariableCombinationGenerator(self.config)
        # Generate combinations - total number is now determined by deterministic vars
        combinations = generator.generate_variable_combinations()
        actual_num_images_planned = len(combinations)
        logger.info(f"Planned number of images based on config: {actual_num_images_planned}")

        # Initialize config converter
        converter = ConfigConverter(self.default_config, self.config)
        print("LEN OF COMBINATIONS : " + str(len(combinations)))

        # List to store results (image path and final config) for metadata
        generation_results = []

        # Generate images for each combination
        output_dir_path = Path(self.config.output_dir)
        successful_generations = 0
        for i, variable_values in enumerate(combinations):
            logger.info(f"Generating image {i+1}/{actual_num_images_planned}")

            # Create image config
            try:
                 image_config = converter.create_image_config(variable_values)
                 # print(" IMAGE CONFIG : " + str(image_config)) # Too verbose

                 # print(f"Image config {i+1}: {yaml.dump(image_config, default_flow_style=False)}")
                 # Generate image
                 # Use a more descriptive name based on index
                 output_path = output_dir_path / f"{self.config.name}_img_{i:05d}.png"
                 result = self._generate_image(image_config, output_path)
                 if result:
                     generation_results.append(result)
                     successful_generations += 1
            except Exception as e:
                logger.error(f"Failed to create config or generate image {i+1}: {e}")
                print(f"Skipping image {i+1} due to error.")
                continue # Skip to next image

        print("Ended generation of " + str(successful_generations) + " images.")
        # Generate metadata
        self._generate_metadata(generation_results) # Pass results list

        logger.info(f"Dataset generation complete: {self.config.name}")

    def _generate_metadata(self, generation_results: list[dict[str, Any]]):
         """Generate detailed metadata file (JSON Lines) for the dataset."""
         if not generation_results:
             logger.warning("No images were generated successfully, skipping metadata generation.")
             return

         num_images_generated = len(generation_results)
         metadata_path = Path(self.config.output_dir) / 'metadata.jsonl' # Use .jsonl extension

         # Save basic overall metadata separately (optional, or could include in jsonl)
         # Example: basic_metadata_path = Path(self.config.output_dir) / 'metadata_summary.json'
         summary_metadata = {
             'name': self.config.name,
             'generation_date': datetime.now().isoformat(),
             'num_images_generated': num_images_generated,
             'seed': self.config.seed,
             'piece_set_used': self.config.piece_set, # Add piece set info
             'original_variable_config': {
                 name: {
                     'variate_type': var.variate_type,
                     'variate_levels': str(var.variate_levels) if not isinstance(var.variate_levels, list | tuple | dict) else var.variate_levels,
                     'randomize': var.randomize,
                     'randomize_percentage': var.randomize_percentage,
                     'n_images': var.n_images
                 }
                 for name, var in self.config.variables.items()
             }
         }
         basic_metadata_path = Path(self.config.output_dir) / 'metadata_summary.json'
         with open(basic_metadata_path, 'w') as f_sum:
              json.dump(summary_metadata, f_sum, indent=2)
         logger.info(f"Generated summary metadata: {basic_metadata_path}")

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
    main()

