"""Module for generating chess images from YAML configurations."""

import os
from typing import Dict, Any, Tuple, Optional
import yaml
import logging
from dataclasses import dataclass
import json

from chess.config.advanced_models import (
    PieceCountModel,
    PieceTypeModel,
    PiecePosition,
)
from chess.config.advanced_generator import ChessConfigGenerator
from chess.generate_chess_image import generate_chess_image
from chess.config.models import MaterialModel, BoardModel

# Import the factories

from chess.pieces.factories import (
    DefaultPieceFactory,
    OldSchoolPieceFactory,
    StonesColorPieceFactory
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ChessConfig:
    """Configuration for chess piece generation."""
    count_config: PieceCountModel
    type_config: PieceTypeModel
    position_config: PiecePosition
    board_config: Dict[str, Any]

class ChessImageGenerator:
    """
    Generates chess images from YAML configurations or dictionaries using advanced models.
    
    This class handles the complete pipeline from configuration to rendered image,
    including scene setup, piece generation, and noise effects.
    
    The configuration can be provided either as:
    1. A path to a YAML file
    2. A dictionary containing the configuration
    
    Example YAML configuration:
    ```yaml
    chess:
      count_config:
        type: "preset"
        value: "medium"
        randomization: true
      type_config:
        type: "random_n"
        value: 3
      position_config:
        type: "spread"
        value: "medium"
        start_point: "center"
      board_config:
        length: 0.7
        width: 0.7
        thickness: 0.05
        location: [0, 0, 0.9]
        border_width: 0.05
        board_material:
          color: [0.26522322393366626, -0.09919445135564392, 0.9, 1.0]
          material_name: "ChessboardMaterial"
          roughness: 0.3
        square_black_material:
          color: [0.1, 0.1, 0.1, 1.0]
          material_name: "BlackSquareMaterial"
          roughness: 0.2
        square_white_material:
          color: [0.9, 0.9, 0.8, 1.0]
          material_name: "WhiteSquareMaterial"
          roughness: 0.2
    
    scene:
      camera:
        distance: 4.0
        angle: "medium"
      lighting:
        type: "medium"
      table:
        shape: "elliptic"
        length: 2.5
        width: 1.8
    
    noise:
      blur: "none"
      table_texture: "low"
      lightning: "none"
    ```
    """
    
    def __init__(self, 
                 config_path: Optional[str] = None, 
                 config_dict: Optional[Dict[str, Any]] = None,
                 piece_set: Optional[str] = None):
        """
        Initialize the generator with either a YAML configuration file or a dictionary.
        
        Args:
            config_path: Path to the YAML configuration file
            config_dict: Dictionary containing the configuration
            piece_set: Explicitly set the piece set to use (overrides config file)
            
        Raises:
            ValueError: If both config_path and config_dict are provided or neither is provided
        """
        if config_path is not None and config_dict is not None:
            raise ValueError("Cannot provide both config_path and config_dict")
        if config_path is None and config_dict is None:
            raise ValueError("Must provide either config_path or config_dict")
            
        self.config_path = config_path
        self.config_dict = config_dict
        self.config = None
        # Prioritize explicitly passed piece_set, otherwise default
        if piece_set:
            self.piece_set = piece_set.lower()
            logger.info(f"Piece set explicitly set to: {self.piece_set}")
        else:
            self.piece_set = "default" # Default if not passed
        self.chess_config = None
        self.scene_config = None
        self.noise_config = None
        
    def load_config(self) -> Dict[str, Any]:
        """
        Load and validate the configuration from either YAML file or dictionary.
        
        Returns:
            Dictionary containing the parsed configuration
            
        Raises:
            FileNotFoundError: If the config file doesn't exist
            yaml.YAMLError: If the YAML is invalid
            ValueError: If the configuration is invalid
        """
        if self.config_path is not None:
            if not os.path.exists(self.config_path):
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
                
            try:
                with open(self.config_path, 'r') as f:
                    # Load the full config here, including dataset level for piece_set
                    self.config = yaml.safe_load(f)
                    if self.config is None:
                        raise ValueError("Configuration file is empty or invalid")
            except yaml.YAMLError as e:
                raise yaml.YAMLError(f"Invalid YAML configuration: {e}")
        else:
            self.config = self.config_dict
            
        # Only try reading from config if piece_set wasn't explicitly passed during init
        if self.piece_set == "default" and self.config and \
           'dataset' in self.config and 'piece_set' in self.config['dataset']:
            self.piece_set = self.config['dataset']['piece_set'].lower()
            logger.info(f"Using piece set from config: {self.piece_set}")
        elif self.piece_set == "default":
            logger.info(f"Piece set not in config or explicitly passed, using default: {self.piece_set}")
            
        # Validate required sections
        required_sections = ['chess', 'scene']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required section in configuration: {section}")
                
        return self.config
    
    def _create_chess_config(self) -> ChessConfig:
        """
        Create chess configuration from the loaded YAML.
        
        Returns:
            ChessConfig object containing piece count, type, and position configurations
            
        Raises:
            ValueError: If the chess configuration is invalid
        """
        chess_config = self.config['chess']
        
        # Create piece count config
        count_type = chess_config['count_config']['type']
        count_value = chess_config['count_config']['value']
        randomization = chess_config['count_config'].get('randomization', False)
        randomization_percentage = chess_config['count_config'].get('randomization_percentage', 0.2)
        
        if count_type == "preset":
            count_config = PieceCountModel.from_preset(count_value, randomization, randomization_percentage)
        elif count_type == "fixed":
            count_config = PieceCountModel.from_fixed(count_value, randomization, randomization_percentage)
        elif count_type == "range":
            min_count, max_count = count_value
            count_config = PieceCountModel.from_range(min_count, max_count, randomization, randomization_percentage)
        elif count_type == "explicit":
            count_config = PieceCountModel.from_explicit(count_value, randomization, randomization_percentage)
        elif count_type == "range_by_type":
            count_config = PieceCountModel.from_range_by_type(count_value, randomization, randomization_percentage)
        else:
            raise ValueError(f"Invalid count configuration type: {count_type}")
            
        # Create piece type config
        type_type = chess_config['type_config']['spec_type']
        # Handle both 'value' and 'preset' keys for type configuration
        type_value = chess_config['type_config'].get('value') or chess_config['type_config'].get('preset') or chess_config['type_config'].get('n_types')
        if type_value is None:
            raise ValueError("Type configuration must have either 'value' or 'preset' key")
        
        if type_type == "preset":
            type_config = PieceTypeModel.from_preset(type_value)
        elif type_type == "explicit":
            type_config = PieceTypeModel.from_explicit(type_value)
        elif type_type == "random_n":
            type_config = PieceTypeModel.from_random_n(type_value)
        else:
            raise ValueError(f"Invalid type configuration type: {type_type}")
            
        # Create position config
        pos_type = chess_config['position_config']['type']
        pos_value = chess_config['position_config']['value']
        
        if pos_type == "spread":
            start_point = chess_config['position_config'].get('start_point', 'center')
            position_config = PiecePosition.from_spread(pos_value, start_point)
        elif pos_type == "bounds":
            position_config = PiecePosition.from_bounds()
        else:
            raise ValueError(f"Invalid position configuration type: {pos_type}")
            
        # Create board config
        board_config = chess_config['board_config']
        
        return ChessConfig(
            count_config=count_config,
            type_config=type_config,
            position_config=position_config,
            board_config=board_config
        )
    
    def _convert_material_config(self, material_dict: Dict[str, Any]) -> MaterialModel:
        """
        Convert a material configuration dictionary to a MaterialModel object.
        
        Args:
            material_dict: Dictionary containing material configuration
            
        Returns:
            MaterialModel object
        """
        if material_dict is None:
            material_dict = {}
            
        if isinstance(material_dict, MaterialModel):
            return material_dict
        
        return MaterialModel(
            color=material_dict.get('color', (0.8, 0.8, 0.8, 1.0)),
            material_name=material_dict.get('material_name'),
            roughness=material_dict.get('roughness', 0.5),
            custom_material=None
        )
    
    def generate_image(
        self,
        output_dir: Optional[str] = None,
        base_filename: str = "chess_scene"
    ) -> Tuple[str, Optional[str], Optional[str], Dict[str, Any], Optional[Dict[str, Any]]]:
        """
        Generate the chess image and return paths and final configurations.
        
        Args:
            output_dir: Directory to save output files (default: current directory)
            base_filename: Base name for output files
            
        Returns:
            Tuple containing:
            - Path to the rendered image
            - Path to the text legend (if generated)
            - Path to the JSON legend (if generated)
            - Final scene configuration dictionary (including camera, render, etc.)
            - Final noise configuration dictionary (or None)
            
        Raises:
            RuntimeError: If configuration hasn't been loaded
        """
        if self.config is None:
            # If config not loaded via file/dict, maybe it was set directly?
            # We need self.piece_set to be determined.
            if self.piece_set:
                logger.warning("Config not loaded, but piece_set was set. Proceeding with direct config.")
            else:
                # Fallback if generate_image is called without load_config and no piece_set known
                logger.error("Configuration must be loaded (or config_dict provided) before generating image, and piece_set must be determined.")
                raise RuntimeError("Configuration not loaded and piece_set unknown.")
        elif not self.piece_set: # Should be set during load_config
            raise RuntimeError("Configuration must be loaded before generating image")
            
        # Create chess configuration
        chess_config = self._create_chess_config()
        
        # Get the input board config dict
        input_board_config_dict = chess_config.board_config

        # Translate row_columns before creating the BoardModel
        if 'row_columns' in input_board_config_dict and isinstance(input_board_config_dict['row_columns'], int):
            rows_cols = input_board_config_dict['row_columns']
            input_board_config_dict['rows'] = rows_cols
            input_board_config_dict['columns'] = rows_cols
            del input_board_config_dict['row_columns'] # Remove the original key
            logger.info(f"Translated row_columns ({rows_cols}) to rows and columns in input dict.")

        # Now create the BoardModel from the potentially modified input dict
        board_model_from_input = BoardModel.from_dict(input_board_config_dict)

        # Get final dimensions to pass to generator
        final_rows = board_model_from_input.rows
        final_columns = board_model_from_input.columns

        # Create chess config generator
        generator = ChessConfigGenerator(
            count_config=chess_config.count_config,
            type_config=chess_config.type_config,
            position_config=chess_config.position_config
        )
        
        # Generate board and pieces configuration (gets default structures)
        board_config, pieces_config = generator.generate_all_configs(rows=final_rows, columns=final_columns)
        
        # --- Update generated board_config with values from input BoardModel --- 
        # This ensures user overrides and correct defaults (like materials) are used.
        board_config['rows'] = board_model_from_input.rows
        board_config['columns'] = board_model_from_input.columns
        board_config['location'] = board_model_from_input.location
        board_config['length'] = board_model_from_input.length
        board_config['width'] = board_model_from_input.width
        board_config['thickness'] = board_model_from_input.thickness
        board_config['border_width'] = board_model_from_input.border_width
        board_config['random_pattern'] = board_model_from_input.random_pattern
        board_config['pattern_seed'] = board_model_from_input.pattern_seed
        # Pass the material DICTIONARIES from the model, not MaterialModel objects
        board_config['board_material'] = board_model_from_input.board_material
        board_config['square_white_material'] = board_model_from_input.square_white_material
        board_config['square_black_material'] = board_model_from_input.square_black_material
        # ------------------------------------------------------------------------

        # Get scene and noise configuration
        scene_config = self.config.get('scene', {})
        noise_config = self.config.get('noise', None)
        
        # --- Instantiate the correct Piece Factory --- 
        if self.piece_set == "old_school":
            piece_factory = OldSchoolPieceFactory()
        elif self.piece_set == "stones_color":
            piece_factory = StonesColorPieceFactory()
        else: # Default case
            piece_factory = DefaultPieceFactory()
        logger.info(f"Using piece factory: {piece_factory.__class__.__name__}")
        # ---------------------------------------------

        # Generate the image
        try:
            (
                scene_path, 
                legend_txt_path, 
                legend_json_path, 
                final_scene_config, 
                final_board_config,  # We don't need board/piece config here, but must capture
                final_pieces_config, # We don't need board/piece config here, but must capture
                final_noise_config
            ) = generate_chess_image(
                scene_config=scene_config,
                board_config=board_config, # Pass the generated board_config dict
                pieces_config=pieces_config,
                piece_factory=piece_factory, # Pass the created factory instance
                noise_config=noise_config,
                output_dir=output_dir,
                base_filename=base_filename
            )
            
            # Return paths and the relevant final configurations
            return scene_path, legend_txt_path, legend_json_path, final_scene_config, final_noise_config
            
        except Exception as e:
            logger.error(f"Error generating chess image: {e}")
            raise

def main():
    """Example usage of the ChessImageGenerator."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate chess images from YAML configuration or JSON dictionary")
    parser.add_argument("--config-path", help="Path to YAML configuration file")
    parser.add_argument("--config-json", help="JSON string containing the configuration")
    parser.add_argument("--output-dir", help="Directory to save output files")
    parser.add_argument("--base-filename", default="chess_scene", help="Base name for output files")
    
    args = parser.parse_args()
    
    if args.config_path is None and args.config_json is None:
        parser.error("Either --config-path or --config-json must be provided")
    if args.config_path is not None and args.config_json is not None:
        parser.error("Cannot provide both --config-path and --config-json")
    
    try:
        if args.config_path:
            generator = ChessImageGenerator(config_path=args.config_path)
        else:
            config_dict = json.loads(args.config_json)
            generator = ChessImageGenerator(config_dict=config_dict)
            
        generator.load_config()
        image_path, txt_legend_path, json_legend_path, final_scene_config, final_noise_config = generator.generate_image(
            output_dir=args.output_dir,
            base_filename=args.base_filename
        )
        
        print(f"Generated image: {image_path}")
        if txt_legend_path:
            print(f"Text legend: {txt_legend_path}")
        if json_legend_path:
            print(f"JSON legend: {json_legend_path}")
            
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    exit(main()) 