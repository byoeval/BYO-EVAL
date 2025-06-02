import argparse
import yaml
import os
import sys
import bpy
import traceback

from poker.config.models import PokerSceneModel
from poker.scene_generator import generate_poker_scene_from_config
from scene_setup.rendering import clear_scene


def load_yaml_config(config_path: str) -> dict:
    """Loads a YAML configuration file."""
    print(f"Loading configuration from: {config_path}")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        if not config:
            raise ValueError("Configuration file is empty or invalid.")
        print("Configuration loaded successfully.")
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file {config_path}: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error in configuration file: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Generate a single poker scene image from a YAML configuration file.")
    parser.add_argument("config_path", type=str, help="Path to the YAML configuration file.")
    
    # Parse arguments 
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    args = parser.parse_args(argv)
    if not args:
        print("Argument parsing failed. Exiting.")
        return

    # --- Load Config ---
    full_config = load_yaml_config(args.config_path)

    # --- Get and Prepare Output Path from Top Level --- 
    output_path_relative = full_config.get('output_path', 'poker/img/default_render.png')
    if not output_path_relative:
        print("Error: Top-level 'output_path' missing in configuration file.")
        print("Using default output path: poker/img/default_render.png")
    
    final_output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), output_path_relative))
    output_dir = os.path.dirname(final_output_path)
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Ensured output directory exists: {output_dir}")
        print(f"Output path set to: {final_output_path}")
    except OSError as e:
        print(f"Error creating output directory {output_dir}: {e}")
        sys.exit(1)
    # -----------------------------------------------------

    # --- Clear default scene --- 
    clear_scene()

    # --- Parse Config into Model --- 
    try:
        print("Parsing configuration into PokerSceneModel...")
        poker_scene_model = PokerSceneModel.from_dict(full_config)
        print("PokerSceneModel created successfully.")
    except (ValueError, TypeError) as e:
        print(f"Error creating PokerSceneModel from config: {e}")
        print(f"Problematic Config Snippet:\\n{yaml.dump(full_config, default_flow_style=False, indent=2, sort_keys=False)[:1000]}...")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during model parsing: {e}")
        traceback.print_exc()
        sys.exit(1)

    # --- Generate Scene (includes general setup, but NOT output path) ---
    try:
        print("Generating poker scene (includes general setup)...")
        loaded_objects = generate_poker_scene_from_config(poker_scene_model)
        num_cards = len(loaded_objects.get('cards', []))
        num_chips = len(loaded_objects.get('chips', []))
        print(f"Scene generation complete. Created {num_cards} cards, {num_chips} chips.")
        # (Optional check for empty generation remains)
        if num_cards == 0 and num_chips == 0 and not poker_scene_model.scene_setup:
             print("Warning: Scene generation resulted in no poker objects and no specific scene setup was requested.")

    except Exception as e:
        print(f"An unexpected error occurred during scene generation: {e}")
        traceback.print_exc()
        sys.exit(1)

    # --- Set Render Filepath and Render --- 
    try:
        # Explicitly set the filepath before rendering
        bpy.context.scene.render.filepath = final_output_path
        print(f"Starting final render to {bpy.context.scene.render.filepath}...")
        bpy.ops.render.render(write_still=True)
        print(f"Render finished successfully! Image saved to {bpy.context.scene.render.filepath}")
    except Exception as e:
        print(f"Error during final rendering: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Unhandled exception in main execution: {e}")
        traceback.print_exc()
        try:
            sys.exit(1)
        except SystemExit:
            os._exit(1) 