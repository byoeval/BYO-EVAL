from typing import Any

import bpy

# --- Grid Drawing Logic ---

def _draw_grid_on_image(image: bpy.types.Image, grid_config: dict[str, Any]) -> bool:
    """
    Draws a grid directly on the provided Blender Image object's pixel data.

    Args:
        image: The bpy.types.Image object to draw on.
        grid_config: A dictionary containing grid parameters:
            granularity (int): Number of divisions for the grid.
            line_thickness (int): Thickness of the grid lines in pixels.
            line_color_rgba (tuple): RGBA color for the grid lines.

    Returns:
        bool: True if drawing was attempted, False if image was invalid.
    """
    if not image:
        print("Error (draw_grid_on_image): Invalid image object provided.")
        return False

    print(f"Executing _draw_grid_on_image for image '{image.name}' with config: {grid_config}")

    n_divisions = grid_config.get("granularity", 2)
    line_thickness = grid_config.get("line_thickness", 1)
    line_color_rgba = grid_config.get("line_color_rgba", (0.0, 0.0, 0.0, 0.8))

    # Use the image's actual dimensions
    width = image.size[0]
    height = image.size[1]

    if width == 0 or height == 0:
        print(f"Error (draw_grid_on_image): Image '{image.name}' has zero width or height.")
        # Try to get scene render resolution as a fallback if this image IS "Render Result"
        # but this function is now generic, so direct scene access is less appropriate here.
        # For loaded images, size should be correct.
        return False

    # Ensure pixels are loaded if they aren't already (especially for loaded images)
    if not image.has_data:
        print(f"Info (draw_grid_on_image): Image '{image.name}' has no data, attempting to load.")
        # This check might be more relevant for file-loaded images.
        # For "Render Result", it should have data if render succeeded.
        # However, this function is now generic.
        # If image.pixels is accessed on an image without data, it can error.
        # For now, assume if size is non-zero, pixels can be accessed.


    try:
        # It's crucial that the image's pixel data is available and correctly sized.
        # For images loaded from file, this should be true after loading.
        # For "Render Result", it's true after rendering.
        pixels = list(image.pixels) # Make a mutable copy
        if len(pixels) != width * height * 4:
            print(f"Error (draw_grid_on_image): Pixel data length mismatch for image '{image.name}'. Expected {width*height*4}, got {len(pixels)}.")
            return False
    except RuntimeError as e:
        print(f"Error (draw_grid_on_image): Could not access pixels for image '{image.name}': {e}")
        return False

    # Draw Horizontal Lines
    if n_divisions > 1 and height > 0:
        spacing_y = height / n_divisions
        for i in range(1, n_divisions):
            y_coord_center = i * spacing_y
            for t_offset in range(line_thickness):
                y_coord = int(y_coord_center - (line_thickness / 2.0) + t_offset)
                if 0 <= y_coord < height:
                    for x in range(width):
                        idx = (y_coord * width + x) * 4
                        if idx + 3 < len(pixels):
                            pixels[idx+0] = line_color_rgba[0]
                            pixels[idx+1] = line_color_rgba[1]
                            pixels[idx+2] = line_color_rgba[2]
                            pixels[idx+3] = line_color_rgba[3]

    # Draw Vertical Lines
    if n_divisions > 1 and width > 0:
        spacing_x = width / n_divisions
        for i in range(1, n_divisions):
            x_coord_center = i * spacing_x
            for t_offset in range(line_thickness):
                x_coord = int(x_coord_center - (line_thickness / 2.0) + t_offset)
                if 0 <= x_coord < width:
                    for y in range(height):
                        idx = (y * width + x_coord) * 4
                        if idx + 3 < len(pixels):
                            pixels[idx+0] = line_color_rgba[0]
                            pixels[idx+1] = line_color_rgba[1]
                            pixels[idx+2] = line_color_rgba[2]
                            pixels[idx+3] = line_color_rgba[3]

    try:
        image.pixels = pixels # Apply the modified pixels back to the image
        image.update() # Mark image for refresh / ensure changes are registered
        print(f"Grid drawn on image '{image.name}' with {n_divisions} divisions.")
        return True
    except RuntimeError as e:
        print(f"Error (draw_grid_on_image): Could not set pixels for image '{image.name}': {e}")
        return False

def add_grid_to_image_file(image_filepath: str, grid_config: dict[str, Any], output_filepath: str | None = None) -> bool:
    """
    Loads an image from a file, draws a grid on it, and saves it.

    Args:
        image_filepath: Path to the input image file.
        grid_config: Configuration for the grid (n_divisions, line_thickness, etc.).
        output_filepath: Optional. Path to save the modified image.
                         If None, overwrites the input file.

    Returns:
        bool: True if successful, False otherwise.
    """
    print(f"Attempting to add grid to image file: {image_filepath}")
    # Ensure the image is not already loaded with the same name, or use a unique internal name
    # For simplicity, we'll load it. If it was already in bpy.data.images, this might get that one.
    # A more robust way for temporary processing might involve unique names or direct pixel libs.
    loaded_image = bpy.data.images.load(filepath=image_filepath)

    if not loaded_image:
        print(f"Error: Could not load image from {image_filepath}")
        return False

    # Check if image was packed, if so unpack it to make pixels accessible
    if loaded_image.packed_file:
        print(f"Info: Image '{loaded_image.name}' is packed. Unpacking...")
        loaded_image.unpack(method='USE_ORIGINAL') # Or 'WRITE_ORIGINAL' if it needs to be saved out

    # Ensure pixel data is accessible after load (size should be non-zero)
    if loaded_image.size[0] == 0 or loaded_image.size[1] == 0:
        print(f"Error: Loaded image '{loaded_image.name}' from {image_filepath} has zero dimensions.")
        bpy.data.images.remove(loaded_image) # Clean up
        return False

    success = _draw_grid_on_image(loaded_image, grid_config)

    if success:
        # Save the modified image
        save_path = output_filepath if output_filepath else image_filepath
        print(f"Saving modified image to: {save_path}")

        # Important: Blender needs to know the file format for saving.
        # We infer it from the original filepath or default to PNG.
        # This might not be robust if output_filepath has a different extension.
        original_format = loaded_image.file_format
        if not original_format and save_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
             if save_path.lower().endswith('.png'):
                 loaded_image.file_format = 'PNG'
             elif save_path.lower().endswith(('.jpg', '.jpeg')):
                 loaded_image.file_format = 'JPEG'
             # Add other common formats if needed
             else:
                 loaded_image.file_format = 'PNG' # Default
        elif not original_format:
            loaded_image.file_format = 'PNG' # Default if cannot infer

        # If we are overwriting, filepath_raw might be set.
        # If saving to a new path, it's better to set filepath_raw.
        loaded_image.filepath_raw = save_path

        try:
            loaded_image.save()
            print(f"Successfully saved modified image to {save_path}")
        except RuntimeError as e:
            print(f"Error saving modified image to {save_path}: {e}")
            success = False

    # Clean up by removing the image from Blender's data unless it's meant to persist
    # For a post-processing step like this, usually, we remove it.
    if loaded_image.users == 0 or loaded_image.name in bpy.data.images: # Only remove if not used elsewhere (e.g. by a material)
         bpy.data.images.remove(loaded_image)


    return success
