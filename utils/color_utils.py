"""
Utility functions for color processing and naming.

This module provides functions for working with colors, including converting RGB values
to color names, calculating color distances, and other color-related utilities.
"""

from typing import Tuple, Dict, List, Union
import math


def RGB_to_color(rgb: Tuple[float, float, float]) -> str:
    """
    Convert RGB values to closest named color.
    
    Args:
        rgb: Tuple of (R, G, B) values in range 0-1
        
    Returns:
        String name of the closest matching color
    """
    # Ensure RGB values are in 0-1 range
    r, g, b = rgb
    
    # Dictionary of common colors with their RGB values (0-1 scale)
    color_map = {
        "black": (0.0, 0.0, 0.0),
        "white": (1.0, 1.0, 1.0),
        "red": (1.0, 0.0, 0.0),
        "green": (0.0, 1.0, 0.0),
        "blue": (0.0, 0.0, 1.0),
        "yellow": (1.0, 1.0, 0.0),
        "cyan": (0.0, 1.0, 1.0),
        "magenta": (1.0, 0.0, 1.0),
        "silver": (0.75, 0.75, 0.75),
        "gray": (0.5, 0.5, 0.5),
        "dark_gray": (0.25, 0.25, 0.25),
        "maroon": (0.5, 0.0, 0.0),
        "olive": (0.5, 0.5, 0.0),
        "dark_green": (0.0, 0.5, 0.0),
        "navy": (0.0, 0.0, 0.5),
        "purple": (0.5, 0.0, 0.5),
        "teal": (0.0, 0.5, 0.5),
        "brown": (0.65, 0.16, 0.16),
        "orange": (1.0, 0.65, 0.0),
        "gold": (1.0, 0.84, 0.0),
        "ivory": (1.0, 1.0, 0.94),
        "beige": (0.96, 0.96, 0.86),
        "tan": (0.82, 0.71, 0.55),
    }
    
    # Function to calculate Euclidean distance between colors
    def color_distance(c1: Tuple[float, float, float], c2: Tuple[float, float, float]) -> float:
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))
    
    # Find the closest color by minimizing the distance
    closest_color = min(color_map.items(), key=lambda x: color_distance(rgb, x[1]))
    
    return closest_color[0]


def hex_to_rgb(hex_color: str) -> Tuple[float, float, float]:
    """
    Convert hexadecimal color string to RGB tuple.
    
    Args:
        hex_color: Color in hex format (e.g., "#FF0000" or "FF0000")
        
    Returns:
        Tuple of (R, G, B) values in range 0-1
    """
    # Remove '#' if present
    hex_color = hex_color.lstrip('#')
    
    # Convert hex to RGB (0-255)
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    
    # Convert to 0-1 range
    return (r / 255.0, g / 255.0, b / 255.0)


def rgb_to_hex(rgb: Tuple[float, float, float]) -> str:
    """
    Convert RGB tuple to hexadecimal color string.
    
    Args:
        rgb: Tuple of (R, G, B) values in range 0-1
        
    Returns:
        Color in hex format (e.g., "#FF0000")
    """
    # Convert 0-1 range to 0-255 range and then to hex
    r, g, b = rgb
    r_int = min(255, max(0, int(r * 255)))
    g_int = min(255, max(0, int(g * 255)))
    b_int = min(255, max(0, int(b * 255)))
    
    return f"#{r_int:02X}{g_int:02X}{b_int:02X}"


def is_dark_color(rgb: Tuple[float, float, float]) -> bool:
    """
    Determine if a color is "dark" based on luminance.
    
    Args:
        rgb: Tuple of (R, G, B) values in range 0-1
        
    Returns:
        True if the color is dark, False otherwise
    """
    # Calculate luminance (perceived brightness)
    # Formula: 0.299*R + 0.587*G + 0.114*B
    r, g, b = rgb
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    
    # Usually, luminance < 0.5 is considered dark
    return luminance < 0.5


def get_contrasting_text_color(rgb: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """
    Get a contrasting color (black or white) for text displayed on the given background color.
    
    Args:
        rgb: Tuple of (R, G, B) values in range 0-1 for the background color
        
    Returns:
        Tuple of (R, G, B) for either black or white, whichever has better contrast
    """
    return (0.0, 0.0, 0.0) if not is_dark_color(rgb) else (1.0, 1.0, 1.0)


def interpolate_colors(color1: Tuple[float, float, float], 
                       color2: Tuple[float, float, float], 
                       factor: float) -> Tuple[float, float, float]:
    """
    Linearly interpolate between two colors.
    
    Args:
        color1: Starting color as (R, G, B) tuple
        color2: Ending color as (R, G, B) tuple
        factor: Interpolation factor (0.0 to 1.0)
        
    Returns:
        Interpolated color as (R, G, B) tuple
    """
    r1, g1, b1 = color1
    r2, g2, b2 = color2
    
    # Ensure factor is in range 0-1
    factor = max(0.0, min(1.0, factor))
    
    r = r1 + factor * (r2 - r1)
    g = g1 + factor * (g2 - g1)
    b = b1 + factor * (b2 - b1)
    
    return (r, g, b)


# Example usage
if __name__ == "__main__":
    # Test RGB to color name
    test_colors = [
        (0.0, 0.0, 0.0),        # Black
        (1.0, 1.0, 1.0),        # White
        (1.0, 0.0, 0.0),        # Red
        (0.0, 1.0, 0.0),        # Green
        (0.0, 0.0, 1.0),        # Blue
        (0.9, 0.9, 0.9),        # Almost white
        (0.1, 0.1, 0.1),        # Almost black
        (0.8, 0.4, 0.2),        # Brownish
        (0.5, 0.0, 0.5),        # Purple
    ]
    
    for color in test_colors:
        color_name = RGB_to_color(color)
        hex_code = rgb_to_hex(color)
        print(f"RGB{color} → {color_name} ({hex_code})")
        
    # Test hex to RGB
    hex_colors = ["#FF0000", "#00FF00", "#0000FF", "#FFFFFF", "#000000", "#FFFF00"]
    for hex_color in hex_colors:
        rgb = hex_to_rgb(hex_color)
        color_name = RGB_to_color(rgb)
        print(f"{hex_color} → RGB{rgb} → {color_name}") 