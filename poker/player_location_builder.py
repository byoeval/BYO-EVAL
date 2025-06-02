import logging
import math
from typing import TYPE_CHECKING, Any

# Conditional import for type hinting to avoid circular dependency
if TYPE_CHECKING:
    from poker.config.models import PlayerDistribution

logger = logging.getLogger(__name__)

def calculate_player_locations(
    distribution_config: 'PlayerDistribution',
    num_players_to_place: int,
    table_conf: dict[str, Any],
    table_height: float
) -> list[tuple[float, float, float]]:
    """
    Calculates the (x, y, z) locations for players based on distribution settings
    and table configuration.

    Args:
        distribution_config: The PlayerDistribution configuration instance.
        num_players_to_place: The number of player locations to generate.
        table_conf: Dictionary containing table configuration (shape, dimensions).
        table_height: The height of the table surface.

    Returns:
        A list of (x, y, z) tuples representing player locations.

    Raises:
        ValueError: If required table dimensions are missing or invalid.
        NotImplementedError: If the table shape is unsupported.
    """
    locations = []
    if num_players_to_place <= 0:
        return locations

    try:
        table_shape = table_conf.get('shape', 'rectangular')
    except AttributeError:
        logger.error("Invalid table_conf format, expected a dictionary.")
        raise ValueError("Invalid table_conf format, expected a dictionary.")

    if table_shape == "circular":
        try:
            # Check for 'diameter' first, then fall back to 'width'
            table_diameter = None
            if 'diameter' in table_conf:
                 table_diameter = float(table_conf['diameter'])
                 logger.debug(f"Using 'diameter' ({table_diameter}) for circular table radius.")
            elif 'width' in table_conf:
                 table_diameter = float(table_conf['width'])
                 logger.debug(f"Using 'width' ({table_diameter}) as diameter fallback for circular table radius.")
            else:
                raise KeyError("Circular layout requires 'diameter' or 'width' in table_conf.")

            if table_diameter <= 0:
                raise ValueError("Table diameter/width must be positive.")
            table_radius = table_diameter / 2.0
        except KeyError as e:
            logger.error(f"{e}")
            raise ValueError(str(e))
        except (ValueError, TypeError) as e:
             logger.error(f"Invalid dimension for circular table: {e}")
             raise ValueError(f"Invalid dimension for circular table: {e}")

        if distribution_config.spacing_degrees is not None:
            angle_step_rad = math.radians(distribution_config.spacing_degrees)
        else:
            angle_step_rad = 2 * math.pi / num_players_to_place

        start_angle_rad = math.radians(distribution_config.start_angle_degrees)
        direction = 1 if distribution_config.trigonometric_direction else -1
        placement_radius = table_radius * 0.75

        for i in range(num_players_to_place):
            current_angle = start_angle_rad + direction * i * angle_step_rad
            loc_x = placement_radius * math.cos(current_angle)
            loc_y = placement_radius * math.sin(current_angle)
            loc_z = table_height + 0.01
            locations.append((loc_x, loc_y, loc_z))

    elif table_shape == "rectangular":
        try:
            length = float(table_conf['length'])
            width = float(table_conf['width'])
            if length <= 0 or width <= 0:
                raise ValueError("Table length and width must be positive.")
        except KeyError:
            logger.error("Rectangular layout requires 'length' and 'width' in table_conf.")
            raise ValueError("Rectangular layout requires 'length' and 'width' in table_conf.")
        except (ValueError, TypeError) as e:
            logger.error(f"Invalid dimensions for rectangular table: {e}")
            raise ValueError(f"Invalid dimensions for rectangular table: {e}")

        # Factor to control how far players are placed from the edge towards the center.
        # 0.0 = at the edge (original position), 1.0 = at the very center (0,0).
        player_centering_factor: float = 0.1 # Example: Move players 10% towards the center
        if not (0.0 <= player_centering_factor < 1.0):
            logger.warning(f"player_centering_factor ({player_centering_factor}) should ideally be between 0.0 and 1.0.")
            player_centering_factor = max(0.0, min(player_centering_factor, 0.99)) # Clamp reasonably

        perimeter = 2 * (length + width)
        spacing = perimeter / num_players_to_place if num_players_to_place > 0 else 0
        inset = 0.1 # Keep original inset for initial path calculation

        half_length = length / 2.0 - inset
        half_width = width / 2.0 - inset
        current_distance = 0.0

        sides = [
            (-half_width, -half_length, 1, 0, width - 2*inset),
            (half_width, -half_length, 0, 1, length - 2*inset),
            (half_width, half_length, -1, 0, width - 2*inset),
            (-half_width, half_length, 0, -1, length - 2*inset)
        ]

        player_idx = 0
        for sx, sy, dx, dy, side_len in sides:
            while current_distance < side_len and player_idx < num_players_to_place:
                # Calculate original position along the inset perimeter
                original_loc_x = sx + dx * current_distance
                original_loc_y = sy + dy * current_distance
                loc_z = table_height + 0.01

                # Apply centering factor to move towards (0,0)
                centered_loc_x = original_loc_x * (1.0 - player_centering_factor)
                centered_loc_y = original_loc_y * (1.0 - player_centering_factor)

                locations.append((centered_loc_x, centered_loc_y, loc_z))
                player_idx += 1
                current_distance += spacing

            current_distance -= side_len

        if player_idx < num_players_to_place and len(locations) < num_players_to_place:
             sx, sy, dx, dy, _ = sides[0]
             loc_x = sx + dx * current_distance
             loc_y = sy + dy * current_distance
             loc_z = table_height + 0.01
             locations.append((loc_x, loc_y, loc_z))
             logger.warning(f"Placed player {player_idx} using wrap-around logic due to potential spacing calculation limits.")

    elif table_shape == "elliptic":
        logger.error("Elliptic table layout is not yet implemented for player placement.")
        raise NotImplementedError("Elliptic table layout not implemented.")
    else:
        logger.error(f"Unsupported table shape: {table_shape}")
        raise ValueError(f"Unsupported table shape: {table_shape}")

    return locations

# Removed build_player_locations_from_config as it's now redundant
