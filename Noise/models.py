from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, List, Union
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class NoiseType(Enum):
    """Enum for noise types."""
    BLUR = "blur"
    LIGHT = "light"
    TABLE_TEXTURE = "table_texture"
    BOARD = "board"
    DISTRACTORS = "distractors"


@dataclass
class BaseNoiseModel:
    """Base model for all noise effects."""
    enabled: bool = True
    intensity: float = 1.0
    seed: Optional[int] = None
    blend_mode: str = "MIX"
    opacity: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert the model to a dictionary format."""
        return {
            "enabled": self.enabled,
            "intensity": self.intensity,
            "seed": self.seed,
            "blend_mode": self.blend_mode,
            "opacity": self.opacity
        }
        
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'BaseNoiseModel':
        """Create a BaseNoiseModel from a dictionary."""
        if not isinstance(config, dict):
            config = {}
            
        default_instance = cls()
        return cls(
            enabled=config.get("enabled", default_instance.enabled),
            intensity=config.get("intensity", default_instance.intensity),
            seed=config.get("seed", default_instance.seed),
            blend_mode=config.get("blend_mode", default_instance.blend_mode),
            opacity=config.get("opacity", default_instance.opacity)
        )


@dataclass
class BlurNoiseModel:
    """Model for blur noise effects.
    
    The blur effect is implemented using camera depth of field.
    String presets: "none", "very_low", "low", "medium", "high", "very_high"
    """
    # Define blur intensity presets
    BLUR_PRESETS = {
        "none": None,     # No blur - special case handled in setup
        "very_low": 9.0,  # Very subtle blur
        "low": 4.0,       # Light blur
        "medium": 2.0,    # Standard blur
        "high": 1.0,      # Strong blur
        "very_high": 0.5  # Very strong blur
    }

    intensity: Union[str, float] = "none"  # Can be string preset or direct float F-stop value

    def to_dict(self) -> Dict[str, Any]:
        """Convert the model to a dictionary format."""
        return {"blur": self.intensity}
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'BlurNoiseModel':
        """Create a BlurNoiseModel from a dictionary."""
        if not isinstance(config, dict):
            config = {}
            
        blur_value = config.get("blur", "none")
        
        # Validate if it's a string preset or a direct number
        if isinstance(blur_value, str):
            if blur_value not in cls.BLUR_PRESETS:
                logger.warning(f"Invalid blur preset '{blur_value}'. Using 'none'. Valid presets: {list(cls.BLUR_PRESETS.keys())}")
                blur_value = "none"
        elif not isinstance(blur_value, (int, float)):
             logger.warning(f"Invalid blur value type '{type(blur_value)}'. Using 'none'.")
             blur_value = "none"

        default_instance = cls()
        return cls(intensity=blur_value)


@dataclass
class LightNoiseModel:
    """Model for light noise effects.
    
    Lighting presets:
    - "very_low": 30% of standard lighting
    - "low": 60% of standard lighting
    - "medium": Standard lighting (100%)
    - "high": 150% of standard lighting
    - "very_high": 200% of standard lighting
    """
    # Define lighting intensity presets
    LIGHTING_PRESETS = {
        "very_low": 0.3,  # 30% of standard lighting
        "low": 0.6,       # 60% of standard lighting
        "medium": 1.0,    # Standard lighting (100%)
        "high": 1.5,      # 150% of standard lighting
        "very_high": 2.0  # 200% of standard lighting
    }

    # Base energy values for standard (medium) lighting
    BASE_KEY_LIGHT_ENERGY = 400
    BASE_FILL_LIGHT_ENERGY = 200
    BASE_BACK_LIGHT_ENERGY = 300

    lighting: str = "medium"  # Lighting intensity preset

    def to_dict(self) -> Dict[str, Any]:
        """Convert the model to a dictionary format."""
        return {"lighting": self.lighting}
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'LightNoiseModel':
        """Create a LightNoiseModel from a dictionary."""
        if not isinstance(config, dict):
            config = {}
            
        default_instance = cls()
        return cls(lighting=config.get("lighting", default_instance.lighting))


@dataclass
class TableTextureNoiseModel:
    """Model for table texture noise effects.
    
    Texture entropy presets:
    - "low": Simple monochrome table
    - "medium": Noisy texture with some color variation
    - "high": Complex texture with shapes and colors
    """
    # Define texture entropy presets
    TEXTURE_ENTROPY_PRESETS = {
        "low": 0,      # Simple monochrome table
        "medium": 1,   # Noisy texture with some color variation
        "high": 2      # Complex texture with shapes and colors
    }

    # Base colors for different entropy levels
    LOW_ENTROPY_COLORS = [
        (0.8, 0.8, 0.8, 1.0),  # Light gray
        (0.6, 0.6, 0.6, 1.0),  # Medium gray
        (0.4, 0.4, 0.4, 1.0),  # Dark gray
        (0.8, 0.7, 0.6, 1.0),  # Light wood
        (0.6, 0.5, 0.4, 1.0),  # Medium wood
        (0.4, 0.3, 0.2, 1.0),  # Dark wood
    ]

    table_texture: str = "medium"  # Texture entropy preset

    def to_dict(self) -> Dict[str, Any]:
        """Convert the model to a dictionary format."""
        return {"table_texture": self.table_texture}
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'TableTextureNoiseModel':
        """Create a TableTextureNoiseModel from a dictionary."""
        if not isinstance(config, dict):
            config = {}
            
        default_instance = cls()
        return cls(table_texture=config.get("table_texture", default_instance.table_texture))


@dataclass
class DistractorNoiseModel:
    """Model for distractor noise effects.
    
    Distractor presets:
    - "none": No distractors
    - "low": Few simple distractors
    - "medium": Moderate number of distractors
    - "high": Many complex distractors
    """
    intensity: str = "none"  # String preset for distractor intensity
    types: List[str] = field(default_factory=lambda: [])  # Types of distractors to include
    count_range: Optional[Tuple[int, int]] = None  # Range of distractors to generate
    position_range: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None  # Range for distractor positions

    def to_dict(self) -> Dict[str, Any]:
        """Convert the model to a dictionary format."""
        result = {
            "distractors": self.intensity,
            "distractor_types": self.types
        }
        if self.count_range:
            result["distractor_count_range"] = self.count_range
        if self.position_range:
            result["distractor_position_range"] = self.position_range
        return result
        
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'DistractorNoiseModel':
        """Create a DistractorNoiseModel from a dictionary."""
        if not isinstance(config, dict):
            config = {}
            
        default_instance = cls()
        return cls(
            intensity=config.get("distractors", default_instance.intensity),
            types=config.get("distractor_types", default_instance.types),
            count_range=config.get("distractor_count_range", default_instance.count_range),
            position_range=config.get("distractor_position_range", default_instance.position_range)
        )


@dataclass
class NoiseConfigModel:
    """Configuration model for all noise effects."""
    blur: Optional[BlurNoiseModel] = None
    light: Optional[LightNoiseModel] = None
    table_texture: Optional[TableTextureNoiseModel] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert the noise config to a dictionary format."""
        result = {}
        if self.blur:
            result.update(self.blur.to_dict())
        if self.light:
            result.update(self.light.to_dict())
        if self.table_texture:
            result.update(self.table_texture.to_dict())
        return result

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'NoiseConfigModel':
        """Create a NoiseConfigModel from a dictionary."""
        if not isinstance(config, dict):
            config = {}
            
        return cls(
            blur=BlurNoiseModel.from_dict(config) if "blur" in config else None,
            light=LightNoiseModel.from_dict(config) if "lighting" in config else None,
            table_texture=TableTextureNoiseModel.from_dict(config) if "table_texture" in config else None
        )
