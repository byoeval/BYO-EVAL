from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TableShape(Enum):
    """Enum for table shapes."""
    RECTANGULAR = "rectangular"
    CIRCULAR = "circular"
    ELLIPTIC = "elliptic"


class TableTexture(Enum):
    """Enum for table textures."""
    WOOD = "wood"
    MARBLE = "marble"
    METAL = "metal"
    PLAIN = "plain"


@dataclass
class MaterialModel:
    """Base model for material properties."""
    color: str | tuple[float, float, float, float] = (0.8, 0.8, 0.8, 1.0)
    roughness: float = 0.5
    material_name: str | None = None
    custom_material: Any | None = None

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> 'MaterialModel':
        """Create a MaterialModel from a dictionary."""
        if not isinstance(config, dict):
            config = {}

        default_instance = cls()
        return cls(
            color=config.get("color", default_instance.color),
            roughness=config.get("roughness", default_instance.roughness),
            material_name=config.get("material_name", default_instance.material_name),
            custom_material=config.get("custom_material", default_instance.custom_material)
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert the model to a dictionary format."""
        return {
            "color": self.color,
            "roughness": self.roughness,
            "material_name": self.material_name,
            "custom_material": self.custom_material
        }


@dataclass
class CameraModel:
    """Model for camera configuration."""
    # Camera distance and angle presets
    DISTANCE_PRESETS = {
        "low": 1.0,
        "medium": 3.0,
        "high": 5.0
    }

    ANGLE_PRESETS = {
        "none": 90.0,  # Looking straight down
        "low": 80.0,
        "medium": 60.0,
        "high": 30.0  # More horizontal view
    }

    distance: str | float = "medium"  # Can be "low", "medium", "high" or a float value
    angle: str | float = "medium"  # Can be "low", "medium", "high" or a float value
    horizontal_angle: float = 0.0
    randomize_distance: bool = False
    randomize_distance_percentage: float = 0.1
    randomize_angle: bool = False
    randomize_angle_percentage: float = 0.1

    def to_dict(self) -> dict[str, Any]:
        """Convert the model to a dictionary format."""
        return {
            "distance": self.distance,
            "angle": self.angle,
            "horizontal_angle": self.horizontal_angle,
            "randomize_distance": self.randomize_distance,
            "randomize_distance_percentage": self.randomize_distance_percentage,
            "randomize_angle": self.randomize_angle,
            "randomize_angle_percentage": self.randomize_angle_percentage
        }

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> 'CameraModel':
        """Create a CameraModel from a dictionary."""
        if not isinstance(config, dict):
            config = {}

        default_instance = cls()
        return cls(
            distance=config.get("distance", default_instance.distance),
            angle=config.get("angle", default_instance.angle),
            horizontal_angle=config.get("horizontal_angle", default_instance.horizontal_angle),
            randomize_distance=config.get("randomize_distance", default_instance.randomize_distance),
            randomize_distance_percentage=config.get("randomize_distance_percentage", default_instance.randomize_distance_percentage),
            randomize_angle=config.get("randomize_angle", default_instance.randomize_angle),
            randomize_angle_percentage=config.get("randomize_angle_percentage", default_instance.randomize_angle_percentage)
        )


@dataclass
class TableModel:
    """Model for table configuration."""
    shape: TableShape = TableShape.RECTANGULAR
    length: float = 2.0
    width: float = 1.0
    height: float = 0.9
    texture: TableTexture = TableTexture.WOOD
    material: MaterialModel = field(default_factory=MaterialModel)

    def to_dict(self) -> dict[str, Any]:
        """Convert the model to a dictionary format."""
        # Only include material if it's a custom material
        material = None
        if self.material.custom_material is not None:
            material = self.material.custom_material

        return {
            "shape": self.shape.value,
            "length": self.length,
            "width": self.width,
            "height": self.height,
            "texture": self.texture.value,
            "material": material
        }

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> 'TableModel':
        """Create a TableModel from a dictionary."""
        if not isinstance(config, dict):
            config = {}

        default_instance = cls()

        # Handle material separately
        material_config = config.get("material", {})
        if not isinstance(material_config, dict):
            material_config = {}

        return cls(
            shape=TableShape(config.get("shape", default_instance.shape.value)),
            length=config.get("length", default_instance.length),
            width=config.get("width", default_instance.width),
            height=config.get("height", default_instance.height),
            texture=TableTexture(config.get("texture", default_instance.texture.value)),
            material=MaterialModel.from_dict(material_config)
        )


@dataclass
class FloorModel:
    """Model for floor configuration."""
    color: tuple[float, float, float, float] = (0.8, 0.8, 0.8, 1.0)
    roughness: float = 0.5
    material: MaterialModel = field(default_factory=MaterialModel)

    def to_dict(self) -> dict[str, Any]:
        """Convert the model to a dictionary format."""
        return {
            "color": self.color,
            "roughness": self.roughness,
            "material": self.material.to_dict()
        }

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> 'FloorModel':
        """Create a FloorModel from a dictionary."""
        if not isinstance(config, dict):
            config = {}

        default_instance = cls()

        # Handle material separately
        material_config = config.get("material", {})
        if not isinstance(material_config, dict):
            material_config = {}

        return cls(
            color=config.get("color", default_instance.color),
            roughness=config.get("roughness", default_instance.roughness),
            material=MaterialModel.from_dict(material_config)
        )


@dataclass
class BackgroundModel:
    """Model for background configuration."""
    color: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0)
    use_hdri: bool = False
    hdri_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert the model to a dictionary format."""
        return {
            "color": self.color,
            "use_hdri": self.use_hdri,
            "hdri_path": self.hdri_path
        }

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> 'BackgroundModel':
        """Create a BackgroundModel from a dictionary."""
        if not isinstance(config, dict):
            config = {}

        default_instance = cls()
        return cls(
            color=config.get("color", default_instance.color),
            use_hdri=config.get("use_hdri", default_instance.use_hdri),
            hdri_path=config.get("hdri_path", default_instance.hdri_path)
        )


@dataclass
class LightingModel:
    """Model for lighting configuration."""
    # Lighting intensity presets
    LIGHTING_PRESETS = {
        "very_low": 0.3,  # 30% of standard lighting
        "low": 0.6,       # 60% of standard lighting
        "medium": 1.0,    # Standard lighting (100%)
        "high": 1.5,      # 150% of standard lighting
        "very_high": 2.0  # 200% of standard lighting
    }

    lighting: str | float = "medium"  # Can be "very_low", "low", "medium", "high", "very_high" or a float multiplier
    key_light_power: float = 300.0
    fill_light_power: float = 50.0
    back_light_power: float = 50.0

    def to_dict(self) -> dict[str, Any]:
        """Convert the model to a dictionary format."""
        return {
            "lighting": self.lighting,
            "key_light_power": self.key_light_power,
            "fill_light_power": self.fill_light_power,
            "back_light_power": self.back_light_power
        }

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> 'LightingModel':
        """Create a LightingModel from a dictionary."""
        if not isinstance(config, dict):
            config = {}

        default_instance = cls()
        return cls(
            lighting=config.get("lighting", default_instance.lighting),
            key_light_power=config.get("key_light_power", default_instance.key_light_power),
            fill_light_power=config.get("fill_light_power", default_instance.fill_light_power),
            back_light_power=config.get("back_light_power", default_instance.back_light_power)
        )


@dataclass
class ResolutionModel:
    """Model for resolution configuration."""
    # Resolution presets
    RESOLUTION_PRESETS = {
        "low": (640, 480),
        "medium": (1280, 720),
        "high": (1920, 1080)
    }

    width: int = 1920
    height: int = 1080
    resolution_percentage: int = 100
    pixel_aspect_x: float = 1.0
    pixel_aspect_y: float = 1.0
    randomize: bool = False
    randomize_percentage: float = 0.1

    def to_dict(self) -> dict[str, Any]:
        """Convert the model to a dictionary format."""
        return {
            "width": self.width,
            "height": self.height,
            "resolution_percentage": self.resolution_percentage,
            "pixel_aspect_x": self.pixel_aspect_x,
            "pixel_aspect_y": self.pixel_aspect_y,
            "randomize": self.randomize,
            "randomize_percentage": self.randomize_percentage
        }

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> 'ResolutionModel':
        """Create a ResolutionModel from a dictionary."""
        if not isinstance(config, dict):
            config = {}

        default_instance = cls()
        return cls(
            width=config.get("width", default_instance.width),
            height=config.get("height", default_instance.height),
            resolution_percentage=config.get("resolution_percentage", default_instance.resolution_percentage),
            pixel_aspect_x=config.get("pixel_aspect_x", default_instance.pixel_aspect_x),
            pixel_aspect_y=config.get("pixel_aspect_y", default_instance.pixel_aspect_y),
            randomize=config.get("randomize", default_instance.randomize),
            randomize_percentage=config.get("randomize_percentage", default_instance.randomize_percentage)
        )


@dataclass
class RenderModel:
    """Model for render configuration."""
    engine: str = "CYCLES"
    samples: int = 128
    exposure: float = 0.0
    file_format: str = "PNG"
    resolution: ResolutionModel = field(default_factory=ResolutionModel)
    output_path: str | None = None
    gpu_enabled: bool = True
    gpus: list[int] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert the model to a dictionary format."""
        base_dict = {
            "engine": self.engine,
            "samples": self.samples,
            "exposure": self.exposure,
            "file_format": self.file_format,
            "resolution": self.resolution.to_dict(),
            "gpu_enabled": self.gpu_enabled,
            "gpus": self.gpus
        }
        if self.output_path:
            base_dict["output_path"] = self.output_path
        return base_dict

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> 'RenderModel':
        """Create a RenderModel from a dictionary."""
        if not isinstance(config, dict):
            config = {}

        default_instance = cls()

        # Handle resolution separately
        resolution_config = {}
        if "resolution_x" in config:
            resolution_config["width"] = config.pop("resolution_x")
        if "resolution_y" in config:
            resolution_config["height"] = config.pop("resolution_y")
        if "width" in config:
            resolution_config["width"] = config.pop("width")
        if "height" in config:
            resolution_config["height"] = config.pop("height")

        return cls(
            engine=config.get("engine", default_instance.engine),
            samples=config.get("samples", default_instance.samples),
            exposure=config.get("exposure", default_instance.exposure),
            file_format=config.get("file_format", default_instance.file_format),
            resolution=ResolutionModel.from_dict(resolution_config),
            output_path=config.get("output_path", default_instance.output_path),
            gpu_enabled=config.get("gpu_enabled", default_instance.gpu_enabled),
            gpus=config.get("gpus", default_instance.gpus)
        )


@dataclass
class GridModel:
    """Model for grid configuration."""
    granularity: int = 10  # Renamed from n_divisions
    line_thickness: int = 1
    line_color_rgba: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.8)  # Default to black

    def to_dict(self) -> dict[str, Any]:
        """Convert the model to a dictionary format."""
        return {
            "granularity": self.granularity,
            "line_thickness": self.line_thickness,
            "line_color_rgba": self.line_color_rgba,
        }

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> 'GridModel':
        """Create a GridModel from a dictionary."""
        if not isinstance(config, dict):
            config = {}

        default_instance = cls()
        return cls(
            granularity=config.get("granularity", default_instance.granularity),
            line_thickness=config.get("line_thickness", default_instance.line_thickness),
            line_color_rgba=config.get("line_color_rgba", default_instance.line_color_rgba),
        )


@dataclass
class SceneSetupModel:
    """Main model combining all scene setup components."""
    camera: CameraModel = field(default_factory=CameraModel)
    table: TableModel = field(default_factory=TableModel)
    floor: FloorModel = field(default_factory=FloorModel)
    background: BackgroundModel = field(default_factory=BackgroundModel)
    lighting: LightingModel = field(default_factory=LightingModel)
    resolution: ResolutionModel = field(default_factory=ResolutionModel)
    render: RenderModel = field(default_factory=RenderModel)
    grid: GridModel = field(default_factory=GridModel)

    def to_dict(self) -> dict[str, Any]:
        """Convert the model to a dictionary format."""
        return {
            "camera": self.camera.to_dict(),
            "table": self.table.to_dict(),
            "floor": self.floor.to_dict(),
            "background": self.background.to_dict(),
            "lighting": self.lighting.to_dict(),
            "resolution": self.resolution.to_dict(),
            "render": self.render.to_dict(),
            "grid": self.grid.to_dict()
        }

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> 'SceneSetupModel':
        """Create a SceneSetupModel from a dictionary."""
        if not isinstance(config, dict):
            config = {}

        return cls(
            camera=CameraModel.from_dict(config.get("camera", {})),
            table=TableModel.from_dict(config.get("table", {})),
            floor=FloorModel.from_dict(config.get("floor", {})),
            background=BackgroundModel.from_dict(config.get("background", {})),
            lighting=LightingModel.from_dict(config.get("lighting", {})),
            resolution=ResolutionModel.from_dict(config.get("resolution", {})),
            render=RenderModel.from_dict(config.get("render", {})),
            grid=GridModel.from_dict(config.get("grid", {}))
        )
