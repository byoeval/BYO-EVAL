from dataclasses import dataclass, field
from typing import Tuple, Union, Optional, Dict, Any, List, TypeVar


T = TypeVar('T')

@dataclass
class MaterialModel:
    """Configuration for material properties."""
    color: Union[str, Tuple[float, float, float, float]] = "white"
    roughness: float = 0.2
    material_name: Optional[str] = None
    custom_material: Optional[Any] = None  # For Blender material object

    def to_dict(self) -> Dict[str, Any]:
        """Convert the config to a dictionary."""
        return {
            'color': self.color,
            'roughness': self.roughness,
            'material_name': self.material_name,
            # Skip custom_material as it's not serializable
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'MaterialModel':
        """Create a MaterialModel from a dictionary."""
        if not isinstance(config, dict):
            config = {}
            
        default_instance = cls()
        return cls(
            color=config.get('color', default_instance.color),
            roughness=config.get('roughness', default_instance.roughness),
            material_name=config.get('material_name', default_instance.material_name),
            custom_material=config.get('custom_material', default_instance.custom_material)
        )

@dataclass
class GeometryModel:
    """Configuration for piece geometry."""
    scale: float = 0.1
    random_rotation: bool = False
    max_rotation_angle: float = 15.0
    seed: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert the config to a dictionary."""
        return {
            'scale': self.scale,
            'random_rotation': self.random_rotation,
            'max_rotation_angle': self.max_rotation_angle,
            'seed': self.seed
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'GeometryModel':
        """Create a GeometryModel from a dictionary."""
        if not isinstance(config, dict):
            config = {}
            
        default_instance = cls()
        return cls(
            scale=config.get('scale', default_instance.scale),
            random_rotation=config.get('random_rotation', default_instance.random_rotation),
            max_rotation_angle=config.get('max_rotation_angle', default_instance.max_rotation_angle),
            seed=config.get('seed', default_instance.seed)
        )

@dataclass
class PieceModel:
    """Configuration for chess piece creation."""
    piece_type: str
    location: Tuple[float, float, float] = (0, 0, 0)
    material: MaterialModel = field(default_factory=MaterialModel)
    geometry: GeometryModel = field(default_factory=GeometryModel)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the config to a dictionary."""
        return {
            'type': self.piece_type,
            'location': self.location,
            'material': self.material.to_dict(),
            'geometry': self.geometry.to_dict()
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'PieceModel':
        """Create a PieceModel from a dictionary."""
        if not isinstance(config, dict):
            config = {}
            
        default_instance = cls(piece_type=config.get('type', 'pawn'))  # piece_type is required but provide a default for complete default instance
        
        # Handle nested models
        material_config = config.get('material', {}) if isinstance(config.get('material'), dict) else {}
        geometry_config = config.get('geometry', {}) if isinstance(config.get('geometry'), dict) else {}
        
        return cls(
            piece_type=config.get('type', default_instance.piece_type),
            location=config.get('location', default_instance.location),
            material=MaterialModel.from_dict(material_config),
            geometry=GeometryModel.from_dict(geometry_config)
        )

@dataclass
class BoardModel:
    """Configuration for chess board."""
    length: float = 0.7
    width: float = 0.7
    thickness: float = 0.05
    location: Tuple[float, float, float] = (0, 0, 0.9)
    border_width: float = 0.2
    rows: int = 8
    columns: int = 8
    random_pattern: bool = False
    pattern_seed: Optional[int] = None
    board_material: Dict[str, Any] = field(default_factory=lambda: {
        "color": (0.4, 0.2, 0.05, 1.0),  # Brown
        "roughness": 0.3,
        "material_name": "ChessboardMaterial"
    })
    square_white_material: Dict[str, Any] = field(default_factory=lambda: {
        "color": (0.9, 0.9, 0.8, 1.0),  # Off-white
        "roughness": 0.2,
        "material_name": "WhiteSquareMaterial"
    })
    square_black_material: Dict[str, Any] = field(default_factory=lambda: {
        "color": (0.1, 0.1, 0.1, 1.0),  # Off-black
        "roughness": 0.2,
        "material_name": "BlackSquareMaterial"
    })

    def to_dict(self) -> Dict[str, Any]:
        """Convert the config to a dictionary."""
        return {
            'length': self.length,
            'width': self.width,
            'thickness': self.thickness,
            'location': self.location,
            'border_width': self.border_width,
            'rows': self.rows,
            'columns': self.columns,
            'random_pattern': self.random_pattern,
            'pattern_seed': self.pattern_seed,
            'board_material': self.board_material,
            'square_white_material': self.square_white_material,
            'square_black_material': self.square_black_material
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'BoardModel':
        """Create a BoardModel from a dictionary."""
        if not isinstance(config, dict):
            config = {}
            
        default_instance = cls()
        return cls(
            length=config.get('length', default_instance.length),
            width=config.get('width', default_instance.width),
            thickness=config.get('thickness', default_instance.thickness),
            location=config.get('location', default_instance.location),
            border_width=config.get('border_width', default_instance.border_width),
            rows=config.get('rows', default_instance.rows),
            columns=config.get('columns', default_instance.columns),
            random_pattern=config.get('random_pattern', default_instance.random_pattern),
            pattern_seed=config.get('pattern_seed', default_instance.pattern_seed),
            board_material=config.get('board_material', default_instance.board_material),
            square_white_material=config.get('square_white_material', default_instance.square_white_material),
            square_black_material=config.get('square_black_material', default_instance.square_black_material)
        )

    def get_material_model(self, material_type: str) -> MaterialModel:
        """
        Create a MaterialModel from the stored material configuration.
        
        Args:
            material_type: One of 'board', 'white', or 'black'
            
        Returns:
            MaterialModel instance
        """
        material_config = {
            'board': self.board_material,
            'white': self.square_white_material,
            'black': self.square_black_material
        }.get(material_type)
        
        if material_config is None:
            raise ValueError(f"Invalid material type: {material_type}")
            
        return MaterialModel.from_dict(material_config)

@dataclass
class RandomizationRange:
    """Configuration for a range of random values."""
    min_value: float
    max_value: float
    step: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert the config to a dictionary."""
        return {
            'min_value': self.min_value,
            'max_value': self.max_value,
            'step': self.step
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'RandomizationRange':
        """Create a RandomizationRange from a dictionary."""
        if not isinstance(config, dict):
            config = {}
            
        default_instance = cls(min_value=0.0, max_value=1.0)  # Provide default values for required fields
        return cls(
            min_value=config.get('min_value', default_instance.min_value),
            max_value=config.get('max_value', default_instance.max_value),
            step=config.get('step', default_instance.step)
        )

@dataclass
class PieceCountRange:
    """Configuration for piece count ranges."""
    min_count: int = 0
    max_count: int = 8
    total_min: Optional[int] = None
    total_max: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert the config to a dictionary."""
        return {
            'min_count': self.min_count,
            'max_count': self.max_count,
            'total_min': self.total_min,
            'total_max': self.total_max
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'PieceCountRange':
        """Create a PieceCountRange from a dictionary."""
        if not isinstance(config, dict):
            config = {}
            
        default_instance = cls()
        return cls(
            min_count=config.get('min_count', default_instance.min_count),
            max_count=config.get('max_count', default_instance.max_count),
            total_min=config.get('total_min', default_instance.total_min),
            total_max=config.get('total_max', default_instance.total_max)
        )

@dataclass
class BoardRandomizationModel:
    """Configuration for board randomization."""
    location_bounds: Tuple[RandomizationRange, RandomizationRange] = field(
        default_factory=lambda: (
            RandomizationRange(-0.4, 0.4),  # x
            RandomizationRange(-0.4, 0.4)   # y
        )
    )
    pattern_randomization: bool = False
    pattern_seed: Optional[int] = None
    rows: int = 8
    columns: int = 8

    def to_dict(self) -> Dict[str, Any]:
        """Convert the config to a dictionary."""
        return {
            'location_bounds': [bound.to_dict() for bound in self.location_bounds],
            'pattern_randomization': self.pattern_randomization,
            'pattern_seed': self.pattern_seed,
            'rows': self.rows,
            'columns': self.columns
        }

    @classmethod
    def from_dict(cls, config: Union[Dict[str, Any], 'BoardRandomizationModel']) -> 'BoardRandomizationModel':
        """
        Create a BoardRandomizationModel from a dictionary or another BoardRandomizationModel.
        
        Args:
            config: Either a dictionary or a BoardRandomizationModel object
            
        Returns:
            BoardRandomizationModel object
        """
        # If config is already a BoardRandomizationModel, return it
        if isinstance(config, BoardRandomizationModel):
            return config
            
        # Handle dictionary input
        location_bounds = config.get('location_bounds', [])
        if isinstance(location_bounds, list) and len(location_bounds) == 2:
            bounds = tuple(RandomizationRange.from_dict(bound) for bound in location_bounds)
        else:
            bounds = (RandomizationRange(-0.4, 0.4), RandomizationRange(-0.4, 0.4))

        return cls(
            location_bounds=bounds,
            pattern_randomization=config.get('pattern_randomization', False),
            pattern_seed=config.get('pattern_seed'),
            rows=config.get('rows', 8),
            columns=config.get('columns', 8)
        )

@dataclass
class PieceRandomizationModel:
    """Configuration for piece randomization."""
    piece_counts: Dict[str, PieceCountRange] = field(default_factory=lambda: {
        "pawn": PieceCountRange(0, 8),
        "rook": PieceCountRange(0, 2),
        "knight": PieceCountRange(0, 2),
        "bishop": PieceCountRange(0, 2),
        "queen": PieceCountRange(0, 1),
        "king": PieceCountRange(0, 1)
    })
    scale_range: RandomizationRange = field(
        default_factory=lambda: RandomizationRange(0.05, 0.12, 0.01)
    )
    allowed_colors: List[Union[str, Tuple[float, float, float, float]]] = field(
        default_factory=lambda: ["white", "black"]
    )
    intra_class_variation: bool = False
    extra_class_variation: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert the config to a dictionary."""
        return {
            'piece_counts': {k: v.to_dict() for k, v in self.piece_counts.items()},
            'scale_range': self.scale_range.to_dict(),
            'allowed_colors': self.allowed_colors,
            'intra_class_variation': self.intra_class_variation,
            'extra_class_variation': self.extra_class_variation
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'PieceRandomizationModel':
        """Create a PieceRandomizationModel from a dictionary."""
        if not isinstance(config, dict):
            config = {}
            
        default_instance = cls()
        
        piece_counts = {
            k: PieceCountRange.from_dict(v) if isinstance(v, dict) else v
            for k, v in config.get('piece_counts', default_instance.piece_counts).items()
        }
        
        scale_range = (RandomizationRange.from_dict(config['scale_range']) 
                      if isinstance(config.get('scale_range'), dict) 
                      else default_instance.scale_range)

        return cls(
            piece_counts=piece_counts,
            scale_range=scale_range,
            allowed_colors=config.get('allowed_colors', default_instance.allowed_colors),
            intra_class_variation=config.get('intra_class_variation', default_instance.intra_class_variation),
            extra_class_variation=config.get('extra_class_variation', default_instance.extra_class_variation)
        )

@dataclass
class DifficultyPreset:
    """Preset configurations for different difficulty levels."""
    name: str
    board_config: BoardRandomizationModel
    piece_config: PieceRandomizationModel

    def to_dict(self) -> Dict[str, Any]:
        """Convert the preset to a dictionary."""
        return {
            'name': self.name,
            'board_config': self.board_config.to_dict(),
            'piece_config': self.piece_config.to_dict()
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'DifficultyPreset':
        """Create a DifficultyPreset from a dictionary."""
        return cls(
            name=config['name'],
            board_config=BoardRandomizationModel.from_dict(config.get('board_config', {})),
            piece_config=PieceRandomizationModel.from_dict(config.get('piece_config', {}))
        )

@dataclass
class PieceCountConfig:
    """Configuration for piece count ranges."""
    pawn: int
    rook: int
    knight: int
    bishop: int
    queen: int
    king: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert the config to a dictionary."""
        return {
            'pawn': self.pawn,
            'rook': self.rook,
            'knight': self.knight,
            'bishop': self.bishop,
            'queen': self.queen,
            'king': self.king
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'PieceCountConfig':
        """Create a PieceCountConfig from a dictionary."""
        if not isinstance(config, dict):
            config = {}
            
        default_instance = cls()
        return cls(
            pawn=config.get('pawn', default_instance.pawn),
            rook=config.get('rook', default_instance.rook),
            knight=config.get('knight', default_instance.knight),
            bishop=config.get('bishop', default_instance.bishop),
            queen=config.get('queen', default_instance.queen),
            king=config.get('king', default_instance.king)
        ) 