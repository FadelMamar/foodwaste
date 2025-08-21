"""
Configuration file for DINOv2 semantic segmentation training
"""

from pathlib import Path
from typing import Tuple
from pydantic import BaseModel, Field

ROOT = Path(__file__).parents[2]

id2label = {
    0: "background",
    1: "candy",
    2: "egg tart",
    3: "french fries",
    4: "chocolate",
    5: "biscuit",
    6: "popcorn",
    7: "pudding",
    8: "ice cream",
    9: "cheese butter",
    10: "cake",
    11: "wine",
    12: "milkshake",
    13: "coffee",
    14: "juice",
    15: "milk",
    16: "tea",
    17: "almond",
    18: "red beans",
    19: "cashew",
    20: "dried cranberries",
    21: "soy",
    22: "walnut",
    23: "peanut",
    24: "egg",
    25: "apple",
    26: "date",
    27: "apricot",
    28: "avocado",
    29: "banana",
    30: "strawberry",
    31: "cherry",
    32: "blueberry",
    33: "raspberry",
    34: "mango",
    35: "olives",
    36: "peach",
    37: "lemon",
    38: "pear",
    39: "fig",
    40: "pineapple",
    41: "grape",
    42: "kiwi",
    43: "melon",
    44: "orange",
    45: "watermelon",
    46: "steak",
    47: "pork",
    48: "chicken duck",
    49: "sausage",
    50: "fried meat",
    51: "lamb",
    52: "sauce",
    53: "crab",
    54: "fish",
    55: "shellfish",
    56: "shrimp",
    57: "soup",
    58: "bread",
    59: "corn",
    60: "hamburg",
    61: "pizza",
    62: "hanamaki baozi",
    63: "wonton dumplings",
    64: "pasta",
    65: "noodles",
    66: "rice",
    67: "pie",
    68: "tofu",
    69: "eggplant",
    70: "potato",
    71: "garlic",
    72: "cauliflower",
    73: "tomato",
    74: "kelp",
    75: "seaweed",
    76: "spring onion",
    77: "rape",
    78: "ginger",
    79: "okra",
    80: "lettuce",
    81: "pumpkin",
    82: "cucumber",
    83: "white radish",
    84: "carrot",
    85: "asparagus",
    86: "bamboo shoots",
    87: "broccoli",
    88: "celery stick",
    89: "cilantro mint",
    90: "snow peas",
    91: "cabbage",
    92: "bean sprouts",
    93: "onion",
    94: "pepper",
    95: "green beans",
    96: "French beans",
    97: "king oyster mushroom",
    98: "shiitake",
    99: "enoki mushroom",
    100: "oyster mushroom",
    101: "white button mushroom",
    102: "salad",
    103: "other ingredients"
}

label2id = {v: k for k, v in id2label.items()}


class DataAugmentationConfig(BaseModel):
    """Data augmentation configuration parameters"""
    horizontal_flip_prob: float = Field(default=0.5, description="Probability of horizontal flip")
    rotation_limit: int = Field(default=45, description="Maximum rotation angle in degrees")
    brightness_contrast_prob: float = Field(default=0.3, description="Probability of brightness/contrast adjustment")
    
class ModelConfig(BaseModel):
    """Model configuration parameters"""
    model_name: str = Field(default="timm/vit_small_patch14_dinov2.lvd142m", description="DINOv2 model to use")
    freeze_backbone: bool = Field(default=True, description="Whether to freeze the backbone")
    use_cls_token: bool = Field(default=False, description="Whether to use the cls token")
    use_patch_classifier: bool = Field(default=False, description="Whether to use the patch classifier")
    dropout: float = Field(default=0.2, description="Dropout probability")
    patch_size: int = Field(default=14, description="Patch size")
    model_layer: int = Field(default=9, description="Layer to use for the patch classifier")
    
class TrainingConfig(BaseModel):
    """Training configuration parameters"""
    # Core training parameters
    learning_rate: float = Field(default=5e-5, description="Learning rate for training")
    num_epochs: int = Field(default=10, description="Number of training epochs")
    batch_size: int = Field(default=8, description="Training batch size")
    num_workers: int = Field(default=4, description="Number of data loading workers")
    device: str = Field(default="cpu", description="Device to use for training")
    lrf: float = Field(default=0.1, description="Learning rate factor for scheduling")
    accelerator: str = Field(default="auto", description="Accelerator to use for training")
    precision: str = Field(default="bf16-mixed", description="Precision to use for training")
    
    # Data parameters
    image_size: int = Field(default=518, description="Input image size")
    train_split: str = Field(default="train", description="Training split name")
    val_split: str = Field(default="validation", description="Validation split name")
    
    # Optimization parameters
    weight_decay: float = Field(default=0.0001, description="Weight decay for optimizer")
    warmup_steps: int = Field(default=500, description="Number of warmup steps")

    # monitor
    monitor: str = Field(default="val_loss", description="Metric to monitor")
    mode: str = Field(default="min", description="Mode to use for monitoring")
    patience: int = Field(default=10, description="Number of epochs to wait before early stopping")
    
    # Reproducibility
    seed: int = Field(default=42, description="Random seed for reproducibility")
    
class LoggingConfig(BaseModel):
    """Logging and saving configuration parameters"""
    save_dir: str = Field(default="checkpoints", description="Directory to save checkpoints")
    eval_interval: int = Field(default=1, description="Evaluation interval in epoch")
    logger_url: str = Field(default="http://localhost:5000", description="Logger server URL")
    experiment_name: str = Field(default="foodwaste-segmentation", description="experiment name")
    run_name: str = Field(default="foodwaste-segmentation", description="run name")
    logger:str = Field(default="mlflow", description="Logger to use")

class DataConfig(BaseModel):
    """Data configuration parameters"""
    dataset_name: str = Field(default="EduardoPacheco/FoodSeg103", description="HuggingFace dataset name")
    cache_dir: str = Field(default=str(ROOT / "data_cache"), description="Directory to cache dataset")
    
    # Normalization parameters (ADE20K standard)
    mean: Tuple[float, float, float] = Field(
        default=(123.675, 116.280, 103.530), 
        description="RGB mean values for normalization"
    )
    std: Tuple[float, float, float] = Field(
        default=(58.395, 57.120, 57.375), 
        description="RGB standard deviation values for normalization"
    )

class MainConfig(BaseModel):
    """Main configuration that combines all sub-configurations"""
    model: ModelConfig = Field(default_factory=ModelConfig, description="Model configuration")
    training: TrainingConfig = Field(default_factory=TrainingConfig, description="Training configuration")
    data: DataConfig = Field(default_factory=DataConfig, description="Data configuration")
    logging: LoggingConfig = Field(default_factory=LoggingConfig, description="Logging configuration")
    augmentation: DataAugmentationConfig = Field(default_factory=DataAugmentationConfig, description="Data augmentation configuration")
    
    def get_training_config(self) -> TrainingConfig:
        """Get training configuration"""
        return self.training
    
    def get_data_config(self) -> DataConfig:
        """Get data configuration"""
        return self.data
    
    def get_model_config(self) -> ModelConfig:
        """Get model configuration"""
        return self.model
    
    def get_logging_config(self) -> LoggingConfig:
        """Get logging configuration"""
        return self.logging
    
    def get_augmentation_config(self) -> DataAugmentationConfig:
        """Get augmentation configuration"""
        return self.augmentation
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary"""
        return self.model_dump()
    
    def save_to_file(self, filepath: str) -> None:
        """Save configuration to file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.model_dump_json(indent=2))
    
    @classmethod
    def from_file(cls, filepath: str) -> 'MainConfig':
        """Load configuration from file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return cls.model_validate_json(f.read())

# Global configuration instances
main_config = MainConfig()

# Backward compatibility - keep the old instances for existing code
training_config = main_config.training
data_config = main_config.data

# Convenience access functions
def get_training_config() -> TrainingConfig:
    """Get training configuration"""
    return main_config.training

def get_data_config() -> DataConfig:
    """Get data configuration"""
    return main_config.data

def get_model_config() -> ModelConfig:
    """Get model configuration"""
    return main_config.model

def get_logging_config() -> LoggingConfig:
    """Get logging configuration"""
    return main_config.logging

def get_augmentation_config() -> DataAugmentationConfig:
    """Get augmentation configuration"""
    return main_config.augmentation
