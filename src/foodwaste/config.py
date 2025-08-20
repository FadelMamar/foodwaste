"""
Configuration file for DINOv2 semantic segmentation training
"""

from pathlib import Path
from typing import Tuple
from pydantic import BaseModel, Field

ROOT = Path(__file__).parents[2]

class DataAugmentationConfig(BaseModel):
    """Data augmentation configuration parameters"""
    horizontal_flip_prob: float = Field(default=0.5, description="Probability of horizontal flip")
    rotation_limit: int = Field(default=45, description="Maximum rotation angle in degrees")
    brightness_contrast_prob: float = Field(default=0.3, description="Probability of brightness/contrast adjustment")
    
class ModelConfig(BaseModel):
    """Model configuration parameters"""
    model_name: str = Field(default="facebook/dinov2-with-registers-small", description="DINOv2 model to use")
    freeze_backbone: bool = Field(default=True, description="Whether to freeze the backbone")
    use_cls_token: bool = Field(default=False, description="Whether to use the cls token")
    
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
    image_size: int = Field(default=1024, description="Input image size")
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
    mlflow_url: str = Field(default="http://localhost:5000", description="MLflow server URL")
    mlflow_experiment_name: str = Field(default="foodwaste-segmentation", description="MLflow experiment name")
    mlflow_run_name: str = Field(default="foodwaste-segmentation", description="MLflow run name")

class DataConfig(BaseModel):
    """Data configuration parameters"""
    dataset_name: str = Field(default="EduardoPacheco/FoodSeg103", description="HuggingFace dataset name")
    cache_dir: str = Field(default="./data_cache", description="Directory to cache dataset")
    
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
