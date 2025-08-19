# 🍽️ FoodWaste - DINOv2 Semantic Segmentation Framework

A powerful and efficient framework for training semantic segmentation models on food waste datasets using Meta's DINOv2 (Data-Efficient Image Transformer v2) architecture. Built with PyTorch Lightning for scalable training and MLflow for experiment tracking.

## 🚀 Features

- **DINOv2 Integration**: Leverage state-of-the-art vision transformer models for semantic segmentation
- **FoodSeg103 Dataset**: Pre-configured for the comprehensive FoodSeg103 dataset (103 food categories)
- **PyTorch Lightning**: Scalable training with automatic mixed precision, distributed training, and callbacks
- **Frozen Backbone Training**: Efficient training by freezing DINOv2 features and training only the classifier
- **Comprehensive Metrics**: IoU (Intersection over Union) and Dice score evaluation
- **MLflow Integration**: Experiment tracking and model versioning
- **CLI Interface**: Easy-to-use command-line tools for training and validation
- **Flexible Configuration**: YAML-based configuration with Pydantic validation
- **Data Augmentation**: Built-in augmentation pipeline with configurable parameters
- **Jupyter Notebooks**: Interactive examples and tutorials

## 🏗️ Architecture

The framework uses a **two-stage approach**:

1. **DINOv2 Backbone**: Pre-trained vision transformer that extracts rich visual features
2. **Linear Classifier**: Simple convolutional classifier that operates on the frozen features

```
Input Image → DINOv2 Backbone (frozen) → Patch Embeddings → Linear Classifier → Segmentation Mask
```

### Model Variants
- `facebook/dinov2-small` (21M parameters)
- `facebook/dinov2-base` (86M parameters) 
- `facebook/dinov2-large` (300M parameters)
- `facebook/dinov2-giant` (1.1B parameters)

## 📦 Installation

### Prerequisites
- Python 3.11+
- PyTorch 2.6.0+
- CUDA-compatible GPU (recommended)

### Install with UV (Recommended)
```bash
# Clone the repository
git clone https://github.com/your-username/foodwaste.git
cd foodwaste

# Install dependencies
uv sync
```

### Install with pip
```bash
pip install -e .
```

## 🚀 Quick Start

### 1. Basic Training
```bash
# Train with default configuration
foodwaste train

# Train with custom config
foodwaste train --config configs/train.yaml

# Override specific parameters
foodwaste train --config configs/train.yaml --override "training.learning_rate=1e-5,training.batch_size=16"
```

### 2. Python API
```python
from foodwaste.config import MainConfig
from foodwaste.train import runner

# Load configuration
config = MainConfig.from_file("configs/train.yaml")

# Start training
runner(config)
```

### 3. Jupyter Notebooks
Explore the `notebooks/` directory for interactive examples:
- `FoodWaste.ipynb` - Main tutorial notebook
- `semantic_segmentation.ipynb` - DINOv2 segmentation examples
- `foreground_segmentation.ipynb` - Foreground detection
- `pca.ipynb` - Feature analysis

## ⚙️ Configuration

The framework uses a hierarchical configuration system defined in `configs/train.yaml`:

### Model Configuration
```yaml
model:
  model_name: "facebook/dinov2-base"
  freeze_backbone: true
  use_cls_token: false
  token_height: 14
  token_width: 14
```

### Training Configuration
```yaml
training:
  learning_rate: 1e-4
  num_epochs: 10
  batch_size: 8
  device: "auto"
  accelerator: "auto"
  precision: "bf16-mixed"
  image_size: 518
```

### Data Configuration
```yaml
data:
  dataset_name: "EduardoPacheco/FoodSeg103"
  cache_dir: "./data_cache"
  mean: [123.675, 116.280, 103.530]
  std: [58.395, 57.120, 57.375]
```

## 📊 Dataset

The framework is pre-configured for the **FoodSeg103** dataset, which includes:

- **103 food categories** + background
- **High-quality segmentation masks**
- **Diverse food types**: fruits, vegetables, meats, desserts, beverages
- **Professional food photography**

### Supported Categories
- **Fruits**: apple, banana, strawberry, orange, watermelon, etc.
- **Vegetables**: tomato, carrot, broccoli, lettuce, etc.
- **Proteins**: steak, chicken, fish, eggs, etc.
- **Grains**: bread, rice, pasta, pizza, etc.
- **Desserts**: cake, ice cream, chocolate, etc.
- **Beverages**: coffee, tea, wine, juice, etc.

## 🎯 Training

### Training Pipeline
1. **Data Loading**: Automatic download and caching of FoodSeg103 dataset
2. **Preprocessing**: Image resizing, normalization, and augmentation
3. **Feature Extraction**: DINOv2 backbone processes images (frozen)
4. **Classification**: Linear classifier predicts segmentation masks
5. **Evaluation**: IoU and Dice score computation
6. **Logging**: MLflow integration for experiment tracking

### Key Training Features
- **Automatic Mixed Precision**: BF16 mixed precision training
- **Learning Rate Scheduling**: Cosine annealing with warmup
- **Early Stopping**: Configurable patience and monitoring
- **Model Checkpointing**: Save best models based on validation metrics
- **Gradient Clipping**: Prevent gradient explosion

### Training Commands
```bash
# Basic training
foodwaste train

# Validate configuration
foodwaste validate --config configs/train.yaml

# Show project info
foodwaste info
```

## 🔧 CLI Commands

| Command | Description |
|---------|-------------|
| `foodwaste train` | Start training with default or specified config |
| `foodwaste validate` | Validate configuration file |
| `foodwaste info` | Display project information |

### CLI Options
- `--config, -c`: Path to configuration YAML file
- `--override, -o`: Override configuration values (key=value format)

## 📈 Monitoring & Logging

### MLflow Integration
```bash
# Start MLflow server
./launch_mlflow.bat

# Access at http://localhost:5000
```

### Logged Metrics
- **Training Loss**: Cross-entropy loss
- **Validation Loss**: Cross-entropy loss  
- **IoU Score**: Intersection over Union (per class and mean)
- **Dice Score**: Dice coefficient (weighted average)
- **Learning Rate**: Current learning rate value
- **Epoch Progress**: Training and validation progress

## 🏃‍♂️ Examples

### Custom Training Configuration
```python
from foodwaste.config import MainConfig, TrainingConfig, ModelConfig

# Create custom configuration
config = MainConfig(
    model=ModelConfig(
        model_name="facebook/dinov2-large",
        freeze_backbone=True
    ),
    training=TrainingConfig(
        learning_rate=5e-5,
        num_epochs=20,
        batch_size=16,
        precision="16-mixed"
    )
)

# Start training
from foodwaste.train import runner
runner(config)
```

### Data Augmentation
```python
from foodwaste.transforms import get_train_transforms

# Custom augmentation
transforms = get_train_transforms(
    image_size=1024,
    horizontal_flip_prob=0.7,
    rotation_limit=30,
    brightness_contrast_prob=0.5
)
```

## 🧪 Development

### Project Structure
```
foodwaste/
├── src/foodwaste/          # Core package
│   ├── __init__.py         # Package initialization
│   ├── cli.py             # Command-line interface
│   ├── config.py          # Configuration management
│   ├── dataset.py         # Dataset classes and data loading
│   ├── models.py          # DINOv2 models and classifiers
│   ├── train.py           # Training logic and Lightning module
│   ├── transforms.py      # Data augmentation pipeline
│   └── utils.py           # Utility functions
├── configs/                # Configuration files
├── examples/               # Example scripts
├── notebooks/              # Jupyter notebooks
├── checkpoints/            # Model checkpoints
├── logs/                   # Training logs
└── runs/                   # MLflow runs
```

### Key Dependencies
- **PyTorch 2.6.0**: Deep learning framework
- **Transformers 4.55.2**: Hugging Face model library
- **Lightning 2.5.3**: Training framework
- **MLflow 3.3.0**: Experiment tracking
- **Datasets 4.0.0**: Hugging Face datasets
- **Albumentations 2.0.8**: Image augmentation
- **TimM 1.0.19**: Vision model utilities

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Meta AI Research** for DINOv2 models
- **Hugging Face** for the Transformers library
- **PyTorch Lightning** team for the training framework
- **FoodSeg103** dataset contributors

## 📚 References

- [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)
- [FoodSeg103: A Dataset for Food Image Segmentation](https://github.com/LARC-CMU-SRI/FoodSeg103)
- [PyTorch Lightning Documentation](https://lightning.ai/docs/pytorch/stable/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/your-username/foodwaste/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/foodwaste/discussions)
- **Documentation**: [Project Wiki](https://github.com/your-username/foodwaste/wiki)

---

**Made with ❤️ for the food waste research community**