#!/usr/bin/env python3
"""
Test script for PyTorch Lightning implementation
"""

import sys
import os
from pathlib import Path
import pytest
import torch

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from foodwaste.lightning_config import LightningTrainingConfig, LightningDataConfig
from foodwaste.train_dinov2_segmentation_lightning import (
    FoodWasteSegmentationModule, 
    FoodWasteDataModule
)


def test_lightning_config():
    """Test Lightning configuration classes"""
    
    # Test training config
    training_config = LightningTrainingConfig()
    assert training_config.model_name == "facebook/dinov2-base"
    assert training_config.num_labels == 104
    assert training_config.learning_rate == 5e-5
    assert training_config.num_epochs == 10
    
    # Test custom training config
    custom_config = LightningTrainingConfig(
        learning_rate=1e-4,
        num_epochs=20,
        batch_size=16
    )
    assert custom_config.learning_rate == 1e-4
    assert custom_config.num_epochs == 20
    assert custom_config.batch_size == 16
    
    # Test data config
    data_config = LightningDataConfig()
    assert data_config.batch_size == 8
    assert data_config.num_workers == 4
    assert data_config.dataset_name == "EduardoPacheco/FoodSeg103"


def test_lightning_module_creation():
    """Test Lightning module creation"""
    
    config = LightningTrainingConfig(
        num_labels=10,  # Smaller number for testing
        learning_rate=1e-4
    )
    
    # Create module
    module = FoodWasteSegmentationModule(config)
    
    # Check that hyperparameters are saved
    assert hasattr(module, 'hparams')
    assert module.config == config
    
    # Check that model is created
    assert hasattr(module, 'model')
    assert hasattr(module, 'train_loss')
    assert hasattr(module, 'val_loss')
    assert hasattr(module, 'val_iou')


def test_lightning_module_forward():
    """Test Lightning module forward pass"""
    
    config = LightningTrainingConfig(
        num_labels=10,
        learning_rate=1e-4
    )
    
    module = FoodWasteSegmentationModule(config)
    
    # Create dummy input
    batch_size = 2
    channels = 3
    height = 224
    width = 224
    
    dummy_images = torch.randn(batch_size, channels, height, width)
    dummy_labels = torch.randint(0, 10, (batch_size, height, width))
    
    # Test forward pass without labels
    outputs = module(pixel_values=dummy_images)
    assert hasattr(outputs, 'logits')
    assert outputs.logits.shape[0] == batch_size
    assert outputs.logits.shape[1] == 10  # num_labels
    
    # Test forward pass with labels
    outputs_with_labels = module(pixel_values=dummy_images, labels=dummy_labels)
    assert hasattr(outputs_with_labels, 'loss')
    assert outputs_with_labels.loss is not None


def test_lightning_module_training_step():
    """Test Lightning module training step"""
    
    config = LightningTrainingConfig(
        num_labels=10,
        learning_rate=1e-4
    )
    
    module = FoodWasteSegmentationModule(config)
    
    # Create dummy batch
    batch_size = 2
    channels = 3
    height = 224
    width = 224
    
    dummy_images = torch.randn(batch_size, channels, height, width)
    dummy_labels = torch.randint(0, 10, (batch_size, height, width))
    batch = (dummy_images, dummy_labels)
    
    # Test training step
    loss = module.training_step(batch, batch_idx=0)
    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad


def test_lightning_module_validation_step():
    """Test Lightning module validation step"""
    
    config = LightningTrainingConfig(
        num_labels=10,
        learning_rate=1e-4
    )
    
    module = FoodWasteSegmentationModule(config)
    
    # Create dummy batch
    batch_size = 2
    channels = 3
    height = 224
    width = 224
    
    dummy_images = torch.randn(batch_size, channels, height, width)
    dummy_labels = torch.randint(0, 10, (batch_size, height, width))
    batch = (dummy_images, dummy_labels)
    
    # Test validation step
    loss = module.validation_step(batch, batch_idx=0)
    assert isinstance(loss, torch.Tensor)


def test_lightning_module_optimizer_config():
    """Test Lightning module optimizer configuration"""
    
    config = LightningTrainingConfig(
        num_labels=10,
        learning_rate=1e-4,
        weight_decay=0.01
    )
    
    module = FoodWasteSegmentationModule(config)
    
    # Configure optimizers
    optimizer_config = module.configure_optimizers()
    
    assert "optimizer" in optimizer_config
    assert "lr_scheduler" in optimizer_config
    
    optimizer = optimizer_config["optimizer"]
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["lr"] == 1e-4
    assert optimizer.param_groups[0]["weight_decay"] == 0.01


def test_lightning_module_callbacks():
    """Test Lightning module callback configuration"""
    
    config = LightningTrainingConfig(
        num_labels=10,
        learning_rate=1e-4,
        save_dir="test_checkpoints",
        monitor_metric="val_iou",
        monitor_mode="max",
        save_top_k=3,
        early_stopping_patience=5
    )
    
    module = FoodWasteSegmentationModule(config)
    
    # Configure callbacks
    callbacks = module.configure_callbacks()
    
    assert len(callbacks) == 3  # ModelCheckpoint, EarlyStopping, LearningRateMonitor
    
    # Check ModelCheckpoint
    checkpoint_callback = callbacks[0]
    assert checkpoint_callback.dirpath == "test_checkpoints"
    assert checkpoint_callback.monitor == "val_iou"
    assert checkpoint_callback.mode == "max"
    assert checkpoint_callback.save_top_k == 3
    
    # Check EarlyStopping
    early_stopping_callback = callbacks[1]
    assert early_stopping_callback.monitor == "val_iou"
    assert early_stopping_callback.mode == "max"
    assert early_stopping_callback.patience == 5


def test_lightning_data_module():
    """Test Lightning data module"""
    
    config = LightningDataConfig(
        batch_size=4,
        num_workers=2,
        cache_dir="./test_cache"
    )
    
    data_module = FoodWasteDataModule(config)
    
    assert data_module.config == config
    assert data_module.train_loader is None
    assert data_module.val_loader is None


def test_lightning_module_metrics():
    """Test Lightning module metrics"""
    
    config = LightningTrainingConfig(
        num_labels=10,
        learning_rate=1e-4
    )
    
    module = FoodWasteSegmentationModule(config)
    
    # Test metrics initialization
    assert hasattr(module, 'train_loss')
    assert hasattr(module, 'val_loss')
    assert hasattr(module, 'val_iou')
    
    # Test metrics are torchmetrics
    import torchmetrics
    assert isinstance(module.train_loss, torchmetrics.MeanMetric)
    assert isinstance(module.val_loss, torchmetrics.MeanMetric)
    assert isinstance(module.val_iou, torchmetrics.JaccardIndex)


if __name__ == "__main__":
    # Run tests
    print("Running Lightning implementation tests...")
    
    try:
        test_lightning_config()
        print("✓ Configuration tests passed")
        
        test_lightning_module_creation()
        print("✓ Module creation tests passed")
        
        test_lightning_module_forward()
        print("✓ Forward pass tests passed")
        
        test_lightning_module_training_step()
        print("✓ Training step tests passed")
        
        test_lightning_module_validation_step()
        print("✓ Validation step tests passed")
        
        test_lightning_module_optimizer_config()
        print("✓ Optimizer configuration tests passed")
        
        test_lightning_module_callbacks()
        print("✓ Callback configuration tests passed")
        
        test_lightning_data_module()
        print("✓ Data module tests passed")
        
        test_lightning_module_metrics()
        print("✓ Metrics tests passed")
        
        print("\n🎉 All tests passed! Lightning implementation is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
