#!/usr/bin/env python3
"""
Example script for training DINOv2 semantic segmentation using PyTorch Lightning
"""

import sys
import os
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from foodwaste.lightning_config import LightningTrainingConfig, LightningDataConfig
from foodwaste.train_dinov2_segmentation_lightning import (
    FoodWasteSegmentationModule, 
    FoodWasteDataModule,
    main
)
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger


def create_custom_config():
    """Create a custom configuration for training"""
    
    # Custom training configuration
    training_config = LightningTrainingConfig(
        model_name="facebook/dinov2-base",
        num_labels=104,
        freeze_backbone=True,
        learning_rate=1e-4,  # Higher learning rate
        weight_decay=0.01,    # Higher weight decay
        warmup_steps=1000,    # More warmup steps
        max_grad_norm=1.0,
        num_epochs=20,        # More epochs
        batch_size=16,        # Larger batch size
        num_workers=8,        # More workers
        save_dir="custom_checkpoints",
        log_interval=50,      # More frequent logging
        eval_interval=250,    # More frequent evaluation
        accelerator="auto",
        devices="auto",
        precision="16-mixed", # Use mixed precision
        seed=123,
        deterministic=True,
        early_stopping_patience=15,
        save_top_k=5,
        monitor_metric="val_iou",
        monitor_mode="max"
    )
    
    # Custom data configuration
    data_config = LightningDataConfig(
        dataset_name="EduardoPacheco/FoodSeg103",
        cache_dir="./custom_data_cache",
        batch_size=16,
        num_workers=8,
        train_split="train",
        val_split="validation"
    )
    
    return training_config, data_config


def train_with_custom_config():
    """Train with custom configuration"""
    
    # Create custom configs
    training_config, data_config = create_custom_config()
    
    # Create data module
    data_module = FoodWasteDataModule(data_config)
    
    # Create model
    model = FoodWasteSegmentationModule(training_config)
    
    # Create logger
    tb_logger = TensorBoardLogger(
        save_dir="lightning_logs",
        name="custom_foodwaste_training",
        version=None
    )
    
    # Create trainer with custom settings
    trainer = L.Trainer(
        max_epochs=training_config.num_epochs,
        accelerator=training_config.accelerator,
        devices=training_config.devices,
        precision=training_config.precision,
        logger=tb_logger,
        callbacks=model.configure_callbacks(),
        log_every_n_steps=training_config.log_interval,
        val_check_interval=training_config.eval_interval / 1000,
        gradient_clip_val=training_config.max_grad_norm,
        deterministic=training_config.deterministic,
        enable_progress_bar=True,
        enable_model_summary=True,
        enable_checkpointing=True,
        num_sanity_val_steps=2,
        check_val_every_n_epoch=1,
        sync_batchnorm=False,
        strategy="auto",
        # Additional custom settings
        accumulate_grad_batches=2,  # Gradient accumulation
        track_grad_norm=2,          # Track gradient norms
        overfit_batches=0.0,        # No overfitting
        reload_dataloaders_every_n_epochs=0,
        use_distributed_sampler=True,
        detect_anomaly=False,
        benchmark=False,
        inference_mode=True
    )
    
    # Train the model
    print("Starting custom training...")
    trainer.fit(model, data_module)
    
    # Validate the model
    print("Running validation...")
    trainer.validate(model, data_module)
    
    # Save the final model
    final_model_path = os.path.join(training_config.save_dir, "custom_final_model.ckpt")
    trainer.save_checkpoint(final_model_path)
    print(f"Custom training completed! Model saved to {final_model_path}")


def train_with_default_config():
    """Train with default configuration"""
    print("Training with default configuration...")
    
    # Import and run the main function with default args
    import argparse
    args = argparse.Namespace()
    args.config = ""
    args.resume = False
    args.checkpoint_path = ""
    
    main(args)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train DINOv2 for semantic segmentation using PyTorch Lightning"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["default", "custom"],
        default="default",
        help="Training mode: 'default' for standard config, 'custom' for custom config"
    )
    
    args = parser.parse_args()
    
    if args.mode == "custom":
        train_with_custom_config()
    else:
        train_with_default_config()
