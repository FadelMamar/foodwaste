#!/usr/bin/env python3
"""
Main training script for DINOv2 semantic segmentation using PyTorch Lightning
"""

import os
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
import torchmetrics

from config import TrainingConfig, DataConfig
from models import create_model
from dataset import load_foodseg103_dataset, create_data_loaders
from utils import (
    set_seed, calculate_metrics, visualize_batch, create_color_map
)


class FoodWasteSegmentationModule(L.LightningModule):
    """PyTorch Lightning module for FoodWaste semantic segmentation"""
    
    def __init__(self, config: LightningTrainingConfig):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        # Create model
        self.model = create_model(
            model_name=config.model_name,
            freeze_backbone=config.freeze_backbone
        )
        
        # Metrics
        self.train_loss = torchmetrics.MeanMetric()
        self.val_loss = torchmetrics.MeanMetric()
        self.val_iou = torchmetrics.JaccardIndex(
            task="multiclass", 
            num_classes=config.num_labels,
            ignore_index=config.ignore_index
        )
        
    def forward(self, pixel_values, labels=None):
        """Forward pass through the model"""
        return self.model(pixel_values=pixel_values, labels=labels)
    
    def training_step(self, batch, batch_idx):
        """Training step"""
        images, labels = batch
        
        # Forward pass
        outputs = self(pixel_values=images, labels=labels)
        loss = outputs.loss
        
        # Log loss
        self.train_loss.update(loss)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        images, labels = batch
        
        # Forward pass
        outputs = self(pixel_values=images, labels=labels)
        loss = outputs.loss
        
        # Log loss
        self.val_loss.update(loss)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        
        # Calculate IoU
        predictions = torch.argmax(outputs.logits, dim=1)
        self.val_iou.update(predictions, labels)
        self.log("val_iou", self.val_iou, prog_bar=True, sync_dist=True)
        
        return loss
    
    def on_train_epoch_end(self):
        """Called at the end of training epoch"""
        # Log epoch metrics
        self.log("train_loss_epoch", self.train_loss.compute(), sync_dist=True)
        self.train_loss.reset()
    
    def on_validation_epoch_end(self):
        """Called at the end of validation epoch"""
        # Log epoch metrics
        self.log("val_loss_epoch", self.val_loss.compute(), sync_dist=True)
        self.log("val_iou_epoch", self.val_iou.compute(), sync_dist=True)
        self.val_loss.reset()
        self.val_iou.reset()
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler"""
        optimizer = AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Calculate total steps
        total_steps = self.trainer.estimated_stepping_batches
        
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=self.config.learning_rate * 0.1
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "step",
                "frequency": 1
            }
        }
    
    def configure_callbacks(self):
        """Configure callbacks"""
        callbacks = []
        
        # Model checkpointing
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.config.save_dir,
            filename="foodwaste-{epoch:02d}-{val_loss:.4f}-{val_iou:.4f}",
            monitor=self.config.monitor_metric,
            mode=self.config.monitor_mode,
            save_top_k=self.config.save_top_k,
            save_last=True,
            verbose=True
        )
        callbacks.append(checkpoint_callback)
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor=self.config.monitor_metric,
            mode=self.config.monitor_mode,
            patience=self.config.early_stopping_patience,
            verbose=True
        )
        callbacks.append(early_stopping)
        
        # Learning rate monitoring
        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)
        
        return callbacks


class FoodWasteDataModule(L.LightningDataModule):
    """PyTorch Lightning data module for FoodWaste dataset"""
    
    def __init__(self, config: LightningDataConfig):
        super().__init__()
        self.config = config
        
        self.train_loader = None
        self.val_loader = None
    
    def setup(self, stage: Optional[str] = None):
        """Setup data loaders"""
        if stage == "fit" or stage is None:
            # Load dataset
            dataset = load_foodseg103_dataset(cache_dir=self.config.cache_dir)
            
            # Create data loaders
            self.train_loader, self.val_loader = create_data_loaders(
                dataset,
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
                train_split=self.config.train_split,
                val_split=self.config.val_split
            )
    
    def train_dataloader(self):
        """Return training data loader"""
        return self.train_loader
    
    def val_dataloader(self):
        """Return validation data loader"""
        return self.val_loader


def setup_logging(log_dir: str = "logs") -> logging.Logger:
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def main(args: argparse.Namespace):
    """Main training function using PyTorch Lightning"""
    # Setup
    logger = setup_logging()
    logger.info("Starting DINOv2 semantic segmentation training with PyTorch Lightning")
    
    # Load configuration
    training_config = LightningTrainingConfig()
    data_config = LightningDataConfig()
    
    # Override config with command line arguments if provided
    if args.config:
        # TODO: Add config file loading logic
        pass
    
    # Set seed for reproducibility
    set_seed(training_config.seed)
    
    # Create output directories
    os.makedirs(training_config.save_dir, exist_ok=True)
    
    # Create data module
    data_module = FoodWasteDataModule(data_config)
    
    # Create model
    model = FoodWasteSegmentationModule(training_config)
    
    # Create logger
    tb_logger = TensorBoardLogger(
        save_dir="lightning_logs",
        name="foodwaste_segmentation",
        version=None
    )
    
    # Create trainer
    trainer = L.Trainer(
        max_epochs=training_config.num_epochs,
        accelerator=training_config.accelerator,
        devices=training_config.devices,
        precision=training_config.precision,
        logger=tb_logger,
        callbacks=model.configure_callbacks(),
        log_every_n_steps=training_config.log_interval,
        val_check_interval=training_config.eval_interval / 1000,  # Convert to fraction
        gradient_clip_val=training_config.max_grad_norm,
        deterministic=training_config.deterministic,
        enable_progress_bar=True,
        enable_model_summary=True,
        enable_checkpointing=True,
        num_sanity_val_steps=2,
        check_val_every_n_epoch=1,
        sync_batchnorm=False,
        strategy="auto"
    )
    
    # Train the model
    logger.info("Starting training...")
    trainer.fit(model, data_module)
    
    # Test the model
    logger.info("Running final validation...")
    trainer.validate(model, data_module)
    
    # Save the final model
    final_model_path = os.path.join(training_config.save_dir, "final_model.ckpt")
    trainer.save_checkpoint(final_model_path)
    logger.info(f"Final model saved to {final_model_path}")
    
    logger.info("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train DINOv2 for semantic segmentation using PyTorch Lightning"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--resume", 
        action="store_true",
        help="Resume training from checkpoint"
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="",
        help="Path to checkpoint for resuming training"
    )
    
    args = parser.parse_args()
    main(args)
