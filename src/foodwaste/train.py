#!/usr/bin/env python3
"""
Main training script for DINOv2 semantic segmentation using PyTorch Lightning
"""

import os

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import MLFlowLogger, TensorBoardLogger
from torchmetrics.segmentation import MeanIoU,DiceScore
from copy import deepcopy

from .config import MainConfig
from .models import Dinov2ForSemanticSegmentation,PatchClassifier
from .dataset import id2label, FoodSegmentationDataModule

from .utils import (
    set_seed
)

from logging import getLogger


LOGGER = getLogger(__name__)

class FoodWasteSegmentationModule(L.LightningModule):
    """PyTorch Lightning module for FoodWaste semantic segmentation"""
    
    def __init__(self, config: MainConfig):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        # Create model
        if config.model.use_patch_classifier:
            self.model = PatchClassifier(config)
        else:
            self.model = Dinov2ForSemanticSegmentation(config)
        
        # Losses
        self.train_loss = nn.CrossEntropyLoss(ignore_index=0)
        self.val_loss = nn.CrossEntropyLoss(ignore_index=0)

        # Metrics
        self.val_metrics = {
            "iou": MeanIoU(
                num_classes=len(id2label),
                include_background=False,
            ),
            "dice": DiceScore(
                num_classes=len(id2label),
                include_background=False,
                average="weighted"
            )
        }
        self.train_metrics = deepcopy(self.val_metrics)

        self.metrics = {
            "train": self.train_metrics,
            "val": self.val_metrics
        }
        
        
    def forward(self, pixel_values,mask:Optional[torch.Tensor]=None):
        """Forward pass through the model"""
        return self.model(pixel_values=pixel_values,mask=mask)
    
    def shared_step(self, batch, stage: str):
        """Shared step for training and validation"""
        images, labels = batch
        outputs = self(pixel_values=images,mask=labels)
        
        # Handle different model output formats
        if isinstance(outputs, tuple):
            # PatchClassifier returns (logits, labels)
            logits, labels = outputs
        else:
            # Dinov2ForSemanticSegmentation returns logits
            logits = outputs

        if stage == "train":
            loss = self.train_loss(logits, labels)
        else:
            loss = self.val_loss(logits, labels)

        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        for name, metric in self.metrics[stage].items():
            metric.update(logits.detach().cpu().argmax(dim=1), labels.detach().cpu())
            #self.log(f"{stage}_{name}", metric, prog_bar=True, on_step=False, on_epoch=True)

        return loss
    
    def training_step(self, batch, batch_idx):
        """Training step"""
        loss = self.shared_step(batch, "train")
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        loss = self.shared_step(batch, "val")
        return loss
    
    def on_validation_epoch_end(self) -> None:
        for name, metric in self.metrics["val"].items():
            score = metric.compute().cpu()
            self.log(f"val_{name}",score, prog_bar=True, on_step=False, on_epoch=True)        
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler"""
        optimizer = AdamW(
            self.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )
                
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.config.training.num_epochs,
            T_mult=1,
            eta_min=self.config.training.learning_rate * self.config.training.lrf,
        )
        
        return [optimizer], [scheduler]
    
    def configure_callbacks(self):
        """Configure callbacks"""
        callbacks = []
        
        # Model checkpointing
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.config.logging.save_dir,
            filename="foodwaste-{epoch:02d}-{val_iou:.4f}",
            monitor=self.config.training.monitor,
            mode=self.config.training.mode,
            save_on_train_epoch_end=False,
            save_last=True,
            verbose=True
        )
        callbacks.append(checkpoint_callback)
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor=self.config.training.monitor,
            mode=self.config.training.mode,
            patience=self.config.training.patience,
            verbose=True
        )
        callbacks.append(early_stopping)
        
        # Learning rate monitoring
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks.append(lr_monitor)
        
        return callbacks


def runner(config: MainConfig):
    """Main training function using PyTorch Lightning"""
    # Setup
    
    LOGGER.info("Starting FoodWaste semantic segmentation training with PyTorch Lightning")
    LOGGER.info(f"Model configuration: {config.model}")
    LOGGER.info(f"Training configuration: {config.training}")

    # Set seed for reproducibility
    set_seed(config.training.seed)

    # Create output directories
    os.makedirs(config.logging.save_dir, exist_ok=True)
    
    # Create data module
    data_module = FoodSegmentationDataModule(batch_size=config.training.batch_size, 
                                            num_workers=config.training.num_workers,
                                            cache_dir=config.data.cache_dir,
                                            dataset_name=config.data.dataset_name,
                                            image_size=config.training.image_size,
                                            mean=config.data.mean,
                                            std=config.data.std,
                                            train_split=config.training.train_split, 
                                            val_split=config.training.val_split)
    
    # Create model
    model = FoodWasteSegmentationModule(config)
    
    # Create logger
    if config.logging.logger == "mlflow":
        logger = MLFlowLogger(
            tracking_uri=config.logging.logger_url,
            experiment_name=config.logging.experiment_name,
            run_name=config.logging.run_name,
            #tags=config.model_dump()
        )
    elif config.logging.logger == "tensorboard":
        logger = TensorBoardLogger(
            save_dir=config.logging.save_dir,
            name=config.logging.experiment_name,
            version=config.logging.run_name,
        )
    else:
        raise ValueError(f"Logger {config.logging.logger} not supported")
    
    # Create trainer
    trainer = L.Trainer(
        max_epochs=config.training.num_epochs,
        accelerator=config.training.accelerator,
        precision=config.training.precision,
        logger=logger,
        #callbacks=model.configure_callbacks(),
        check_val_every_n_epoch=config.logging.eval_interval,
        num_sanity_val_steps=2,
    )
    
    # Train the model
    LOGGER.info("Starting training...")
    trainer.fit(model, data_module)
    
    # Test the model
    #logger.info("Running final validation...")
    #trainer.validate(model, data_module)
    
    # Save the final model
    final_model_path = trainer.checkpoint_callback.best_model_path
    LOGGER.info(f"Final model saved to {final_model_path}")
    
    LOGGER.info("Training completed!")
