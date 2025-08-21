"""
Dataset classes for semantic segmentation
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any, Tuple
from datasets import load_dataset
from lightning import LightningDataModule
import torch.nn as nn

from .transforms import get_train_transforms, get_val_transforms, ADE_MEAN, ADE_STD
from .config import id2label,label2id

class SegmentationDataset(Dataset):
    """Custom dataset for semantic segmentation"""
    
    def __init__(self, dataset, transform, split="train"):
        """
        Args:
            dataset: HuggingFace dataset
            transform: Data transformation pipeline
            split: Dataset split ('train' or 'validation')
        """
        self.dataset = dataset[split]
        self.transform = transform
        self.split = split
                
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        transformed = self.transform(image=np.array(item["image"]), mask=np.array(item["label"]))
        image, target = torch.tensor(transformed['image']), torch.LongTensor(transformed['mask'])

        # convert to C, H, W
        image = image.permute(2,0,1)
        
        return image, target
    
    def get_label_mappings(self):
        """Get label ID to label name mappings"""
        return id2label, label2id


class FoodSegmentationDataModule(LightningDataModule):
    """Lightning DataModule for FoodSeg103 semantic segmentation dataset"""
    
    def __init__(
        self,
        cache_dir: str = "./data_cache",
        dataset_name: str = "EduardoPacheco/FoodSeg103",
        batch_size: int = 8,
        num_workers: int = 0,
        train_split: str = "train",
        val_split: str = "validation",
        image_size: int = 518,
        mean: Tuple[float, float, float] = ADE_MEAN,
        std: Tuple[float, float, float] = ADE_STD
    ):
        super().__init__()
        self.cache_dir = cache_dir
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_split
        self.val_split = val_split
        
        # Initialize transforms
        self.train_transform = get_train_transforms(mean=mean, std=std, image_size=image_size)
        self.val_transform = get_val_transforms(mean=mean, std=std, image_size=image_size)
        
        # Dataset attributes
        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None
        
    def prepare_data(self):
        """Download and prepare the dataset"""
        # Load FoodSeg103 dataset
        self.dataset = load_dataset(self.dataset_name, cache_dir=self.cache_dir)
    
    def collate_fn(self, inputs):
        """Custom collate function for batching"""
        batch = dict()
        batch["pixel_values"] = torch.stack([i[0] for i in inputs], dim=0)
        batch["labels"] = torch.stack([i[1] for i in inputs], dim=0)
        return batch
        
    def setup(self, stage: str):
        """Setup datasets for different stages"""
        # Assign train/val datasets for use in dataloaders
        if self.dataset is None:
            self.prepare_data()
        if stage == "fit":
            self.train_dataset = SegmentationDataset(
                self.dataset, self.train_transform, split=self.train_split
            )
            self.val_dataset = SegmentationDataset(
                self.dataset, self.val_transform, split=self.val_split
            )
        if stage == "validate":
            self.val_dataset = SegmentationDataset(
                self.dataset, self.val_transform, split=self.val_split
            )
                        
    def train_dataloader(self):
        """Training data loader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
            #collate_fn=self.collate_fn,
            persistent_workers=self.num_workers>0
        )
        
    def val_dataloader(self):
        """Validation data loader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            #collate_fn=self.collate_fn,
            persistent_workers=self.num_workers>0
        )
                
    def get_label_mappings(self):
        """Get label ID to label name mappings"""
        if self.dataset is not None:
            return id2label, label2id
        return None, None

