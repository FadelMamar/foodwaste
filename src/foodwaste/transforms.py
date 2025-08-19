"""
Data transformation functions for semantic segmentation
"""

import numpy as np
import albumentations as A
from typing import Tuple, Dict, Any

ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
ADE_STD = np.array([58.395, 57.120, 57.375]) / 255

def get_train_transforms() -> A.Compose:
    """Get training data transformations"""
    train_transform = A.Compose([
        A.Resize(width=448, height=448),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=ADE_MEAN, std=ADE_STD),
    ])
    return train_transform


def get_val_transforms() -> A.Compose:
    """Get validation data transformations"""
    val_transform = A.Compose([
    A.Resize(width=448, height=448),
    A.Normalize(mean=ADE_MEAN, std=ADE_STD),
    ])
    return val_transform


def apply_transforms(
    image: np.ndarray,
    mask: np.ndarray,
    transforms: A.Compose
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply transformations to image and mask"""
    transformed = transforms(image=image, mask=mask)
    return transformed['image'], transformed['mask']


def denormalize_image(normalized_image: np.ndarray) -> np.ndarray:
    """Denormalize image for visualization"""
    mean = np.array(ADE_MEAN)[:, None, None]
    std = np.array(ADE_STD)[:, None, None]
    
    denormalized = (normalized_image * std) + mean
    denormalized = np.clip(denormalized, 0, 255).astype(np.uint8)
    return denormalized
