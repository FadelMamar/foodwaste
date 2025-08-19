"""
Utility functions for visualization and evaluation
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Dict, List, Tuple, Optional
from PIL import Image
import evaluate
from transforms import denormalize_image


def create_color_map(num_classes: int, seed: int = 42) -> Dict[int, List[int]]:
    """Create a random color map for visualization"""
    np.random.seed(seed)
    color_map = {
        k: list(np.random.choice(range(256), size=3)) 
        for k in range(num_classes)
    }
    return color_map


def visualize_segmentation(
    image: np.ndarray,
    segmentation_map: np.ndarray,
    color_map: Dict[int, List[int]],
    alpha: float = 0.7,
    save_path: Optional[str] = None
) -> None:
    """
    Visualize segmentation results
    
    Args:
        image: Input image [H, W, C]
        segmentation_map: Segmentation map [H, W]
        color_map: Color mapping for classes
        alpha: Transparency for overlay
        save_path: Path to save visualization
    """
    # Create colored segmentation map
    colored_map = np.zeros_like(image)
    for class_id, color in color_map.items():
        mask = segmentation_map == class_id
        colored_map[mask] = color
    
    # Create overlay
    overlay = image * (1 - alpha) + colored_map * alpha
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    # Segmentation map
    axes[1].imshow(segmentation_map, cmap="tab20")
    axes[1].set_title("Segmentation Map")
    axes[1].axis("off")
    
    # Overlay
    axes[2].imshow(overlay)
    axes[2].set_title("Segmentation Overlay")
    axes[2].axis("off")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    plt.show()


def visualize_batch(
    batch: Dict[str, torch.Tensor],
    predictions: torch.Tensor,
    color_map: Dict[int, List[int]],
    num_samples: int = 4,
    save_path: Optional[str] = None
) -> None:
    """
    Visualize a batch of images with predictions
    
    Args:
        batch: Input batch with pixel_values and labels
        predictions: Model predictions [batch_size, num_classes, H, W]
        color_map: Color mapping for classes
        num_samples: Number of samples to visualize
        save_path: Path to save visualization
    """
    num_samples = min(num_samples, batch["pixel_values"].shape[0])
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Get image and labels
        image = batch["pixel_values"][i]
        labels = batch["labels"][i]
        pred = predictions[i]
        
        # Convert to numpy and denormalize
        image_np = image.permute(1, 2, 0).numpy()
        image_np = denormalize_image(image_np)
        
        labels_np = labels.numpy()
        pred_np = pred.argmax(dim=0).numpy()
        
        # Original image
        axes[i, 0].imshow(image_np)
        axes[i, 0].set_title(f"Sample {i+1} - Original")
        axes[i, 0].axis("off")
        
        # Ground truth
        axes[i, 1].imshow(labels_np, cmap="tab20")
        axes[i, 1].set_title(f"Sample {i+1} - Ground Truth")
        axes[i, 1].axis("off")
        
        # Prediction
        axes[i, 2].imshow(pred_np, cmap="tab20")
        axes[i, 2].set_title(f"Sample {i+1} - Prediction")
        axes[i, 2].axis("off")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    plt.show()


def calculate_metrics(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    ignore_index: int = 0
) -> Dict[str, float]:
    """
    Calculate segmentation metrics
    
    Args:
        predictions: Model predictions [batch_size, num_classes, H, W]
        labels: Ground truth labels [batch_size, H, W]
        num_classes: Number of classes
        ignore_index: Index to ignore in calculations
    
    Returns:
        Dictionary with metrics
    """
    # Convert to numpy
    pred_np = predictions.argmax(dim=1).cpu().numpy()
    labels_np = labels.cpu().numpy()
    
    # Flatten
    pred_flat = pred_np.flatten()
    labels_flat = labels_np.flatten()
    
    # Remove ignored indices
    valid_mask = labels_flat != ignore_index
    pred_valid = pred_flat[valid_mask]
    labels_valid = labels_flat[valid_mask]
    
    # Calculate metrics
    metric = evaluate.load("mean_iou")
    metrics = metric.compute(
        predictions=pred_valid,
        references=labels_valid,
        num_labels=num_classes
    )
    
    return metrics


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    save_path: str
) -> None:
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, save_path)


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: str
) -> Tuple[int, float]:
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return epoch, loss


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
