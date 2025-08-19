"""
Dataset classes for semantic segmentation
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any, Tuple
from datasets import load_dataset
from transforms import get_train_transforms, get_val_transforms
from lightning import LightningDataModule

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
        batch_size: int = 8,
        num_workers: int = 0,
        train_split: str = "train",
        val_split: str = "validation"
    ):
        super().__init__()
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_split
        self.val_split = val_split
        
        # Initialize transforms
        self.train_transform = get_train_transforms()
        self.val_transform = get_val_transforms()
        
        # Dataset attributes
        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None
        
    def prepare_data(self):
        """Download and prepare the dataset"""
        # Load FoodSeg103 dataset
        self.dataset = load_dataset("EduardoPacheco/FoodSeg103", cache_dir=self.cache_dir)
    
    def collate_fn(self, inputs):
        """Custom collate function for batching"""
        batch = dict()
        batch["pixel_values"] = torch.stack([i[0] for i in inputs], dim=0)
        batch["labels"] = torch.stack([i[1] for i in inputs], dim=0)
        return batch
        
    def setup(self, stage: str):
        """Setup datasets for different stages"""
        # Assign train/val datasets for use in dataloaders
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
            collate_fn=self.collate_fn
        )
        
    def val_dataloader(self):
        """Validation data loader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )
                
    def get_label_mappings(self):
        """Get label ID to label name mappings"""
        if self.dataset is not None:
            return self.dataset.features["labels"].feature.int2str, self.dataset.features["labels"].feature.str2int
        return None, None



