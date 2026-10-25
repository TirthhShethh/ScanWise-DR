"""
preprocessing.py – Image transforms for ScanWise / IDRiD.

Applies ImageNet-normalised preprocessing. Training set gets augmentation.
Test/inference gets deterministic centre-crop only.
"""

from __future__ import annotations

from typing import Tuple

import torchvision.transforms as T

# IDRiD images are high-resolution fundus photos (typically 4288 × 2848).
# We resize to 512 on the longer side, then centre-crop to INPUT_SIZE.
# For EfficientNet-B0 the canonical input is 224×224.
# We use 384×384 to preserve more retinal detail while fitting in CPU RAM.

INPUT_SIZE   = 384          # final square input to the network
RESIZE_TO    = 420          # resize shorter-side before crop

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def train_transforms() -> T.Compose:
    """Augmentation pipeline for training data only."""
    return T.Compose([
        T.Resize(RESIZE_TO, interpolation=T.InterpolationMode.BICUBIC),
        T.RandomCrop(INPUT_SIZE),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=20),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def val_transforms() -> T.Compose:
    """Deterministic transforms for validation / test / inference."""
    return T.Compose([
        T.Resize(RESIZE_TO, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(INPUT_SIZE),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def denormalize(tensor):
    """
    Reverse ImageNet normalisation for display.
    Returns a (H, W, 3) numpy uint8 array.
    """
    import numpy as np
    t = tensor.clone().cpu().float()
    mean = T.functional.to_tensor([IMAGENET_MEAN]).view(3, 1, 1)
    std  = T.functional.to_tensor([IMAGENET_STD ]).view(3, 1, 1)
    # Use inline computation instead of transforms
    mean_t = __import__('torch').tensor(IMAGENET_MEAN).view(3, 1, 1)
    std_t  = __import__('torch').tensor(IMAGENET_STD ).view(3, 1, 1)
    t = t * std_t + mean_t
    t = t.clamp(0, 1)
    arr = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return arr


PREPROCESSING_CONFIG = {
    "input_size":      INPUT_SIZE,
    "resize_to":       RESIZE_TO,
    "imagenet_mean":   IMAGENET_MEAN,
    "imagenet_std":    IMAGENET_STD,
    "interpolation":   "BICUBIC",
}
