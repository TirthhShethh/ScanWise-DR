"""
models.py – Model definitions for ScanWise.

Provides EfficientNet-B0 (primary) and ResNet-50 (comparison) via timm.
Each model head is replaced for the target number of classes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


def build_efficientnet(num_classes: int, pretrained: bool = True) -> nn.Module:
    """EfficientNet-B0 with a fresh classification head."""
    try:
        import timm
        model = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            num_classes=num_classes,
        )
    except Exception:
        # Fallback: torchvision EfficientNet-B0
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def build_resnet50(num_classes: int, pretrained: bool = True) -> nn.Module:
    """ResNet-50 with a fresh classification head."""
    from torchvision.models import resnet50, ResNet50_Weights
    weights = ResNet50_Weights.DEFAULT if pretrained else None
    model = resnet50(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def freeze_backbone(model: nn.Module, model_name: str = "efficientnet") -> None:
    """Freeze all layers except the classification head."""
    if model_name == "efficientnet":
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
    elif model_name == "resnet50":
        for name, param in model.named_parameters():
            if "fc" not in name:
                param.requires_grad = False


def unfreeze_all(model: nn.Module) -> None:
    """Unfreeze every parameter for full fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

CHECKPOINT_META_KEYS = [
    "epoch", "best_val_acc", "class_names", "num_classes",
    "task", "model_name", "preprocessing_config", "train_metrics", "val_metrics",
]


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[Any],
    scheduler: Optional[Any],
    meta: Dict[str, Any],
) -> None:
    """Save model state + training metadata to *path*."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        **meta,
    }
    torch.save(payload, path)


def load_checkpoint(path: Path, model: nn.Module, device: torch.device) -> Dict[str, Any]:
    """Load model weights from checkpoint and return the metadata dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return {k: ckpt.get(k) for k in CHECKPOINT_META_KEYS}
