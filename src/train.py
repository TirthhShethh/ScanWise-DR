"""
train.py – Training pipeline for ScanWise / IDRiD disease grading.

Usage
-----
# Smoke test (5 images, 2 epochs):
python -m src.train --smoke-test

# Full training with EfficientNet-B0:
python -m src.train --task dr --model efficientnet --epochs 30

# Resume interrupted training:
python -m src.train --task dr --model efficientnet --epochs 30 --resume

Run `python -m src.train --help` for all options.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset       import IDRiDDataset, DR_GRADES, ME_GRADES
from src.preprocessing import train_transforms, val_transforms, PREPROCESSING_CONFIG
from src.models        import (
    build_efficientnet, build_resnet50,
    freeze_backbone, unfreeze_all,
    save_checkpoint, load_checkpoint,
)

MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42

def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if model_name == "efficientnet":
        return build_efficientnet(num_classes, pretrained=pretrained)
    elif model_name == "resnet50":
        return build_resnet50(num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    training: bool,
    scaler=None,
) -> Tuple[float, float]:
    """Run one epoch. Returns (avg_loss, accuracy)."""
    model.train(training)
    total_loss = 0.0
    correct    = 0
    total      = 0

    with torch.set_grad_enabled(training):
        for imgs, labels in loader:
            imgs   = imgs.to(device)
            labels = labels.to(device)

            if training and scaler is not None:
                with torch.amp.autocast("cuda"):
                    logits = model(imgs)
                    loss   = criterion(logits, labels)
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(imgs)
                loss   = criterion(logits, labels)
                if training:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            preds        = logits.argmax(dim=1)
            correct     += (preds == labels).sum().item()
            total       += labels.size(0)
            total_loss  += loss.item() * labels.size(0)

    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(
    task:       str   = "dr",
    model_name: str   = "efficientnet",
    epochs:     int   = 30,
    batch_size: int   = 16,
    lr:         float = 1e-4,
    freeze_epochs: int = 5,
    smoke_test: bool  = False,
    resume:     bool  = False,
    pretrained: bool  = True,
) -> Path:
    """
    Train the model and save best checkpoint.

    Returns path to best checkpoint.
    """
    seed_everything(SEED)
    device = get_device()
    print(f"[train] Device: {device}")
    print(f"[train] Task: {task}  Model: {model_name}  Epochs: {epochs}")

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
    train_ds = IDRiDDataset(split="train", task=task, transform=train_transforms())
    test_ds  = IDRiDDataset(split="test",  task=task, transform=val_transforms())

    if smoke_test:
        n = min(20, len(train_ds))
        m = min(10, len(test_ds))
        train_ds = Subset(train_ds, list(range(n)))
        test_ds  = Subset(test_ds,  list(range(m)))
        epochs   = max(epochs, 2)
        print(f"[train] SMOKE TEST: {n} train, {m} test, {epochs} epochs")
    else:
        print(train_ds.summary())

    sampler = train_ds.weighted_sampler() if not smoke_test else None
    shuffle = False if sampler else True

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    # ------------------------------------------------------------------
    # Model, loss, optimizer
    # ------------------------------------------------------------------
    if smoke_test:
        base = train_ds.dataset if hasattr(train_ds, "dataset") else train_ds
        num_classes = base.num_classes
        class_names = base.class_names
    else:
        num_classes = train_ds.num_classes
        class_names = train_ds.class_names

    model = build_model(model_name, num_classes, pretrained=pretrained)

    # Class-weighted cross-entropy for imbalance
    if smoke_test:
        cw = torch.ones(num_classes)
    else:
        cw = train_ds.class_weights()
    criterion = nn.CrossEntropyLoss(weight=cw.to(device))

    # Phase 1: freeze backbone, train head only
    freeze_backbone(model, model_name)
    model.to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr * 10,   # higher LR for head
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=freeze_epochs)

    ckpt_path = MODELS_DIR / f"{model_name}_{task}_best.pth"
    start_epoch   = 0
    best_val_acc  = 0.0
    train_history = {"loss": [], "acc": [], "val_loss": [], "val_acc": []}

    # Resume
    if resume and ckpt_path.exists():
        meta = load_checkpoint(ckpt_path, model, device)
        start_epoch  = meta.get("epoch", 0) + 1
        best_val_acc = meta.get("best_val_acc", 0.0)
        print(f"[train] Resumed from epoch {start_epoch}, best val acc={best_val_acc:.4f}")

    # Optional mixed precision (CUDA only)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for epoch in range(start_epoch, epochs):
        t0 = time.time()

        # Switch to full fine-tuning after freeze_epochs
        if epoch == freeze_epochs:
            print(f"[train] Epoch {epoch}: unfreezing backbone")
            unfreeze_all(model)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs - freeze_epochs
            )

        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, training=True, scaler=scaler
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, None, device, training=False
        )
        scheduler.step()

        train_history["loss"].append(train_loss)
        train_history["acc"].append(train_acc)
        train_history["val_loss"].append(val_loss)
        train_history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch+1:3d}/{epochs} | "
            f"Train loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"Val loss={val_loss:.4f} acc={val_acc:.4f} | "
            f"{elapsed:.1f}s"
        )

        # Save best checkpoint
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                ckpt_path, model, optimizer, scheduler,
                meta={
                    "epoch":               epoch,
                    "best_val_acc":        best_val_acc,
                    "class_names":         class_names,
                    "num_classes":         num_classes,
                    "task":                task,
                    "model_name":          model_name,
                    "preprocessing_config": PREPROCESSING_CONFIG,
                    "train_metrics":       {"loss": train_loss, "acc": train_acc},
                    "val_metrics":         {"loss": val_loss,   "acc": val_acc},
                    "train_history":       train_history,
                },
            )
            print(f"  [OK] Saved best checkpoint (val_acc={best_val_acc:.4f})")

        # Always save latest checkpoint for resumption
        save_checkpoint(
            MODELS_DIR / f"{model_name}_{task}_latest.pth",
            model, optimizer, scheduler,
            meta={
                "epoch":               epoch,
                "best_val_acc":        best_val_acc,
                "class_names":         class_names,
                "num_classes":         num_classes,
                "task":                task,
                "model_name":          model_name,
                "preprocessing_config": PREPROCESSING_CONFIG,
                "train_history":       train_history,
            },
        )

    print(f"\n[train] Finished. Best val acc: {best_val_acc:.4f}")
    print(f"[train] Checkpoint: {ckpt_path}")
    return ckpt_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train ScanWise DR grading model")
    parser.add_argument("--task",       default="dr",          choices=["dr", "me"],
                        help="Classification task: dr=retinopathy, me=macular edema")
    parser.add_argument("--model",      default="efficientnet", choices=["efficientnet", "resnet50"],
                        help="Backbone architecture")
    parser.add_argument("--epochs",     default=30, type=int,  help="Total training epochs")
    parser.add_argument("--batch-size", default=16, type=int,  help="Mini-batch size")
    parser.add_argument("--lr",         default=1e-4, type=float, help="Learning rate (fine-tune phase)")
    parser.add_argument("--freeze-epochs", default=5, type=int,
                        help="Epochs to train head only before unfreezing backbone")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Quick run with 20 images and 2 epochs")
    parser.add_argument("--resume",     action="store_true",
                        help="Resume from latest checkpoint")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Do not use ImageNet pre-trained weights")
    args = parser.parse_args()

    train(
        task=args.task,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        freeze_epochs=args.freeze_epochs,
        smoke_test=args.smoke_test,
        resume=args.resume,
        pretrained=not args.no_pretrained,
    )


if __name__ == "__main__":
    main()
