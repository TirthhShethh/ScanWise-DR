"""
evaluate.py – Compute and persist evaluation metrics for a trained ScanWise model.

Usage
-----
python -m src.evaluate --task dr --model efficientnet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset       import IDRiDDataset
from src.preprocessing import val_transforms
from src.models        import build_efficientnet, build_resnet50, load_checkpoint
from src.train         import get_device, MODELS_DIR


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    task:       str = "dr",
    model_name: str = "efficientnet",
    batch_size: int = 16,
    split:      str = "test",
) -> Dict:
    """
    Load the best checkpoint, run on *split*, return metrics dict.
    Saves results to models/{model_name}_{task}_eval.json.
    """
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score,
        classification_report, confusion_matrix,
        roc_auc_score, f1_score,
    )

    device = get_device()
    ckpt_path = MODELS_DIR / f"{model_name}_{task}_best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {ckpt_path}. Train first with:\n"
            f"  python -m src.train --task {task} --model {model_name}"
        )

    ds = IDRiDDataset(split=split, task=task, transform=val_transforms())
    num_classes = ds.num_classes
    class_names = ds.class_names

    if model_name == "efficientnet":
        from src.models import build_efficientnet
        model = build_efficientnet(num_classes, pretrained=False)
    else:
        from src.models import build_resnet50
        model = build_resnet50(num_classes, pretrained=False)

    meta = load_checkpoint(ckpt_path, model, device)

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    all_preds  = []
    all_labels = []
    all_probs  = []

    model.eval()
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Evaluating"):
            imgs = imgs.to(device)
            logits = model(imgs)
            probs  = torch.softmax(logits, dim=1)
            preds  = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    y_true   = np.array(all_labels)
    y_pred   = np.array(all_preds)
    y_prob   = np.array(all_probs)

    acc     = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    report  = classification_report(
        y_true, y_pred,
        target_names=[class_names[i] for i in sorted(class_names)],
        zero_division=0,
        output_dict=True,
    )
    cm = confusion_matrix(y_true, y_pred).tolist()

    # ROC-AUC (one-vs-rest for multiclass)
    try:
        roc_auc = roc_auc_score(
            y_true, y_prob,
            multi_class="ovr",
            average="macro",
            labels=sorted(class_names.keys()),
        )
    except Exception:
        roc_auc = None

    results = {
        "task":              task,
        "model_name":        model_name,
        "split":             split,
        "accuracy":          acc,
        "balanced_accuracy": bal_acc,
        "macro_f1":          macro_f1,
        "roc_auc_ovr":       roc_auc,
        "classification_report": report,
        "confusion_matrix":  cm,
        "y_true":            y_true.tolist(),
        "y_pred":            y_pred.tolist(),
        "y_prob":            y_prob.tolist(),
        "class_names":       {str(k): v for k, v in class_names.items()},
        "checkpoint_meta":   {k: v for k, v in meta.items() if k != "preprocessing_config"},
    }

    out_path = MODELS_DIR / f"{model_name}_{task}_eval.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[evaluate] Results saved to {out_path}")
    print(f"  Accuracy:          {acc:.4f}")
    print(f"  Balanced Accuracy: {bal_acc:.4f}")
    print(f"  Macro F1:          {macro_f1:.4f}")
    if roc_auc:
        print(f"  ROC-AUC (OvR):     {roc_auc:.4f}")

    return results


# ---------------------------------------------------------------------------
# Misclassified examples
# ---------------------------------------------------------------------------

def get_misclassified(
    results: Dict,
    n: int = 12,
    split: str = "test",
    task: str = "dr",
) -> List[Dict]:
    """Return up to *n* misclassified example dicts with image paths."""
    from src.dataset import load_labels
    df = load_labels(split)
    label_col = "retinopathy_grade" if task == "dr" else "me_grade"
    df = df.dropna(subset=[label_col]).reset_index(drop=True)

    y_true = results["y_true"]
    y_pred = results["y_pred"]
    class_names = results["class_names"]

    examples = []
    for i, (t, p) in enumerate(zip(y_true, y_pred)):
        if t != p and i < len(df):
            examples.append({
                "image_path": df.iloc[i]["image_path"],
                "image_id":   df.iloc[i]["image_id"],
                "true_label": class_names.get(str(t), str(t)),
                "pred_label": class_names.get(str(p), str(p)),
                "true_idx":   t,
                "pred_idx":   p,
                "confidence": results["y_prob"][i][p],
            })
        if len(examples) >= n:
            break
    return examples


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate ScanWise model")
    parser.add_argument("--task",       default="dr",          choices=["dr", "me"])
    parser.add_argument("--model",      default="efficientnet", choices=["efficientnet", "resnet50"])
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--split",      default="test",        choices=["train", "test"])
    args = parser.parse_args()
    evaluate_model(task=args.task, model_name=args.model,
                   batch_size=args.batch_size, split=args.split)


if __name__ == "__main__":
    main()
