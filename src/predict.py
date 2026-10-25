"""
predict.py – Single-image inference for ScanWise.

Usage
-----
python -m src.predict path/to/retinal_image.jpg --task dr --model efficientnet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset       import DR_GRADES, ME_GRADES
from src.preprocessing import val_transforms
from src.models        import (
    build_efficientnet, build_resnet50, load_checkpoint,
)
from src.train         import get_device, MODELS_DIR


# Cache loaded models to avoid repeated disk I/O in Streamlit
_MODEL_CACHE: Dict[str, object] = {}


def load_model_for_inference(
    task:       str = "dr",
    model_name: str = "efficientnet",
) -> tuple:
    """Load and cache model + metadata. Returns (model, meta, device)."""
    cache_key = f"{model_name}_{task}"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    device    = get_device()
    ckpt_path = MODELS_DIR / f"{model_name}_{task}_best.pth"

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No checkpoint found at {ckpt_path}.\n"
            f"Train first: python -m src.train --task {task} --model {model_name}"
        )

    class_names = DR_GRADES if task == "dr" else ME_GRADES
    num_classes = len(class_names)

    if model_name == "efficientnet":
        model = build_efficientnet(num_classes, pretrained=False)
    else:
        model = build_resnet50(num_classes, pretrained=False)

    meta = load_checkpoint(ckpt_path, model, device)
    model.eval()

    result = (model, meta, device)
    _MODEL_CACHE[cache_key] = result
    return result


def predict_image(
    image_input,         # PIL Image or file path (str/Path)
    task:       str = "dr",
    model_name: str = "efficientnet",
) -> Dict:
    """
    Run inference on a single image.

    Returns
    -------
    dict with keys:
        predicted_class     – int
        predicted_label     – str
        probabilities       – {class_label: float}
        top3                – [(label, prob), ...]
        model_name          – str
        task                – str
    """
    model, meta, device = load_model_for_inference(task, model_name)
    transform           = val_transforms()

    if isinstance(image_input, (str, Path)):
        img = Image.open(image_input).convert("RGB")
    else:
        img = image_input.convert("RGB")

    # Validate image
    w, h = img.size
    if w < 64 or h < 64:
        raise ValueError(f"Image too small ({w}×{h}). Minimum 64×64 required.")

    tensor = transform(img).unsqueeze(0).to(device)   # (1, 3, H, W)

    class_names = DR_GRADES if task == "dr" else ME_GRADES

    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1).squeeze(0).cpu()

    pred_idx   = int(probs.argmax())
    pred_label = class_names[pred_idx]

    prob_dict = {class_names[i]: float(probs[i]) for i in sorted(class_names)}
    top3 = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)[:3]

    return {
        "predicted_class": pred_idx,
        "predicted_label": pred_label,
        "probabilities":   prob_dict,
        "top3":            top3,
        "model_name":      model_name,
        "task":            task,
        "raw_logits":      logits.squeeze(0).cpu().tolist(),
    }


def clear_model_cache() -> None:
    """Force reload of models (useful after new training run)."""
    _MODEL_CACHE.clear()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(description="Run ScanWise prediction on a single image")
    parser.add_argument("image",   help="Path to the retinal image file")
    parser.add_argument("--task",  default="dr", choices=["dr", "me"])
    parser.add_argument("--model", default="efficientnet", choices=["efficientnet", "resnet50"])
    args = parser.parse_args()

    result = predict_image(args.image, task=args.task, model_name=args.model)
    print(f"\nPrediction: {result['predicted_label']} (class {result['predicted_class']})")
    print("Probabilities:")
    for label, prob in result["probabilities"].items():
        bar = "#" * int(prob * 30)
        print(f"  {label:25s}: {prob:.4f}  {bar}")


if __name__ == "__main__":
    main()
