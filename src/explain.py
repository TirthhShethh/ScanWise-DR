"""
explain.py – Grad-CAM implementation for ScanWise.

Generates saliency heatmaps for EfficientNet-B0 and ResNet-50
that show which retinal regions influenced the model's prediction.

DISCLAIMER: Highlighted regions indicate model attention, not
proven diagnostic locations. This is a research prototype.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


# ---------------------------------------------------------------------------
# Grad-CAM implementation
# ---------------------------------------------------------------------------

class GradCAM:
    """
    Generic Grad-CAM that works with any CNN.

    Args:
        model:       The trained PyTorch model.
        target_layer: The convolutional layer to hook (e.g. model.features[-1]).
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model        = model
        self.target_layer = target_layer
        self._gradients   = None
        self._activations = None
        self._hooks: list = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(module, input, output):
            self._activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self._gradients = grad_output[0].detach()

        self._hooks.append(
            self.target_layer.register_forward_hook(forward_hook)
        )
        self._hooks.append(
            self.target_layer.register_full_backward_hook(backward_hook)
        )

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def generate(
        self,
        input_tensor: torch.Tensor,   # (1, 3, H, W)
        target_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, np.ndarray]:
        """
        Compute Grad-CAM heatmap.

        Returns
        -------
        heatmap   : (H, W) float32 in [0, 1]
        pred_class: predicted class index
        probs     : (num_classes,) softmax probabilities
        """
        self.model.eval()
        input_tensor = input_tensor.clone().requires_grad_(True)

        logits = self.model(input_tensor)
        probs  = F.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()

        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        self.model.zero_grad()
        score = logits[0, target_class]
        score.backward()

        # Global average pool the gradients (GAP)
        grads   = self._gradients   # (1, C, h, w)
        acts    = self._activations # (1, C, h, w)
        weights = grads.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        cam = (weights * acts).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # Normalise to [0, 1]
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        else:
            cam = np.zeros_like(cam)

        return cam.astype(np.float32), target_class, probs


# ---------------------------------------------------------------------------
# Layer resolver
# ---------------------------------------------------------------------------

def get_target_layer(model: nn.Module, model_name: str) -> nn.Module:
    """Return the last convolutional feature layer for Grad-CAM."""
    if model_name == "efficientnet":
        # timm EfficientNet
        if hasattr(model, "features"):
            return model.features[-1]
        # torchvision EfficientNet
        if hasattr(model, "features"):
            return model.features[-1]
        # fallback: traverse named modules
        conv_layers = [
            m for _, m in model.named_modules()
            if isinstance(m, (nn.Conv2d, nn.Sequential))
        ]
        return conv_layers[-3] if len(conv_layers) >= 3 else conv_layers[-1]
    elif model_name == "resnet50":
        return model.layer4[-1]
    else:
        # Heuristic: last Conv2d
        convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
        return convs[-1]


# ---------------------------------------------------------------------------
# Overlay helpers
# ---------------------------------------------------------------------------

def apply_colormap(heatmap: np.ndarray) -> np.ndarray:
    """Convert (H, W) float [0,1] heatmap to (H, W, 3) uint8 RGB jet colormap."""
    import matplotlib
    import matplotlib.cm as cm
    try:
        import matplotlib
        colormap = matplotlib.colormaps["jet"]
    except (KeyError, AttributeError):
        colormap = cm.get_cmap("jet")
    rgba = colormap(heatmap)        # (H, W, 4)
    rgb  = (rgba[:, :, :3] * 255).astype(np.uint8)
    return rgb


def overlay_heatmap(
    original_rgb: np.ndarray,   # (H, W, 3) uint8
    heatmap: np.ndarray,        # (h, w) float [0,1]
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Resize heatmap to match original image and blend as overlay.

    Returns (H, W, 3) uint8.
    """
    H, W = original_rgb.shape[:2]
    heatmap_resized = np.array(
        Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
            (W, H), Image.BILINEAR
        )
    ) / 255.0

    colored = apply_colormap(heatmap_resized)          # (H, W, 3)
    overlay = (
        alpha * colored.astype(np.float32)
        + (1 - alpha) * original_rgb.astype(np.float32)
    ).clip(0, 255).astype(np.uint8)
    return overlay


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------

def generate_gradcam(
    model:        nn.Module,
    model_name:   str,
    image_tensor: torch.Tensor,      # (1, 3, H, W)
    original_img: Image.Image,       # original PIL for overlay
    target_class: Optional[int] = None,
    device:       torch.device  = torch.device("cpu"),
) -> Dict:
    """
    Run Grad-CAM and return a results dict with numpy arrays.

    Returns
    -------
    dict:
        heatmap          : (h, w) float [0,1]
        heatmap_colored  : (H, W, 3) uint8 jet-colored
        overlay          : (H, W, 3) uint8 blended
        original_np      : (H, W, 3) uint8
        pred_class       : int
        probs            : list[float]
    """
    layer = get_target_layer(model, model_name)
    cam   = GradCAM(model, layer)

    tensor = image_tensor.to(device)
    try:
        heatmap, pred_class, probs = cam.generate(tensor, target_class=target_class)
    finally:
        cam.remove_hooks()

    original_np = np.array(original_img.convert("RGB"))

    heatmap_full = np.array(
        Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
            (original_np.shape[1], original_np.shape[0]), Image.BILINEAR
        )
    ) / 255.0

    heatmap_colored = apply_colormap(heatmap_full)
    overlay         = overlay_heatmap(original_np, heatmap, alpha=0.45)

    return {
        "heatmap":         heatmap,
        "heatmap_colored": heatmap_colored,
        "overlay":         overlay,
        "original_np":     original_np,
        "pred_class":      pred_class,
        "probs":           probs.tolist(),
    }
