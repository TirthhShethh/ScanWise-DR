"""
tests/test_all.py – Automated test suite for ScanWise.

Run: python -m pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from io import BytesIO

import numpy as np
import pytest
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def train_df():
    from src.dataset import load_labels
    return load_labels("train")

@pytest.fixture(scope="session")
def test_df():
    from src.dataset import load_labels
    return load_labels("test")

@pytest.fixture(scope="session")
def real_image_path(train_df):
    """Return the path to the first valid training image."""
    return train_df.iloc[0]["image_path"]

@pytest.fixture(scope="session")
def real_pil_image(real_image_path):
    return Image.open(real_image_path).convert("RGB")

@pytest.fixture(scope="session")
def fake_retinal_pil():
    """Synthetic 512×384 RGB image for testing without requiring actual data."""
    arr = (np.random.rand(384, 512, 3) * 255).astype(np.uint8)
    return Image.fromarray(arr)

@pytest.fixture(scope="session")
def tiny_model():
    """EfficientNet-B0 with random weights (no pretrained download)."""
    from src.models import build_efficientnet
    return build_efficientnet(num_classes=5, pretrained=False)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Dataset discovery and label parsing
# ─────────────────────────────────────────────────────────────────────────────

class TestDatasetLoading:
    def test_train_labels_loads(self, train_df):
        assert len(train_df) > 0, "Training CSV is empty"

    def test_test_labels_loads(self, test_df):
        assert len(test_df) > 0, "Test CSV is empty"

    def test_train_has_required_columns(self, train_df):
        for col in ("image_id", "retinopathy_grade", "me_grade", "image_path"):
            assert col in train_df.columns, f"Missing column: {col}"

    def test_retinopathy_grades_valid(self, train_df):
        grades = set(train_df["retinopathy_grade"].dropna().astype(int).unique())
        assert grades.issubset({0, 1, 2, 3, 4}), f"Unexpected grades: {grades}"

    def test_me_grades_valid(self, train_df):
        grades = set(train_df["me_grade"].dropna().astype(int).unique())
        assert grades.issubset({0, 1, 2}), f"Unexpected ME grades: {grades}"

    def test_image_paths_exist(self, train_df):
        missing = train_df["image_path"].isna().sum()
        assert missing == 0, f"{missing} training images are missing from disk"

    def test_train_count_range(self, train_df):
        assert 400 <= len(train_df) <= 420, f"Unexpected train size: {len(train_df)}"

    def test_test_count_range(self, test_df):
        assert 100 <= len(test_df) <= 110, f"Unexpected test size: {len(test_df)}"

    def test_no_data_leakage(self, train_df, test_df):
        """
        IDRiD uses the same image numbering in both splits (001-103 appear in both).
        True split integrity check: verify that the ACTUAL FILE PATHS don't overlap.
        Train uses .../a. Training Set/ and test uses .../b. Testing Set/.
        """
        train_paths = set(train_df["image_path"].dropna().tolist())
        test_paths  = set(test_df["image_path"].dropna().tolist())
        overlap     = train_paths & test_paths
        assert len(overlap) == 0, f"Data leakage via file paths: {overlap}"

    def test_class_to_index_mapping(self):
        from src.dataset import DR_GRADES, ME_GRADES
        assert DR_GRADES[0] == "No DR"
        assert DR_GRADES[4] == "Proliferative DR"
        assert ME_GRADES[0] == "No ME Risk"
        assert ME_GRADES[2] == "High ME Risk"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Dataset class
# ─────────────────────────────────────────────────────────────────────────────

class TestIDRiDDataset:
    def test_dr_dataset_len(self):
        from src.dataset import IDRiDDataset
        ds = IDRiDDataset(split="train", task="dr")
        assert len(ds) >= 400

    def test_me_dataset_len(self):
        from src.dataset import IDRiDDataset
        ds = IDRiDDataset(split="train", task="me")
        assert len(ds) >= 400

    def test_dataset_item_shape(self):
        from src.dataset import IDRiDDataset
        from src.preprocessing import val_transforms
        ds = IDRiDDataset(split="train", task="dr", transform=val_transforms())
        img, lbl = ds[0]
        assert img.shape == (3, 384, 384), f"Wrong shape: {img.shape}"
        assert isinstance(lbl, int)

    def test_invalid_split_raises(self):
        from src.dataset import IDRiDDataset
        with pytest.raises(ValueError):
            IDRiDDataset(split="val", task="dr")

    def test_class_weights_shape(self):
        from src.dataset import IDRiDDataset
        ds = IDRiDDataset(split="train", task="dr")
        cw = ds.class_weights()
        assert cw.shape == (5,)
        assert (cw > 0).all()

    def test_weighted_sampler_length(self):
        from src.dataset import IDRiDDataset
        ds = IDRiDDataset(split="train", task="dr")
        sampler = ds.weighted_sampler()
        assert len(sampler) == len(ds)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

class TestPreprocessing:
    def test_val_transform_output_shape(self, fake_retinal_pil):
        from src.preprocessing import val_transforms, INPUT_SIZE
        tf = val_transforms()
        tensor = tf(fake_retinal_pil)
        assert tensor.shape == (3, INPUT_SIZE, INPUT_SIZE)

    def test_train_transform_output_shape(self, fake_retinal_pil):
        from src.preprocessing import train_transforms, INPUT_SIZE
        tf = train_transforms()
        tensor = tf(fake_retinal_pil)
        assert tensor.shape == (3, INPUT_SIZE, INPUT_SIZE)

    def test_normalisation_range(self, fake_retinal_pil):
        from src.preprocessing import val_transforms
        tf = val_transforms()
        tensor = tf(fake_retinal_pil)
        # After ImageNet norm, values can be outside [0,1] — just check finite
        assert torch.isfinite(tensor).all()
        assert tensor.dtype == torch.float32

    def test_val_transform_deterministic(self, fake_retinal_pil):
        from src.preprocessing import val_transforms
        tf = val_transforms()
        t1 = tf(fake_retinal_pil)
        t2 = tf(fake_retinal_pil)
        assert torch.allclose(t1, t2)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Models
# ─────────────────────────────────────────────────────────────────────────────

class TestModels:
    def test_efficientnet_output_shape(self, tiny_model, fake_retinal_pil):
        from src.preprocessing import val_transforms
        tf  = val_transforms()
        x   = tf(fake_retinal_pil).unsqueeze(0)
        out = tiny_model(x)
        assert out.shape == (1, 5), f"Wrong output: {out.shape}"

    def test_resnet50_output_shape(self, fake_retinal_pil):
        from src.models import build_resnet50
        from src.preprocessing import val_transforms
        model = build_resnet50(num_classes=5, pretrained=False)
        tf    = val_transforms()
        x     = tf(fake_retinal_pil).unsqueeze(0)
        out   = model(x)
        assert out.shape == (1, 5)

    def test_freeze_backbone(self):
        from src.models import build_efficientnet, freeze_backbone
        model = build_efficientnet(num_classes=5, pretrained=False)
        freeze_backbone(model, "efficientnet")
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # Only the classifier head should be trainable
        assert trainable < 50_000, f"Too many trainable params after freeze: {trainable}"

    def test_checkpoint_save_load(self, tiny_model):
        import tempfile, os
        from src.models import save_checkpoint, load_checkpoint
        with tempfile.TemporaryDirectory() as td:
            ckpt = Path(td) / "test.pth"
            save_checkpoint(ckpt, tiny_model, optimizer=None, scheduler=None,
                            meta={"epoch": 0, "best_val_acc": 0.5, "class_names": {0:"A"},
                                  "num_classes": 5, "task": "dr", "model_name": "efficientnet",
                                  "preprocessing_config": {}, "train_metrics": {}, "val_metrics": {}})
            assert ckpt.exists()

            from src.models import build_efficientnet
            model2 = build_efficientnet(num_classes=5, pretrained=False)
            meta   = load_checkpoint(ckpt, model2, torch.device("cpu"))
            assert meta["epoch"] == 0
            assert meta["best_val_acc"] == 0.5

    def test_missing_checkpoint_raises(self):
        from src.models import build_efficientnet, load_checkpoint
        model = build_efficientnet(num_classes=5, pretrained=False)
        with pytest.raises(FileNotFoundError):
            load_checkpoint(Path("nonexistent.pth"), model, torch.device("cpu"))


# ─────────────────────────────────────────────────────────────────────────────
# 5. Prediction
# ─────────────────────────────────────────────────────────────────────────────

class TestPrediction:
    def test_predict_requires_checkpoint(self, fake_retinal_pil):
        """predict_image should raise FileNotFoundError if no checkpoint."""
        from src.predict import predict_image, clear_model_cache, MODELS_DIR
        clear_model_cache()
        # Only test if the checkpoint doesn't exist
        ckpt = MODELS_DIR / "efficientnet_dr_best.pth"
        if not ckpt.exists():
            with pytest.raises(FileNotFoundError):
                predict_image(fake_retinal_pil, task="dr", model_name="efficientnet")
        else:
            result = predict_image(fake_retinal_pil, task="dr", model_name="efficientnet")
            assert "predicted_class" in result
            assert 0 <= result["predicted_class"] <= 4
            probs = list(result["probabilities"].values())
            assert abs(sum(probs) - 1.0) < 1e-4

    def test_predict_small_image_raises(self):
        """Images smaller than 64×64 should be rejected."""
        from src.predict import predict_image, MODELS_DIR
        ckpt = MODELS_DIR / "efficientnet_dr_best.pth"
        if not ckpt.exists():
            pytest.skip("No checkpoint available")
        tiny = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
        with pytest.raises(ValueError, match="too small"):
            predict_image(tiny, task="dr", model_name="efficientnet")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Grad-CAM
# ─────────────────────────────────────────────────────────────────────────────

class TestGradCAM:
    def test_gradcam_output_shape(self, tiny_model, fake_retinal_pil):
        from src.preprocessing import val_transforms
        from src.explain import generate_gradcam
        tf     = val_transforms()
        tensor = tf(fake_retinal_pil).unsqueeze(0)
        result = generate_gradcam(
            tiny_model, "efficientnet", tensor, fake_retinal_pil,
            target_class=0, device=torch.device("cpu"),
        )
        assert "heatmap" in result
        assert "overlay" in result
        assert result["heatmap"].ndim == 2
        assert result["overlay"].dtype == np.uint8

    def test_gradcam_heatmap_range(self, tiny_model, fake_retinal_pil):
        from src.preprocessing import val_transforms
        from src.explain import generate_gradcam
        tf     = val_transforms()
        tensor = tf(fake_retinal_pil).unsqueeze(0)
        result = generate_gradcam(
            tiny_model, "efficientnet", tensor, fake_retinal_pil,
            target_class=0, device=torch.device("cpu"),
        )
        assert result["heatmap"].min() >= 0.0
        assert result["heatmap"].max() <= 1.0 + 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# 7. Corruption and invalid upload handling
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_corrupt_image_detection(self):
        """A file with garbage bytes should not be openable as a valid image."""
        import tempfile, os
        with tempfile.TemporaryDirectory() as td:
            bad_file = Path(td) / "corrupt.jpg"
            bad_file.write_bytes(b"\xff\xd8" + b"\x00" * 100)  # truncated JPEG
            with pytest.raises(Exception):
                Image.open(bad_file).verify()

    def test_non_image_file_raises(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            txt_file = Path(td) / "notimage.txt"
            txt_file.write_text("this is not an image")
            with pytest.raises(Exception):
                Image.open(txt_file).convert("RGB")

    def test_grayscale_converted_to_rgb(self, fake_retinal_pil):
        gray = fake_retinal_pil.convert("L")
        rgb  = gray.convert("RGB")
        assert rgb.mode == "RGB"
        from src.preprocessing import val_transforms
        tf     = val_transforms()
        tensor = tf(rgb)
        assert tensor.shape[0] == 3


# ─────────────────────────────────────────────────────────────────────────────
# 8. Smoke-test training (real data, 2 epochs)
# ─────────────────────────────────────────────────────────────────────────────

class TestSmokeTraining:
    @pytest.mark.slow
    def test_smoke_training_creates_checkpoint(self, tmp_path):
        """
        Run 2-epoch smoke training with real IDRiD data.
        Marks as slow so it can be skipped with: pytest -m "not slow"
        """
        from src.train import train, MODELS_DIR
        import shutil

        ckpt = train(
            task="dr",
            model_name="efficientnet",
            epochs=2,
            batch_size=4,
            smoke_test=True,
            pretrained=False,
        )
        assert ckpt.exists(), f"Checkpoint was not created at {ckpt}"

        # Verify checkpoint is loadable
        import torch
        from src.models import build_efficientnet, load_checkpoint
        model = build_efficientnet(num_classes=5, pretrained=False)
        meta  = load_checkpoint(ckpt, model, torch.device("cpu"))
        assert meta["num_classes"] == 5
        assert meta["task"] == "dr"


# ─────────────────────────────────────────────────────────────────────────────
# 9. Evaluation metrics
# ─────────────────────────────────────────────────────────────────────────────

class TestEvaluationMetrics:
    def test_metrics_on_perfect_predictions(self):
        from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
        y_true = [0, 1, 2, 3, 4]
        y_pred = [0, 1, 2, 3, 4]
        assert accuracy_score(y_true, y_pred) == 1.0
        assert f1_score(y_true, y_pred, average="macro") == 1.0

    def test_metrics_on_all_wrong(self):
        from sklearn.metrics import accuracy_score
        y_true = [0, 1, 2, 3, 4]
        y_pred = [4, 3, 1, 0, 2]
        assert accuracy_score(y_true, y_pred) == 0.0

    def test_softmax_probabilities_sum_to_one(self, tiny_model, fake_retinal_pil):
        from src.preprocessing import val_transforms
        import torch.nn.functional as F
        tf  = val_transforms()
        x   = tf(fake_retinal_pil).unsqueeze(0)
        out = tiny_model(x)
        prob = F.softmax(out, dim=1)
        assert abs(prob.sum().item() - 1.0) < 1e-5
        assert (prob >= 0).all()
        assert (prob <= 1).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not slow"])
