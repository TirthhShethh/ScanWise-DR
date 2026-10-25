"""
audit_dataset.py – Inspect the IDRiD dataset and write reports/dataset_audit.md.
Run: python scripts/audit_dataset.py
"""

from __future__ import annotations

import hashlib
import sys
import io
from pathlib import Path

# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

from src.dataset import (
    DISEASE_GRADING_ROOT, IMG_TRAIN, IMG_TEST,
    TRAIN_CSV, TEST_CSV, DR_GRADES, ME_GRADES,
    load_labels,
)


def file_hash(path: Path, block: int = 65536) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()


def audit() -> None:
    lines = []

    def p(s: str = "") -> None:
        try:
            print(s)
        except UnicodeEncodeError:
            print(s.encode('ascii', 'replace').decode('ascii'))
        lines.append(s)

    p("# ScanWise - Dataset Audit Report")
    p()
    p("## Dataset: IDRiD (Indian Diabetic Retinopathy Image Dataset)")
    p()
    p("**Source**: Porwal et al. (2018) - https://idrid.grand-challenge.org/")
    p("**Type**: Colour fundus photographs of the retina")
    p("**Tasks supported**:")
    p("  - **A. Segmentation** - Pixel-level lesion masks (Microaneurysms, Haemorrhages, Hard/Soft Exudates, Optic Disc)")
    p("  - **B. Disease Grading** - Image-level DR severity (0-4) and Macular Edema risk (0-2) [Primary Task]")
    p("  - **C. Localization** - Optic disc and fovea centre coordinates")
    p()
    p("---")
    p()
    p("## Section B: Disease Grading (Primary Task)")
    p()

    # Load CSVs
    train_df = load_labels("train")
    test_df  = load_labels("test")

    p("### Image Counts")
    p("| Split | Images |")
    p("|-------|--------|")
    p(f"| Train | {len(train_df)} |")
    p(f"| Test  | {len(test_df)} |")
    p(f"| **Total** | **{len(train_df)+len(test_df)}** |")
    p()

    # Class distribution - DR
    p("### Retinopathy Grade Distribution")
    p("| Grade | Name | Train | Test | Total |")
    p("|-------|------|-------|------|-------|")
    train_dr = train_df["retinopathy_grade"].value_counts().sort_index()
    test_dr  = test_df["retinopathy_grade"].value_counts().sort_index()
    for g, name in DR_GRADES.items():
        tr = train_dr.get(g, 0)
        te = test_dr.get(g, 0)
        p(f"| {g} | {name} | {tr} | {te} | {tr+te} |")
    p()

    # Class distribution - ME
    p("### Macular Edema Risk Distribution")
    p("| Grade | Name | Train | Test | Total |")
    p("|-------|------|-------|------|-------|")
    train_me = train_df["me_grade"].value_counts().sort_index()
    test_me  = test_df["me_grade"].value_counts().sort_index()
    for g, name in ME_GRADES.items():
        tr = train_me.get(g, 0)
        te = test_me.get(g, 0)
        p(f"| {g} | {name} | {tr} | {te} | {tr+te} |")
    p()

    # Image format check
    p("### Image Format and Integrity")
    from PIL import Image as PILImage
    corrupt  = []
    duplicates_hash = {}
    sizes = []
    all_paths = (
        list(train_df["image_path"].dropna()) +
        list(test_df["image_path"].dropna())
    )
    print(f"Checking {len(all_paths)} images for integrity...")
    for pstr in all_paths:
        pp = Path(pstr)
        try:
            with PILImage.open(pp) as img:
                img.verify()
            with PILImage.open(pp) as img:
                sizes.append(img.size)
            h = file_hash(pp)
            if h in duplicates_hash:
                duplicates_hash[h].append(pp.name)
            else:
                duplicates_hash[h] = [pp.name]
        except Exception as e:
            corrupt.append((pp.name, str(e)))

    p(f"- **Format**: JPEG (.jpg)")
    p(f"- **Corrupt images**: {len(corrupt)}")
    p(f"- **Duplicate images** (by MD5): {sum(1 for v in duplicates_hash.values() if len(v)>1)}")
    if sizes:
        ws = [s[0] for s in sizes]
        hs = [s[1] for s in sizes]
        p(f"- **Width range**: {min(ws)} - {max(ws)} px")
        p(f"- **Height range**: {min(hs)} - {max(hs)} px")
    p()

    # Missing labels
    train_missing = train_df["image_path"].isna().sum()
    test_missing  = test_df["image_path"].isna().sum()
    p("### Missing Labels / Images")
    p(f"- Train images with missing path: {train_missing}")
    p(f"- Test images with missing path:  {test_missing}")
    p()

    p("---")
    p()
    p("## Dataset Design Notes")
    p()
    p("- **Official split preserved**: The dataset provides an official train/test split that we use as-is.")
    p("- **No patient-ID leakage risk**: Each IDRiD image corresponds to a unique patient eye.")
    p("- **Class imbalance**: Grade 0 (No DR) and Grade 2 (Moderate DR) dominate. Handled via weighted sampler + class-weighted loss.")
    p("- **Primary task chosen**: Disease Grading (B) - provides image-level labels suitable for CNN classification.")
    p("- **Input size**: Images resized to 384x384 for training to preserve retinal detail.")
    p()
    p("---")
    p()
    p("## Other Dataset Sections")
    p()
    p("### A. Segmentation")

    seg_root = PROJECT_ROOT / "A.%20Segmentation" / "A. Segmentation"
    lesion_types = [
        "1. Microaneurysms", "2. Haemorrhages",
        "3. Hard Exudates", "4. Soft Exudates", "5. Optic Disc"
    ]
    for lt in lesion_types:
        tr = seg_root / "2. All Segmentation Groundtruths" / "a. Training Set" / lt
        te = seg_root / "2. All Segmentation Groundtruths" / "b. Testing Set" / lt
        n_tr = len(list(tr.glob("*.tif"))) if tr.exists() else 0
        n_te = len(list(te.glob("*.tif"))) if te.exists() else 0
        p(f"- {lt}: {n_tr} train masks, {n_te} test masks")
    p()
    p("### C. Localization")
    p("- Optic disc centre coordinates (CSV)")
    p("- Fovea centre coordinates (CSV)")
    p()

    out = REPORTS_DIR / "dataset_audit.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nAudit written to {out}")


if __name__ == "__main__":
    audit()
