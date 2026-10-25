# ScanWise - Dataset Audit Report

## Dataset: IDRiD (Indian Diabetic Retinopathy Image Dataset)

**Source**: Porwal et al. (2018) - https://idrid.grand-challenge.org/
**Type**: Colour fundus photographs of the retina
**Tasks supported**:
  - **A. Segmentation** - Pixel-level lesion masks (Microaneurysms, Haemorrhages, Hard/Soft Exudates, Optic Disc)
  - **B. Disease Grading** - Image-level DR severity (0-4) and Macular Edema risk (0-2) [Primary Task]
  - **C. Localization** - Optic disc and fovea centre coordinates

---

## Section B: Disease Grading (Primary Task)

### Image Counts
| Split | Images |
|-------|--------|
| Train | 413 |
| Test  | 103 |
| **Total** | **516** |

### Retinopathy Grade Distribution
| Grade | Name | Train | Test | Total |
|-------|------|-------|------|-------|
| 0 | No DR | 134 | 34 | 168 |
| 1 | Mild DR | 20 | 5 | 25 |
| 2 | Moderate DR | 136 | 32 | 168 |
| 3 | Severe DR | 74 | 19 | 93 |
| 4 | Proliferative DR | 49 | 13 | 62 |

### Macular Edema Risk Distribution
| Grade | Name | Train | Test | Total |
|-------|------|-------|------|-------|
| 0 | No ME Risk | 177 | 45 | 222 |
| 1 | Moderate ME Risk | 41 | 10 | 51 |
| 2 | High ME Risk | 195 | 48 | 243 |

### Image Format and Integrity
- **Format**: JPEG (.jpg)
- **Corrupt images**: 0
- **Duplicate images** (by MD5): 6
- **Width range**: 4288 - 4288 px
- **Height range**: 2848 - 2848 px

### Missing Labels / Images
- Train images with missing path: 0
- Test images with missing path:  0

---

## Dataset Design Notes

- **Official split preserved**: The dataset provides an official train/test split that we use as-is.
- **No patient-ID leakage risk**: Each IDRiD image corresponds to a unique patient eye.
- **Class imbalance**: Grade 0 (No DR) and Grade 2 (Moderate DR) dominate. Handled via weighted sampler + class-weighted loss.
- **Primary task chosen**: Disease Grading (B) - provides image-level labels suitable for CNN classification.
- **Input size**: Images resized to 384x384 for training to preserve retinal detail.

---

## Other Dataset Sections

### A. Segmentation
- 1. Microaneurysms: 54 train masks, 27 test masks
- 2. Haemorrhages: 53 train masks, 27 test masks
- 3. Hard Exudates: 54 train masks, 27 test masks
- 4. Soft Exudates: 26 train masks, 14 test masks
- 5. Optic Disc: 54 train masks, 27 test masks

### C. Localization
- Optic disc centre coordinates (CSV)
- Fovea centre coordinates (CSV)
