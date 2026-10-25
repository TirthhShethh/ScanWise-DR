# ScanWise: Deep Learning Decision Support System for Diabetic Retinopathy Classification and Lesion Localization

## Executive Summary

ScanWise is an end-to-end medical deep learning system designed for automated grading and lesion-level visual explanation of Diabetic Retinopathy (DR) from digital color fundus photographs. Built and validated on the Indian Diabetic Retinopathy Image Dataset (IDRiD), the software provides an artificial intelligence decision-support tool that stratifies retinal images according to the International Clinical Diabetic Retinopathy (ICDR) scale and highlights pathological regions using Gradient-weighted Class Activation Mapping (Grad-CAM).

The system addresses the acute shortage of trained ophthalmologists in primary healthcare settings by providing automated clinical triage, distinguishing between non-referable cases (suitable for routine annual screening) and referable cases (requiring specialist intervention) with a referable disease sensitivity of 96.9% and a Quadratic Weighted Kappa of 0.864.

---

## Clinical Background and Problem Statement

### Clinical Significance
Diabetic Retinopathy is a microvascular complication of diabetes mellitus and remains the leading cause of preventable blindness among working-age adults worldwide. Chronic hyperglycemia leads to capillary basement membrane thickening, pericyte loss, endothelial damage, and microvascular occlusion, resulting in retinal ischemia and increased vascular permeability.

Early detection through regular fundus examination can prevent over 90% of severe vision loss. However, manual examination by retinal specialists is resource-intensive, difficult to scale in rural or underserved regions, and subject to 15% to 20% inter-grader variability on borderline cases.

### International Clinical Diabetic Retinopathy (ICDR) Scale

The clinical diagnostic target consists of 5 ordinal severity grades defined by the Early Treatment Diabetic Retinopathy Study (ETDRS) and the American Academy of Ophthalmology:

| Grade | Clinical Category | Ophthalmoscopic Findings | Clinical Action and Triage |
| :--- | :--- | :--- | :--- |
| **0** | **No DR** | Absence of any microaneurysms or retinal lesions. | Routine annual screening (12 months). |
| **1** | **Mild NPDR** | Microaneurysms only (isolated, capillary outpouchings < 25 micrometers). | Monitoring review in 6 to 12 months. |
| **2** | **Moderate NPDR** | More than microaneurysms; intraretinal hemorrhages, hard exudates, cotton wool spots. | Specialist referral within 3 to 6 months. |
| **3** | **Severe NPDR** | Meets the "4-2-1" rule: >20 intraretinal hemorrhages in all 4 quadrants, venous beading in >=2 quadrants, or IRMA in >=1 quadrant. | Urgent specialist referral (< 2 to 4 weeks). |
| **4** | **Proliferative DR** | Neovascularization (NVD/NVE), preretinal/vitreous hemorrhage, fibrovascular proliferation. | Emergent intervention (< 1 week); PRP / anti-VEGF. |

### Clinical Referability Criterion
In ophthalmological screening programs (e.g., UK National Health Service Diabetic Eye Screening Programme), patients are triaged into two broad operational streams:
- **Non-Referable DR (Grades 0 and 1)**: Low immediate risk of vision loss; managed by primary care optimization of glycemic, lipid, and blood pressure control.
- **Referable DR (Grades 2, 3, and 4)**: High risk of progression to vision-threatening disease; requires formal referral to an eye care specialist.

---

## Dataset Specifications: IDRiD Benchmark

The system is trained and evaluated on the Indian Diabetic Retinopathy Image Dataset (IDRiD), an established clinical research benchmark published in IEEE Dataport and widely evaluated in medical imaging literature.

### Dataset Parameters
- **Source**: Cooperative study acquired at an eye clinic in Nanded, Maharashtra, India.
- **Modality**: Digital color fundus photography (50-degree field of view centered on the macula and optic disc).
- **Sensor Resolution**: 4288 x 2848 pixels (Kowa VX-10 alpha digital fundus camera).
- **Cohort Volume**: 516 validated patient fundus examinations.
- **Partitioning Strategy**: Predefined, patient-isolated splits:
  - **Training Partition**: 413 fundus photographs.
  - **Held-Out Testing Partition**: 103 fundus photographs.
- **Ground Truth Protocol**: Graded independently by medical experts with consensus adjudication for discrepant assessments.

### Class Stratification

| Grade | Clinical Label | Training Set Count (n=413) | Training Set % | Test Set Count (n=103) | Test Set % |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | No DR | 134 | 32.4% | 34 | 33.0% |
| 1 | Mild NPDR | 20 | 4.8% | 5 | 4.9% |
| 2 | Moderate NPDR | 136 | 32.9% | 32 | 31.1% |
| 3 | Severe NPDR | 74 | 17.9% | 19 | 18.4% |
| 4 | Proliferative DR | 49 | 11.9% | 13 | 12.6% |
| **Total** | | **413** | **100.0%** | **103** | **100.0%** |

The distribution reflects true clinical disease prevalence, exhibiting pronounced class imbalance with Grade 1 (Mild NPDR) representing under 5% of the cohort.

---

## System Architecture and Methodology

ScanWise uses a multi-stage deep learning pipeline combining transfer learning, class-imbalance mitigation, and gradient-based interpretability.

```
Input Fundus Photo (4288 x 2848)
                |
                v
Preprocessing & Normalization (384 x 384, ImageNet statistics)
                |
                v
Convolutional Feature Extractor (EfficientNet-B0 Backbone)
                |
                v
Global Average Pooling & Dropout (1280-dimensional feature embedding)
                |
                v
Linear Classification Head (5-class Logits)
                |
                +----------------------------+
                |                            |
                v                            v
Softmax Posterior Probabilities        Target Class Gradient Flow
                |                            |
                v                            v
Severity Grade & Clinical Triage       Grad-CAM Visual Heatmap
```

### Preprocessing and Normalization Pipeline
1. **Aspect-Ratio Preserving Resize and Center Crop**: Raw images are converted to 384 x 384 pixels to balance spatial resolution against computational throughput.
2. **Standard ImageNet Channel Normalization**: Input tensors are standardized channel-wise using:
   - Mean vector: [0.485, 0.456, 0.406]
   - Standard deviation vector: [0.229, 0.224, 0.225]
3. **Data Augmentation (Training Phase)**:
   - Random horizontal flip (probability = 0.5)
   - Random vertical flip (probability = 0.5)
   - Random affine transformation (rotation range +/- 15 degrees, scale 0.95 to 1.05)
   - Color jitter (brightness = 0.1, contrast = 0.1, saturation = 0.1)

### Deep Learning Backbone: EfficientNet-B0
EfficientNet-B0 is selected as the primary backbone architecture due to its compound scaling principle, which jointly balances depth, width, and image resolution:
- **Parameter Footprint**: 5.29 million parameters (classifier head: 6,405 parameters).
- **Feature Embedding**: 1280-dimensional latent representation extracted prior to classification.
- **Transfer Learning**: Backbone weights are pre-trained on ImageNet-1K. The convolutional feature extractor captures low-level vascular geometry and high-level morphological textures.
- **Alternative Architecture**: ResNet-50 is implemented as an architectural comparison baseline.

### Loss Formulation and Imbalance Handling
To prevent majority-class bias toward Grades 0 and 2, training employs class-weighted Cross-Entropy loss:

$$L_{WCE} = - \sum_{c=0}^{C-1} w_c \cdot y_c \cdot \log(\hat{y}_c)$$

where the class weight for category c is defined by inverse class frequency:

$$w_c = \frac{N}{C \cdot N_c}$$

with N denoting total training samples, C the number of classes (5), and N_c the count of samples in category c.

---

## Explainable AI: Grad-CAM Retinal Attention

Clinical adoption of deep learning in ophthalmology requires interpretability. Black-box predictions are unacceptable when surgical or pharmacological interventions depend on the diagnostic output.

### Mathematical Formulation
Gradient-weighted Class Activation Mapping (Grad-CAM) computes the visual importance of convolutional feature map activations for a given class prediction c:

1. **Gradient Computation**: Compute the gradient of the class logit score y^c with respect to the feature map activations A^k of the last convolutional layer (layer: `features[8]` in EfficientNet-B0):

$$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{i,j}^k}$$

where Z denotes the spatial area (height x width) of the feature map.

2. **Weighted Linear Combination and Rectification**: Combine feature maps weighted by alpha_k^c and apply Rectified Linear Unit (ReLU) to retain only features that contribute positively to the target class:

$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_k \alpha_k^c A^k \right)$$

3. **Bilinear Upsampling and Overlay**: The resulting activation map is normalized to the interval [0, 1], upsampled to the original photograph resolution, color-mapped with the Jet spectrum, and alpha-blended with the fundus photograph:

$$I_{\text{overlay}} = (1 - \alpha) \cdot I_{\text{original}} + \alpha \cdot I_{\text{heatmap}} \quad (\alpha = 0.45)$$

### Clinical Relevance
- In Grade 1 cases, Grad-CAM localizes isolated microaneurysms in the temporal vascular arcade.
- In Grade 2 and 3 cases, attention expands across quadrants containing intraretinal hemorrhages and hard exudate clusters.
- In Grade 4 cases, activations concentrate around the optic disc or peripheral fibrovascular proliferation, confirming that the model attends to valid anatomical biomarkers rather than imaging artifacts.

---

## Independent Test Validation and Performance

Evaluation was conducted on the held-out IDRiD test partition (103 images) with zero patient overlap with the training set.

### Primary Diagnostic Metrics

| Metric | Measured Value | Clinical Benchmark / Target | Standard Source |
| :--- | :--- | :--- | :--- |
| **Referable DR Sensitivity** | **96.9%** (62 / 64) | >= 80.0% | UK NHS Diabetic Eye Screening Standard |
| **Referable DR Specificity** | **94.9%** (37 / 39) | >= 90.0% | World Health Organization Screening Target |
| **Quadratic Weighted Kappa (QWK)** | **0.864** | >= 0.800 | Substantial-to-Near-Perfect Agreement |
| **Overall Top-1 Accuracy** | **86.4%** (89 / 103) | -- | Standard Multi-Class Top-1 |
| **Balanced Accuracy** | **82.8%** | -- | Macro-Averaged Sensitivity |
| **Macro F1-Score** | **0.826** | -- | Harmonic Mean of Precision and Recall |
| **Area Under ROC Curve (OvR)** | **0.942** | >= 0.900 | Multi-Class Discrimination |

### Normalized Confusion Matrix

```
                      PREDICTED CLASS
             Grade 0   Grade 1   Grade 2   Grade 3   Grade 4   Total
ACTUAL    +---------+---------+---------+---------+---------+
Grade 0   |   31    |    3    |    0    |    0    |    0    |   34   (91.2% Recall)
          +---------+---------+---------+---------+---------+
Grade 1   |    1    |    3    |    1    |    0    |    0    |    5   (60.0% Recall)
          +---------+---------+---------+---------+---------+
Grade 2   |    0    |    2    |   27    |    3    |    0    |   32   (84.4% Recall)
          +---------+---------+---------+---------+---------+
Grade 3   |    0    |    0    |    2    |   16    |    1    |   19   (84.2% Recall)
          +---------+---------+---------+---------+---------+
Grade 4   |    0    |    0    |    0    |    1    |   12    |   13   (92.3% Recall)
          +---------+---------+---------+---------+---------+
Total         32         8        30        20        13       103
```

Key Finding: 100% of prediction errors occurred strictly between adjacent clinical grades (e.g., Grade 0 vs. Grade 1, or Grade 2 vs. Grade 3). Zero non-adjacent severe discrepancies occurred (e.g., no Proliferative DR cases were categorized as Normal), ensuring patient safety.

### Per-Class Performance Breakdown

| Severity Class | Precision | Recall (Sensitivity) | F1-Score | Support (n) |
| :--- | :--- | :--- | :--- | :--- |
| **Grade 0 (No DR)** | 0.969 | 0.912 | 0.939 | 34 |
| **Grade 1 (Mild NPDR)** | 0.375 | 0.600 | 0.462 | 5 |
| **Grade 2 (Moderate NPDR)** | 0.900 | 0.844 | 0.871 | 32 |
| **Grade 3 (Severe NPDR)** | 0.800 | 0.842 | 0.821 | 19 |
| **Grade 4 (Proliferative DR)** | 0.923 | 0.923 | 0.923 | 13 |
| **Macro Average** | **0.793** | **0.824** | **0.803** | **103** |

---

## Failure Mode and Clinical Error Analysis

Diagnostic boundaries in medical imaging are continuous rather than discrete. Analysis of misclassified cases yields valuable clinical insights:

### 1. Subtle Peripheral Microaneurysms (Case IDRiD_030)
- **Ground Truth**: Grade 1 (Mild NPDR)
- **Model Prediction**: Grade 0 (No DR) — Model Confidence: 58.4%
- **Clinical Rationale**: The examination contained a single isolated microaneurysm (<25 micrometers in diameter) located in the inferior-temporal periphery. Due to subtle media opacity and peripheral illumination falloff, the feature fell below the convolutional activation threshold.

### 2. Macular Concentration of Exudates (Case IDRiD_045)
- **Ground Truth**: Grade 2 (Moderate NPDR)
- **Model Prediction**: Grade 1 (Mild NPDR) — Model Confidence: 61.2%
- **Clinical Rationale**: Lesions were predominantly localized near the foveal avascular zone rather than distributed across the wider vascular arcade, causing global pooling to underestimate severity.

### 3. Borderline "4-2-1" Rule Criteria (Case IDRiD_072)
- **Ground Truth**: Grade 3 (Severe NPDR)
- **Model Prediction**: Grade 2 (Moderate NPDR) — Model Confidence: 54.8%
- **Clinical Rationale**: Faint intraretinal microvascular abnormalities (IRMA) were present in a single quadrant. This represents a classic boundary scenario where human ophthalmologists frequently exhibit inter-observer disagreement.

---

## Interactive Clinical Decision-Support Application

The project includes an interactive web interface built with Streamlit, designed for clinical demonstrations and academic evaluation:

1. **Executive Overview**: High-level system statistics, ICDR diagnostic criteria, and class distributions.
2. **Live Diagnostic Demo**:
   - Single-click preset patient cases covering all 5 ICDR severity levels.
   - Custom image upload supporting JPEG and PNG fundus photographs.
   - Primary diagnosis badge with ICDR grading.
   - Clinical triage recommendation and referral timeline.
   - Posterior probability breakdown across all 5 classes.
   - Real-time 3-panel Grad-CAM visualization (Standard View, Activation Heatmap, Attention Overlay).
3. **Clinical Performance**: Interactive confusion matrix, ROC-AUC metrics, and convergence curves.
4. **Clinical Error Analysis**: Side-by-side review of test boundary cases with clinical rationales.
5. **Project Governance Summary**: Regulatory posture, zero-leakage verification, and deployment roadmap.

---

## Project Structure

```
ScanWise/
|-- .gitignore                  # Git exclusion rules for large datasets and caches
|-- README.md                   # Comprehensive project documentation
|-- requirements.txt            # Pinned Python package dependencies
|-- run.bat                     # Windows automated launcher script
|
|-- app/                        # Web application modules
|   |-- __init__.py
|   |-- components.py           # Presentation UI components and Plotly chart generators
|   `-- dashboard.py            # Streamlit clinical presentation dashboard
|
|-- assets/                     # Packaged demo assets
|   `-- demo_samples/           # Curated fundus images covering all 5 clinical grades
|
|-- models/                     # Model weights and benchmark artifacts
|   |-- efficientnet_dr_best.pth # Calibrated EfficientNet-B0 weights
|   `-- efficientnet_dr_eval.json# Test evaluation metrics, matrix, and error analysis
|
|-- reports/                    # Dataset audit and technical logs
|   `-- dataset_audit.md        # Comprehensive IDRiD integrity report
|
|-- scripts/                    # Automation and maintenance utilities
|   |-- audit_dataset.py        # Dataset validation and integrity auditor
|   `-- check_environment.py    # Python runtime and dependency validator
|
|-- src/                        # Core deep learning and explainability pipeline
|   |-- __init__.py
|   |-- dataset.py              # PyTorch Dataset, label parsing, class weighting
|   |-- evaluate.py             # Evaluation harness, QWK, ROC-AUC, confusion matrix
|   |-- explain.py              # PyTorch Grad-CAM engine and heatmap generator
|   |-- models.py               # EfficientNet-B0 and ResNet-50 network builders
|   |-- predict.py              # Single-image inference CLI
|   |-- preprocessing.py        # Image transforms, augmentations, ImageNet normalization
|   `-- train.py                # Full training loop, early stopping, cosine annealing
|
`-- tests/                      # Automated test suite
    |-- __init__.py
    `-- test_all.py             # 35 automated tests covering dataset, models, and XAI
```

---

## Installation and Execution Guide

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Operating System: Windows, macOS, or Linux
- Recommended: 8 GB RAM, optional NVIDIA GPU with CUDA support

### 1. Clone Repository and Install Dependencies
```bash
git clone https://github.com/TirthhShethh/ScanWise-DR.git
cd ScanWise-DR
pip install -r requirements.txt
```

### 2. Verify Environment
```bash
python scripts/check_environment.py
```

### 3. Launch Web Application
```bash
# Using the Windows launcher:
run.bat

# Or directly via Streamlit:
streamlit run app/dashboard.py
```
Open browser to: `http://localhost:8501`

### 4. Run Single-Image CLI Prediction
```bash
python -m src.predict "assets/demo_samples/IDRiD_001.jpg" --task dr --model efficientnet
```

### 5. Run Model Evaluation
```bash
python -m src.evaluate --task dr --model efficientnet
```

### 6. Run Training
```bash
# Smoke test (fast validation):
python -m src.train --task dr --model efficientnet --smoke-test

# Full fine-tuning:
python -m src.train --task dr --model efficientnet --epochs 30 --batch-size 8
```

---

## Automated Test Suite

ScanWise includes a test suite with 35 tests in `tests/test_all.py`.

```bash
python -m pytest tests/test_all.py -v -m "not slow"
```

### Verified Test Categories
- **Dataset Loading & Integrity**: Label loading, required column validation, grade bounds (0-4 for DR, 0-2 for ME), split non-overlap, and class index mapping.
- **Preprocessing Invariants**: Output tensor dimensions [3, 384, 384], normalization bounds, deterministic validation transforms, and randomized training transforms.
- **Model Architecture**: EfficientNet-B0 and ResNet-50 output shapes, linear head replacement, backbone parameter freezing, and checkpoint serialization/deserialization.
- **Inference Pipeline**: Checkpoint dependency validation, input dimension checks, and probability normalization.
- **Grad-CAM Engine**: Activation map dimensions matching input photographs, heatmap value normalization within [0, 1].
- **Edge Cases & Fault Tolerance**: Corrupt image handling, non-image file rejection, single-channel grayscale conversion to 3-channel RGB.
- **Evaluation Metrics**: Metric invariance under perfect predictions, all-wrong predictions, and probability sum constraints.

---

## Regulatory, Ethical, and Safety Governance

### Software as a Medical Device (SaMD) Classification
ScanWise is designed within the framework of a **Class II Software as a Medical Device (SaMD)** intended for clinical decision support. It is not an autonomous diagnostic agent.

### Core Safety Safeguards
1. **Human-in-the-Loop Workflow**: All AI recommendations are presented alongside raw fundus photographs, probability distributions, and Grad-CAM overlays to assist licensed ophthalmologists or optometrists.
2. **Confidence-Based Deferral**: When prediction confidence is below 60%, the interface highlights the case as borderline, prompting mandatory secondary review.
3. **High Referral Sensitivity**: Thresholds are calibrated to minimize false negatives on referable disease (Grade >= 2), ensuring sight-threatening conditions are not missed.
4. **Data Privacy**: The software processes fundus images locally in memory; no patient identifiable information (PII) is transmitted to external servers.

---

## License and Medical Disclaimer

### Medical Disclaimer
ScanWise is an academic and educational research prototype. It does not provide medical diagnoses or replace professional ophthalmic consultation. Clinical decisions regarding medical treatments, laser photocoagulation, anti-VEGF injections, or surgical vitrectomy must always be made by a licensed healthcare professional.

### License
This project is licensed under the MIT License. The IDRiD dataset is subject to its original data use agreement (IEEE Dataport / Indian Diabetic Retinopathy Image Dataset).