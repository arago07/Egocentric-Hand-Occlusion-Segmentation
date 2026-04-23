# Overcoming Domain Gap in Egocentric Vision Segmentation

[![Project Status: WIP](https://img.shields.io/badge/Project%20Status-WIP-yellow.svg)](#)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c.svg)](https://pytorch.org/)

## 1. Problem Statement
Current state-of-the-art Semantic Segmentation models (e.g., DeepLabV3) pre-trained on large-scale exocentric (third-person) datasets like MS COCO or Pascal VOC often suffer from a significant **Domain Gap** when deployed in egocentric (first-person) environments.

The primary challenges in egocentric vision include:
* **Severe Hand Occlusion:** The user's hand often obstructs the majority of the target object's surface, distorting its canonical shape.
* **Extreme Perspective & Distortion:** Objects appear with high radial distortion and varying scales due to proximity to the camera lens.
* **Dynamic Lighting Conditions:** 1st-person views are subject to harsh backlighting, shadows, and varying color temperatures that differ from standard dataset distributions.
* **Object Transparency:** Transparent objects (e.g., glass cups) lack distinct visual features, as background pixels pass through the object, making feature extraction particularly difficult.

This project aims to bridge this gap through a controlled, domain-specific fine-tuning approach.

## 2. Hypothesis
By introducing a small-scale (N=60) but highly controlled egocentric dataset that specifically targets **"Hand-Object Interaction"** and diverse lighting conditions, we can improve the segmentation mIoU of a pre-trained model through targeted fine-tuning. We further investigate whether a **small number of carefully selected "hard examples"** can compensate for unseen lighting distributions more efficiently than large-scale data augmentation.

## 3. Experimental Design

### 3.1. Dataset Specification
* **Total Samples:** 60 Egocentric Images (baseline) → expanded in subsequent experiments.
* **Target Classes:** `hand`, `bottle`, `cup` (Mapping to COCO-equivalent classes).
* **Interaction Distribution (by visual inspection):**
  * **Firm Grip:** Severe occlusion where the hand covers a large portion of the object.
  * **Light Touch:** Partial occlusion to learn edge boundaries.
  * **No Interaction:** Baseline egocentric view.

### 3.2. Data Split by Lighting Session
The data is split by **"Lighting Sessions"** to evaluate generalization across unseen illumination conditions. Images taken in the same session are kept within the same split to prevent scene-level leakage.

| Split | Images | Condition (Illumination) | Purpose |
| :--- | :--- | :--- | :--- |
| **Train** | 40 | Natural, Fluorescent, Desk Lamp (Color A) | Primary Feature Learning |
| **Val** | 10 | Desk Lamp (Color B) | Intermediate Evaluation |
| **Test** | 10 | **Strong Backlight** | **Unseen Environment Evaluation** |

> **Note:** The Val and Test splits are drawn from **different lighting sessions** than Train. This is an intentional domain gap setup to evaluate generalization to unseen illumination — not a conventional random split.

## 4. Methodology (Pipeline)

* **Annotation:** Fine-grained polygon annotation using **CVAT (COCO 1.0 format)**.
* **Augmentation:** **Albumentations** library for joint image-mask transforms (Resize, HorizontalFlip, RandomRotate90, ColorJitter).
* **Model:** **DeepLabV3** with a **ResNet-50** backbone (PyTorch, pretrained on COCO).
* **Classes (4):** `Background (0)`, `Hand (1)`, `Bottle (2)`, `Cup (3)`.
* **Hyperparameters:**
  * Optimizer: Adam (lr = 0.001)
  * Loss Function: CrossEntropyLoss
  * Batch Size: 4
  * Epochs: 5

## 5. Experimental Results

### 📊 5.1. Quantitative Results (mIoU on the respective validation set)

Evaluation uses **pixel-level intersection/union accumulation** across the validation set (PASCAL VOC–style), which is more stable than per-image IoU averaging for small datasets.

| Class | Exp 1: Baseline (40 Imgs) | Exp 2: Hard Example Mining (43 Imgs) |
| :--- | :---: | :---: |
| **Background** | 83.65% | 68.72% |
| **Hand** | 47.62% | 28.41% |
| **Bottle** | 50.26% | 47.08% |
| **Cup** | N/A* | N/A* |
| **Mean IoU (valid classes)** | **45.38%** | **36.05%** |

> **\*Cup IoU is reported as N/A due to insufficient sample size**: the Cup class appeared in only 1–2 images within each validation set, which is statistically unreliable for mIoU reporting. This limitation motivates the subsequent expansion of cup-specific data in later experiments.
>
> **⚠ Important caveat on Exp 1 vs Exp 2 comparison:** The validation sets used for Exp 1 and Exp 2 are **not identical** (Exp 1 evaluated on a yellow desk lamp session with 10 images; Exp 2 evaluated on a subset of 7 images due to a revised split for introducing hard examples). Therefore, the numerical difference between Exp 1 and Exp 2 in the table above should **not be interpreted as a direct performance comparison**. Establishing a unified evaluation protocol is part of ongoing work.

### 🖼️ 5.2. Qualitative Results: Visual Comparison (Backlight Case)

| **Condition** | **Exp 1: Baseline (Failure)** | **Exp 2: Hard Example (Improved)** |
| :---: | :---: | :---: |
| **Extreme Backlight** | <img src="images/results/experiment1_failed_backlight.png" width="350" height="350" style="display:block; object-fit: cover; object-position: center;"> | <img src="images/results/experiment2_successful_backlight.png" width="350" height="350" style="display:block; object-fit: cover; object-position: center;"> |
| **Observation** | Model failed to detect the hand, relying only on color cues. | Model successfully captured the **hand silhouette** despite minimal color information. |

While the quantitative comparison is limited (see caveat above), the qualitative improvement in backlight scenarios is visually unambiguous and reproducible across multiple test images.

## 6. Key Insights & Limitations

### 💡 6.1. Data Efficiency through Edge Case Targeting
Adding just **3 targeted backlight images** qualitatively resolved the severe failure mode observed in Exp 1. This suggests that identifying and targeting edge cases may be a **data-efficient strategy** for enhancing robustness, though rigorous quantitative validation requires a unified evaluation protocol.

### ⚠️ 6.2. Evaluation Protocol Limitations Identified
During experiment review, the following methodological issues were identified and are being addressed in subsequent experiments:

* **Insufficient Cup samples in validation:** 1–2 samples is too small for meaningful mIoU computation. Additional cup-specific data collection is underway.
* **Non-matching validation sets between Exp 1 and Exp 2:** Direct numerical comparison between the two experiments is not valid. Future experiments will adopt a fixed test set across all conditions.
* **Per-batch vs. dataset-level IoU:** Earlier versions computed IoU on a single batch; the current reported values use proper dataset-level pixel accumulation.

### ⚠️ 6.3. Transparent Object Challenge
Transparent cups allow background pixels to pass through, providing minimal color or texture cues. Combined with severe hand occlusion in egocentric views, this remains the primary open challenge of the project.

## 7. Next Steps

1. **Expanded Cup Dataset:** Systematic collection of additional transparent cup images varying across form factor (cylindrical, mug-shaped, stemmed, pint), transparency level (empty/liquid-filled), occlusion pattern, and lighting conditions.
2. **Unified Evaluation Protocol:** Fixed held-out test set used consistently across all experimental conditions to enable direct comparison.
3. **Hand-Free Diagnostic Set:** A subset of cup-only images (without hand) to diagnose potential spurious correlation between hand and cup predictions.
4. **Strategic Loss Design:** Evaluating class-weighted loss and augmentation strategies targeting transparent object boundaries.

---

## 8. How to Run

**1. Environment Setup**
```bash
pip install torch torchvision torchaudio albumentations opencv-python pycocotools
```

**2. Execution**

The pipeline is implemented in `Egocentric_Hand_Segmentation_DeepLabV3.ipynb`, structured for top-to-bottom execution. Experiments 3 and beyond (in-progress) are included in the same notebook but should be considered work-in-progress until the updated evaluation protocol is finalized.
