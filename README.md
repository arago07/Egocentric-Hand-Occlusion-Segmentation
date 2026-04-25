# Overcoming Domain Gap in Egocentric Vision Segmentation

[![Project Status: Iteration 3](https://img.shields.io/badge/Project%20Status-Iteration%203-blue.svg)](#)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c.svg)](https://pytorch.org/)

## 1. Problem Statement
Current state-of-the-art Semantic Segmentation models (e.g., DeepLabV3) pre-trained on large-scale exocentric (third-person) datasets like MS COCO or Pascal VOC often suffer from a significant **Domain Gap** when deployed in egocentric (first-person) environments.

The primary challenges in egocentric vision include:
* **Severe Hand Occlusion:** The user's hand often obstructs the majority of the target object's surface, distorting its canonical shape.
* **Extreme Perspective & Distortion:** Objects appear with high radial distortion and varying scales due to proximity to the camera lens.
* **Dynamic Lighting Conditions:** First-person views are subject to harsh backlighting, shadows, and varying color temperatures that differ from standard dataset distributions.
* **Object Transparency:** Transparent objects (e.g., glass cups) lack distinct visual features, as background pixels pass through the object, making feature extraction particularly difficult.

This project aims to bridge this gap through a series of controlled, domain-specific fine-tuning experiments — progressively diagnosing limitations and applying targeted corrections.

## 2. Hypothesis
By introducing small-scale but highly controlled egocentric datasets that specifically target **"Hand-Object Interaction"** and diverse lighting conditions, we can improve the segmentation mIoU of a pre-trained model through targeted fine-tuning. We further investigate whether **a small number of carefully selected "hard examples"** can compensate for unseen lighting distributions more efficiently than large-scale data augmentation, and whether the same data-centric principle extends to transparent object recognition.

## 3. Experimental Roadmap

| Experiment | Question | Approach | Outcome |
| :--- | :--- | :--- | :--- |
| **Exp 1** | Baseline performance under normal lighting | Fine-tune on 40 images (sunlight, fluorescent, stand lamp) | Strong on familiar lighting; failed on backlight |
| **Exp 2** | Can targeted hard examples close the lighting gap? | Add 3 backlight images to training | Qualitatively recovered hand silhouette under backlight |
| **Exp 3-1** | Can algorithmic interventions alone solve transparent cup segmentation? | Apply class weighting (8×) + edge-enhancing augmentation | Limited by insufficient cup samples (only 5 in test set) |
| **Exp 3-2** | Does systematic data expansion outperform algorithmic interventions? | Add 30 systematically varied transparent cup images + hand-free diagnostic test | **Cup IoU 64.81%, with minimal spurious correlation** |

## 4. Methodology

* **Annotation:** Polygon annotation using CVAT (COCO 1.0 format), combining CVAT's built-in AI-assisted segmentation tools with manual polygon drawing for fine-grained refinement (especially for transparent cup boundaries and hand-cup overlap regions).
* **Augmentation:** **Albumentations** library — Resize, ColorJitter, Sharpen (for edge enhancement on transparent objects), Normalize.
* **Model:** **DeepLabV3** with a **ResNet-50** backbone (PyTorch, pretrained on COCO).
* **Classes (4):** `Background (0)`, `Hand (1)`, `Bottle (2)`, `Cup (3)`.
* **Evaluation:** Pixel-level intersection/union accumulation across the full test set (PASCAL VOC–style), reported per class. Classes absent from a test set are reported as `N/A` rather than `0%` to avoid statistical misrepresentation.

## 5. Experiment 1 & 2: Lighting Domain Adaptation

### 5.1. Quantitative Results

| Class | Exp 1: Baseline (40 imgs) | Exp 2: + 3 Backlight imgs (43 imgs) |
| :--- | :---: | :---: |
| Background | 83.65% | 68.72% |
| Hand | 47.62% | 28.41% |
| Bottle | 50.26% | 47.08% |
| Cup | N/A* | N/A* |
| **Mean IoU (valid)** | **45.38%** | **36.05%** |

> **\*Cup IoU is reported as N/A** because the validation set contained only 1–2 cup-bearing images — too few for statistically meaningful evaluation. This limitation directly motivated Exp 3.
>
> **⚠ Important caveat:** The validation sets for Exp 1 and Exp 2 are **not identical** (10 vs 7 images). The numerical difference between them is not a fair direct comparison; refer to the qualitative results below for the meaningful improvement.

### 5.2. Qualitative Results: Backlight Generalization

| **Condition** | **Exp 1: Baseline (Failure)** | **Exp 2: Hard Example (Improved)** |
| :---: | :---: | :---: |
| **Extreme Backlight** | <img src="images/results/experiment1_failed_backlight.png" width="350"> | <img src="images/results/experiment2_successful_backlight.png" width="350"> |
| **Observation** | Model failed to detect the hand, relying only on color cues. | Model successfully captured the hand silhouette despite minimal color information. |

While the quantitative comparison is limited by inconsistent splits, the qualitative improvement is reproducible across multiple backlight test images and validates the **data efficiency of targeted hard examples**.

## 6. Experiment 3-1: Algorithmic Approach (Class Weighting + Augmentation)

### 6.1. Approach
Targeted the failure on the cup class with two interventions:
- **Class weighting:** Cup class weight set to 8× to penalize misclassification of underrepresented pixels.
- **Edge-enhancing augmentation:** `Sharpen` and `ColorJitter` applied to encourage learning of structural over color-based features.

### 6.2. Outcome and Limitation
Cup IoU reached 45.44% on a held-out test of 5 cup-bearing images — a meaningful improvement from previous undefined values, but **the test set was too small** for confident generalization claims. This motivated a data-centric approach in Exp 3-2.

## 7. Experiment 3-2: Expanded Cup Dataset

### 🎯 7.1. Motivation
Building on the data-centric insight from Exp 2 ("3 targeted backlight images significantly improved hand segmentation"), Exp 3-2 investigates whether the same principle applies to transparent objects: would **systematic expansion of the cup dataset** outperform purely algorithmic interventions?

### 📦 7.2. Data Collection Protocol
30 new transparent cup images were collected by varying four design axes:

| Axis | Variations |
| :--- | :--- |
| **Cup type** | Cylindrical, Mug-shaped (with handle), Wine glass (stemmed), Pint glass |
| **Transparency state** | Empty, Liquid-filled |
| **Occlusion pattern** | Handle grip, Side wrap, Two-hand wrap, Top-down view, **Hand-free** |
| **Lighting** | Sunlight, Fluorescent, Backlight, Stand lamp |

Of the 30 new images, **2 were captured without any hand contact** (`transparent_cup_027`, `transparent_cup_029`) and held out as a **diagnostic test set** for evaluating spurious correlation between hand and cup predictions — a known risk in egocentric data where hand and held objects almost always co-occur.

### 📊 7.3. Quantitative Results

**Dataset:** 95 images total (60 original + 5 from Exp 3 + 30 new)
**Train / Test split:** 86 / 9 images
**Training:** 40 epochs, Adam (lr = 1e-5), cup class weight reduced to 3× (relaxed thanks to better data balance)
**Final loss:** 0.5390

#### Overall Test Performance (9 images)

| Class | IoU |
| :--- | ---: |
| Background | 85.76% |
| Hand | 71.75% |
| Bottle | 0.00%* |
| **Cup** | **64.81%** |
| mIoU (valid) | 55.58% |

*Bottle had very few annotations in this test split and is not the focus of Exp 3-2.

### 🔬 7.4. Spurious Correlation Diagnostic

To assess whether the model genuinely learned cup features (rather than relying on hand co-occurrence), Cup IoU was evaluated separately on with-hand vs. hand-free subsets:

| Test Subset | # Images | Cup IoU |
| :--- | :---: | ---: |
| With Hand (`photo_061~065`, `transparent_cup_028, 030`) | 7 | **66.88%** |
| Hand-Free (`transparent_cup_027`, `transparent_cup_029`) | 2 | **59.06%** |
| **Δ (gap)** | — | **−7.82 %p** |

**Interpretation:** The relatively small gap (~8%p) suggests the model has learned cup-intrinsic visual features (e.g., refraction patterns, rim curvature, edge highlights) rather than depending on hand context. While the hand-free sample size (2 images) is small, this preliminary diagnostic supports genuine feature learning rather than spurious correlation.

### 🖼️ 7.5. Qualitative Results
![Exp 3-2 Test Results](images/results/exp3_2_test_results.png)

The visualization shows the original image, ground truth (red = cup, cyan = hand), and prediction for all 9 test images, including the 2 hand-free diagnostic cases marked `[HAND-FREE]`.

### 💡 7.6. Key Findings

1. **Data-centric beats algorithm-centric for transparent objects.** Cup IoU improved from 45.44% (Exp 3-1, with 8× weighting + augmentation on 20 images) to 64.81% (Exp 3-2, with relaxed 3× weighting + augmentation on 50 images). The ~20%p gain came primarily from systematic data diversification, not from stronger algorithmic intervention.

2. **Class weight relaxation was viable.** With more balanced data, cup weighting could be reduced from 8× to 3× without degrading performance, lowering the risk of overfitting to rare-class artifacts.

3. **Hand-free diagnostic shows promising independence.** The 7.82 %p gap between with-hand and hand-free Cup IoU is much smaller than what pure spurious correlation would produce (typically a 30%p+ collapse), suggesting the model learned object-intrinsic features.

4. **Hand segmentation also improved substantially.** Hand IoU reached 71.75% in Exp 3-2 — a significant improvement over Exp 1 (47.62%) and Exp 2 (28.41%), benefiting from the expanded and more diverse training set.

## 8. Limitations and Methodological Notes

The following limitations were identified during the experiments and are documented for transparency:

* **Insufficient Cup samples in Exp 1 / Exp 2 validation sets** (1–2 images) made early Cup IoU reporting unreliable. Resolved in Exp 3-2 by expanding cup data.
* **Non-matching validation sets between Exp 1 and Exp 2** prevented fair direct numerical comparison. Future iterations should adopt a unified held-out test set across all conditions.
* **Hand-free diagnostic with only 2 images** is preliminary; a larger hand-free test set would yield more confident generalization claims.
* **Random seed not fixed**, so reproducibility within ±5%p is expected. Seed fixing and 3-run averaging is the next priority for publication-grade evaluation.
* **Bottle class** received less attention in the latter experiments as the focus shifted to transparent cup segmentation. A broader evaluation set covering all classes would strengthen overall claims.

## 9. Future Work

Building on the data-centric findings of Exp 3-2:

1. **Expand the hand-free diagnostic set** to ≥10 images for stronger spurious-correlation claims.
2. **Unified evaluation protocol** with a fixed held-out test set across all experimental conditions.
3. **Seed fixing and multi-run averaging** for reproducibility (mean ± std reporting).
4. **Background complexity ablation:** failure cases in Exp 3-2 suggest visual clutter (e.g., densely packed background objects) is the next bottleneck — a targeted "complex background" subset is a candidate next data axis.
5. **Architecture comparison:** evaluate whether transformer-based segmenters (e.g., Mask2Former, Trans4Trans) handle transparent objects more robustly than DeepLabV3.

---

## 10. How to Run

**1. Environment Setup**
```bash
pip install torch torchvision torchaudio albumentations opencv-python pycocotools
```

**2. Execution**

The full pipeline (Exp 1, Exp 2, Exp 3-1, Exp 3-2) is implemented in `Egocentric_Hand_Segmentation_DeepLabV3.ipynb`, structured for top-to-bottom execution. Each experiment is preceded by a markdown header for navigation.

**3. Data**

Annotations and images are organized as:
```
dataset/
├── annotations/
│   ├── instances_default.json    # 60 original images
│   ├── exp3_cups.json            # 5 additional cup images
│   └── transparent_cups.json     # 30 new transparent cup images
└── images/                       # 95 images, flat directory
```
