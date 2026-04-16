# Overcoming Domain Gap in Egocentric Vision Segmentation

[![Project Status: WIP](https://img.shields.io/badge/Project%20Status-WIP-yellow.svg)](#)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c.svg)](https://pytorch.org/)

## 1. Problem Statement
Current state-of-the-art Semantic Segmentation models (e.g., DeepLabV3) pre-trained on large-scale exocentric (third-person) datasets like MS COCO or Pascal VOC often suffer from a significant **Domain Gap** when deployed in egocentric (first-person) environments.

The primary challenges in egocentric vision include:
* **Severe Hand Occlusion:** The user's hand often obstructs the majority of the target object's surface, distorting its canonical shape.
* **Extreme Perspective & Distortion:** Objects appear with high radial distortion and varying scales due to proximity to the camera lens.
* **Dynamic Lighting Conditions:** 1st-person views are subject to harsh backlighting, shadows, and varying color temperatures that differ from standard dataset distributions.

This project aims to bridge this gap through a controlled, domain-specific fine-tuning approach.

## 2. Hypothesis
By introducing a small-scale (N=60) but highly controlled egocentric dataset that specifically targets **"Hand-Object Interaction"** and **"Illumination Invariance,"** we can significantly improve the segmentation mIoU of a pre-trained model. Furthermore, we hypothesize that the model can achieve **Zero-Shot Generalization** in completely unseen lighting environments if the training data is strategically sampled across diverse spectral distributions.

## 3. Experimental Design
To ensure rigorous scientific validation, the dataset and experiment are designed as follows:

### 3.1. Dataset Specification
* **Total Samples:** 60 High-Resolution Egocentric Images.
* **Target Classes:** `hand`, `bottle`, `cup` (Mapping to COCO-equivalent classes).
* **Interaction Ratios:**
  * **Firm Grip (70%):** Severe occlusion where the hand covers >50% of the object.
  * **Light Touch (15%):** Partial occlusion to learn edge boundaries.
  * **No Interaction (15%):** Baseline egocentric view to prevent spurious correlation.

### 3.2. Strategic Data Split (4:1:1)
The data is split by **"Lighting Sessions"** to prevent Scene Leakage and evaluate generalization.

| Split | Images | Condition (Illumination) | Purpose |
| :--- | :--- | :--- | :--- |
| **Train** | 40 | Natural, Fluorescent, Backlight, Desk Lamp (Color A) | Primary Feature Learning |
| **Val** | 10 | Desk Lamp (Color B - Low Intensity) | Early Stopping & Hyperparameter Tuning |
| **Test** | 10 | **Desk Lamp (Color B - High Intensity)** | **Zero-Shot Evaluation (Unseen Environment)** |

### 3.3. Scene Leakage Prevention
Images taken in the same session (same background/lighting) are never split across Train and Test sets. This forces the model to learn the semantic features of the objects rather than memorizing the background textures.

## 4. Methodology (Pipeline)

* **Annotation:** Fine-grained polygon annotation using **CVAT (COCO 1.0 format)**.
* **Augmentation:** **Albumentations** library for joint spatial transforms (ColorJitter for lighting invariance, Horizontal Flip, etc.) to maintain mask-image integrity while simulating extreme environments.
* **Model:** **DeepLabV3** with a **ResNet-50** backbone (PyTorch).
* **Classes (4):** `Background (0)`, `Hand (1)`, `Bottle (2)`, `Cup (3)`.
* **Hyperparameters:**
  * Optimizer: Adam (lr = 0.001)
  * Loss Function: CrossEntropyLoss
  * Batch Size: 4
  * Epochs: 5 (Initial setup)

## 5. Experimental Results: The Power of Hard Example Mining

Initially, we hypothesized that the model could achieve zero-shot generalization in unseen lighting. However, the baseline model severely struggled under extreme backlight. By injecting just **3 Hard Example images**, we significantly enhanced the model's robustness.

### 📊 5.1. Quantitative Results (mIoU)

| Class | Exp 1: Baseline (40 Imgs) | Exp 2: Hard Example Mining (43 Imgs) | Improvement |
| :--- | :---: | :---: | :---: |
| **Background** | 83.66% | 83.25% | - |
| **Hand** | 42.64% | **55.24%** | **+12.60%p** |
| **Bottle** | 45.11% | **47.96%** | +2.85%p |
| **Cup** | 0.05% | 0.00% | - |
| **🏆 Mean IoU** | **42.87%** | **46.61%** | **🚀 +3.74%p** |

### 🖼️ 5.2. Qualitative Results: Visual Comparison (Backlight Case)

| **Condition** | **Exp 1: Baseline (Failure)** | **Exp 2: Hard Example (Success)** |
| :---: | :---: | :---: |
| **Extreme Backlight** | <img src="images/results/experiment1_failed_backlight.png" width="350" height="350" style="display:block; object-fit: cover; object-position: center;"> | <img src="images/results/experiment2_successful_backlight.png" width="350" height="350" style="display:block; object-fit: cover; object-position: center;"> |
| **Observation** | Model failed to detect the hand, relying only on color cues. | Model successfully captured the **hand silhouette** despite zero color info. |

## 6. Key Insights & Limitations

### 💡 6.1. Data Efficiency through Edge Case Targeting
Instead of blindly increasing dataset size, adding **3 targeted images** solved a domain failure. This proves that targeting Edge Cases is a highly **data-efficient strategy** for enhancing robustness.

### ⚠️ 6.2. Challenge: Intrinsic Transparency
* **Observation:** 'Cup' class yielded **0% mIoU**.
* **Analysis:** Unlike bottles, **transparent cups** allow background pixels to pass through. The lack of distinct visual features, combined with severe occlusion, remains a critical challenge.

## 7. Next Steps: Strategic Troubleshooting

Before advancing to broader theoretical architectures, a troubleshooting phase is planned to overcome the current 'Cup' recognition failure:
1. **Data Augmentation:** Utilizing `ColorJitter` or `MixUp` to force the model to focus on the subtle edge boundaries of transparent objects.
2. **Loss Weighting:** Applying `Focal Loss` or class-specific weights to increase the penalty for misclassifying the 'Cup' pixels, forcing the model to prioritize this underrepresented and challenging class.

---
## 8. How to Run

**1. Environment Setup**
```bash
pip install torch torchvision torchaudio albumentations opencv-python pycocotools
```

**2. Execution**

The entire pipeline (Data loading, Model initialization, Exp 1 Training, Exp 2 Hard Example Mining, and mIoU Evaluation) is fully implemented in the provided Colab Notebook (Egocentric_Hand_Segmentation_DeepLabV3.ipynb). It is structured for a seamless top-to-bottom execution.
