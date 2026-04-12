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
* **Annotation:** Fine-grained polygon annotation using **CVAT (COCO 1.0 format)** with AI-assisted segmentation (SAM).
* **Augmentation:** **Albumentations** library for joint spatial transforms (Horizontal Flip, Rotation, Elastic Transform) to maintain mask-image integrity.
* **Model:** **DeepLabV3** with a ResNet-101 backbone.
  * *Strategy:* Freeze the backbone weights and fine-tune only the classifier head to adapt to the 3-class egocentric domain.

## 5. Sample Data (Hard Examples)
> *Note: These samples represent the "Hard Examples" used to train the model's robustness against backlighting and severe occlusion.*

| Firm Grip & Severe Occlusion | Harsh Backlighting | Shadow & Low Light |
| :---: | :---: | :---: |
| <img src="https://github.com/user-attachments/assets/375232a2-1869-46cc-9de6-388256f8e43e" alt="Firm Grip Sample" width="250"> | <img src="https://github.com/user-attachments/assets/2c8a5a23-9cd4-412b-8ba3-3155ffd89acf" alt="Sample 2" width="250"> | <img src="https://github.com/user-attachments/assets/282cac39-1f39-44f3-9872-7d33057dd1da" alt="Sample 3" width="250"> |
| *[Firm Grip of a Bottle]* | *[A Bottle with Strong Backlight]* | *[A Bottle with Low Light]* |

---
## 6. How to Run (TBA)
*Requirements: PyTorch, Torchvision, Albumentations, OpenCV*
```bash
# Installation and execution instructions will be updated upon completion of the training script.
