# Satellite Image Change Detection using Semi-Supervised Learning and U-Net

A **multi-approach satellite image change detection framework** built using **No-Ground-Truth Learning**, **Semi-Supervised Learning (SSL)**, and **Deep Learning (U-Net)** on the **LEVIR-CD dataset**.

The objective is to detect structural changes between **bi-temporal satellite image pairs** captured at different timestamps.

This project explores and compares **three different approaches** for change detection.

---

## Open in Google Colab

### 1. Change Detection Without Ground Truth

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SivaRohithReddy07/Satellite-Image-Change-Detection/blob/main/Change_Detection_With_No_GT.ipynb)

A **label-free self-training framework** where pseudo-labels are generated automatically using difference images without requiring manual annotations.

**Results (Average of 5 Test Images)**

| Metric | Value |
|--------|--------|
| Accuracy | **0.8019** |
| F1 Score | **0.0947** |
| IoU | **0.0523** |

---

### 2. Semi-Supervised Change Detection using Limited Labels

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SivaRohithReddy07/Satellite-Image-Change-Detection/blob/main/Change_Detection_Using_Some_GT.ipynb)

Uses a **small labeled subset** together with a large unlabeled pool through **iterative pseudo-label refinement** using Self-Training SSL.

**Results (Average of 5 Test Images)**

| Metric | Value |
|--------|--------|
| Accuracy | **0.8991** |
| F1 Score | **0.3054** |
| IoU | **0.1886** |

---

### 3. U-Net Model Training

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SivaRohithReddy07/Satellite-Image-Change-Detection/blob/main/Change_Detection_U-Net_Training.ipynb)

Implements a **custom 6-channel U-Net architecture** using **patch-based training (1024×1024 → 256×256 patches)** and **BCE + Dice hybrid loss**.

**Final Test Results**

| Metric | Value |
|--------|--------|
| Precision | **0.9068** |
| Recall | **0.8459** |
| F1 Score | **0.8753** |
| IoU | **0.7782** |

---

### 4. U-Net Visualization and Inference

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SivaRohithReddy07/Satellite-Image-Change-Detection/blob/main/Change_Detection_U-Net_Visualization.ipynb)

Loads trained model weights and performs **full-image reconstruction** by stitching predictions from **16 non-overlapping patches**.

---

### 5. Model Comparison

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SivaRohithReddy07/Satellite-Image-Change-Detection/blob/main/Comparison.ipynb)

Compares predictions from different approaches and analyzes qualitative performance differences.

---

## Approaches Implemented

### 1. No Ground Truth (No-GT) Change Detection

A label-free learning strategy where:

- Difference images are computed between satellite image pairs
- High-confidence pixels are selected automatically
- Pseudo-labels are generated without manual annotation
- An **MLP classifier** is trained using **Self-Training SSL**

---

### 2. Semi-Supervised Learning using Limited Ground Truth

A hybrid framework where:

- Only a **small subset of labeled pixels** is used
- Remaining data is treated as unlabeled
- High-confidence pseudo-labels are progressively added
- **Class balancing** is used to improve recall on change regions

---

### 3. U-Net Based Change Detection

A fully supervised **deep learning segmentation framework** using a **custom PyTorch U-Net architecture**.

#### Training Strategy

- Native **1024×1024 resolution**
- Split into **16 patches of 256×256**
- 6-channel input representation (**Image A + Image B**)
- Patch-wise training for memory efficiency
- Full-resolution reconstruction during inference

---

## Dataset

This project uses the **LEVIR-CD Dataset**, a benchmark dataset for satellite image change detection.

**Download Dataset:**  
https://www.kaggle.com/datasets/mdrifaturrahman33/levir-cd

### Expected Dataset Structure

```text
LEVIR CD/
├── train/
│   ├── A/
│   ├── B/
│   └── label/
│
├── val/
│   ├── A/
│   ├── B/
│   └── label/
│
└── test/
    ├── A/
    ├── B/
    └── label/
```

Place the dataset inside **Google Drive** before running the notebooks.

---

## Model

The trained U-Net model weights are publicly available.

Download the model:

[best_model.pth](https://drive.google.com/file/d/17QqFhsuRQjvEms_ObvoypaTDAAfG1Phg/view?usp=sharing)

After downloading, place the model file inside:

```text
/content/drive/My Drive/Change Detection/
```

This matches the notebook path configuration and allows inference notebooks to run without modifying paths.

---
## Project Structure

```text
Satellite-Image-Change-Detection/
│── Change_Detection_With_No_GT.ipynb
│── Change_Detection_Using_Some_GT.ipynb
│── Change_Detection_U-Net_Training.ipynb
│── Change_Detection_U-Net_Visualization.ipynb
│── Comparison.ipynb
│
├── Output Images/
│   ├── A.png
│   ├── B.png
│   ├── GT.png
│   ├── No_GT_Predicted.png
│   ├── Some_GT_Predicted.png
│   └── U-Net_Predicted.png
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Requirements

Google Colab already contains most required dependencies.

For local execution:

```bash
pip install -r requirements.txt
```

Recommended environment:

- **Google Colab**
- Python 3.x
- GPU runtime for U-Net training

---

## Key Features

- No-Ground-Truth Learning
- Semi-Supervised Learning (SSL)
- U-Net Based Deep Learning
- Patch-Based Training & Reconstruction
- Self-Training with Pseudo Labels
- IoU / F1 Based Evaluation
- Google Colab Ready

---

## Notes

- The **LEVIR-CD dataset is not included** due to repository size limitations.
- `best_model.pth` is not included due to GitHub file size limitations.
- Google Colab is recommended for execution.

---

## License

This project is licensed under the **MIT License**.