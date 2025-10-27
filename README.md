# 🧠 Resolving Autism Spectrum Disorder (ASD) using fMRI and MLP

This repository contains the implementation of our course project:
**“Resolving Autism Spectrum Disorder through Brain Topologies using fMRI and Multilayer Perceptron (MLP)”**

Developed under **Course Project – Image and Video Processing (Group B4)**, IIIT Allahabad.

---

## 📄 Overview

The goal of this project is to automatically detect **Autism Spectrum Disorder (ASD)** using resting-state **fMRI** data from the **ABIDE** dataset.
The system computes connectivity features between brain regions (ROIs) and classifies subjects as **ASD** or **Typical Control (TC)** using a **Multilayer Perceptron (MLP)**.

This project adapts and simplifies the multi-atlas ensemble concept proposed in **MADE-for-ASD (Liu et al., 2024)** to a single, interpretable MLP-based workflow.

---

## 📊 Dataset

* **Dataset Used:** ABIDE I (Autism Brain Imaging Data Exchange)
* **Data Type:** Resting-state fMRI
* **Subjects:** ≈ 1000 from 17 global sites
* **Demographics:** Age, Gender, and Site details included
* **Phenotype Files:**

  ```
  data/phenotypes/
      Phenotypic_V1_0b.csv
      Phenotypic_V1_0b_preprocessed1.csv
  ```

---

## ⚙️ Environment Setup

> Python 3.8 + recommended (Linux / Mac / Colab supported)

```bash
!pip install -U pip
!pip install -r requirements.txt
!pip install nilearn networkx nibabel scikit-learn matplotlib seaborn h5py
```

---

## 🧩 Workflow

### 🪄 Step 1: Data Download and Preparation

Download the ABIDE dataset and generate connectivity matrices:

```bash
!python download_abide.py --pipeline=cpac --strategy=filt_global
!python prepare_data.py --folds=3 --whole cc200 aal ez
```

This produces `.hdf5` data files containing:

```
Subject_ID, Connectivity_Matrix, Demographic_Info
```

---

### ⚙️ Step 2: Model Training

Train the **MLP classifier** for ASD vs TC prediction:

```bash
!python nn.py --whole cc200 aal ez
```

**Model Details**

| Layer   | Units | Activation | Notes       |
| ------- | ----- | ---------- | ----------- |
| Dense 1 | 128   | ReLU       | Dropout 0.3 |
| Dense 2 | 64    | ReLU       | BatchNorm   |
| Dense 3 | 32    | ReLU       | –           |
| Output  | 2     | Softmax    | ASD / TC    |

Optimizer = Adam (lr = 0.001), Loss = CrossEntropyLoss

---

### 🧠 Step 3: Model Evaluation

```bash
!python nn_evaluate.py
```

Outputs: Accuracy, ROC-AUC, Confusion Matrix, Classification Report, and plots.

---

## 🧠 Running on Test Data

To run quickly on test data:

```python
# In B4/prepare_data.py
pheno = pd.read_csv("data/phenotypes/test_phenotype.csv")
```

This reduces runtime significantly while keeping the pipeline identical.

---

## 📈 Results Summary

| Model                 | Dataset | Accuracy (%)  | ROC-AUC  |
| :-------------------- | :------ | :------------ | :------- |
| MADE-for-ASD (2024)   | ABIDE   | 96.4 (subset) | 0.97     |
| Autoencoder + MLP     | ABIDE   | 74.8          | 0.75     |
| **Proposed MLP (B4)** | ABIDE   | **75.2**      | **0.76** |

---

## 📊 Visualizations

* Confusion Matrix
* ROC Curve
* Feature Importance (PCA components)
* Accuracy vs Epoch plot

---

## 🧱 Folder Structure

```
B4/
├── data/
│   ├── phenotypes/
│   │   ├── Phenotypic_V1_0b.csv
│   │   ├── Phenotypic_V1_0b_preprocessed1.csv
│   │   └── test_phenotype.csv
│   ├── cc200/  aal/  ez/
├── models/
│   └── mlp_model.pt
├── results/
│   ├── accuracy_curve.png
│   ├── confusion_matrix.png
│   └── roc_curve.png
├── prepare_data.py
├── nn.py
├── nn_evaluate.py
├── download_abide.py
└── README.md
```

---

## 🔬 Citation

> X. Liu, M. Hasan, T. Gedeon, and M. Hossain,
> “MADE-for-ASD: A Multi-Atlas Deep Ensemble Network for Diagnosing Autism Spectrum Disorder,”
> *Computers in Biology and Medicine*, 182, 109083, 2024.

---

## 🧩 Acknowledgment

This project adapts the structure of **MADE-for-ASD (2024)** into a simplified educational MLP pipeline.
Developed by **Group B4 – IIIT Allahabad**
under the guidance of **Prof. Anupam Shukla**.



