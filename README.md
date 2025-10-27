
# 🧠 Resolving Autism Spectrum Disorder (ASD) using fMRI and MLP  

This repository contains the implementation of our course project:  
**"Resolving Autism Spectrum Disorder through Brain Topologies using fMRI and Multilayer Perceptron (MLP)"**  

Developed under **Course Project – Image and Video Processing (Group B4)**, IIIT Allahabad.

---

## 📄 Overview  

The goal of this project is to automatically detect **Autism Spectrum Disorder (ASD)** using resting-state **fMRI data** from the **ABIDE** dataset.  
The system computes **connectivity features between brain regions (ROIs)** and classifies subjects as **ASD** or **Typical Control (TC)** using a **Multilayer Perceptron (MLP)**.  

This project adapts and simplifies the multi-atlas ensemble concept proposed in  
**MADE-for-ASD (Liu et al., 2024)** to a single, interpretable MLP-based workflow.

---

## 📊 Dataset  

- **Dataset Used:** ABIDE I (Autism Brain Imaging Data Exchange)  
- **Data Type:** Resting-state fMRI  
- **Subjects:** ~1000 from 17 global sites  
- **Demographics:** Age, Gender, and Site details included  

**Phenotype Files:**  
```

data/phenotypes/
Phenotypic_V1_0b.csv
Phenotypic_V1_0b_preprocessed1.csv

````

---

## ⚙️ Environment Setup  

> **Python 3.8+ recommended** (Linux / Mac preferred)

Install all dependencies:
```bash
!pip install -U pip
!pip install -r requirements.txt
!pip install nilearn networkx nibabel scikit-learn matplotlib seaborn h5py
````

Required libraries:

```python
import os, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, classification_report
import networkx as nx
import h5py
```

---

## 🧩 Workflow

Run **download_abide.py** to download the raw data:

```bash
!python download_abide.py --pipeline=cpac --strategy=filt_global
```

To show demographic information of ABIDE I:

```bash
!python pheno_info.py
```

---

### 🧠 Step 1: Data Preparation

The `prepare_data.py` script computes correlation matrices between Regions of Interest (ROIs) for each subject.

```bash
!python prepare_data.py --folds=3 --whole cc200 aal ez
```

This produces `.hdf5` data files containing:

```
Subject_ID, Connectivity_Matrix, Demographic_Info
```

---

### ⚙️ Step 2: Model Training

Train the **MLP classifier** for ASD vs. TC prediction:

```bash
!python nn.py --whole cc200 aal ez
```

**Model Details:**

* Input: PCA-reduced brain connectivity features
* Layers: [128 → 64 → 32 → 2]
* Activation: ReLU
* Dropout: 0.3
* Optimizer: Adam
* Loss: CrossEntropyLoss

---

### 📈 Step 3: Model Evaluation

Evaluate trained models on test data:

```bash
!python nn_evaluate.py
```

**Generates:**

* Accuracy and ROC-AUC metrics
* Confusion matrix and ROC plots
* Classification report summary

---

## 🧠 Running on Test Data

If you want to run the model on **test data** for quick execution:

1. Open:

   ```bash
   B4/prepare_data.py
   ```
2. Locate:

   ```python
   pheno = pd.read_csv("data/phenotypes/Phenotypic_V1_0b_preprocessed1.csv")
   ```
3. Replace with:

   ```python
   pheno = pd.read_csv("data/phenotypes/test_phenotype.csv")
   ```
4. Save and rerun the pipeline.

💡 *Running on the full dataset may take several hours — for demo purposes, use the test phenotype file.*

---

## 📊 Results Summary

| Model                 | Dataset | Accuracy (%)  | ROC-AUC  |
| --------------------- | ------- | ------------- | -------- |
| MADE-for-ASD (2024)   | ABIDE   | 96.4 (subset) | 0.97     |
| Autoencoder + MLP     | ABIDE   | 74.8          | 0.75     |
| **Proposed MLP (B4)** | ABIDE   | **75.2**      | **0.76** |

---

## 📈 Visualizations

* Confusion Matrix
* ROC Curve
* Feature Importance (PCA components)
* Training Accuracy vs Epoch plot

---

## 🧱 Folder Structure

```
B4/
│
├── data/
│   ├── phenotypes/
│   │   ├── Phenotypic_V1_0b.csv
│   │   ├── Phenotypic_V1_0b_preprocessed1.csv
│   │   └── test_phenotype.csv
│   ├── cc200/
│   ├── aal/
│   └── ez/
│
├── models/
│   └── mlp_model.pt
│
├── results/
│   ├── accuracy_curve.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│
├── prepare_data.py
├── nn.py
├── nn_evaluate.py
├── Experiments.ipynb
└── README.md
```

---

## 🔬 Citation

If this project is referenced in future work, please cite the base paper:

> X. Liu, M. Hasan, T. Gedeon, and M. Hossain,
> “**MADE-for-ASD: A Multi-Atlas Deep Ensemble Network for Diagnosing Autism Spectrum Disorder**,”
> *Computers in Biology and Medicine*, 182, 109083, 2024.

---

## 🧩 Acknowledgment

This project adapts the structure of **MADE-for-ASD (2024)** to a simplified, educational implementation for course learning purposes.

Developed by **Group B4 – IIIT Allahabad**
under the guidance of **Prof. Anupam Shukla**.

```




