# Threat Detection System Using Machine Learning

**Final Year Project**

A machine-learning pipeline for network threat detection: network flow records are preprocessed, models are trained on labeled data, and predictions are evaluated and demonstrated for binary classification (normal vs. attack traffic).

---

## Introduction

Modern networks face constant intrusion attempts; automated detection helps analysts prioritize real threats. This project implements an **intelligence tier** that learns patterns from labeled network traffic and classifies new flows as **normal** or **attack**.

The work uses the **UNSW-NB15** benchmark dataset, which provides realistic network flow features and labels suitable for training and testing intrusion-detection models. The system focuses on **supervised learning**: models are trained on historical flows with known outcomes, then applied to held-out test data and small interactive demos.

**Scope:** Binary classification (`label`: 0 = normal, 1 = attack). Multi-class attack categories (`attack_cat`) are removed from features and not predicted in the current pipeline.

---

## Methodology

### Dataset and preparation

- **Source:** UNSW-NB15 training and testing CSVs (placed under `data/` as referenced by the scripts; if your files live in `DATA/`, use the same filenames or align the folder name with the paths in the code).
- **Cleaning:** Rows with missing values are dropped; identifier columns such as `id` are removed when present.
- **Features and target:** Features exclude `label` and `attack_cat`. The target is `label`.
- **Categorical variables:** Object-type columns are **one-hot encoded**. Train and test feature matrices are built from a **combined** frame so both splits share identical columns after encoding.
- **Scaling:** **StandardScaler** (zero mean, unit variance) is fit on the training set and applied to the test set to avoid data leakage.

### Models

| Model | Role | Notes |
|--------|------|--------|
| **Random Forest** | Primary classifier | `RandomForestClassifier` with 100 trees; trained on the full preprocessed training set. |
| **Support Vector Machine (SVM)** | Secondary classifier | `SVC` with a **linear** kernel; trained on the **first 10,000** training samples for faster turnaround during development and demos. |

Trained models are persisted with **Joblib** as `rf_model.joblib` and `svm_model.joblib` in the project root.

### Evaluation and demo

- **Evaluation** (`evaluate_system.py`): Loads both models, predicts on the preprocessed test set, and reports **accuracy, precision, recall, and F1**. Saves a bar chart comparing accuracy (`accuracy_comparison.png`).
- **Demo** (`demo_detection.py`): Loads the Random Forest model and prints predictions for a **small sample** of test rows alongside actual labels.
- **Visualizations** (`visualize_results.py`): After training, loads saved models and the same preprocessed test data; writes **confusion matrices** for Random Forest and SVM and a **top-15 feature importance** bar chart for Random Forest.

---

## How to Run the Scripts

### Prerequisites

- **Python 3.10+** (3.11 is used in development)
- **Dependencies:**

Install everything at once (recommended):

```bash
pip install -r requirements.txt
```

Or individually:

```text
pandas
scikit-learn
joblib
matplotlib
seaborn
```

If downloads are slow or time out, install large packages separately with a longer timeout, for example:

```bash
pip install numpy scipy --default-timeout=600
pip install -r requirements.txt --default-timeout=600
```

### Data layout

Ensure the UNSW-NB15 CSVs are available at the paths expected by the scripts:

- `data/UNSW_NB15_training-set.csv`
- `data/UNSW_NB15_testing-set.csv`

The current scripts expect CSVs under **`DATA/`** (see constants at the top of each script). If you use `data/` instead, update the paths consistently.

### Execution order

1. **Train models** (generates `rf_model.joblib` and `svm_model.joblib`):

   ```bash
   python train_models.py
   ```

2. **Evaluate on the test set** (prints metrics; may show a plot and write `accuracy_comparison.png`):

   ```bash
   python evaluate_system.py
   ```

3. **Quick demo** (sample predictions vs. actual labels):

   ```bash
   python demo_detection.py
   ```

4. **Plots** (confusion matrices + feature importance; requires trained `*_model.joblib` files):

   ```bash
   python visualize_results.py
   ```

5. **Preprocessing only** (optional sanity check of shapes and pipeline):

   ```bash
   python preprocess.py
   ```

   Adjust the paths inside `preprocess.py` under `if __name__ == "__main__":` if your CSV locations differ.

### Windows (PowerShell)

From the project folder:

```powershell
cd path\to\threat-detection-system
python train_models.py
python evaluate_system.py
python demo_detection.py
python visualize_results.py
```

---

## Outputs

| Output | Description |
|--------|-------------|
| `rf_model.joblib` | Trained Random Forest model |
| `svm_model.joblib` | Trained SVM model |
| `accuracy_comparison.png` | Bar chart of model accuracies (from `evaluate_system.py`) |
| `confusion_rf.png` | Random Forest confusion matrix (`visualize_results.py`) |
| `confusion_svm.png` | SVM confusion matrix (`visualize_results.py`) |
| `feature_importance.png` | Top 15 Random Forest feature importances (`visualize_results.py`) |

---

## Project Structure (overview)

```text
threat-detection-system/
├── DATA/                   # UNSW-NB15 CSVs
├── requirements.txt        # Python dependencies
├── preprocess.py           # Loading, encoding, scaling
├── train_models.py         # Train RF and SVM; save Joblib models
├── evaluate_system.py      # Metrics and accuracy bar chart
├── demo_detection.py       # Small-sample prediction demo
├── visualize_results.py    # Confusion matrices and feature importance
├── rf_model.joblib         # Produced after training
├── svm_model.joblib        # Produced after training
└── README.md
```

---

*This README describes the implementation as shipped for the final year project submission.*
