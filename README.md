# BioGraphX
[![DOI](https://img.shields.io/badge/DOI-10.64898%252F2026.01.21.700873-blue)](https://doi.org/10.64898/2026.01.21.700873)
![Python Version](https://img.shields.io/badge/Python-3.7+-blue.svg)
![License](https://img.shields.io/badge/License-Academic-blue.svg)

**BioGraphX** is a physicochemical graph encoding framework that converts protein sequences into explainable 157-dimensional biophysical feature vectors. It combines residue interaction graph modeling, hybrid interaction scoring, motif-based localization profiling, and frustration analysis for downstream classification and inference.

---

## 🚀 What’s New

* Added `inference.py` for BioGraphX+ESM hybrid model prediction
* Added `esm_embeddings.py` for ESM embedding extraction from protein sequences
* Added complete prediction workflow with model checkpoint inference
* Updated documentation for end-to-end feature generation and inference

---

## 🧠 Overview

BioGraphX supports two main workflows:

1. **Feature extraction** from protein sequences using the BioGraphX encoding pipeline
2. **Hybrid inference** combining BioGraphX physics features with ESM embeddings via `inference.py`

The repository contains:

* `BioGraphX-Encoding/src/biographx/` — core feature extraction modules
* `BioGraphX-Encoding/src/run.py` — example entrypoint for batch feature extraction
* `inference.py` — prediction script for the trained BioGraphX_Hybrid_Improved model
* `esm_embeddings.py` — script for generating `.npz` ESM embeddings

---

## ✅ Requirements

Recommended Python environment:

* Python 3.8+ (3.14 may not be compatible with all packages)
* GPU recommended for ESM embedding extraction and inference

Install required packages:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install torch torchvision torchaudio scikit-learn tqdm transformers
```

> If you want a CPU-only PyTorch install, use the official PyTorch CPU index from https://pytorch.org.

---

## 📦 Feature Extraction Workflow

### 1) Prepare input CSV

Your input CSV must contain a `Sequence` column with amino acid sequences. Example:

```csv
ACC,Sequence,OtherMeta
P12345,MKTIIALSYIFCLVFADYKDDDDK,foo
Q67890,MSYQGHGHHHKSGLSDLK,bar
```

### 2) Run the integrated pipeline

The easiest option is to use `BioGraphX-Encoding/src/run.py`, but note this file uses hardcoded paths. Update `input_file` and `output_file` before running.

```powershell
cd BioGraphX\BioGraphX-Encoding\src
python run.py
```

Alternatively, use the pipeline directly from Python:

```python
from biographx.pipeline import run_integrated_pipeline

run_integrated_pipeline(
    input_file="/path/to/proteins.csv",
    output_file="/path/to/encoded_features.csv",
    chunk_size=500,
    n_jobs=8,
)
```

### 3) Output format

The output CSV preserves all original columns except `Sequence`. It appends 157 feature columns named according to the BioGraphX feature registry.

> Important: `Sequence` is used for encoding and is dropped from the output.

---

## 🧬 ESM Embedding Generation

Use `esm_embeddings.py` to produce per-protein `.npz` files for ESM embeddings.

### Required input format

The script expects a CSV with columns:

* `ACC` — unique protein identifier
* `Sequence_main` — amino acid sequence

### Configure the script

Open `esm_embeddings.py` and set:

* `CSV_PATH` — path to your input CSV
* `OUTPUT_DIR` — destination directory for `.npz` embeddings
* `MODEL_NAME` — ESM model name (default: `facebook/esm2_t36_3B_UR50D`)
* `PART_TO_RUN` / `TOTAL_PARTS` — use when splitting a very large dataset
* `BATCH_SIZE` / `NUM_WORKERS` — tune for your machine

### Run embedding extraction

```powershell
python esm_embeddings.py
```

### Result

Each protein will be saved as:

```text
OUTPUT_DIR/<ACC>.npz
```

The `.npz` contains a compressed `embedding` array of shape `[sequence_length, 2560]`.

---

## 🧪 Prediction / Inference Workflow

`inference.py` produces localization prediction probabilities and binary labels using the hybrid BioGraphX+ESM model.

### 1) Prepare physics feature CSV

Your feature CSV must include:

* `ACC` — unique protein identifier
* physics feature columns extracted by BioGraphX

`inference.py` automatically excludes known metadata and target columns, then uses the remaining columns as physics inputs.

### 2) Prepare ESM embeddings

Provide one or more directories containing `.npz` files named by `ACC`.

Example:

```text
esm_dir_1/P12345.npz
esm_dir_1/Q67890.npz
```

### 3) Run inference

```powershell
python inference.py \
  --csv_path /path/to/encoded_features.csv \
  --esm_dirs /path/to/esm_dir_1 /path/to/esm_dir_2 \
  --model_path /path/to/model_fold_0.pth \
  --output predictions.csv \
  --threshold 0.5 \
  --batch_size 64 \
  --device cuda
```

### 4) Output columns

The inference output CSV contains:

* `ACC`
* `<class>_prob` — sigmoid probability for each target class
* `<class>_pred` — binary prediction using the threshold
* `physics_gate` — learned physics contribution weight
* `esm_gate` — learned ESM contribution weight

If the model checkpoint uses a different number of classes than the default 11, generic columns `class_0`, `class_1`, ... are used.

---

## 🏋️ Model Weights

Pre-trained model checkpoints for the BioGraphX_Hybrid_Improved architecture are included in the repository. These models were trained on the FGNNSol dataset for subcellular localization prediction.

Available checkpoints:
* `model_fold_0.pth` — Best performing model from cross-validation fold 0
* Additional folds available as needed

Use these with `inference.py` for out-of-the-box predictions.

---

## 🔍 Structural Proxy Validation (fGNNSol Benchmark)

The `BioGraphX-Encoding/Structure Validation/` folder contains the cross-dataset validation of BioGraphX structural proxy features on the E. coli eSol solubility benchmark, demonstrating that sequence-derived constraint graphs capture genuine structural signal without requiring 3D coordinates.

### Background

To validate that BioGraphX graph encodings function as effective structural proxies, we benchmarked the standalone 157 features (no ESM embeddings, no deep learning) against fGNNSol [1], a state-of-the-art method that uses AlphaFold3-derived 3D structural features (~620 dimensions), ESM-C embeddings (1,152 dimensions), and a dual-stream GNN architecture. Despite using 11.6× fewer features and no 3D coordinates, BioGraphX achieves competitive recall (0.726 vs. 0.734) at the standard solubility threshold.

### Run Validation

```powershell
cd BioGraphX-Encoding/Structure Validation
python validate.py
```

This script:

* Loads the eSol benchmark dataset
* Extracts 157 BioGraphX features from protein sequences
* Trains an XGBoost regressor with 5 random seeds (2024–2028)
* Reports mean ± std for regression metrics (R², Pearson r, RMSE) and classification metrics (accuracy, precision, recall, F1, AUC, MCC)
* Performs feature importance analysis on the best-performing seed
* Validates convergence with fGNNSol's reported biophysical determinants

### Reference

[1] Song, W., Xu, B., Zhang, D., & Li, M. (2026). fGNNSol: A fused graph neural network with AlphaFold3 and ESM-C embeddings for accurate protein solubility prediction. Nature Machine Intelligence, 8, 120–132.

---

## 🧩 Notes

* `inference.py` will print a warning and use zero ESM embeddings for proteins without matching `.npz` files.
* Make sure the `ACC` values in your feature CSV exactly match the `.npz` filenames.
* `esm_embeddings.py` is GPU-intensive. Use a machine with at least one GPU for best performance.
* If your model or dataset uses a different feature order, verify the physics feature CSV before inference.

---

## 🔧 Troubleshooting

* `ModuleNotFoundError: No module named 'numpy'` — activate the virtual environment and reinstall requirements.
* `CSV must contain an 'ACC' column` — add the `ACC` identifier column to your CSV.
* `Model expects X features` — verify the physics feature CSV contains the expected number of BioGraphX columns.

---

## 📚 Citation

If you use BioGraphX in your research, please cite:

**BioGraphX:**
Saeed, A., & Abbas, W. (2026). BioGraphX: Bridging the sequence–structure gap via physicochemical graph encoding for Interpretable subcellular localization prediction. https://doi.org/10.64898/2026.01.21.700873



