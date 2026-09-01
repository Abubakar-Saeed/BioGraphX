# BioGraphX
[![DOI](https://img.shields.io/badge/DOI-10.64898%252F2026.01.21.700873-blue)](https://doi.org/10.64898/2026.01.21.700873)
![Python Version](https://img.shields.io/badge/Python-3.7+-blue.svg)
![License](https://img.shields.io/badge/License-Academic-blue.svg)

**BioGraphX** is a physicochemical graph encoding framework that converts protein sequences into explainable 157-dimensional biophysical feature vectors. It combines residue interaction graph modeling, hybrid interaction scoring, motif-based localization profiling, and frustration analysis for downstream classification and inference.

---

## 🚀 What's New

**Added**
* `BioGraphX_Training_Code.py` and `inference.py` are now callable two ways: a `--help`-documented CLI, and a plain importable function (`train_single_fold`, `run_inference`) for use from a notebook or another script.
* `esm_embeddings.py` gained the same treatment — CLI flags (`--csv-path`, `--output-dir`, `--part`/`--total-parts`, ...) plus a Python-importable `extract_esm_embeddings()`.
* Training now saves the fitted `StandardScaler` alongside each fold's checkpoint (`scaler_fold_{N}.pkl`), together with the exact feature-column list/order it was fit on.
* A full **Training** walkthrough below — the training script existed before but was never documented in this README.

**Fixed**
* `inference.py` no longer fits a *new* `StandardScaler` on whatever CSV you're predicting on. It now loads the training-time scaler and its feature-column list, and reuses `BioGraphX_Hybrid` from `BioGraphX_Training_Code.py` directly instead of a second, hand-maintained copy of the architecture (`BioGraphX_Hybrid_Improved`) that could silently drift out of sync with it.
* `pipeline.py`: `adaptive_extract_features()`'s sliding-window strategy (sequences >10,000 residues) used `concurrent.futures` without importing it — crashed every time that path was hit.
* `graph_engine.py`: the zero-edge fallback in `extract_basic_graph_features()` padded with one too many zeros, which silently shifted every feature after it by one column for single-residue inputs (`len(interaction_rules) + 1` instead of `len(interaction_rules)`).
* `esm_embeddings.py`: `collate_fn` closed over a local `tokenizer`, which crashes with `Can't pickle local object` the moment a `DataLoader` worker process is spawned (`num_workers>0`) — the default multiprocessing start method on Windows/macOS, and available on Linux too. This is unrelated to CPU vs. GPU; it fires regardless of which device the model itself runs on. Fixed by making `collate_fn` module-level, with each process (main and workers, via `worker_init_fn`) building its own tokenizer.
* `esm_embeddings.py`: `np.array_split(df, total_parts)` returns plain `ndarray`s rather than DataFrames on current numpy/pandas, breaking the `df_part['ACC']` access right after. Now splits the row-index array and slices with `.iloc` instead.
* `requirements.txt` was missing `torch`, `scikit-learn`, `transformers`, and `tqdm`, even though `BioGraphX_Training_Code.py`, `inference.py`, and `esm_embeddings.py` all import them.
* Removed a dead `physics_bypass` submodule from `BioGraphX_Hybrid` — defined in `__init__`, never referenced in `forward()`.

**Removed**
* The previously-shipped pretrained checkpoints (`model-weights/model_fold_0.pth` ... `model_fold_4.pth`). They were trained against a 158-feature version of the encoder and are no longer compatible with the current 157-feature pipeline (confirmed from the checkpoints' own `phys_branch.0.weight` shape). Train fresh ones with `BioGraphX_Training_Code.py` — see **Training** below.

Every fix above was verified end-to-end against synthetic data (encoded features → training a fold → inference on the resulting checkpoint) before being committed, not just read through.

---

## 🧠 Overview

BioGraphX supports two main workflows:

1. **Feature extraction** from protein sequences using the BioGraphX encoding pipeline
2. **Hybrid inference** combining BioGraphX physics features with ESM embeddings via `inference.py`

The repository contains:

* `BioGraphX-Encoding/src/biographx/` — core feature extraction modules
* `BioGraphX-Encoding/targeting_rules.py` — motif heuristics and targeting rules used by the localization profiler
* `BioGraphX-Encoding/src/run.py` — example entrypoint for batch feature extraction
* `BioGraphX_Training_Code.py` — trains the Gated Hybrid (physics + ESM) localization model, one cross-validation fold at a time
* `inference.py` — prediction script for the trained Gated Hybrid model
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

Your input CSV must contain a `Sequence` column with amino acid sequences. Any other columns (`ACC`, `Kingdom`, `Partition`, the 11 localization target columns, ...) are preserved as-is and carried through to the output — this is also the format `BioGraphX_Training_Code.py` expects downstream, so it's worth including `ACC` and `Partition` from the start if you're heading toward training. Example:

```csv
ACC,Sequence,Kingdom,Partition,Cytoplasm,Nucleus,Extracellular,Cell membrane,Mitochondrion,Endoplasmic reticulum,Lysosome/Vacuole,Golgi apparatus,Peroxisome,Plastid,Membrane
P12345,MKTIIALSYIFCLVFADYKDDDDK,Animal,0,1,0,0,0,0,0,0,0,0,0,0
Q67890,MSYQGHGHHHKSGLSDLK,Plant,1,0,1,0,0,0,0,0,0,0,0,0
```

### 2) Run the integrated pipeline

Use `BioGraphX-Encoding/src/run.py` with command-line options to specify input and output file paths.

```powershell
cd BioGraphX\BioGraphX-Encoding\src
python run.py --input-file path\to\proteins.csv --output-file path\to\BioGraphXEncodedFeatures.csv
```

The pipeline also uses localization rules defined in `BioGraphX-Encoding/targeting_rules.py`, which provides the motif heuristics and canonical targeting patterns used by the `MotifProfiler` during feature extraction.

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
* `Sequence_main` — amino acid sequence (override with `--sequence-col` if your CSV names it differently)

### Run embedding extraction

```powershell
python esm_embeddings.py --csv-path proteins.csv --output-dir esm_embeddings_ml --batch-size 15
```

Split a large dataset across multiple runs/GPUs with `--part` / `--total-parts`, and see `python esm_embeddings.py --help` for the rest (`--model-name`, `--num-workers`, `--max-length`).

Or call it directly from Python:

```python
from esm_embeddings import extract_esm_embeddings

extract_esm_embeddings(csv_path="proteins.csv", output_dir="esm_embeddings_ml")
```

### Result

Each protein will be saved as:

```text
OUTPUT_DIR/<ACC>.npz
```

The `.npz` contains a compressed `embedding` array of shape `[sequence_length, 2560]`.

---

## 🏋️ Training

`BioGraphX_Training_Code.py` trains the Gated Hybrid model (biophysical features + ESM-2, fused with a learned per-sample gate) for one cross-validation fold, using the `Partition` column in the encoded CSV to hold that fold out for validation.

```powershell
python BioGraphX_Training_Code.py `
  --csv-path BioGraphXEncodedFeatures.csv `
  --esm-dirs esm_embeddings_ml `
  --fold 0 `
  --output-dir results\hybrid
```

Or directly from Python:

```python
from BioGraphX_Training_Code import train_single_fold

results = train_single_fold(
    fold_num=0,
    csv_path="BioGraphXEncodedFeatures.csv",
    esm_dirs=["esm_embeddings_ml"],
    output_dir="results/hybrid",
)
```

Each run writes, to `--output-dir`:

* `best_model_fold_{N}.pth` — the checkpoint with the best validation MCC
* `scaler_fold_{N}.pkl` — the `StandardScaler` fitted on the training split, plus the exact feature-column list/order it was fit on. **`inference.py` requires this** — a model trained on standardized features needs new data standardized the same way, not re-fit on whatever's being predicted.
* `gate_history_fold_{N}.csv` — per-epoch physics/ESM gate means
* `fold_{N}_results.csv` — final metrics (accuracy, Jaccard, micro/macro F1, per-class MCC, optimized thresholds)

See `python BioGraphX_Training_Code.py --help` for the remaining hyperparameters (`--epochs`, `--batch-size`, `--lr`, `--physics-lr-multiplier`, `--gate-regularization-epochs`).

No pretrained checkpoints are distributed in this repository; train the fold(s) you need with the command above.

---

## 🧪 Prediction / Inference Workflow

`inference.py` produces localization prediction probabilities and binary labels using the hybrid BioGraphX+ESM model.

### 1) Prepare physics feature CSV

Your feature CSV must include:

* `ACC` — unique protein identifier
* the physics feature columns extracted by BioGraphX

`inference.py` reads the exact feature-column list from `scaler_fold_{N}.pkl` (saved during training) rather than re-deriving it - it raises a clear error if any of those columns are missing from your CSV, instead of silently scoring on the wrong columns.

### 2) Prepare ESM embeddings

Provide one or more directories containing `.npz` files named by `ACC`.

Example:

```text
esm_dir_1/P12345.npz
esm_dir_1/Q67890.npz
```

### 3) Run inference

```powershell
python inference.py `
  --csv-path /path/to/encoded_features.csv `
  --esm-dirs /path/to/esm_dir_1 /path/to/esm_dir_2 `
  --model-path /path/to/best_model_fold_0.pth `
  --scaler-path /path/to/scaler_fold_0.pkl `
  --output-csv predictions.csv `
  --threshold 0.5 `
  --batch-size 64
```

Or directly from Python:

```python
from inference import run_inference

df = run_inference(
    csv_path="encoded_features.csv",
    esm_dirs=["esm_dir_1", "esm_dir_2"],
    model_path="best_model_fold_0.pth",
    scaler_path="scaler_fold_0.pkl",
)
```

Model dimensions (physics feature count, hidden size, number of classes) are recovered from the checkpoint's own weight shapes, and device (`cuda`/`cpu`) is auto-detected unless `--device`/`device=` is given.

### 4) Output columns

The inference output CSV contains:

* `ACC`
* `<class>_prob` — sigmoid probability for each target class
* `<class>_pred` — binary prediction using the threshold
* `physics_gate` — learned physics contribution weight
* `esm_gate` — learned ESM contribution weight

If the model checkpoint uses a different number of classes than the default 11, generic columns `class_0`, `class_1`, ... are used.

---

## 🔍 Structural Proxy Validation (fGNNSol Benchmark)

The `BioGraphX-Encoding/Structure Validation/` folder contains the cross-dataset validation of BioGraphX structural proxy features on the E. coli eSol solubility benchmark, demonstrating that sequence-derived constraint graphs capture genuine structural signal without requiring 3D coordinates.

### Background

To validate that BioGraphX graph encodings function as effective structural proxies, we benchmarked the standalone 157 features (no ESM embeddings, no deep learning) against fGNNSol, a state-of-the-art method that uses AlphaFold3-derived 3D structural features (~620 dimensions), ESM-C embeddings (1,152 dimensions), and a dual-stream GNN architecture. Despite using 11.6× fewer features and no 3D coordinates, BioGraphX achieves competitive recall (0.726 vs. 0.734) at the standard solubility threshold.

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

### Reference

Song, G., et al. Protein Solubility Prediction Using Fused Graph Convolutional Networks and Improved Attention Networks with AlphaFold3-Derived Features. J. Chem. Inf. Model., 2025. DOI: 10.1021/acs.jcim.5c02262"


---

## 📚 Citation

If you use BioGraphX in your research, please cite:

**BioGraphX:**
Saeed, A., & Abbas, W. (2026). BioGraphX: Bridging the sequence–structure gap via physicochemical graph encoding for Interpretable subcellular localization prediction. https://doi.org/10.1093/bioadv/vbag181



