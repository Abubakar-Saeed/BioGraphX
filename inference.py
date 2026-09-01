#!/usr/bin/env python3
"""
BioGraphX - unified inference for the Gated Hybrid subcellular-localization model.

Loads a checkpoint produced by `BioGraphX_Training_Code.py` (`train_single_fold`)
together with the StandardScaler it saved (`scaler_fold_{N}.pkl`), and predicts
per-protein localization probabilities across the 11 target compartments.

Using the training-time scaler (rather than fitting a new one on the inference
CSV) matters: a model trained on features standardized against the training
distribution will not produce meaningful outputs if inference features are
standardized against a different (and possibly tiny) inference batch instead.

Usage
-----
    # CLI
    python inference.py \\
        --csv-path new_proteins_encoded.csv \\
        --esm-dirs esm_embeddings/new \\
        --model-path results/hybrid/best_model_fold_0.pth \\
        --scaler-path results/hybrid/scaler_fold_0.pkl \\
        --output-csv predictions.csv

    # Directly from Python
    from inference import run_inference
    df = run_inference(
        csv_path="new_proteins_encoded.csv",
        esm_dirs=["esm_embeddings/new"],
        model_path="results/hybrid/best_model_fold_0.pth",
        scaler_path="results/hybrid/scaler_fold_0.pkl",
    )
"""

import argparse
import glob
import os
import warnings

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from BioGraphX_Training_Code import BioGraphX_Hybrid, TARGET_COLS as DEFAULT_TARGET_COLS

warnings.filterwarnings("ignore")


class InferenceDataset(Dataset):
    def __init__(self, entry_ids, physics_array, esm_index, esm_dim=2560):
        self.entry_ids = entry_ids
        self.physics = torch.tensor(physics_array, dtype=torch.float32)
        self.esm_index = esm_index
        self.esm_dim = esm_dim

    def __len__(self):
        return len(self.entry_ids)

    def __getitem__(self, idx):
        entry_id = self.entry_ids[idx]
        path = self.esm_index.get(entry_id, None)
        if path:
            try:
                data = np.load(path)
                esm_tensor = torch.from_numpy(data["embedding"]).float()
                if esm_tensor.dim() == 1:
                    esm_tensor = esm_tensor.unsqueeze(0)  # [esm_dim] -> [1, esm_dim]
            except Exception as e:
                print(f"Warning: failed to load ESM for {entry_id}: {e}")
                esm_tensor = torch.zeros((1, self.esm_dim), dtype=torch.float32)
        else:
            print(f"Warning: no ESM embedding found for {entry_id}, using zeros")
            esm_tensor = torch.zeros((1, self.esm_dim), dtype=torch.float32)
        return esm_tensor, self.physics[idx], entry_id


def collate_fn(batch):
    esm, phys, ids = zip(*batch)
    lengths = torch.tensor([x.size(0) for x in esm])
    esm_padded = torch.nn.utils.rnn.pad_sequence(esm, batch_first=True)
    mask = torch.arange(esm_padded.size(1))[None, :] < lengths[:, None]
    return esm_padded, mask, torch.stack(phys), list(ids)


def build_esm_index(dir_list):
    """Index ESM .npz files by ACC."""
    index = {}
    missing_dirs = []
    for directory in dir_list:
        if os.path.exists(directory):
            for f in glob.glob(os.path.join(directory, "*.npz")):
                entry_id = os.path.basename(f).replace(".npz", "")
                index[entry_id] = f
        else:
            missing_dirs.append(directory)

    if missing_dirs:
        print(f"Warning: directories not found: {missing_dirs}")
    print(f"Found {len(index)} ESM embeddings across {len(dir_list) - len(missing_dirs)} directories")
    return index


def _infer_dims_from_checkpoint(state_dict):
    """Recover (biographx_dim, shared_dim, num_classes) from a checkpoint's
    weight shapes, so inference doesn't need the training hyperparameters
    passed in separately."""
    w_phys0 = state_dict["phys_branch.0.weight"]  # shape: (shared_dim*2, biographx_dim)
    shared_dim_x2, biographx_dim = w_phys0.shape
    shared_dim = shared_dim_x2 // 2

    classifier_linear_keys = [k for k in state_dict if k.startswith("classifier.") and k.endswith(".weight")]
    last_key = sorted(classifier_linear_keys, key=lambda k: int(k.split(".")[1]))[-1]
    num_classes = state_dict[last_key].shape[0]

    return biographx_dim, shared_dim, num_classes


def run_inference(
    csv_path,
    esm_dirs,
    model_path,
    scaler_path,
    output_csv=None,
    threshold=0.5,
    batch_size=64,
    esm_dim=2560,
    num_workers=4,
    device=None,
    target_cols=None,
):
    """
    Run Gated Hybrid inference on a CSV of BioGraphX-encoded proteins.

    Args:
        csv_path: CSV with an `ACC` column and the 157 physics feature columns
            (as produced by `BioGraphX-Encoding/src/run.py`). Extra columns
            (Sequence, Partition, target labels, etc.) are ignored.
        esm_dirs: Directory or list of directories of cached `{ACC}.npz` ESM
            embeddings (as produced by `esm_embeddings.py`).
        model_path: Checkpoint saved by `train_single_fold` (a raw state_dict
            .pth, optionally wrapped in {"model_state_dict": ...}).
        scaler_path: The `scaler_fold_{N}.pkl` saved alongside that checkpoint
            - a dict with keys "scaler" (fitted StandardScaler) and
            "feature_columns" (the exact columns/order used at training time).
        output_csv: If given, write predictions here.
        threshold: Decision threshold for binary predictions.
        esm_dim: Expected ESM embedding hidden size.
        device: "cuda"/"cpu"; auto-detected if omitted.
        target_cols: Override the 11 default compartment names (e.g. if the
            checkpoint was trained on a different label set).

    Returns:
        pandas.DataFrame with per-class probabilities/predictions and gate weights.
    """
    if isinstance(esm_dirs, str):
        esm_dirs = [esm_dirs]
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    print("=" * 70)
    print("BioGraphX Gated Hybrid - Inference")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Model: {model_path}")
    print(f"Scaler: {scaler_path}")
    print(f"CSV: {csv_path}")
    print(f"ESM dirs: {esm_dirs}")
    print("=" * 70)

    # ---- 1. Load CSV data ----
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} entries, {len(df.columns)} columns")
    if "ACC" not in df.columns:
        raise KeyError("CSV must contain an 'ACC' column with protein identifiers!")
    entry_ids = df["ACC"].values

    # ---- 2. Load the training-time scaler + its exact feature-column list ----
    scaler_bundle = joblib.load(scaler_path)
    scaler = scaler_bundle["scaler"]
    feature_cols = scaler_bundle["feature_columns"]

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input CSV is missing {len(missing)} feature column(s) the model was trained on: "
            f"{missing[:10]}{'...' if len(missing) > 10 else ''}"
        )
    print(f"Using {len(feature_cols)} feature columns from the training-time scaler")

    phys_data_scaled = scaler.transform(df[feature_cols].values.astype(np.float32))

    # ---- 3. Build ESM index ----
    esm_index = build_esm_index(esm_dirs)
    found_esm = sum(1 for eid in entry_ids if eid in esm_index)
    print(f"ESM coverage: {found_esm}/{len(entry_ids)} proteins ({100*found_esm/len(entry_ids):.1f}%)")
    if found_esm < len(entry_ids):
        print(f"  {len(entry_ids) - found_esm} protein(s) missing ESM embeddings (will use zeros)")

    # ---- 4. Data loader ----
    dataset = InferenceDataset(entry_ids, phys_data_scaled, esm_index, esm_dim=esm_dim)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )

    # ---- 5. Load model (dimensions recovered from the checkpoint itself) ----
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    biographx_dim, shared_dim, num_classes = _infer_dims_from_checkpoint(state_dict)
    print(f"Inferred from checkpoint: physics_dim={biographx_dim}, shared_dim={shared_dim}, num_classes={num_classes}")

    if biographx_dim != len(feature_cols):
        raise ValueError(
            f"Checkpoint expects {biographx_dim} physics features but the scaler/CSV provide "
            f"{len(feature_cols)}. This model and scaler were not trained together."
        )

    model = BioGraphX_Hybrid(
        esm_dim=esm_dim, biographx_dim=biographx_dim, num_classes=num_classes, shared_dim=shared_dim,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded.")

    # ---- 6. Run inference ----
    print(f"Running inference on {len(entry_ids)} proteins...")
    all_logits, all_ids, all_gates = [], [], []
    with torch.no_grad():
        for esm_padded, mask, phys, ids in loader:
            esm_padded, mask, phys = esm_padded.to(device), mask.to(device), phys.to(device)
            logits, gate_weights = model(esm_padded, mask, phys)
            all_logits.append(logits.cpu().numpy())
            all_gates.append(gate_weights.cpu().numpy())
            all_ids.extend(ids)

    logits = np.concatenate(all_logits, axis=0)
    gates = np.concatenate(all_gates, axis=0)
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= threshold).astype(int)

    print("Inference complete.")
    print(f"  Physics gate mean: {gates[:, 0].mean():.3f}")
    print(f"  ESM gate mean: {gates[:, 1].mean():.3f}")
    print(f"  Avg positive predictions per protein: {preds.sum(axis=1).mean():.2f}/{num_classes}")

    # ---- 7. Build output DataFrame ----
    class_names = list(target_cols) if target_cols else list(DEFAULT_TARGET_COLS)
    if len(class_names) != num_classes:
        print(f"Note: checkpoint has {num_classes} classes but {len(class_names)} names were given; "
              f"using generic class_i names instead.")
        class_names = [f"class_{i}" for i in range(num_classes)]

    out_df = pd.DataFrame({"ACC": all_ids})
    for i, name in enumerate(class_names):
        out_df[f"{name}_prob"] = probs[:, i]
    for i, name in enumerate(class_names):
        out_df[f"{name}_pred"] = preds[:, i]
    out_df["physics_gate"] = gates[:, 0]
    out_df["esm_gate"] = gates[:, 1]

    if output_csv:
        out_df.to_csv(output_csv, index=False)
        print(f"Predictions saved to: {output_csv}")
    print("=" * 70)

    return out_df


def main():
    parser = argparse.ArgumentParser(description="BioGraphX Gated Hybrid inference.")
    parser.add_argument("--csv-path", required=True, help="CSV with ACC + physics feature columns.")
    parser.add_argument("--esm-dirs", nargs="+", required=True, help="Directories of cached {ACC}.npz embeddings.")
    parser.add_argument("--model-path", required=True, help="Trained checkpoint (.pth).")
    parser.add_argument("--scaler-path", required=True, help="scaler_fold_{N}.pkl saved during training.")
    parser.add_argument("--output-csv", default="predictions.csv")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--esm-dim", type=int, default=2560)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None, help="cuda or cpu (default: auto-detect).")
    args = parser.parse_args()

    run_inference(
        csv_path=args.csv_path,
        esm_dirs=args.esm_dirs,
        model_path=args.model_path,
        scaler_path=args.scaler_path,
        output_csv=args.output_csv,
        threshold=args.threshold,
        batch_size=args.batch_size,
        esm_dim=args.esm_dim,
        num_workers=args.num_workers,
        device=args.device,
    )


if __name__ == "__main__":
    main()
