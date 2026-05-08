#!/usr/bin/env python3
"""
inference.py — Run predictions with BioGraphX_Hybrid_Improved model.
StandardScaler is fitted on-the-fly from the input CSV file.

Usage:
    python inference.py \
        --csv_path /path/to/features.csv \
        --esm_dirs /path/to/esm_embeddings_dir1 /path/to/esm_embeddings_dir2 \
        --model_path best_model_improved_fold_0.pth \
        --output predictions.csv \
        [--threshold 0.5] [--device cuda] [--batch_size 64]
"""

import os
import glob
import argparse
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# -------------------------------------------------------------------
# MODEL DEFINITION (exact architecture from training)
# -------------------------------------------------------------------
class BioGraphX_Hybrid_Improved(nn.Module):
    def __init__(self, esm_dim=2560, biographx_dim=157, num_classes=11, shared_dim=1024):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(esm_dim, shared_dim),
            nn.Tanh(),
            nn.Linear(shared_dim, 1)
        )
        self.esm_bottleneck = nn.Sequential(
            nn.Linear(esm_dim, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.phys_branch = nn.Sequential(
            nn.Linear(biographx_dim, shared_dim * 2),
            nn.BatchNorm1d(shared_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(shared_dim * 2, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(shared_dim, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
        )
        self.gate_controller = nn.Sequential(
            nn.Linear(shared_dim * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
            nn.Sigmoid()
        )
        self.classifier = nn.Sequential(
            nn.Linear(shared_dim * 2, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, esm_seq, mask, phys):
        attn_logits = self.attn(esm_seq)
        mask_expanded = mask.unsqueeze(-1).to(attn_logits.device)
        attn_logits = attn_logits.masked_fill(~mask_expanded, float('-inf'))
        attn_weights = F.softmax(attn_logits, dim=1)
        esm_vec = torch.sum(esm_seq * attn_weights, dim=1)
        esm_feat = self.esm_bottleneck(esm_vec)

        phys_feat = self.phys_branch(phys)
        combined = torch.cat([esm_feat, phys_feat], dim=1)

        gate_weights = self.gate_controller(combined)
        physics_gate = gate_weights[:, 0:1]
        esm_gate = gate_weights[:, 1:2]

        gated_esm = esm_feat * esm_gate.expand_as(esm_feat)
        gated_phys = phys_feat * physics_gate.expand_as(phys_feat)
        final_features = torch.cat([gated_esm, gated_phys], dim=1)

        return self.classifier(final_features), gate_weights


# -------------------------------------------------------------------
# DATASET & COLLATE
# -------------------------------------------------------------------
class InferenceDataset(Dataset):
    def __init__(self, entry_ids, physics_array, esm_index):
        self.entry_ids = entry_ids
        self.physics = torch.tensor(physics_array, dtype=torch.float32)
        self.esm_index = esm_index

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
                    esm_tensor = esm_tensor.unsqueeze(0)  # [2560] -> [1, 2560]
            except Exception as e:
                print(f"Warning: failed to load ESM for {entry_id}: {e}")
                esm_tensor = torch.zeros((1, 2560), dtype=torch.float32)
        else:
            print(f"Warning: No ESM embedding found for {entry_id}, using zeros")
            esm_tensor = torch.zeros((1, 2560), dtype=torch.float32)
        return esm_tensor, self.physics[idx], entry_id


def collate_fn(batch):
    esm, phys, ids = zip(*batch)
    lengths = torch.tensor([x.size(0) for x in esm])
    esm_padded = torch.nn.utils.rnn.pad_sequence(esm, batch_first=True)
    mask = torch.arange(esm_padded.size(1))[None, :] < lengths[:, None]
    return esm_padded, mask, torch.stack(phys), list(ids)


# -------------------------------------------------------------------
# UTILITIES
# -------------------------------------------------------------------
def build_esm_index(dir_list):
    """Index ESM .npz files by ACC."""
    index = {}
    missing_dirs = []
    for directory in dir_list:
        if os.path.exists(directory):
            files = glob.glob(os.path.join(directory, "*.npz"))
            for f in files:
                entry_id = os.path.basename(f).replace(".npz", "")
                index[entry_id] = f
        else:
            missing_dirs.append(directory)
    
    if missing_dirs:
        print(f"Warning: Directories not found: {missing_dirs}")
    
    print(f"Found {len(index)} ESM embeddings across {len(dir_list) - len(missing_dirs)} directories")
    return index


def get_feature_columns(df):
    """
    Identify physics feature columns by excluding metadata and target columns.
    """
    # Columns to exclude (metadata + target labels)
    exclude_cols = [
        "ACC",
        # Target columns
        "Cytoplasm", "Nucleus", "Extracellular", "Cell membrane",
        "Mitochondrion", "Endoplasmic reticulum", "Lysosome/Vacuole",
        "Golgi apparatus", "Peroxisome", "Plastid", "Membrane"
    ]
    
    # Also exclude 'Partition' if present
    if "Partition" in df.columns:
        exclude_cols.append("Partition")
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    if not feature_cols:
        raise ValueError(
            "No physics feature columns found! Check that your CSV contains "
            "columns beyond the known metadata/target columns."
        )
    
    print(f"Identified {len(feature_cols)} physics feature columns")
    print(f"Sample features: {feature_cols[:5]}...")
    
    return feature_cols


def fit_scaler_from_csv(df, feature_cols):
    """
    Fit StandardScaler from the input CSV data.
    Returns the fitted scaler and transformed data.
    """
    print("\n📊 Fitting StandardScaler on input data...")
    
    # Extract physics features
    physics_data = df[feature_cols].values.astype(np.float32)
    
    # Fit scaler
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(physics_data)
    
    print(f"  Fitted scaler on {len(df)} samples, {len(feature_cols)} features")
    print(f"  Before scaling - Mean: {physics_data.mean():.4f}, Std: {physics_data.std():.4f}")
    print(f"  After scaling  - Mean: {scaled_data.mean():.4f}, Std: {scaled_data.std():.4f}")
    
    return scaler, scaled_data


# -------------------------------------------------------------------
# MAIN INFERENCE FUNCTION
# -------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Inference with BioGraphX Hybrid Improved model"
    )
    parser.add_argument("--csv_path", required=True, 
                       help="CSV file with ACC column and physics features")
    parser.add_argument("--esm_dirs", nargs="+", required=True, 
                       help="One or more directories containing ESM .npz embeddings")
    parser.add_argument("--model_path", required=True, 
                       help="Trained model checkpoint (.pth)")
    parser.add_argument("--output", default="predictions.csv", 
                       help="Output CSV file path (default: predictions.csv)")
    parser.add_argument("--threshold", type=float, default=0.5, 
                       help="Threshold for binary predictions (default: 0.5)")
    parser.add_argument("--batch_size", type=int, default=64, 
                       help="Batch size (default: 64)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use (default: auto-detect)")
    parser.add_argument("--save_scaler", type=str, default=None,
                       help="Optionally save the fitted scaler to a file for reuse")
    
    args = parser.parse_args()
    device = torch.device(args.device)
    
    print("=" * 70)
    print("🚀 BioGraphX Hybrid Improved - Inference")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Model: {args.model_path}")
    print(f"CSV: {args.csv_path}")
    print(f"ESM dirs: {args.esm_dirs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Threshold: {args.threshold}")
    print("=" * 70)

    # ---- 1. Load CSV data ----
    print("\n📂 Loading CSV data...")
    df = pd.read_csv(args.csv_path)
    print(f"  Loaded {len(df)} entries, {len(df.columns)} columns")
    
    if "ACC" not in df.columns:
        raise KeyError("CSV must contain an 'ACC' column with protein identifiers!")
    
    entry_ids = df["ACC"].values
    print(f"  ACC column found with {len(entry_ids)} protein IDs")

    # ---- 2. Identify feature columns ----
    feature_cols = get_feature_columns(df)

    # ---- 3. Fit scaler on input data ----
    scaler, phys_data_scaled = fit_scaler_from_csv(df, feature_cols)

    # ---- 4. Optionally save the scaler for future use ----
    if args.save_scaler:
        import pickle
        with open(args.save_scaler, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"💾 Saved scaler to: {args.save_scaler}")

    # ---- 5. Build ESM index ----
    print("\n🔍 Indexing ESM embeddings...")
    esm_index = build_esm_index(args.esm_dirs)
    
    # Check how many proteins have ESM embeddings
    found_esm = sum(1 for eid in entry_ids if eid in esm_index)
    print(f"  ESM coverage: {found_esm}/{len(entry_ids)} proteins ({100*found_esm/len(entry_ids):.1f}%)")
    if found_esm < len(entry_ids):
        print(f"  ⚠️  {len(entry_ids) - found_esm} proteins missing ESM embeddings (will use zeros)")

    # ---- 6. Create DataLoader ----
    print("\n📦 Creating data loader...")
    dataset = InferenceDataset(entry_ids, phys_data_scaled, esm_index)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True if device.type == "cuda" else False,
    )

    # ---- 7. Load model ----
    print("\n🧠 Loading model...")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Infer model dimensions from checkpoint
    # phys_branch.0.weight shape: (shared_dim*2, biographx_dim)
    if "phys_branch.0.weight" in checkpoint:
        w_phys0 = checkpoint["phys_branch.0.weight"]
    else:
        # Try model_state_dict format
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        w_phys0 = state_dict["phys_branch.0.weight"]
    
    shared_dim_x2, biographx_dim = w_phys0.shape
    shared_dim = shared_dim_x2 // 2
    
    # Get num_classes from classifier last layer
    if "classifier.7.weight" in checkpoint:
        w_class_last = checkpoint["classifier.7.weight"]
    else:
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        w_class_last = state_dict["classifier.7.weight"]
    num_classes = w_class_last.shape[0]
    
    print(f"  Inferred dimensions:")
    print(f"    Physics features: {biographx_dim}")
    print(f"    Shared dim: {shared_dim}")
    print(f"    Num classes: {num_classes}")
    
    # Check feature dimension match
    if biographx_dim != len(feature_cols):
        print(f"\n  ⚠️  WARNING: Model expects {biographx_dim} features, but CSV has {len(feature_cols)}")
        print(f"  This may cause errors. Please verify your CSV matches the training data.")
    
    # Initialize model
    model = BioGraphX_Hybrid_Improved(
        esm_dim=2560,
        biographx_dim=biographx_dim,
        num_classes=num_classes,
        shared_dim=shared_dim,
    ).to(device)
    
    # Load state dict (handle both raw state dict and wrapped checkpoint)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print("  Model loaded successfully ✓")

    # ---- 8. Run inference ----
    print(f"\n🔮 Running inference on {len(entry_ids)} proteins...")
    all_logits = []
    all_ids = []
    all_gates = []
    
    with torch.no_grad():
        for batch_idx, (esm_padded, mask, phys, ids) in enumerate(loader):
            esm_padded = esm_padded.to(device)
            mask = mask.to(device)
            phys = phys.to(device)
            
            logits, gate_weights = model(esm_padded, mask, phys)
            
            all_logits.append(logits.cpu().numpy())
            all_gates.append(gate_weights.cpu().numpy())
            all_ids.extend(ids)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {(batch_idx + 1) * args.batch_size}/{len(entry_ids)} samples...")

    # ---- 9. Post-process results ----
    logits = np.concatenate(all_logits, axis=0)
    gates = np.concatenate(all_gates, axis=0)
    probs = 1 / (1 + np.exp(-logits))  # sigmoid to get probabilities
    preds = (probs >= args.threshold).astype(int)
    
    print(f"\n📊 Inference complete!")
    print(f"  Physics gate mean: {gates[:, 0].mean():.3f}")
    print(f"  ESM gate mean: {gates[:, 1].mean():.3f}")
    print(f"  Avg positive predictions per protein: {preds.sum(axis=1).mean():.2f}/{num_classes}")

    # ---- 10. Build output DataFrame ----
    TARGET_COLS = [
        "Cytoplasm", "Nucleus", "Extracellular", "Cell membrane",
        "Mitochondrion", "Endoplasmic reticulum", "Lysosome/Vacuole",
        "Golgi apparatus", "Peroxisome", "Plastid", "Membrane"
    ]
    
    # Adjust if model has different number of classes
    if len(TARGET_COLS) != num_classes:
        print(f"  Note: Model has {num_classes} classes, but hardcoded names have {len(TARGET_COLS)}")
        print(f"  Using generic class names")
        TARGET_COLS = [f"class_{i}" for i in range(num_classes)]
    
    out_df = pd.DataFrame({"ACC": all_ids})
    
    # Add probabilities
    for i, name in enumerate(TARGET_COLS):
        if i < num_classes:
            out_df[f"{name}_prob"] = probs[:, i]
    
    # Add binary predictions
    for i, name in enumerate(TARGET_COLS):
        if i < num_classes:
            out_df[f"{name}_pred"] = preds[:, i]
    
    # Add gate weights
    out_df["physics_gate"] = gates[:, 0]
    out_df["esm_gate"] = gates[:, 1]
    
    # Save
    out_df.to_csv(args.output, index=False)
    print(f"\n💾 Predictions saved to: {args.output}")
    print(f"  Columns: {list(out_df.columns)}")
    print(f"  Rows: {len(out_df)}")
    print("=" * 70)
    print("✅ Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()