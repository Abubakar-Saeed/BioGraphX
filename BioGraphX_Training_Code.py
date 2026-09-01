"""
Author: Abubakar Saeed
Created: January 2026
Last Modified: February 2026

Description:
    Hybrid neural network architecture for subcellular localization prediction
    integrating ESM protein language model embeddings with BioGraphX biophysical features.

    Implements adaptive gating mechanism that learns to weight ESM transformer features
    against physics-based feature representations dynamically per sample. Applies a
    gate-regularization term for the first few epochs to discourage the gate from
    collapsing onto a single pathway before both branches are informative.

    All distance-based features in the physics branch originate from linear sequence
    positions, NOT 3D spatial coordinates.

Usage
-----
    # CLI
    python BioGraphX_Training_Code.py \\
        --csv-path BioGraphXEncodedFeatures.csv \\
        --esm-dirs esm_embeddings_ml \\
        --fold 0 \\
        --output-dir results/hybrid

    # Directly from Python
    from BioGraphX_Training_Code import train_single_fold
    results = train_single_fold(
        fold_num=0,
        csv_path="BioGraphXEncodedFeatures.csv",
        esm_dirs=["esm_embeddings_ml"],
        output_dir="results/hybrid",
    )
"""

import argparse
import glob
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, matthews_corrcoef
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings('ignore')

# ==========================================
# TASK DEFINITION (fixed by the problem, not a per-run setting)
# ==========================================

# Target localization compartments (11-class multi-label)
TARGET_COLS = [
    "Cytoplasm", "Nucleus", "Extracellular", "Cell membrane",
    "Mitochondrion", "Endoplasmic reticulum", "Lysosome/Vacuole",
    "Golgi apparatus", "Peroxisome", "Plastid", "Membrane"
]
NUM_CLASSES = len(TARGET_COLS)

# Metadata columns to exclude from the feature matrix
META_COLS = ['ACC', 'Kingdom', 'Partition', 'Sequence'] + TARGET_COLS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-label classification with class imbalance.

    Down-weights well-classified examples and focuses training on hard,
    misclassified instances. Particularly effective for multi-label problems
    with severe class imbalance.
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # Probability of correct classification
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss


# ==========================================
# 1. DATA PREPARATION
# ==========================================

def build_esm_index(dir_list):
    """
    Build lookup index mapping protein accession IDs to ESM embedding file paths.

    Args:
        dir_list: List of directories containing .npz embedding files

    Returns:
        dict: Mapping {accession_id: file_path}
    """
    index = {}
    for directory in dir_list:
        if os.path.exists(directory):
            files = glob.glob(os.path.join(directory, "*.npz"))
            for f in files:
                entry_id = os.path.basename(f).replace(".npz", "")
                index[entry_id] = f
    print(f"Found {len(index)} ESM embeddings")
    return index


class BioGraphX_Dataset(Dataset):
    """
    PyTorch Dataset for combined ESM embeddings and BioGraphX physics features.

    Args:
        df: DataFrame containing protein metadata and accession IDs
        esm_index: Dictionary mapping accession IDs to ESM embedding paths
        physics_array: Preprocessed BioGraphX feature matrix (n_samples, n_features)
        label_matrix: Multi-label binary matrix (n_samples, n_classes)
    """
    def __init__(self, df, esm_index, physics_array, label_matrix, esm_dim=2560):
        self.entry_ids = df["ACC"].values
        self.labels = torch.tensor(label_matrix, dtype=torch.float32)
        self.esm_index = esm_index
        self.physics = torch.tensor(physics_array, dtype=torch.float32)
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
                # Ensure 2D shape: [sequence_length, embedding_dim]
                if esm_tensor.dim() == 1:
                    esm_tensor = esm_tensor.unsqueeze(0)  # [2560] -> [1, 2560]
            except Exception as e:
                print(f"Warning: error loading {entry_id}: {e}")
                esm_tensor = torch.zeros((1, self.esm_dim), dtype=torch.float32)
        else:
            esm_tensor = torch.zeros((1, self.esm_dim), dtype=torch.float32)

        return esm_tensor, self.physics[idx], self.labels[idx]


def collate_fn(batch):
    """
    Custom collation function for variable-length ESM sequences.

    Pads sequences to maximum length in batch and creates attention mask.

    Returns:
        esm_padded: Padded ESM embeddings (batch, max_len, esm_dim)
        mask: Boolean mask indicating valid positions (batch, max_len)
        physics: Stacked physics feature vectors (batch, physics_dim)
        labels: Stacked label matrices (batch, n_classes)
    """
    esm, phys, labels = zip(*batch)
    lengths = torch.tensor([x.size(0) for x in esm])
    esm_padded = torch.nn.utils.rnn.pad_sequence(esm, batch_first=True)
    mask = torch.arange(esm_padded.size(1))[None, :] < lengths[:, None]
    return esm_padded, mask, torch.stack(phys), torch.stack(labels)


# ==========================================
# 2. MODEL WITH ADAPTIVE GATING
# ==========================================

class BioGraphX_Hybrid(nn.Module):
    """
    Hybrid architecture with adaptive gating between ESM and physics pathways.

    Architecture:
        - ESM pathway: Attention-weighted pooling of per-residue embeddings
        - Physics pathway: Deep MLP with batch normalization
        - Gate controller: Learns sample-specific weighting between pathways
        - Classifier: Combined feature classification head
    """
    def __init__(self, esm_dim=2560, biographx_dim=157, num_classes=11, shared_dim=1024):
        super().__init__()

        # -----------------------------------------------------------------
        # ESM PATHWAY: Transformer embedding aggregation
        # -----------------------------------------------------------------
        # Attention mechanism for per-position weighting
        self.attn = nn.Sequential(
            nn.Linear(esm_dim, shared_dim),
            nn.Tanh(),
            nn.Linear(shared_dim, 1)
        )

        # Bottleneck projection
        self.esm_bottleneck = nn.Sequential(
            nn.Linear(esm_dim, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # -----------------------------------------------------------------
        # PHYSICS BRANCH: Deeper architecture for biophysical features
        # -----------------------------------------------------------------
        self.phys_branch = nn.Sequential(
            # Expansion layer: capture feature interactions
            nn.Linear(biographx_dim, shared_dim * 2),
            nn.BatchNorm1d(shared_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Intermediate compression
            nn.Linear(shared_dim * 2, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Project to match ESM dimension
            nn.Linear(shared_dim, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
        )

        # -----------------------------------------------------------------
        # GATE CONTROLLER: Sample-specific adaptive weighting
        # -----------------------------------------------------------------
        # Learns when to trust physics vs. ESM based on combined features
        self.gate_controller = nn.Sequential(
            nn.Linear(shared_dim * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 2),  # Output: [physics_gate, esm_gate]
            nn.Sigmoid()        # Gates bounded to [0,1]
        )

        # -----------------------------------------------------------------
        # CLASSIFIER: Multi-label output head
        # -----------------------------------------------------------------
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
        """
        Forward pass with adaptive gating.

        Args:
            esm_seq: ESM embeddings (batch, seq_len, esm_dim)
            mask: Attention mask (batch, seq_len)
            phys: Physics feature vectors (batch, physics_dim)

        Returns:
            logits: Classification logits (batch, num_classes)
            gate_weights: Sample-specific gate values (batch, 2)
        """
        # -----------------------------------------------------------------
        # ESM processing: Attention-weighted pooling
        # -----------------------------------------------------------------
        attn_logits = self.attn(esm_seq)
        mask_expanded = mask.unsqueeze(-1).to(attn_logits.device)
        attn_logits = attn_logits.masked_fill(~mask_expanded, float('-inf'))
        attn_weights = F.softmax(attn_logits, dim=1)
        esm_vec = torch.sum(esm_seq * attn_weights, dim=1)
        esm_feat = self.esm_bottleneck(esm_vec)

        # -----------------------------------------------------------------
        # Physics processing: Deep MLP
        # -----------------------------------------------------------------
        phys_feat = self.phys_branch(phys)

        # -----------------------------------------------------------------
        # Adaptive gating: Sample-specific pathway weighting
        # -----------------------------------------------------------------
        combined = torch.cat([esm_feat, phys_feat], dim=1)
        gate_weights = self.gate_controller(combined)  # [batch, 2]
        physics_gate = gate_weights[:, 0:1]  # [batch, 1]
        esm_gate = gate_weights[:, 1:2]      # [batch, 1]

        # Apply element-wise gating
        gated_esm = esm_feat * esm_gate.expand_as(esm_feat)
        gated_phys = phys_feat * physics_gate.expand_as(phys_feat)

        # -----------------------------------------------------------------
        # Classification
        # -----------------------------------------------------------------
        final_features = torch.cat([gated_esm, gated_phys], dim=1)

        return self.classifier(final_features), gate_weights


# ==========================================
# 3. UTILITY FUNCTIONS
# ==========================================

def evaluate_validation_mcc(model, val_loader, device=DEVICE):
    """Evaluate validation Matthews Correlation Coefficient with a 0.5 threshold."""
    model.eval()
    all_logits, all_labels = [], []

    with torch.no_grad():
        for esm, mask, phys, labels in val_loader:
            esm, mask, phys = esm.to(device), mask.to(device), phys.to(device)
            out, _ = model(esm, mask, phys)
            all_logits.append(out.cpu().numpy())
            all_labels.append(labels.numpy())

    logits = np.vstack(all_logits)
    labels = np.vstack(all_labels)
    predictions = (torch.sigmoid(torch.tensor(logits)) > 0.5).numpy()
    return matthews_corrcoef(labels.reshape(-1), predictions.reshape(-1))


def optimize_thresholds_mcc(model, train_loader, num_classes=NUM_CLASSES, device=DEVICE):
    """
    Optimize per-class decision thresholds to maximize MCC on training data.

    Performs grid search over [0.05, 0.95] for each class independently.
    """
    model.eval()
    all_logits, all_labels = [], []

    with torch.no_grad():
        for esm, mask, phys, labels in train_loader:
            esm, mask, phys = esm.to(device), mask.to(device), phys.to(device)
            out, _ = model(esm, mask, phys)
            all_logits.append(out.cpu().numpy())
            all_labels.append(labels.numpy())

    logits = np.vstack(all_logits)
    labels = np.vstack(all_labels)

    optimal_thresholds = []
    for class_idx in range(num_classes):
        best_mcc = -1.0
        best_thresh = 0.5

        for thresh in np.arange(0.05, 0.96, 0.01):
            preds = (torch.sigmoid(torch.tensor(logits[:, class_idx])) > thresh).numpy()
            mcc = matthews_corrcoef(labels[:, class_idx], preds)
            if mcc > best_mcc:
                best_mcc = mcc
                best_thresh = thresh

        optimal_thresholds.append(best_thresh)

    return optimal_thresholds


def evaluate_with_thresholds(model, val_loader, thresholds, num_classes=NUM_CLASSES, device=DEVICE):
    """
    Evaluate model using per-class optimized thresholds.

    Computes accuracy, Jaccard, micro-F1, macro-F1, per-class MCC, and
    average prediction cardinality.
    """
    model.eval()
    all_logits, all_labels = [], []

    with torch.no_grad():
        for esm, mask, phys, labels in val_loader:
            esm, mask, phys = esm.to(device), mask.to(device), phys.to(device)
            out, _ = model(esm, mask, phys)
            all_logits.append(out.cpu().numpy())
            all_labels.append(labels.numpy())

    logits = np.vstack(all_logits)
    labels = np.vstack(all_labels)

    # Apply per-class thresholds
    predictions = []
    for class_idx in range(num_classes):
        pred = (torch.sigmoid(torch.tensor(logits[:, class_idx])) > thresholds[class_idx]).numpy()
        predictions.append(pred.reshape(-1, 1))

    predictions = np.hstack(predictions)

    metrics = {}
    metrics['accuracy'] = accuracy_score(labels, predictions)
    metrics['jaccard'] = jaccard_score(labels, predictions, average='samples', zero_division=0)
    metrics['micro_f1'] = f1_score(labels, predictions, average='samples', zero_division=0)

    class_f1s = [f1_score(labels[:, i], predictions[:, i], zero_division=0) for i in range(num_classes)]
    metrics['macro_f1'] = np.mean(class_f1s)

    class_mccs = [matthews_corrcoef(labels[:, i], predictions[:, i]) for i in range(num_classes)]
    metrics['class_mccs'] = class_mccs

    metrics['avg_pred_labels'] = np.mean(np.sum(predictions, axis=1))

    return metrics


def monitor_gate_weights(model, val_loader, device=DEVICE):
    """Monitor gate activation statistics during validation."""
    model.eval()
    all_gates = []

    with torch.no_grad():
        for esm, mask, phys, _ in val_loader:
            esm, mask, phys = esm.to(device), mask.to(device), phys.to(device)
            _, gates = model(esm, mask, phys)
            all_gates.append(gates.cpu().numpy())

    gates = np.vstack(all_gates)
    return gates[:, 0].mean(), gates[:, 1].mean(), gates


def evaluate_and_monitor_gates(model, val_loader, device=DEVICE):
    """Single-pass validation with simultaneous MCC and gate monitoring."""
    model.eval()
    all_logits, all_labels, all_gates = [], [], []

    with torch.no_grad():
        for esm, mask, phys, labels in val_loader:
            esm, mask, phys = esm.to(device), mask.to(device), phys.to(device)
            out, gates = model(esm, mask, phys)
            all_logits.append(out.cpu().numpy())
            all_labels.append(labels.numpy())
            all_gates.append(gates.cpu().numpy())

    logits = np.vstack(all_logits)
    labels = np.vstack(all_labels)
    predictions = (torch.sigmoid(torch.tensor(logits)) > 0.5).numpy()
    mcc = matthews_corrcoef(labels.reshape(-1), predictions.reshape(-1))

    gates = np.vstack(all_gates)
    return mcc, gates[:, 0].mean(), gates[:, 1].mean(), gates


# ==========================================
# 4. TRAINING WITH GATE REGULARIZATION
# ==========================================

def train_single_fold(
    fold_num,
    csv_path="BioGraphXEncodedFeatures.csv",
    esm_dirs=("esm_embeddings_ml",),
    output_dir=".",
    fixed_epochs=35,
    batch_size=64,
    learning_rate=1e-5,
    physics_lr_multiplier=3.0,
    gate_regularization_epochs=10,
    esm_dim=2560,
    num_workers=8,
    device=DEVICE,
):
    """
    Train and evaluate a single cross-validation fold of the Gated Hybrid model.

    For the first `gate_regularization_epochs` epochs, an auxiliary loss term
    nudges the physics gate toward 0.5 to discourage the gate from collapsing
    onto a single pathway before both branches have learned anything useful.

    Args:
        fold_num: Validation fold index (must match a value present in the
            CSV's `Partition` column).
        csv_path: Path to the BioGraphX-encoded CSV (ACC, Kingdom, Partition,
            the 11 target columns, and the 157 physics feature columns -
            produced by `BioGraphX-Encoding/src/run.py`).
        esm_dirs: Directory (or list of directories) of cached `{ACC}.npz`
            ESM embeddings, as produced by `esm_embeddings.py`.
        output_dir: Directory to write the checkpoint, fitted scaler, gate
            history, and results summary to.
        fixed_epochs: Number of training epochs.
        batch_size, learning_rate, physics_lr_multiplier: Optimizer settings.
        gate_regularization_epochs: Number of initial epochs during which the
            physics-gate regularization term is applied.
        esm_dim: Expected ESM embedding hidden size.
        num_workers: DataLoader worker processes.
        device: "cuda" or "cpu".

    Returns:
        dict: Metrics, thresholds, and gate statistics for this fold.
    """
    if isinstance(esm_dirs, str):
        esm_dirs = [esm_dirs]
    os.makedirs(output_dir, exist_ok=True)

    print(f"TRAINING FOLD {fold_num} (HYBRID)")
    print("=" * 60)

    # -----------------------------------------------------------------
    # Data Loading and Preparation
    # -----------------------------------------------------------------
    esm_index = build_esm_index(esm_dirs)
    df = pd.read_csv(csv_path)

    print(f"Dataset loaded: {len(df)} total entries")
    print(f"Target columns ({NUM_CLASSES}): {TARGET_COLS}")

    if 'Partition' not in df.columns:
        raise ValueError("No 'Partition' column found in dataset!")
    part_col = 'Partition'

    unique_partitions = sorted(df[part_col].unique())
    print(f"Unique partitions in data: {unique_partitions}")

    if fold_num not in unique_partitions:
        print(f"Warning: fold {fold_num} not in data partitions. Using fold {unique_partitions[0]} instead.")
        fold_num = unique_partitions[0]

    train_df = df[df[part_col] != fold_num].reset_index(drop=True)
    val_df = df[df[part_col] == fold_num].reset_index(drop=True)

    print(f"\nFold {fold_num} Split:")
    print(f"  Training partitions: {sorted(train_df[part_col].unique())} - {len(train_df)} proteins")
    print(f"  Validation partition: {fold_num} - {len(val_df)} proteins")

    feat_cols = [col for col in df.columns if col not in META_COLS + [part_col]]
    print(f"\nFeature columns: {len(feat_cols)} physics features")
    print(f"Sample features: {feat_cols[:5]}...")

    # Standardize features - the fitted scaler is saved so inference.py can
    # apply the identical transform to new data instead of refitting on it.
    scaler = StandardScaler()
    train_phys = scaler.fit_transform(train_df[feat_cols]).astype(np.float32)
    val_phys = scaler.transform(val_df[feat_cols]).astype(np.float32)
    scaler_path = os.path.join(output_dir, f"scaler_fold_{fold_num}.pkl")
    joblib.dump({"scaler": scaler, "feature_columns": feat_cols}, scaler_path)
    print(f"Fitted scaler saved to: {scaler_path}")

    y_train = train_df[TARGET_COLS].values.astype(np.float32)
    y_val = val_df[TARGET_COLS].values.astype(np.float32)

    train_loader = DataLoader(
        BioGraphX_Dataset(train_df, esm_index, train_phys, y_train, esm_dim=esm_dim),
        batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=num_workers
    )
    val_loader = DataLoader(
        BioGraphX_Dataset(val_df, esm_index, val_phys, y_val, esm_dim=esm_dim),
        batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    # -----------------------------------------------------------------
    # Model Initialization
    # -----------------------------------------------------------------
    model = BioGraphX_Hybrid(
        esm_dim=esm_dim, biographx_dim=len(feat_cols), num_classes=NUM_CLASSES
    ).to(device)
    print(f"\nModel initialized with {len(feat_cols)} physics features and {NUM_CLASSES} classes")

    criterion = FocalLoss(alpha=0.25, gamma=2.0).to(device)

    # -----------------------------------------------------------------
    # Parameter Grouping with Differential Learning Rates
    # -----------------------------------------------------------------
    physics_params, esm_params, gate_params, classifier_params = [], [], [], []
    for name, param in model.named_parameters():
        if 'phys_branch' in name:
            physics_params.append(param)
        elif 'attn' in name or 'esm_bottleneck' in name:
            esm_params.append(param)
        elif 'gate' in name:
            gate_params.append(param)
        else:
            classifier_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': physics_params, 'lr': learning_rate * physics_lr_multiplier},
        {'params': esm_params, 'lr': learning_rate},
        {'params': gate_params, 'lr': learning_rate * 0.5},
        {'params': classifier_params, 'lr': learning_rate}
    ], weight_decay=1e-2)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=fixed_epochs)

    # -----------------------------------------------------------------
    # Training Loop
    # -----------------------------------------------------------------
    print(f"\nTraining for {fixed_epochs} fixed epochs...")
    print("-" * 60)

    best_val_mcc = -1.0
    best_model_state = None
    best_epoch = 0
    gate_history = []

    for epoch in range(fixed_epochs):
        model.train()
        train_loss = 0

        for esm, mask, phys, labels in train_loader:
            esm, mask, phys, labels = (
                esm.to(device), mask.to(device), phys.to(device), labels.to(device)
            )

            optimizer.zero_grad()
            outputs, gate_weights = model(esm, mask, phys)
            loss = criterion(outputs, labels)

            # Gate regularization (first `gate_regularization_epochs` epochs only):
            # nudges the physics gate toward 0.5 to prevent pathway collapse
            # before both branches are informative.
            if epoch < gate_regularization_epochs:
                physics_gate = gate_weights[:, 0].mean()
                physics_encouragement = F.mse_loss(
                    physics_gate, torch.tensor(0.5).to(device)
                )
                loss = loss + 0.1 * physics_encouragement

            loss.backward()
            torch.nn.utils.clip_grad_norm_(physics_params, 5.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        val_mcc, physics_gate_mean, esm_gate_mean, gates = evaluate_and_monitor_gates(
            model, val_loader, device=device
        )
        gate_history.append({
            'epoch': epoch,
            'physics_gate': physics_gate_mean,
            'esm_gate': esm_gate_mean,
            'gate_ratio': esm_gate_mean / (physics_gate_mean + 1e-8)
        })

        checkpoint_path = os.path.join(output_dir, f"best_model_fold_{fold_num}.pth")
        if val_mcc > best_val_mcc:
            best_val_mcc = val_mcc
            best_model_state = model.state_dict().copy()
            best_epoch = epoch
            torch.save(best_model_state, checkpoint_path)

        gate_info = f"Gates: P={physics_gate_mean:.3f}, E={esm_gate_mean:.3f}"
        print(f"Epoch {epoch+1:2d}/{fixed_epochs}: Loss={train_loss/len(train_loader):.4f}, "
              f"Val MCC={val_mcc:.4f} (Best: {best_val_mcc:.4f}) {gate_info}")

        if epoch == gate_regularization_epochs - 1:
            print("\nGate regularization phase complete. Continuing full training...")

    # -----------------------------------------------------------------
    # Post-Training Analysis
    # -----------------------------------------------------------------
    model.load_state_dict(best_model_state)
    print(f"\nBest model from epoch {best_epoch+1} (MCC: {best_val_mcc:.4f})")
    checkpoint_path = os.path.join(output_dir, f"best_model_fold_{fold_num}.pth")
    torch.save(best_model_state, checkpoint_path)
    print(f"Best model saved to: {checkpoint_path}")

    gate_df = pd.DataFrame(gate_history)
    gate_history_path = os.path.join(output_dir, f"gate_history_fold_{fold_num}.csv")
    gate_df.to_csv(gate_history_path, index=False)

    print("\nFinal Gate Analysis:")
    final_physics_gate, final_esm_gate, _ = monitor_gate_weights(model, val_loader, device=device)
    gate_ratio = final_esm_gate / final_physics_gate
    print(f"  Physics gate mean: {final_physics_gate:.3f}")
    print(f"  ESM gate mean: {final_esm_gate:.3f}")
    print(f"  Gate ratio (ESM/Physics): {gate_ratio:.2f}:1")
    print(f"  Physics contribution: {final_physics_gate/(final_physics_gate+final_esm_gate)*100:.1f}%")

    print("\nOptimizing thresholds on training data...")
    optimal_thresholds = optimize_thresholds_mcc(model, train_loader, device=device)
    print(f"Optimal thresholds: {[f'{t:.2f}' for t in optimal_thresholds]}")
    print(f"Average threshold: {np.mean(optimal_thresholds):.2f}")

    print("\nEvaluating with optimized thresholds...")
    metrics = evaluate_with_thresholds(model, val_loader, optimal_thresholds, device=device)

    results = {
        'fold': fold_num,
        'accuracy': metrics['accuracy'],
        'jaccard': metrics['jaccard'],
        'micro_f1': metrics['micro_f1'],
        'macro_f1': metrics['macro_f1'],
        'avg_pred_labels': metrics['avg_pred_labels'],
        'class_mccs': metrics['class_mccs'],
        'thresholds': optimal_thresholds,
        'best_epoch': best_epoch + 1,
        'best_val_mcc': best_val_mcc,
        'final_physics_gate': final_physics_gate,
        'final_esm_gate': final_esm_gate,
        'gate_ratio': gate_ratio,
        'physics_contribution_pct': final_physics_gate/(final_physics_gate+final_esm_gate)*100,
        'num_classes': NUM_CLASSES,
        'target_columns': TARGET_COLS,
        'train_size': len(train_df),
        'val_size': len(val_df),
        'model_type': 'hybrid',
        'checkpoint_path': checkpoint_path,
        'scaler_path': scaler_path,
        'timestamp': datetime.now().isoformat()
    }

    results_df = pd.DataFrame([results])
    results_file = os.path.join(output_dir, f"fold_{fold_num}_results.csv")
    results_df.to_csv(results_file, index=False)

    print(f"\nFold {fold_num} Results:")
    print("-" * 50)
    print(f"  Micro F1:          {metrics['micro_f1']:.4f}")
    print(f"  Macro F1:          {metrics['macro_f1']:.4f}")
    print(f"  Accuracy:          {metrics['accuracy']:.4f}")
    print(f"  Best MCC:          {best_val_mcc:.4f}")
    print(f"  Physics gate:      {final_physics_gate:.3f}")
    print(f"  ESM gate:          {final_esm_gate:.3f}")
    print(f"  Physics %:         {results['physics_contribution_pct']:.1f}%")
    print(f"  Best epoch:        {best_epoch + 1}")
    print(f"\nResults saved to: {results_file}")

    return results


# ==========================================
# MAIN EXECUTION
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Train the BioGraphX Gated Hybrid localization model for one fold.")
    parser.add_argument("--csv-path", default="BioGraphXEncodedFeatures.csv",
                        help="CSV produced by BioGraphX-Encoding/src/run.py (with ACC, Kingdom, Partition, "
                             "target columns, and 157 physics feature columns).")
    parser.add_argument("--esm-dirs", nargs="+", default=["esm_embeddings_ml"],
                        help="One or more directories of cached {ACC}.npz ESM embeddings.")
    parser.add_argument("--fold", type=int, default=0, help="Validation fold (must match a Partition value).")
    parser.add_argument("--output-dir", default="results/hybrid")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--physics-lr-multiplier", type=float, default=3.0)
    parser.add_argument("--gate-regularization-epochs", type=int, default=10)
    parser.add_argument("--esm-dim", type=int, default=2560)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    print("HYBRID TRAINING")
    print("=" * 70)
    print(f"Target Classes: {NUM_CLASSES}")
    print(f"Fixed Epochs: {args.epochs}")
    print(f"Device: {DEVICE}")
    print(f"Fold Number: {args.fold}")
    print(f"Physics LR Multiplier: {args.physics_lr_multiplier}x")
    print("=" * 70)

    import time
    start_time = time.time()

    try:
        results = train_single_fold(
            fold_num=args.fold,
            csv_path=args.csv_path,
            esm_dirs=args.esm_dirs,
            output_dir=args.output_dir,
            fixed_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            physics_lr_multiplier=args.physics_lr_multiplier,
            gate_regularization_epochs=args.gate_regularization_epochs,
            esm_dim=args.esm_dim,
            num_workers=args.num_workers,
        )

        total_time = time.time() - start_time
        hours, remainder = divmod(int(total_time), 3600)
        minutes, seconds = divmod(remainder, 60)

        print(f"\nFOLD {args.fold} COMPLETED SUCCESSFULLY!")
        print(f"Total time: {hours}h {minutes}m {seconds}s")
        print(f"Micro F1: {results['micro_f1']:.4f}")
        print(f"Physics Contribution: {results['physics_contribution_pct']:.1f}%")
        print(f"Gate Ratio: {results['gate_ratio']:.2f}:1")

        summary_path = os.path.join(args.output_dir, f"fold_{args.fold}_summary.txt")
        with open(summary_path, 'w') as f:
            f.write(f"Hybrid - Fold {args.fold} Summary\n")
            f.write(f"Time: {datetime.now().isoformat()}\n")
            f.write(f"Total time: {hours}h {minutes}m {seconds}s\n")
            f.write(f"Micro F1: {results['micro_f1']:.4f}\n")
            f.write(f"Accuracy: {results['accuracy']:.4f}\n")
            f.write(f"Physics Contribution: {results['physics_contribution_pct']:.1f}%\n")
            f.write(f"Gate Ratio: {results['gate_ratio']:.2f}:1\n")
            f.write(f"Best epoch: {results['best_epoch']}\n")
            f.write(f"Best MCC: {results['best_val_mcc']:.4f}\n")

    except Exception as e:
        print(f"\nERROR during fold {args.fold}: {e}")
        import traceback
        traceback.print_exc()

        os.makedirs(args.output_dir, exist_ok=True)
        error_path = os.path.join(args.output_dir, f"fold_{args.fold}_error.txt")
        with open(error_path, 'w') as f:
            f.write(f"Error at: {datetime.now().isoformat()}\n")
            f.write(f"Fold: {args.fold}\n")
            f.write(f"Error: {str(e)}\n")
            f.write(traceback.format_exc())


if __name__ == "__main__":
    main()
