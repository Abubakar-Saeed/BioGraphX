"""
Author: Abubakar Saeed
Created: January 2026
Last Modified: February 2026

Description:
    Hybrid neural network architecture for subcellular localization prediction
    integrating ESM protein language model embeddings with BioGraphX biophysical features.
    
    Implements adaptive gating mechanism that learns to weight ESM transformer features
    against physics-based feature representations dynamically per sample. Features
    physics-first warm-up training strategy to establish meaningful biophysical
    representations before full joint optimization.

    All distance-based features in physics branch originate from linear sequence
    positions, NOT 3D spatial coordinates.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import glob
import json
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, matthews_corrcoef, jaccard_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION FOR SINGLE FOLD
# ==========================================

# Input data paths
CSV_PATH = "BioGraphXEncodedFeatures.csv"
ESM_DIRS = [
    "esm_embeddings_ml/.npz"
]

# MANUAL FOLD SELECTION: CHANGE FOR EACH CROSS-VALIDATION RUN
FOLD_NUM = 4  # Values: 0, 1, 2, 3, 4

# Target localization compartments (11-class multi-label)
TARGET_COLS = [
    "Cytoplasm", "Nucleus", "Extracellular", "Cell membrane", 
    "Mitochondrion", "Endoplasmic reticulum", "Lysosome/Vacuole", 
    "Golgi apparatus", "Peroxisome", "Plastid", "Membrane"
]

NUM_CLASSES = len(TARGET_COLS)

# Metadata columns to exclude from feature matrix
META_COLS = ['ACC', 'Kingdom', 'Partition', 'Sequence'] + TARGET_COLS

# Hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 1e-5
PHYSICS_LR_MULTIPLIER = 3.0  # Higher learning rate for physics branch initialization
FIXED_EPOCHS = 35            # Deterministic training schedule
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 8


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
    def __init__(self, df, esm_index, physics_array, label_matrix):
        self.entry_ids = df["ACC"].values
        self.labels = torch.tensor(label_matrix, dtype=torch.float32)
        self.esm_index = esm_index
        self.physics = torch.tensor(physics_array, dtype=torch.float32)

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
                    esm_tensor = esm_tensor.unsqueeze(0)  # [2560] → [1, 2560]
            except Exception as e:
                print(f"⚠️ Error loading {entry_id}: {e}")
                esm_tensor = torch.zeros((1, 2560), dtype=torch.float32)
        else:
            esm_tensor = torch.zeros((1, 2560), dtype=torch.float32)
        
        return esm_tensor, self.physics[idx], self.labels[idx]


def collate_fn(batch):
    """
    Custom collation function for variable-length ESM sequences.
    
    Pads sequences to maximum length in batch and creates attention mask.
    
    Returns:
        esm_padded: Padded ESM embeddings (batch, max_len, 2560)
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
    hybrid architecture with adaptive gating between ESM and physics pathways.
    
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
        #  PHYSICS BRANCH: Deeper architecture for biophysical features
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
        #  GATE CONTROLLER: Sample-specific adaptive weighting
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
        
         # bypass connection 
        self.physics_bypass = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.ReLU()
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
def evaluate_validation_mcc(model, val_loader):
    """
    Evaluate validation Matthews Correlation Coefficient with 0.5 threshold.
    
    Args:
        model: PyTorch model
        val_loader: Validation DataLoader
        
    Returns:
        float: MCC score aggregated across all predictions
    """
    model.eval()
    all_logits, all_labels = [], []
    
    with torch.no_grad():
        for esm, mask, phys, labels in val_loader:
            esm, mask, phys = esm.to(DEVICE), mask.to(DEVICE), phys.to(DEVICE)
            out, _ = model(esm, mask, phys)
            all_logits.append(out.cpu().numpy())
            all_labels.append(labels.numpy())
    
    logits = np.vstack(all_logits)
    labels = np.vstack(all_labels)
    predictions = (torch.sigmoid(torch.tensor(logits)) > 0.5).numpy()
    mcc = matthews_corrcoef(labels.reshape(-1), predictions.reshape(-1))
    return mcc


def optimize_thresholds_mcc(model, train_loader):
    """
    Optimize per-class decision thresholds to maximize MCC on training data.
    
    Performs grid search over [0.05, 0.95] for each class independently.
    
    Args:
        model: PyTorch model
        train_loader: Training DataLoader
        
    Returns:
        list: Optimal thresholds for each class
    """
    model.eval()
    all_logits, all_labels = [], []
    
    with torch.no_grad():
        for esm, mask, phys, labels in train_loader:
            esm, mask, phys = esm.to(DEVICE), mask.to(DEVICE), phys.to(DEVICE)
            out, _ = model(esm, mask, phys)
            all_logits.append(out.cpu().numpy())
            all_labels.append(labels.numpy())
    
    logits = np.vstack(all_logits)
    labels = np.vstack(all_labels)
    
    optimal_thresholds = []
    for class_idx in range(NUM_CLASSES):
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


def evaluate_with_thresholds(model, val_loader, thresholds):
    """
    Evaluate model using per-class optimized thresholds.
    
    Computes comprehensive metrics: accuracy, Jaccard, micro-F1, macro-F1,
    per-class MCC, and average prediction cardinality.
    
    Args:
        model: PyTorch model
        val_loader: Validation DataLoader
        thresholds: List of per-class decision thresholds
        
    Returns:
        dict: Dictionary containing all evaluation metrics
    """
    model.eval()
    all_logits, all_labels = [], []
    
    with torch.no_grad():
        for esm, mask, phys, labels in val_loader:
            esm, mask, phys = esm.to(DEVICE), mask.to(DEVICE), phys.to(DEVICE)
            out, _ = model(esm, mask, phys)
            all_logits.append(out.cpu().numpy())
            all_labels.append(labels.numpy())
    
    logits = np.vstack(all_logits)
    labels = np.vstack(all_labels)
    
    # Apply per-class thresholds
    predictions = []
    for class_idx in range(NUM_CLASSES):
        pred = (torch.sigmoid(torch.tensor(logits[:, class_idx])) > thresholds[class_idx]).numpy()
        predictions.append(pred.reshape(-1, 1))
    
    predictions = np.hstack(predictions)
    
    # Compute metrics
    metrics = {}
    metrics['accuracy'] = accuracy_score(labels, predictions)
    metrics['jaccard'] = jaccard_score(labels, predictions, average='samples', zero_division=0)
    metrics['micro_f1'] = f1_score(labels, predictions, average='samples', zero_division=0)
    
    class_f1s = []
    for i in range(NUM_CLASSES):
        f1 = f1_score(labels[:, i], predictions[:, i], zero_division=0)
        class_f1s.append(f1)
    metrics['macro_f1'] = np.mean(class_f1s)
    
    class_mccs = []
    for i in range(NUM_CLASSES):
        mcc = matthews_corrcoef(labels[:, i], predictions[:, i])
        class_mccs.append(mcc)
    metrics['class_mccs'] = class_mccs
    
    metrics['avg_pred_labels'] = np.mean(np.sum(predictions, axis=1))
    
    return metrics


def monitor_gate_weights(model, val_loader):
    """
    Monitor gate activation statistics during validation.
    
    Args:
        model: PyTorch model
        val_loader: Validation DataLoader
        
    Returns:
        tuple: (physics_gate_mean, esm_gate_mean, all_gates)
    """
    model.eval()
    all_gates = []
    
    with torch.no_grad():
        for esm, mask, phys, _ in val_loader:
            esm, mask, phys = esm.to(DEVICE), mask.to(DEVICE), phys.to(DEVICE)
            _, gates = model(esm, mask, phys)
            all_gates.append(gates.cpu().numpy())
    
    gates = np.vstack(all_gates)
    physics_gate_mean = gates[:, 0].mean()
    esm_gate_mean = gates[:, 1].mean()
    
    return physics_gate_mean, esm_gate_mean, gates


def evaluate_and_monitor_gates(model, val_loader):
    """
    Single-pass validation with simultaneous MCC and gate monitoring.
    
    Args:
        model: PyTorch model
        val_loader: Validation DataLoader
        
    Returns:
        tuple: (mcc, physics_gate_mean, esm_gate_mean, all_gates)
    """
    model.eval()
    all_logits, all_labels, all_gates = [], [], []
    
    with torch.no_grad():
        for esm, mask, phys, labels in val_loader:
            esm, mask, phys = esm.to(DEVICE), mask.to(DEVICE), phys.to(DEVICE)
            out, gates = model(esm, mask, phys)
            all_logits.append(out.cpu().numpy())
            all_labels.append(labels.numpy())
            all_gates.append(gates.cpu().numpy())
    
    # Calculate MCC with 0.5 threshold
    logits = np.vstack(all_logits)
    labels = np.vstack(all_labels)
    predictions = (torch.sigmoid(torch.tensor(logits)) > 0.5).numpy()
    mcc = matthews_corrcoef(labels.reshape(-1), predictions.reshape(-1))
    
    # Calculate gate statistics
    gates = np.vstack(all_gates)
    physics_gate_mean = gates[:, 0].mean()
    esm_gate_mean = gates[:, 1].mean()
    
    return mcc, physics_gate_mean, esm_gate_mean, gates


# ==========================================
# 4. TRAINING WITH PHYSICS-FIRST STRATEGY
# ==========================================

def train_single_fold(fold_num, fixed_epochs=FIXED_EPOCHS):
    """
    Train and evaluate single fold with physics-first warm-up strategy.
    
    Two-phase training:
        Phase 1 (epochs 1-5): Physics branch warm-up with gate regularization
        Phase 2 (epochs 6-35): Full joint optimization
    
    Args:
        fold_num: Validation fold index (0-4)
        fixed_epochs: Number of training epochs
        
    Returns:
        dict: Results dictionary containing metrics, thresholds, and gate statistics
    """
    print(f"🚀 TRAINING FOLD {fold_num} (HYBRID)")
    print("=" * 60)
    
    # -----------------------------------------------------------------
    # Data Loading and Preparation
    # -----------------------------------------------------------------
    esm_index = build_esm_index(ESM_DIRS)
    df = pd.read_csv(CSV_PATH)
    
    print(f"Dataset loaded: {len(df)} total entries")
    print(f"Target columns ({NUM_CLASSES}): {TARGET_COLS}")
    
    # Determine partition column for cross-validation
    if 'Partition' in df.columns:
        part_col = 'Partition'
        print(f"Using partition column: {part_col}")
    else:
        raise ValueError("No partition column found in dataset!")
    
    # Verify fold existence
    unique_partitions = sorted(df[part_col].unique())
    print(f"Unique partitions in data: {unique_partitions}")
    
    if fold_num not in unique_partitions:
        print(f" Warning: Fold {fold_num} not in data partitions. Using fold 0 instead.")
        fold_num = 0
    
    # Split data for this fold
    train_df = df[df[part_col] != fold_num].reset_index(drop=True)
    val_df = df[df[part_col] == fold_num].reset_index(drop=True)
    
    print(f"\nFold {fold_num} Split:")
    print(f"  Training partitions: {sorted(train_df[part_col].unique())} - {len(train_df)} proteins")
    print(f"  Validation partition: {fold_num} - {len(val_df)} proteins")
    
    # Identify feature columns (exclude metadata and targets)
    all_exclude_cols = META_COLS + [part_col]
    feat_cols = [col for col in df.columns if col not in all_exclude_cols]
    
    print(f"\nFeature columns: {len(feat_cols)} physics features")
    print(f"Sample features: {feat_cols[:5]}...")
    
    # Standardize features
    scaler = StandardScaler()
    train_phys = scaler.fit_transform(train_df[feat_cols]).astype(np.float32)
    val_phys = scaler.transform(val_df[feat_cols]).astype(np.float32)
    
    # Extract labels
    y_train = train_df[TARGET_COLS].values.astype(np.float32)
    y_val = val_df[TARGET_COLS].values.astype(np.float32)
    
    # Create DataLoaders
    train_loader = DataLoader(
        BioGraphX_Dataset(train_df, esm_index, train_phys, y_train),
        batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        BioGraphX_Dataset(val_df, esm_index, val_phys, y_val),
        batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )
    
    # -----------------------------------------------------------------
    # Model Initialization
    # -----------------------------------------------------------------
    model = BioGraphX_Hybrid(
        biographx_dim=len(feat_cols),
        num_classes=NUM_CLASSES
    ).to(DEVICE)
    print(f"\nModel initialized with {len(feat_cols)} physics features and {NUM_CLASSES} classes")
    
    # Loss function
    criterion = FocalLoss(alpha=0.25, gamma=2.0).to(DEVICE)
    
    # -----------------------------------------------------------------
    # Parameter Grouping with Differential Learning Rates
    # -----------------------------------------------------------------
    physics_params = []
    esm_params = []
    gate_params = []
    classifier_params = []
    
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
        {'params': physics_params, 'lr': LEARNING_RATE * PHYSICS_LR_MULTIPLIER},
        {'params': esm_params, 'lr': LEARNING_RATE},
        {'params': gate_params, 'lr': LEARNING_RATE * 0.5},
        {'params': classifier_params, 'lr': LEARNING_RATE}
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
    
    # PHASE 1: Physics branch warm-up (epochs 0-4)
    print("\n PHASE 1: Physics branch warm-up (epochs 1-5)")
    
    for epoch in range(fixed_epochs):
        # Training
        model.train()
        train_loss = 0
        
        for esm, mask, phys, labels in train_loader:
            esm, mask, phys, labels = (
                esm.to(DEVICE), mask.to(DEVICE), phys.to(DEVICE), labels.to(DEVICE)
            )
            
            optimizer.zero_grad()
            outputs, gate_weights = model(esm, mask, phys)
            loss = criterion(outputs, labels)
            
            # Physics encouragement loss (first 10 epochs only)
            # Regularizes gates toward 0.5 to prevent pathway collapse
            if epoch < 10:
                physics_gate = gate_weights[:, 0].mean()
                physics_encouragement = F.mse_loss(
                    physics_gate, torch.tensor(0.5).to(DEVICE)
                )
                loss = loss + 0.1 * physics_encouragement
            
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(physics_params, 5.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        scheduler.step()
        
        # Validation
        val_mcc, physics_gate_mean, esm_gate_mean, gates = evaluate_and_monitor_gates(
            model, val_loader
        )
        gate_history.append({
            'epoch': epoch,
            'physics_gate': physics_gate_mean,
            'esm_gate': esm_gate_mean,
            'gate_ratio': esm_gate_mean / (physics_gate_mean + 1e-8)
        })
        
        # Save best model based on validation MCC
        if val_mcc > best_val_mcc:
            best_val_mcc = val_mcc
            best_model_state = model.state_dict().copy()
            best_epoch = epoch
            torch.save(best_model_state, f"best_model_fold_{fold_num}.pth")
        
        # Progress reporting
        gate_info = f"Gates: P={physics_gate_mean:.3f}, E={esm_gate_mean:.3f}"
        print(f"Epoch {epoch+1:2d}/{fixed_epochs}: Loss={train_loss/len(train_loader):.4f}, "
              f"Val MCC={val_mcc:.4f} (Best: {best_val_mcc:.4f}) {gate_info}")
        
        # Phase transition notification
        if epoch == 4:
            print("\n Physics warm-up complete. Starting full training...")
    
    # -----------------------------------------------------------------
    # Post-Training Analysis
    # -----------------------------------------------------------------
    # Load best model
    model.load_state_dict(best_model_state)
    print(f"\nBest model from epoch {best_epoch+1} (MCC: {best_val_mcc:.4f})")
    torch.save(best_model_state, f"best_model_fold_{fold_num}.pth")
    print(f" Best model saved to: best_model_fold_{fold_num}.pth")
    
    # Save gate history
    gate_df = pd.DataFrame(gate_history)
    gate_df.to_csv(f"gate_history_fold_{fold_num}.csv", index=False)
    
    # Final gate analysis
    print(f"\n🔬 Final Gate Analysis:")
    final_physics_gate, final_esm_gate, _ = monitor_gate_weights(model, val_loader)
    gate_ratio = final_esm_gate / final_physics_gate
    print(f"  Physics gate mean: {final_physics_gate:.3f}")
    print(f"  ESM gate mean: {final_esm_gate:.3f}")
    print(f"  Gate ratio (ESM/Physics): {gate_ratio:.2f}:1")
    print(f"  Physics contribution: {final_physics_gate/(final_physics_gate+final_esm_gate)*100:.1f}%")
    
    # Optimize thresholds on training data
    print("\nOptimizing thresholds on training data...")
    optimal_thresholds = optimize_thresholds_mcc(model, train_loader)
    print(f"Optimal thresholds: {[f'{t:.2f}' for t in optimal_thresholds]}")
    print(f"Average threshold: {np.mean(optimal_thresholds):.2f}")
    
    # Final evaluation with optimized thresholds
    print("\nEvaluating with optimized thresholds...")
    metrics = evaluate_with_thresholds(model, val_loader, optimal_thresholds)
    
    # -----------------------------------------------------------------
    # Results Compilation
    # -----------------------------------------------------------------
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
        'timestamp': datetime.now().isoformat()
    }
    
    # Save results
    results_df = pd.DataFrame([results])
    results_file = f"fold_{fold_num}_results.csv"
    results_df.to_csv(results_file, index=False)
    
    print(f"\n Fold {fold_num} Results :")
    print("-" * 50)
    print(f"  Micro F1:          {metrics['micro_f1']:.4f}")
    print(f"  Macro F1:          {metrics['macro_f1']:.4f}")
    print(f"  Accuracy:          {metrics['accuracy']:.4f}")
    print(f"  Best MCC:          {best_val_mcc:.4f}")
    print(f"  Physics gate:      {final_physics_gate:.3f}")
    print(f"  ESM gate:          {final_esm_gate:.3f}")
    print(f"  Physics %:         {results['physics_contribution_pct']:.1f}%")
    print(f"  Best epoch:        {best_epoch + 1}")
    print(f"\n Results saved to: {results_file}")
    
    # Compare with baseline
    print(f"\n ANALYSIS:")
    print(f"  Physics-only Micro F1: 0.6020 (baseline)")
    
    return results


# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    print(" HYBRID TRAINING")
    print("=" * 70)
    print(f"Target Classes: {NUM_CLASSES}")
    print(f"Fixed Epochs: {FIXED_EPOCHS}")
    print(f"Device: {DEVICE}")
    print(f"Fold Number: {FOLD_NUM}")
    print(f"Physics LR Multiplier: {PHYSICS_LR_MULTIPLIER}x")
    print("=" * 70)
    
    # Start timing
    import time
    start_time = time.time()
    
    try:
        results = train_single_fold(FOLD_NUM)
        
        # Calculate total runtime
        total_time = time.time() - start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        
        print(f"\n FOLD {FOLD_NUM} COMPLETED SUCCESSFULLY!")
        print(f" Total time: {hours}h {minutes}m {seconds}s")
        print(f" Micro F1: {results['micro_f1']:.4f}")
        print(f" Physics Contribution: {results['physics_contribution_pct']:.1f}%")
        print(f" Gate Ratio: {results['gate_ratio']:.2f}:1")
        
        # Save summary
        with open(f'fold_{FOLD_NUM}_summary.txt', 'w') as f:
            f.write(f"Hybrid - Fold {FOLD_NUM} Summary\n")
            f.write(f"Time: {datetime.now().isoformat()}\n")
            f.write(f"Total time: {hours}h {minutes}m {seconds}s\n")
            f.write(f"Micro F1: {results['micro_f1']:.4f}\n")
            f.write(f"Accuracy: {results['accuracy']:.4f}\n")
            f.write(f"Physics Contribution: {results['physics_contribution_pct']:.1f}%\n")
            f.write(f"Gate Ratio: {results['gate_ratio']:.2f}:1\n")
            f.write(f"Best epoch: {results['best_epoch']}\n")
            f.write(f"Best MCC: {results['best_val_mcc']:.4f}\n")
        
    except Exception as e:
        print(f"\n❌ ERROR during fold {FOLD_NUM}: {e}")
        import traceback
        traceback.print_exc()
        
        # Save error information
        with open(f'fold_{FOLD_NUM}_error.txt', 'w') as f:
            f.write(f"Error at: {datetime.now().isoformat()}\n")
            f.write(f"Fold: {FOLD_NUM}\n")
            f.write(f"Error: {str(e)}\n")
            f.write(traceback.format_exc())
