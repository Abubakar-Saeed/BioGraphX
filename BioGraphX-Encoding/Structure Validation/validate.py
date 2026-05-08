import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    r2_score, mean_squared_error, accuracy_score,
    precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef
)
from scipy.stats import pearsonr

# =====================
# 1. Load data & define features (using last 158 columns)
# =====================
train_df = pd.read_csv("/kaggle/input/datasets/abubakarsaeed1/biographx-psp-data/eSol_train_Encoded.csv")
test_val_df = pd.read_csv("/kaggle/input/datasets/abubakarsaeed1/biographx-psp-data/eSol_test_Encoded.csv")

target_col = 'solubility'

# Use the LAST 158 columns as features (excludes 'solubility' and any ID/metadata columns)
feature_cols = train_df.columns[-158:].tolist()

# Verify we have exactly 158 features
assert len(feature_cols) == 158, f"Expected 158 feature columns, got {len(feature_cols)}"

# Store feature names for importance output
FEATURE_NAMES = feature_cols
# =====================

X_train = train_df[feature_cols]
y_train = train_df[target_col]

# FGNNSol split: validation = first 268, test = remaining 392
val_df = test_val_df.iloc[:268]
test_df = test_val_df.iloc[268:]

X_val, y_val = val_df[feature_cols], val_df[target_col]
X_test, y_test = test_df[feature_cols], test_df[target_col]

# =====================
# 2. 5‑seed loop
# =====================
seeds = [2024, 2025, 2026, 2027, 2028]

# Storage for metrics across seeds
metrics = {
    'R2': [], 'Pearson': [], 'RMSE': [],
    'ACC': [], 'Precision': [], 'Recall': [],
    'F1': [], 'AUC': [], 'MCC': []
}

# To track the best model by recall
best_recall = -1.0
best_model = None
best_seed = None

for seed in seeds:
    print(f"Training with seed {seed}...")

    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=50,
        random_state=seed,
        n_jobs=-1,
        verbosity=0
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    preds = model.predict(X_test)

    # Regression
    r2 = r2_score(y_test, preds)
    pearson_val, _ = pearsonr(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    # Binary classification (0.5 threshold)
    y_test_bin = (y_test >= 0.5).astype(int)
    preds_bin = (preds >= 0.5).astype(int)

    acc = accuracy_score(y_test_bin, preds_bin)
    prec = precision_score(y_test_bin, preds_bin, zero_division=0)
    rec = recall_score(y_test_bin, preds_bin, zero_division=0)
    f1 = f1_score(y_test_bin, preds_bin, zero_division=0)
    auc = roc_auc_score(y_test_bin, preds)
    mcc = matthews_corrcoef(y_test_bin, preds_bin)

    # Store
    metrics['R2'].append(r2)
    metrics['Pearson'].append(pearson_val)
    metrics['RMSE'].append(rmse)
    metrics['ACC'].append(acc)
    metrics['Precision'].append(prec)
    metrics['Recall'].append(rec)
    metrics['F1'].append(f1)
    metrics['AUC'].append(auc)
    metrics['MCC'].append(mcc)

    print(f"  R²={r2:.4f}  Recall={rec:.4f}")

    # Track best model by recall
    if rec > best_recall:
        best_recall = rec
        best_model = model
        best_seed = seed

# =====================
# 3. Mean ± Std
# =====================
mean_metrics = {k: np.mean(v) for k, v in metrics.items()}
std_metrics  = {k: np.std(v)  for k, v in metrics.items()}

print("\n" + "="*70)
print("Five‑Seed Evaluation (XGBoost with BioGraphX features)")
print("="*70)
print(f"{'Metric':<12} | {'Our Mean ± Std':<22} | {'FGNNSol Mean'}")
print("-"*55)
print(f"{'R2':<12} | {mean_metrics['R2']:.4f} ± {std_metrics['R2']:.4f}      | 0.578")
print(f"{'Pearson':<12} | {mean_metrics['Pearson']:.4f} ± {std_metrics['Pearson']:.4f}      | 0.763")
print(f"{'RMSE':<12} | {mean_metrics['RMSE']:.4f} ± {std_metrics['RMSE']:.4f}      | 0.207")
print(f"{'ACC':<12} | {mean_metrics['ACC']:.4f} ± {std_metrics['ACC']:.4f}      | 0.785")
print(f"{'Precision':<12} | {mean_metrics['Precision']:.4f} ± {std_metrics['Precision']:.4f}      | 0.812")
print(f"{'Recall':<12} | {mean_metrics['Recall']:.4f} ± {std_metrics['Recall']:.4f}      | 0.734")
print(f"{'F1':<12} | {mean_metrics['F1']:.4f} ± {std_metrics['F1']:.4f}      | 0.771")
print(f"{'AUC':<12} | {mean_metrics['AUC']:.4f} ± {std_metrics['AUC']:.4f}      | 0.898")
print(f"{'MCC':<12} | {mean_metrics['MCC']:.4f} ± {std_metrics['MCC']:.4f}      | 0.589")
print("-"*55)

# =====================
# 4. Feature importance from best model (highest recall)
# =====================
if best_model is not None:
    importances = best_model.feature_importances_
    # Sort and pick top 10
    indices = np.argsort(importances)[::-1][:10]
    top_features = [FEATURE_NAMES[i] for i in indices]
    top_importances = importances[indices]

    print(f"\nTop 10 Features (Best Model, Seed={best_seed}, Recall={best_recall:.4f}):")
    print("-" * 50)
    for rank, (feat, imp) in enumerate(zip(top_features, top_importances), 1):
        print(f"{rank:2d}. {feat:<40s} {imp:.4f}")
else:
    print("\nNo best model found (error).")