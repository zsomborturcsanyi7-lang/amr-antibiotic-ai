"""AMR Prediction Model - Baseline.

Protein sequence -> Antibiotic resistance profile prediction.
Uses dipeptide composition features + multi-label XGBoost.
"""
import csv
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, hamming_loss, accuracy_score,
    f1_score, roc_auc_score, precision_recall_curve
)
from xgboost import XGBClassifier

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
MODELS_DIR = Path(__file__).parent.parent / "models"
RESULTS_DIR = Path(__file__).parent.parent / "results"

# Standard 20 amino acids
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

def compute_dipeptide_composition(seq):
    """Compute dipeptide (2-mer) composition of a protein sequence.
    
    Returns a vector of length 400 (20x20) with frequencies of each dipeptide.
    """
    dipep = np.zeros(400)
    if len(seq) < 2:
        return dipep
    
    for i in range(len(seq) - 1):
        aa1 = seq[i]
        aa2 = seq[i+1]
        if aa1 in AA_TO_IDX and aa2 in AA_TO_IDX:
            idx = AA_TO_IDX[aa1] * 20 + AA_TO_IDX[aa2]
            dipep[idx] += 1
    
    # Normalize to frequencies
    total = dipep.sum()
    if total > 0:
        dipep /= total
    
    return dipep

def compute_aac(seq):
    """Amino acid composition (20 features)."""
    comp = np.zeros(20)
    for aa in seq:
        if aa in AA_TO_IDX:
            comp[AA_TO_IDX[aa]] += 1
    total = comp.sum()
    if total > 0:
        comp /= total
    return comp

def compute_combined_features(seq):
    """Combine AAC + dipeptide composition."""
    return np.concatenate([compute_aac(seq), compute_dipeptide_composition(seq)])

def load_data(csv_path):
    """Load labeled AMR sequences."""
    sequences = []
    labels = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            seq = row['sequence'].strip()
            drug_classes = [dc.strip() for dc in row['drug_classes'].split(';') if dc.strip()]
            if not seq or not drug_classes:
                continue
            sequences.append(seq)
            labels.append(drug_classes)
    return sequences, labels

def main():
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    
    print("AMR Prediction Model - Training")
    print("=" * 60)
    
    # 1. Load data
    print("\n[1/6] Loading data...")
    seqs, labels = load_data(DATA_DIR / "amr_sequences.csv")
    print(f"  {len(seqs)} sequences loaded")
    
    # 2. Filter to top drug classes
    print("\n[2/6] Filtering to top drug classes...")
    label_counts = Counter()
    for lbl_set in labels:
        for lbl in lbl_set:
            label_counts[lbl] += 1
    
    # Keep classes with at least 50 examples
    MIN_SAMPLES = 50
    top_classes = [cls for cls, count in label_counts.most_common() if count >= MIN_SAMPLES]
    print(f"  Keeping {len(top_classes)} drug classes (>= {MIN_SAMPLES} samples):")
    for cls in top_classes:
        print(f"    {cls}: {label_counts[cls]}")
    
    # Filter labels
    filtered_labels = [[l for l in lbl_set if l in top_classes] for lbl_set in labels]
    # Remove sequences with no labels after filtering
    valid_indices = [i for i, lbl_set in enumerate(filtered_labels) if lbl_set]
    seqs = [seqs[i] for i in valid_indices]
    filtered_labels = [filtered_labels[i] for i in valid_indices]
    print(f"  {len(seqs)} sequences after filtering")
    
    # 3. Feature extraction
    print("\n[3/6] Extracting features...")
    X = np.array([compute_combined_features(s) for s in seqs])
    
    # Multi-label binarizer
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(filtered_labels)
    
    print(f"  Features: {X.shape} ({X.shape[1]} dimensions)")
    print(f"  Labels: {y.shape} ({y.shape[1]} classes)")
    print(f"  Classes: {list(mlb.classes_)}")
    print(f"  Label density: {y.sum() / (y.shape[0] * y.shape[1]):.3f}")
    
    # 4. Train/test split
    print("\n[4/6] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"  Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    
    # 5. Train models
    print("\n[5/6] Training models...")
    results = {}
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Model 1: Random Forest (binary relevance)
    print("\n  [a] Random Forest...")
    rf = MultiOutputClassifier(
        RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    )
    rf.fit(X_train_scaled, y_train)
    y_pred_rf = rf.predict(X_test_scaled)
    y_prob_rf = np.column_stack([est.predict_proba(X_test_scaled)[:, 1] 
                                  for est in rf.estimators_])
    
    # Model 2: XGBoost (binary relevance)
    print("  [b] XGBoost...")
    xgb = MultiOutputClassifier(
        XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, 
                      random_state=42, n_jobs=-1, verbosity=0)
    )
    xgb.fit(X_train_scaled, y_train)
    y_pred_xgb = xgb.predict(X_test_scaled)
    y_prob_xgb = np.column_stack([est.predict_proba(X_test_scaled)[:, 1] 
                                   for est in xgb.estimators_])
    
    # 6. Evaluate
    print("\n[6/6] Evaluating...")
    
    models_eval = {
        'RandomForest': (y_pred_rf, y_prob_rf),
        'XGBoost': (y_pred_xgb, y_prob_xgb),
    }
    
    for name, (y_pred, y_prob) in models_eval.items():
        print(f"\n{'='*40}")
        print(f"  {name}")
        print(f"{'='*40}")
        
        # Overall metrics
        hamming = hamming_loss(y_test, y_pred)
        subset_acc = accuracy_score(y_test, y_pred)
        micro_f1 = f1_score(y_test, y_pred, average='micro')
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        
        print(f"  Hamming Loss:    {hamming:.4f} (lower is better)")
        print(f"  Subset Accuracy: {subset_acc:.4f}")
        print(f"  Micro F1:        {micro_f1:.4f}")
        print(f"  Macro F1:        {macro_f1:.4f}")
        
        # Per-class metrics
        print(f"\n  Per-class performance:")
        for i, cls in enumerate(mlb.classes_):
            # Handle potential NaN in AUC
            try:
                auc = roc_auc_score(y_test[:, i], y_prob[:, i])
            except ValueError:
                auc = float('nan')
            
            # Calculate sensitivity (recall) and specificity
            tp = ((y_pred[:, i] == 1) & (y_test[:, i] == 1)).sum()
            tn = ((y_pred[:, i] == 0) & (y_test[:, i] == 0)).sum()
            fp = ((y_pred[:, i] == 1) & (y_test[:, i] == 0)).sum()
            fn = ((y_pred[:, i] == 0) & (y_test[:, i] == 1)).sum()
            
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            print(f"    {cls:<35s} AUC={auc:.3f} Sens={sens:.3f} Spec={spec:.3f} "
                  f"Support={y_test[:, i].sum():.0f}")
        
        results[name] = {
            'hamming_loss': hamming,
            'subset_accuracy': subset_acc,
            'micro_f1': micro_f1,
            'macro_f1': macro_f1,
        }
    
    # Save best model
    print(f"\n{'='*40}")
    print("  Saving best model (XGBoost)...")
    
    with open(MODELS_DIR / 'mlb.pkl', 'wb') as f:
        pickle.dump(mlb, f)
    with open(MODELS_DIR / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open(MODELS_DIR / 'xgb_model.pkl', 'wb') as f:
        pickle.dump(xgb, f)
    
    # Save results
    with open(RESULTS_DIR / 'baseline_results.json', 'w') as f:
        json.dump({
            'n_samples': len(seqs),
            'n_features': X.shape[1],
            'n_classes': len(mlb.classes_),
            'classes': list(mlb.classes_),
            'top_drug_classes': top_classes,
            'results': {k: {kk: float(vv) for kk, vv in v.items()} 
                       for k, v in results.items()}
        }, f, indent=2)
    
    print(f"\n  Models saved to {MODELS_DIR}")
    print(f"  Results saved to {RESULTS_DIR}")
    print(f"\n  Best model: XGBoost (Micro F1={results['XGBoost']['micro_f1']:.4f})")

if __name__ == '__main__':
    main()
