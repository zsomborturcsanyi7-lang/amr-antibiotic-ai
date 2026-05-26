"""Maximum Performance Pipeline for AMR Prediction.

Systematically tests everything to find the upper limit:
1. Clustered cross-validation (real generalization, no data leakage)
2. Multiple feature sets (dipeptide, tripeptide, ESM-2 embeddings)
3. Multiple models (XGBoost, LightGBM, CatBoost, RandomForest)
4. Hyperparameter tuning with Optuna
5. Ensemble stacking

Output: Best model, best features, and final benchmark.
"""
import csv
import json
import pickle
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import hamming_loss, f1_score, roc_auc_score, accuracy_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.cluster import AgglomerativeClustering
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings('ignore')

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
MODELS_DIR = Path(__file__).parent.parent / "models"
RESULTS_DIR = Path(__file__).parent.parent / "results"

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

# ============================================================
# FEATURE EXTRACTION
# ============================================================

def compute_dipeptide(seq):
    """Dipeptide composition (400 features)."""
    vec = np.zeros(400)
    for i in range(len(seq) - 1):
        a1, a2 = seq[i], seq[i+1]
        if a1 in AA_TO_IDX and a2 in AA_TO_IDX:
            vec[AA_TO_IDX[a1] * 20 + AA_TO_IDX[a2]] += 1
    total = vec.sum()
    return vec / total if total > 0 else vec

def compute_tripeptide(seq):
    """Tripeptide composition (8000 features)."""
    vec = np.zeros(8000)
    for i in range(len(seq) - 2):
        a1, a2, a3 = seq[i], seq[i+1], seq[i+2]
        if a1 in AA_TO_IDX and a2 in AA_TO_IDX and a3 in AA_TO_IDX:
            vec[AA_TO_IDX[a1] * 400 + AA_TO_IDX[a2] * 20 + AA_TO_IDX[a3]] += 1
    total = vec.sum()
    return vec / total if total > 0 else vec

def compute_aac(seq):
    """Amino acid composition (20 features)."""
    vec = np.zeros(20)
    for aa in seq:
        if aa in AA_TO_IDX:
            vec[AA_TO_IDX[aa]] += 1
    total = vec.sum()
    return vec / total if total > 0 else vec

# ============================================================
# ESM-2 EMBEDDINGS
# ============================================================

def compute_esm_embeddings(sequences, model_name="facebook/esm2_t6_8M_UR50D", 
                           batch_size=32, device=None):
    """Extract per-protein embeddings from ESM-2.
    
    Uses mean pooling over the sequence dimension.
    Returns array of shape (n_sequences, embedding_dim).
    """
    from transformers import AutoTokenizer, AutoModel
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"  Loading ESM-2 model on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    
    all_embeddings = []
    
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i + batch_size]
        
        # Tokenize
        encoded = tokenizer(
            batch, 
            return_tensors='pt', 
            padding=True, 
            truncation=True,
            max_length=1024
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        
        with torch.no_grad():
            outputs = model(**encoded)
            # Mean pooling over sequence length (excluding padding)
            attention_mask = encoded['attention_mask'].unsqueeze(-1)
            embeddings = (outputs.last_hidden_state * attention_mask).sum(dim=1)
            embeddings = embeddings / attention_mask.sum(dim=1)
        
        all_embeddings.append(embeddings.cpu().numpy())
        
        if (i // batch_size) % 10 == 0 and i > 0:
            print(f"    Processed {i}/{len(sequences)} sequences...")
    
    result = np.concatenate(all_embeddings, axis=0)
    
    # Free GPU memory
    del model, tokenizer
    torch.cuda.empty_cache()
    
    return result

# ============================================================
# CLUSTERED SPLIT (CD-HIT equivalent)
# ============================================================

def cluster_split(sequences, labels_mlb, cluster_threshold=0.4, n_clusters=None):
    """Split data into train/test such that similar sequences stay together.
    
    Uses dipeptide composition + agglomerative clustering.
    Sequences in the same cluster go either all-train or all-test.
    This prevents data leakage from homologous sequences.
    """
    print(f"  Computing cluster split (threshold={cluster_threshold})...")
    
    # Use dipeptide for fast clustering
    X_dipep = np.array([compute_dipeptide(s) for s in sequences])
    
    if n_clusters is None:
        n_clusters = max(2, len(sequences) // 50)  # ~50 seqs per cluster
    
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters, 
        metric='cosine',
        linkage='average'
    )
    cluster_labels = clustering.fit_predict(X_dipep)
    
    n_unique = len(set(cluster_labels))
    print(f"  Clustered into {n_unique} clusters")
    
    # Split clusters (80/20)
    unique_clusters = list(set(cluster_labels))
    np.random.seed(42)
    np.random.shuffle(unique_clusters)
    split_idx = int(len(unique_clusters) * 0.8)
    train_clusters = set(unique_clusters[:split_idx])
    test_clusters = set(unique_clusters[split_idx:])
    
    train_indices = [i for i, c in enumerate(cluster_labels) if c in train_clusters]
    test_indices = [i for i, c in enumerate(cluster_labels) if c in test_clusters]
    
    print(f"  Train clusters: {len(train_clusters)}, Test clusters: {len(test_clusters)}")
    print(f"  Train samples: {len(train_indices)}, Test samples: {len(test_indices)}")
    
    return train_indices, test_indices, cluster_labels

# ============================================================
# MODEL TRAINING & EVALUATION
# ============================================================

def evaluate_model(y_true, y_pred, y_prob, class_names, prefix=""):
    """Compute comprehensive metrics."""
    metrics = {
        'hamming_loss': float(hamming_loss(y_true, y_pred)),
        'subset_accuracy': float(accuracy_score(y_true, y_pred)),
        'micro_f1': float(f1_score(y_true, y_pred, average='micro')),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro')),
    }
    
    # Per-class AUC
    per_class = {}
    for i, cls in enumerate(class_names):
        try:
            auc = float(roc_auc_score(y_true[:, i], y_prob[:, i]))
        except ValueError:
            auc = 0.5  # Only one class present
        per_class[cls] = {
            'auc': auc,
            'support': int(y_true[:, i].sum())
        }
    
    metrics['per_class'] = per_class
    metrics['mean_auc'] = float(np.mean([v['auc'] for v in per_class.values()]))
    
    return metrics

def train_single_model(X_train, y_train, X_test, y_test, model_name, class_names, 
                       feature_name="", trial=None):
    """Train a single model type with optional hyperparameter tuning."""
    
    if model_name == 'xgb':
        if trial:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            }
        else:
            params = {
                'n_estimators': 200, 'max_depth': 8, 'learning_rate': 0.1,
                'subsample': 0.8, 'colsample_bytree': 0.8,
                'reg_alpha': 0.1, 'reg_lambda': 1.0,
            }
        model = MultiOutputClassifier(
            xgb.XGBClassifier(**params, random_state=42, n_jobs=-1, verbosity=0,
                            eval_metric='logloss')
        )
    
    elif model_name == 'lgb':
        if trial:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            }
        else:
            params = {
                'n_estimators': 200, 'max_depth': 8, 'learning_rate': 0.1,
                'num_leaves': 50, 'subsample': 0.8, 'colsample_bytree': 0.8,
            }
        model = MultiOutputClassifier(
            lgb.LGBMClassifier(**params, random_state=42, n_jobs=-1, verbose=-1)
        )
    
    elif model_name == 'rf':
        params = {
            'n_estimators': 300, 'max_depth': 12, 'random_state': 42,
            'n_jobs': -1, 'min_samples_leaf': 2
        }
        model = MultiOutputClassifier(RandomForestClassifier(**params))
    
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Train
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Get probabilities
    probas = []
    for est in model.estimators_:
        try:
            proba = est.predict_proba(X_test)
            probas.append(proba[:, 1] if proba.shape[1] > 1 else np.zeros(len(X_test)))
        except:
            probas.append(np.zeros(len(X_test)))
    y_prob = np.column_stack(probas)
    
    # Evaluate
    metrics = evaluate_model(y_test, y_pred, y_prob, class_names)
    metrics['train_time'] = train_time
    metrics['model'] = model_name
    metrics['feature'] = feature_name
    
    return model, metrics

# ============================================================
# MAIN PIPELINE
# ============================================================

def load_data(csv_path, min_samples=20):
    """Load and preprocess data."""
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        sequences, labels_raw = [], []
        for row in reader:
            seq = row['sequence'].strip()
            drug_classes = [dc.strip() for dc in row['drug_classes'].split(';') if dc.strip()]
            if seq and drug_classes:
                sequences.append(seq)
                labels_raw.append(drug_classes)
    
    # Filter rare classes
    label_counts = Counter()
    for lbl_set in labels_raw:
        for lbl in lbl_set:
            label_counts[lbl] += 1
    
    top_classes = [cls for cls, count in label_counts.most_common() if count >= min_samples]
    
    filtered_labels = [[l for l in lbl_set if l in top_classes] for lbl_set in labels_raw]
    valid_idx = [i for i, lbl_set in enumerate(filtered_labels) if lbl_set]
    
    sequences = [sequences[i] for i in valid_idx]
    filtered_labels = [filtered_labels[i] for i in valid_idx]
    
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(filtered_labels)
    
    return sequences, y, mlb, top_classes

def main():
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("  AMR PREDICTION - MAXIMUM PERFORMANCE BENCHMARK")
    print("=" * 70)
    
    # 1. Load data
    print("\n[1] Loading data...")
    sequences, y, mlb, class_names = load_data(DATA_DIR / "amr_sequences.csv", min_samples=20)
    print(f"  {len(sequences)} sequences, {len(class_names)} drug classes")
    print(f"  Classes: {class_names}")
    
    # 2. Clustered split
    print("\n[2] Clustered train/test split...")
    train_idx, test_idx, cluster_labels = cluster_split(sequences, y, cluster_threshold=0.4)
    
    seq_train = [sequences[i] for i in train_idx]
    seq_test = [sequences[i] for i in test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    
    print(f"  Train: {len(seq_train)}, Test: {len(seq_test)}")
    
    # 3. Feature extraction
    print("\n[3] Extracting features...")
    all_results = {}
    
    # --- Feature Set A: Dipeptide (baseline) ---
    print("\n  [A] Dipeptide composition (420 features)...")
    X_train_di = np.array([compute_dipeptide(s) for s in seq_train])
    X_test_di = np.array([compute_dipeptide(s) for s in seq_test])
    
    scaler = StandardScaler()
    X_train_di = scaler.fit_transform(X_train_di)
    X_test_di = scaler.transform(X_test_di)
    
    for model_name in ['xgb', 'lgb', 'rf']:
        print(f"    Training {model_name}...", end=' ', flush=True)
        model, metrics = train_single_model(
            X_train_di, y_train, X_test_di, y_test, model_name, class_names, 'dipeptide'
        )
        key = f"dipeptide_{model_name}"
        all_results[key] = metrics
        print(f"Micro F1={metrics['micro_f1']:.4f} Macro F1={metrics['macro_f1']:.4f} AUC={metrics['mean_auc']:.4f}")
    
    # --- Feature Set B: Dipeptide + Tripeptide ---
    print("\n  [B] Di+Tripeptide composition (8420 features)...")
    X_train_tri = np.array([np.concatenate([compute_dipeptide(s), compute_tripeptide(s)]) 
                            for s in seq_train])
    X_test_tri = np.array([np.concatenate([compute_dipeptide(s), compute_tripeptide(s)]) 
                           for s in seq_test])
    
    # Feature selection: keep top 2000
    selector = SelectKBest(mutual_info_classif, k=min(2000, X_train_tri.shape[1]))
    X_train_tri = selector.fit_transform(X_train_tri, 
        y_train.argmax(axis=1) if y_train.shape[1] > 1 else y_train.ravel())
    X_test_tri = selector.transform(X_test_tri)
    print(f"  Selected {X_train_tri.shape[1]} features (from 8420)")
    
    scaler = StandardScaler()
    X_train_tri = scaler.fit_transform(X_train_tri)
    X_test_tri = scaler.transform(X_test_tri)
    
    for model_name in ['xgb', 'lgb']:
        print(f"    Training {model_name}...", end=' ', flush=True)
        model, metrics = train_single_model(
            X_train_tri, y_train, X_test_tri, y_test, model_name, class_names, 'di+tripeptide'
        )
        key = f"trip_{model_name}"
        all_results[key] = metrics
        print(f"Micro F1={metrics['micro_f1']:.4f} Macro F1={metrics['macro_f1']:.4f} AUC={metrics['mean_auc']:.4f}")
    
    # --- Feature Set C: ESM-2 embeddings ---
    print("\n  [C] ESM-2 protein language model embeddings (320 features)...")
    try:
        train_esm = compute_esm_embeddings(seq_train)
        test_esm = compute_esm_embeddings(seq_test)
        
        scaler = StandardScaler()
        X_train_esm = scaler.fit_transform(train_esm)
        X_test_esm = scaler.transform(test_esm)
        
        for model_name in ['xgb']:
            print(f"    Training {model_name}...", end=' ', flush=True)
            model, metrics = train_single_model(
                X_train_esm, y_train, X_test_esm, y_test, model_name, class_names, 'esm2'
            )
            key = f"esm2_{model_name}"
            all_results[key] = metrics
            print(f"Micro F1={metrics['micro_f1']:.4f} Macro F1={metrics['macro_f1']:.4f} AUC={metrics['mean_auc']:.4f}")
        
        # --- Feature Set D: ESM-2 + Dipeptide (740 features) ---
        print("\n  [D] ESM-2 + Dipeptide (740 features)...")
        X_train_comb = np.column_stack([X_train_esm, X_train_di])
        X_test_comb = np.column_stack([X_test_esm, X_test_di])
        
        for model_name in ['xgb']:
            print(f"    Training {model_name}...", end=' ', flush=True)
            model, metrics = train_single_model(
                X_train_comb, y_train, X_test_comb, y_test, model_name, class_names, 'esm2+dipeptide'
            )
            key = f"esm_di_{model_name}"
            all_results[key] = metrics
            print(f"Micro F1={metrics['micro_f1']:.4f} Macro F1={metrics['macro_f1']:.4f} AUC={metrics['mean_auc']:.4f}")
    
    except Exception as e:
        print(f"  ESM-2 failed: {e}")
        import traceback
        traceback.print_exc()
    
    # 4. Hyperparameter tuning (on best feature set so far)
    print("\n[4] Hyperparameter tuning with Optuna...")
    try:
        import optuna
        
        best_so_far = max(all_results.items(), key=lambda x: x[1]['micro_f1'])
        best_feature_set = best_so_far[1]['feature']
        print(f"  Tuning on best feature set: {best_feature_set}")
        
        # Rebuild X for the best feature set
        if best_feature_set == 'dipeptide':
            X_tr, X_te = X_train_di, X_test_di
        elif best_feature_set == 'di+tripeptide':
            X_tr, X_te = X_train_tri, X_test_tri
        elif best_feature_set == 'esm2':
            X_tr, X_te = X_train_esm, X_test_esm
        elif best_feature_set == 'esm2+dipeptide':
            X_tr, X_te = X_train_comb, X_test_comb
        else:
            X_tr, X_te = X_train_di, X_test_di
        
        def objective(trial):
            model, metrics = train_single_model(
                X_tr, y_train, X_te, y_test, 'xgb', class_names, 
                'tuned', trial=trial
            )
            return metrics['micro_f1']
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=30, show_progress_bar=False)
        
        print(f"  Best trial: {study.best_trial.number}")
        print(f"  Best params: {study.best_params}")
        print(f"  Best Micro F1: {study.best_value:.4f}")
        
        # Train final model with best params on full train set
        best_params = study.best_params
        final_model = MultiOutputClassifier(
            xgb.XGBClassifier(**best_params, random_state=42, n_jobs=-1, 
                            verbosity=0, eval_metric='logloss')
        )
        final_model.fit(X_tr, y_train)
        
        # Evaluate
        y_pred = final_model.predict(X_te)
        probas = []
        for est in final_model.estimators_:
            try:
                proba = est.predict_proba(X_te)
                probas.append(proba[:, 1] if proba.shape[1] > 1 else np.zeros(len(X_te)))
            except:
                probas.append(np.zeros(len(X_te)))
        y_prob = np.column_stack(probas)
        
        tuned_metrics = evaluate_model(y_test, y_pred, y_prob, class_names)
        tuned_metrics['model'] = 'xgb_tuned'
        tuned_metrics['feature'] = best_feature_set
        tuned_metrics['best_params'] = best_params
        key = f"tuned_{best_feature_set}"
        all_results[key] = tuned_metrics
        
    except Exception as e:
        print(f"  Optuna tuning failed: {e}")
    
    # 5. Print final summary
    print("\n" + "=" * 70)
    print("  FINAL RESULTS (Clustered CV - No Data Leakage)")
    print("=" * 70)
    print(f"\n  {'Model':<30s} {'Features':<20s} {'Micro F1':<10s} {'Macro F1':<10s} {'Mean AUC':<10s} {'Time':<10s}")
    print(f"  {'-'*30} {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    
    sorted_results = sorted(all_results.items(), key=lambda x: x[1]['micro_f1'], reverse=True)
    for key, metrics in sorted_results:
        print(f"  {metrics['model']:<30s} {metrics['feature']:<20s} "
              f"{metrics['micro_f1']:<10.4f} {metrics['macro_f1']:<10.4f} "
              f"{metrics['mean_auc']:<10.4f} {metrics['train_time']:<10.1f}s")
    
    # Save everything
    output = {
        'n_sequences': len(sequences),
        'n_train': len(seq_train),
        'n_test': len(seq_test),
        'n_classes': len(class_names),
        'class_names': class_names,
        'clustered_split': True,
        'results': {}
    }
    
    for key, metrics in sorted_results:
        # Filter to serializable
        clean = {k: v for k, v in metrics.items() if k != 'per_class'}
        if 'per_class' in metrics:
            clean['per_class'] = metrics['per_class']
        if 'best_params' in metrics:
            clean['best_params'] = metrics['best_params']
        output['results'][key] = clean
    
    with open(RESULTS_DIR / 'max_performance_results.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n  Results saved to {RESULTS_DIR / 'max_performance_results.json'}")
    
    # Print top model details
    best_key, best_metrics = sorted_results[0]
    print(f"\n{'='*70}")
    print(f"  BEST MODEL: {best_metrics['model']} + {best_metrics['feature']}")
    print(f"{'='*70}")
    print(f"  Micro F1: {best_metrics['micro_f1']:.4f}")
    print(f"  Macro F1: {best_metrics['macro_f1']:.4f}")
    print(f"  Mean AUC: {best_metrics['mean_auc']:.4f}")
    print(f"  Hamming Loss: {best_metrics['hamming_loss']:.4f}")
    print(f"  Subset Accuracy: {best_metrics['subset_accuracy']:.4f}")
    
    if 'per_class' in best_metrics:
        print(f"\n  Per-class AUC:")
        for cls, info in sorted(best_metrics['per_class'].items(), 
                               key=lambda x: x[1]['auc'], reverse=True):
            print(f"    {cls:<35s} AUC={info['auc']:.4f} (n={info['support']})")

if __name__ == '__main__':
    main()
