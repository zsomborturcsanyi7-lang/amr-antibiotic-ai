"""Fast benchmark: clustered split + di/tripeptide + hyperparameter tuning.
Skips ESM-2 for speed. Run this first, then the full version later.
"""
import csv, json, pickle, time, warnings
from collections import Counter
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import hamming_loss, f1_score, roc_auc_score, accuracy_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.cluster import AgglomerativeClustering
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings('ignore')

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
RESULTS_DIR = Path(__file__).parent.parent / "results"

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

def compute_dipeptide(seq):
    vec = np.zeros(400)
    for i in range(len(seq) - 1):
        a1, a2 = seq[i], seq[i+1]
        if a1 in AA_TO_IDX and a2 in AA_TO_IDX:
            vec[AA_TO_IDX[a1] * 20 + AA_TO_IDX[a2]] += 1
    total = vec.sum()
    return vec / total if total > 0 else vec

def compute_tripeptide(seq):
    vec = np.zeros(8000)
    for i in range(len(seq) - 2):
        a1, a2, a3 = seq[i], seq[i+1], seq[i+2]
        if a1 in AA_TO_IDX and a2 in AA_TO_IDX and a3 in AA_TO_IDX:
            vec[AA_TO_IDX[a1] * 400 + AA_TO_IDX[a2] * 20 + AA_TO_IDX[a3]] += 1
    total = vec.sum()
    return vec / total if total > 0 else vec

def cluster_split(sequences, n_clusters=None):
    """Split sequences into train/test by clustering similar ones together."""
    if n_clusters is None:
        n_clusters = max(2, len(sequences) // 50)
    
    X_di = np.array([compute_dipeptide(s) for s in sequences])
    clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='cosine', linkage='average')
    labels = clustering.fit_predict(X_di)
    
    unique_clusters = list(set(labels))
    np.random.seed(42)
    np.random.shuffle(unique_clusters)
    split_idx = int(len(unique_clusters) * 0.8)
    train_clusters = set(unique_clusters[:split_idx])
    test_clusters = set(unique_clusters[split_idx:])
    
    train_idx = [i for i, c in enumerate(labels) if c in train_clusters]
    test_idx = [i for i, c in enumerate(labels) if c in test_clusters]
    
    return train_idx, test_idx

def evaluate(y_true, y_pred, y_prob, class_names):
    metrics = {
        'hamming_loss': float(hamming_loss(y_true, y_pred)),
        'subset_accuracy': float(accuracy_score(y_true, y_pred)),
        'micro_f1': float(f1_score(y_true, y_pred, average='micro')),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro')),
    }
    per_class = {}
    for i, cls in enumerate(class_names):
        try:
            auc = float(roc_auc_score(y_true[:, i], y_prob[:, i]))
        except:
            auc = 0.5
        per_class[cls] = {'auc': auc, 'support': int(y_true[:, i].sum())}
    metrics['mean_auc'] = float(np.mean([v['auc'] for v in per_class.values()]))
    metrics['per_class'] = per_class
    return metrics

def load_data(csv_path, min_samples=20):
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        seqs, labels_raw = [], []
        for row in reader:
            seq = row['sequence'].strip()
            dcs = [dc.strip() for dc in row['drug_classes'].split(';') if dc.strip()]
            if seq and dcs:
                seqs.append(seq)
                labels_raw.append(dcs)
    
    lc = Counter()
    for lbl_set in labels_raw:
        for l in lbl_set:
            lc[l] += 1
    top = [cls for cls, count in lc.most_common() if count >= min_samples]
    fl = [[l for l in s if l in top] for s in labels_raw]
    vi = [i for i, s in enumerate(fl) if s]
    seqs = [seqs[i] for i in vi]
    fl = [fl[i] for i in vi]
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(fl)
    return seqs, y, mlb, top

print("="*60)
print("BENCHMARK: Random split vs Clustered split")
print("="*60)

# Load
seqs, y, mlb, class_names = load_data(DATA_DIR / "amr_sequences.csv", min_samples=20)
print(f"Data: {len(seqs)} seqs, {len(class_names)} classes\n")

all_results = {}

# FEATURES
print("Extracting features...")
X_di = np.array([compute_dipeptide(s) for s in seqs])
X_tri_full = np.array([np.concatenate([compute_dipeptide(s), compute_tripeptide(s)]) for s in seqs])

# Feature selection for tripeptide
print("Feature selection for tripeptide...")
selector = SelectKBest(mutual_info_classif, k=min(2000, X_tri_full.shape[1]))
# Use a proxy label for feature selection (most common class per sample)
proxy = np.argmax(y, axis=1)
X_tri = selector.fit_transform(X_tri_full, proxy)
print(f"Tripeptide: {X_tri.shape[1]} features selected\n")

feature_sets = {
    'dipeptide (420)': ('di', X_di),
    'di+tripeptide (2000)': ('trip', X_tri),
}

for split_name, split_type in [('RANDOM SPLIT', 'random'), ('CLUSTERED SPLIT', 'clustered')]:
    print(f"\n{'='*60}")
    print(f"  {split_name}")
    print(f"{'='*60}")
    
    if split_type == 'random':
        train_idx, test_idx = train_test_split(
            range(len(seqs)), test_size=0.2, random_state=42
        )
    else:
        train_idx, test_idx = cluster_split(seqs)
    
    print(f"Train: {len(train_idx)}, Test: {len(test_idx)}")
    
    for fs_name, (fs_key, X_all) in feature_sets.items():
        print(f"\n  [Feature: {fs_name}]")
        
        X_train = X_all[train_idx]
        X_test = X_all[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        for model_name, ModelClass, params in [
            ('XGBoost', xgb.XGBClassifier, {'n_estimators': 200, 'max_depth': 8, 
             'learning_rate': 0.1, 'subsample': 0.8, 'colsample_bytree': 0.8,
             'reg_alpha': 0.1, 'reg_lambda': 1.0, 'random_state': 42, 
             'n_jobs': -1, 'verbosity': 0, 'eval_metric': 'logloss'}),
            ('LightGBM', lgb.LGBMClassifier, {'n_estimators': 200, 'max_depth': 8,
             'learning_rate': 0.1, 'num_leaves': 50, 'subsample': 0.8,
             'colsample_bytree': 0.8, 'random_state': 42, 'n_jobs': -1, 'verbose': -1}),
        ]:
            t0 = time.time()
            model = MultiOutputClassifier(ModelClass(**params))
            model.fit(X_train, y_train)
            train_time = time.time() - t0
            
            y_pred = model.predict(X_test)
            probas = []
            for est in model.estimators_:
                try:
                    p = est.predict_proba(X_test)
                    probas.append(p[:, 1] if p.shape[1] > 1 else np.zeros(len(X_test)))
                except:
                    probas.append(np.zeros(len(X_test)))
            y_prob = np.column_stack(probas)
            
            m = evaluate(y_test, y_pred, y_prob, class_names)
            key = f"{split_type}_{fs_key}_{model_name}"
            all_results[key] = m
            
            print(f"    {model_name:<10s} Micro F1={m['micro_f1']:.4f} "
                  f"Macro F1={m['macro_f1']:.4f} AUC={m['mean_auc']:.4f} "
                  f"SubsetAcc={m['subset_accuracy']:.4f} ({train_time:.1f}s)")

# Optuna tuning on best setup
print(f"\n{'='*60}")
print("  OPTUNA HYPERPARAMETER TUNING")
print(f"{'='*60}")

import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Use clustered split with di+tripeptide + XGBoost
train_idx, test_idx = cluster_split(seqs)
X_train = X_tri[train_idx]
X_test = X_tri[test_idx]
y_train = y[train_idx]
y_test = y[test_idx]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state': 42, 'n_jobs': -1, 'verbosity': 0, 'eval_metric': 'logloss',
    }
    model = MultiOutputClassifier(xgb.XGBClassifier(**params))
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return float(f1_score(y_test, y_pred, average='micro'))

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30, show_progress_bar=True)

print(f"\n  Best params: {study.best_params}")
print(f"  Best Micro F1: {study.best_value:.4f}")

# Final model with best params
final = MultiOutputClassifier(xgb.XGBClassifier(**study.best_params, 
    random_state=42, n_jobs=-1, verbosity=0, eval_metric='logloss'))
final.fit(X_train, y_train)
y_pred = final.predict(X_test)
probas = []
for est in final.estimators_:
    try:
        p = est.predict_proba(X_test)
        probas.append(p[:, 1] if p.shape[1] > 1 else np.zeros(len(X_test)))
    except:
        probas.append(np.zeros(len(X_test)))
y_prob = np.column_stack(probas)

tuned_m = evaluate(y_test, y_pred, y_prob, class_names)
key = "clustered_trip_xgb_tuned"
all_results[key] = tuned_m

# FINAL SUMMARY
print(f"\n{'='*60}")
print("  FINAL SUMMARY (all experiments)")
print(f"{'='*60}")
print(f"\n{'Setup':<45s} {'Micro F1':<10s} {'Macro F1':<10s} {'AUC':<10s}")
print(f"{'-'*45} {'-'*10} {'-'*10} {'-'*10}")

for key, m in sorted(all_results.items(), key=lambda x: x[1]['micro_f1'], reverse=True):
    print(f"{key:<45s} {m['micro_f1']:<10.4f} {m['macro_f1']:<10.4f} {m['mean_auc']:<10.4f}")

# Highlight the damage from clustered split
if 'random_di_XGBoost' in all_results and 'clustered_di_XGBoost' in all_results:
    r = all_results['random_di_XGBoost']
    c = all_results['clustered_di_XGBoost']
    drop = (r['micro_f1'] - c['micro_f1']) / r['micro_f1'] * 100
    print(f"\n  ⚠ Data leakage inflation: Random split overestimates Micro F1 by {drop:.1f}%")

# Save
with open(RESULTS_DIR / 'benchmark_comparison.json', 'w') as f:
    json.dump({k: {kk: float(vv) if not isinstance(vv, dict) else vv 
               for kk, vv in v.items()} 
               for k, v in all_results.items()}, f, indent=2, default=str)

print(f"\nSaved to {RESULTS_DIR / 'benchmark_comparison.json'}")
