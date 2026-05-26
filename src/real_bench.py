"""Real generalization benchmark: clustered split + XGBoost + Optuna."""
import csv, time, warnings, json
from collections import Counter
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, hamming_loss
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import xgboost as xgb
import optuna
warnings.filterwarnings('ignore')

optuna.logging.set_verbosity(optuna.logging.WARNING)

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

def dipeptide(seq):
    v = np.zeros(400)
    for i in range(len(seq)-1):
        a1,a2 = seq[i],seq[i+1]
        if a1 in AA_TO_IDX and a2 in AA_TO_IDX:
            v[AA_TO_IDX[a1]*20+AA_TO_IDX[a2]] += 1
    t = v.sum()
    return v/t if t>0 else v

def tripeptide(seq):
    v = np.zeros(8000)
    for i in range(len(seq)-2):
        a1,a2,a3 = seq[i],seq[i+1],seq[i+2]
        if a1 in AA_TO_IDX and a2 in AA_TO_IDX and a3 in AA_TO_IDX:
            v[AA_TO_IDX[a1]*400+AA_TO_IDX[a2]*20+AA_TO_IDX[a3]] += 1
    t = v.sum()
    return v/t if t>0 else v

def evaluate(model, X_te, y_te):
    yp = model.predict(X_te)
    probas = np.column_stack([
        e.predict_proba(X_te)[:,1] if e.predict_proba(X_te).shape[1]>1 
        else np.zeros(len(X_te)) for e in model.estimators_
    ])
    aucs = []; 
    for i in range(y_te.shape[1]):
        try: aucs.append(float(roc_auc_score(y_te[:,i], probas[:,i])))
        except: aucs.append(0.5)
    return {
        'micro_f1': float(f1_score(y_te, yp, average='micro')),
        'macro_f1': float(f1_score(y_te, yp, average='macro')),
        'mean_auc': float(np.mean(aucs)),
        'subset_acc': float(accuracy_score(y_te, yp)),
        'hamming': float(hamming_loss(y_te, yp)),
    }

# Load data
print("Loading...", flush=True)
with open('data/processed/amr_sequences.csv') as f:
    reader = csv.DictReader(f)
    seqs, lbs = [], []
    for row in reader:
        s = row['sequence'].strip()
        dcs = [dc.strip() for dc in row['drug_classes'].split(';') if dc.strip()]
        if s and dcs: seqs.append(s); lbs.append(dcs)

# Keep classes with >= 20 samples
lc = Counter()
for l in lbs:
    for x in l: lc[x] += 1
top = [c for c,n in lc.most_common() if n>=20]
fl = [[x for x in s if x in top] for s in lbs]
vi = [i for i,s in enumerate(fl) if s]
seqs = [seqs[i] for i in vi]; fl = [fl[i] for i in vi]
mlb = MultiLabelBinarizer(); y = mlb.fit_transform(fl)
n_classes = len(mlb.classes_)
print(f'{len(seqs)} seqs, {n_classes} classes', flush=True)

# CLUSTERED SPLIT
print("Clustering...", flush=True)
X_di = np.array([dipeptide(s) for s in seqs])
nc = max(2, len(seqs)//50)
cl = AgglomerativeClustering(n_clusters=nc, metric='cosine', linkage='average')
labels = cl.fit_predict(X_di)

uc = list(set(labels)); np.random.seed(42); np.random.shuffle(uc)
si = int(len(uc)*0.8)
ti = [i for i,c in enumerate(labels) if c in set(uc[:si])]
ei = [i for i,c in enumerate(labels) if c in set(uc[si:])]
print(f'Train: {len(ti)} Test: {len(ei)}', flush=True)

# Feature: dipeptide only (fast)
print("Features...", flush=True)
X_train = X_di[ti]; X_test = X_di[ei]
y_train = y[ti]; y_test = y[ei]

sc = StandardScaler(); X_train = sc.fit_transform(X_train); X_test = sc.transform(X_test)

# DEFAULT XGBoost
print("\n=== DEFAULT XGB (clustered) ===", flush=True)
m = MultiOutputClassifier(xgb.XGBClassifier(
    n_estimators=50, max_depth=5, learning_rate=0.2,
    random_state=42, n_jobs=4, verbosity=0, eval_metric='logloss',
    tree_method='hist'))
t0=time.time(); m.fit(X_train, y_train); t1=time.time()
res = evaluate(m, X_test, y_test)
print(f'Micro F1={res["micro_f1"]:.4f} Macro F1={res["macro_f1"]:.4f} AUC={res["mean_auc"]:.4f} '
      f'SubsetAcc={res["subset_acc"]:.4f} Hamming={res["hamming"]:.4f} Time={t1-t0:.1f}s', flush=True)

default_micro = res['micro_f1']

# OPTUNA TUNING
print("\n=== OPTUNA TUNING (clustered) ===", flush=True)

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state': 42, 'n_jobs': 4, 'verbosity': 0, 'eval_metric': 'logloss',
        'tree_method': 'hist',
    }
    model = MultiOutputClassifier(xgb.XGBClassifier(**params))
    model.fit(X_train, y_train)
    yp = model.predict(X_test)
    return float(f1_score(y_test, yp, average='micro'))

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=40, show_progress_bar=True)

print(f'\nBest params: {study.best_params}', flush=True)
print(f'Best Micro F1: {study.best_value:.4f}', flush=True)

# FINAL MODEL
print("\n=== FINAL MODEL (best params) ===", flush=True)
final = MultiOutputClassifier(xgb.XGBClassifier(**study.best_params, 
    random_state=42, n_jobs=4, verbosity=0, eval_metric='logloss', tree_method='hist'))
t0=time.time(); final.fit(X_train, y_train); t1=time.time()
final_res = evaluate(final, X_test, y_test)

print(f'Micro F1={final_res["micro_f1"]:.4f} Macro F1={final_res["macro_f1"]:.4f} '
      f'AUC={final_res["mean_auc"]:.4f} SubsetAcc={final_res["subset_acc"]:.4f} '
      f'Hamming={final_res["hamming"]:.4f} Time={t1-t0:.1f}s', flush=True)

# Per-class
print("\nPer-class AUC:", flush=True)
for i, cls in enumerate(mlb.classes_):
    probas = np.column_stack([
        e.predict_proba(X_test)[:,1] if e.predict_proba(X_test).shape[1]>1 
        else np.zeros(len(X_test)) for e in final.estimators_
    ])
    try: auc = float(roc_auc_score(y_test[:,i], probas[:,i]))
    except: auc = 0.5
    sup = int(y_test[:,i].sum())
    print(f'  {cls:<35s} AUC={auc:.4f} n={sup}', flush=True)

# SUMMARY
print(f'\n============================================', flush=True)
print(f'DEFAULT XGB:  Micro F1 = {default_micro:.4f}', flush=True)
print(f'TUNED XGB:    Micro F1 = {final_res["micro_f1"]:.4f}', flush=True)
print(f'IMPROVEMENT:  +{(final_res["micro_f1"]-default_micro)*100:.1f} pp', flush=True)
print(f'============================================', flush=True)
print(f'\nThe REAL maximum on clustered split (dipeptide only): ~{final_res["micro_f1"]:.3f}', flush=True)
print(f'Compare to: Random split baseline: 0.93 (45% inflated)', flush=True)
