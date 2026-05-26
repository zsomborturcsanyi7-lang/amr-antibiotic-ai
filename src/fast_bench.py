"""Fast benchmark: RANDOM vs CLUSTERED split with light XGBoost."""
import csv, time, warnings
from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, hamming_loss
from sklearn.cluster import AgglomerativeClustering
import xgboost as xgb
warnings.filterwarnings('ignore')

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

def quick_eval(model, X_te, y_te):
    yp = model.predict(X_te)
    probas = np.column_stack([
        e.predict_proba(X_te)[:,1] if e.predict_proba(X_te).shape[1]>1 
        else np.zeros(len(X_te)) 
        for e in model.estimators_
    ])
    aucs = []
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

# Fast XGB params (50 trees, shallow)
XGB_FAST = {
    'n_estimators': 50, 'max_depth': 5, 'learning_rate': 0.2,
    'subsample': 0.8, 'colsample_bytree': 0.8,
    'random_state': 42, 'n_jobs': 4, 'verbosity': 0, 'eval_metric': 'logloss',
    'tree_method': 'hist', 'max_bin': 128,
}

print("Loading data...", flush=True)
with open('data/processed/amr_sequences.csv') as f:
    reader = csv.DictReader(f)
    seqs, lbs = [], []
    for row in reader:
        s = row['sequence'].strip()
        dcs = [dc.strip() for dc in row['drug_classes'].split(';') if dc.strip()]
        if s and dcs: seqs.append(s); lbs.append(dcs)

lc = Counter()
for l in lbs:
    for x in l: lc[x] += 1
top = [c for c,n in lc.most_common() if n>=20]
fl = [[x for x in s if x in top] for s in lbs]
vi = [i for i,s in enumerate(fl) if s]
seqs = [seqs[i] for i in vi]; fl = [fl[i] for i in vi]
mlb = MultiLabelBinarizer(); y = mlb.fit_transform(fl)
print(f'{len(seqs)} seqs, {len(mlb.classes_)} classes', flush=True)

print("Dipeptide features...", flush=True)
X = np.array([dipeptide(s) for s in seqs])

# RANDOM SPLIT
print("\n=== RANDOM SPLIT ===", flush=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
sc = StandardScaler(); X_tr = sc.fit_transform(X_tr); X_te = sc.transform(X_te)

m = MultiOutputClassifier(xgb.XGBClassifier(**XGB_FAST))
t0 = time.time(); m.fit(X_tr, y_tr); t1 = time.time()
r = quick_eval(m, X_te, y_te)
print(f'Micro F1={r["micro_f1"]:.4f} Macro F1={r["macro_f1"]:.4f} AUC={r["mean_auc"]:.4f} SubsetAcc={r["subset_acc"]:.4f} Hamming={r["hamming"]:.4f} Time={t1-t0:.1f}s', flush=True)

# CLUSTERED SPLIT
print("\n=== CLUSTERED SPLIT ===", flush=True)
nc = max(2, len(seqs)//50)
print(f'Clustering {len(seqs)} seqs into {nc} clusters...', flush=True)
cl = AgglomerativeClustering(n_clusters=nc, metric='cosine', linkage='average')
labels = cl.fit_predict(X)

uc = list(set(labels)); np.random.seed(42); np.random.shuffle(uc)
si = int(len(uc)*0.8); tc = set(uc[:si]); ec = set(uc[si:])
ti = [i for i,c in enumerate(labels) if c in tc]
ei = [i for i,c in enumerate(labels) if c in ec]
print(f'Train: {len(ti)} Test: {len(ei)} ({len(tc)}/{len(ec)} clusters)', flush=True)

X_tr, X_te = X[ti], X[ei]; y_tr, y_te = y[ti], y[ei]
sc = StandardScaler(); X_tr = sc.fit_transform(X_tr); X_te = sc.transform(X_te)

m = MultiOutputClassifier(xgb.XGBClassifier(**XGB_FAST))
t0 = time.time(); m.fit(X_tr, y_tr); t1 = time.time()
c = quick_eval(m, X_te, y_te)
print(f'Micro F1={c["micro_f1"]:.4f} Macro F1={c["macro_f1"]:.4f} AUC={c["mean_auc"]:.4f} SubsetAcc={c["subset_acc"]:.4f} Hamming={c["hamming"]:.4f} Time={t1-t0:.1f}s', flush=True)

# COMPARISON
drop = (r['micro_f1'] - c['micro_f1']) / r['micro_f1'] * 100
print(f'\n============================================', flush=True)
print(f'RANDOM  Micro F1: {r["micro_f1"]:.4f}  AUC: {r["mean_auc"]:.4f}', flush=True)
print(f'CLUSTER Micro F1: {c["micro_f1"]:.4f}  AUC: {c["mean_auc"]:.4f}', flush=True)
print(f'Data leakage inflation: {drop:.1f}%', flush=True)
print(f'============================================', flush=True)
