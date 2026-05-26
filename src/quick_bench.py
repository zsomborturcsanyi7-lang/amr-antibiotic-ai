"""Quick benchmark: RANDOM vs CLUSTERED split with XGBoost."""
import csv, time, warnings, json
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

def evaluate(model, X_te, y_te, mlb):
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

print("="*60, flush=True)
print("AMR BASELINE: Random vs Clustered Split", flush=True)
print("="*60, flush=True)

# Load data
print("Loading...", flush=True)
with open('data/processed/amr_sequences.csv') as f:
    reader = csv.DictReader(f)
    seqs, lbs = [], []
    for row in reader:
        s = row['sequence'].strip()
        dcs = [dc.strip() for dc in row['drug_classes'].split(';') if dc.strip()]
        if s and dcs:
            seqs.append(s)
            lbs.append(dcs)

lc = Counter()
for l in lbs:
    for x in l: lc[x] += 1
top = [c for c,n in lc.most_common() if n>=20]
fl = [[x for x in s if x in top] for s in lbs]
vi = [i for i,s in enumerate(fl) if s]
seqs = [seqs[i] for i in vi]
fl = [fl[i] for i in vi]
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(fl)
print(f'{len(seqs)} seqs, {len(mlb.classes_)} classes', flush=True)

# Dipeptide features
print("Dipeptide features...", flush=True)
X = np.array([dipeptide(s) for s in seqs])
print(f'  {X.shape}', flush=True)

# ---- RANDOM SPLIT ----
print("\n--- RANDOM SPLIT ---", flush=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
sc = StandardScaler()
X_tr = sc.fit_transform(X_tr)
X_te = sc.transform(X_te)

model = MultiOutputClassifier(xgb.XGBClassifier(
    n_estimators=200, max_depth=8, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, n_jobs=-1, verbosity=0, eval_metric='logloss'
))
t0 = time.time()
model.fit(X_tr, y_tr)
train_time = time.time() - t0
res_random = evaluate(model, X_te, y_te, mlb)
res_random['train_time'] = train_time
print(f'Micro F1: {res_random["micro_f1"]:.4f}', flush=True)
print(f'Macro F1: {res_random["macro_f1"]:.4f}', flush=True)
print(f'Mean AUC: {res_random["mean_auc"]:.4f}', flush=True)
print(f'Subset Acc: {res_random["subset_acc"]:.4f}', flush=True)
print(f'Hamming: {res_random["hamming"]:.4f}', flush=True)
print(f'Train: {train_time:.1f}s', flush=True)

# ---- CLUSTERED SPLIT ----
print("\n--- CLUSTERED SPLIT ---", flush=True)
nc = max(2, len(seqs)//50)
print(f'Clustering {len(seqs)} seqs into {nc} clusters...', flush=True)
cl = AgglomerativeClustering(n_clusters=nc, metric='cosine', linkage='average')
labels = cl.fit_predict(X)
n_uniq = len(set(labels))
print(f'{n_uniq} unique clusters', flush=True)

uc = list(set(labels))
np.random.seed(42)
np.random.shuffle(uc)
si = int(len(uc)*0.8)
tc = set(uc[:si])
ec = set(uc[si:])
ti = [i for i,c in enumerate(labels) if c in tc]
ei = [i for i,c in enumerate(labels) if c in ec]
print(f'Train: {len(ti)}, Test: {len(ei)} ({len(tc)}/{len(ec)} clusters)', flush=True)

X_tr, X_te = X[ti], X[ei]
y_tr, y_te = y[ti], y[ei]
sc = StandardScaler()
X_tr = sc.fit_transform(X_tr)
X_te = sc.transform(X_te)

t0 = time.time()
model.fit(X_tr, y_tr)
train_time = time.time() - t0
res_clustered = evaluate(model, X_te, y_te, mlb)
res_clustered['train_time'] = train_time
print(f'Micro F1: {res_clustered["micro_f1"]:.4f}', flush=True)
print(f'Macro F1: {res_clustered["macro_f1"]:.4f}', flush=True)
print(f'Mean AUC: {res_clustered["mean_auc"]:.4f}', flush=True)
print(f'Subset Acc: {res_clustered["subset_acc"]:.4f}', flush=True)
print(f'Hamming: {res_clustered["hamming"]:.4f}', flush=True)
print(f'Train: {train_time:.1f}s', flush=True)

# Comparison
drop = (res_random['micro_f1'] - res_clustered['micro_f1']) / res_random['micro_f1'] * 100
print(f'\n{"="*60}', flush=True)
print(f'DATA LEAKAGE IMPACT:', flush=True)
print(f'  Random split Micro F1:  {res_random["micro_f1"]:.4f}', flush=True)
print(f'  Clustered split Micro F1: {res_clustered["micro_f1"]:.4f}', flush=True)
print(f'  Inflation: {drop:.1f}%', flush=True)
print(f'{"="*60}', flush=True)

print("\nDONE", flush=True)
