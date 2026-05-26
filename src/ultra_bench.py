"""ULTRA-FAST benchmark: Random vs Clustered split."""
import csv, time, warnings
from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, hamming_loss
from sklearn.cluster import AgglomerativeClustering
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

def evaluate(model, X_te, y_te):
    yp = model.predict(X_te)
    probas = model.predict_proba(X_te)
    aucs = []
    for i in range(y_te.shape[1]):
        try: aucs.append(float(roc_auc_score(y_te[:,i], probas[:,i])))
        except: aucs.append(0.5)
    return f1_score(y_te, yp, average='micro'), f1_score(y_te, yp, average='macro'), float(np.mean(aucs))

print("Loading...", flush=True)
with open('data/processed/amr_sequences.csv') as f:
    reader = csv.DictReader(f)
    seqs, lbs = [], []
    for row in reader:
        s = row['sequence'].strip()
        dcs = [dc.strip() for dc in row['drug_classes'].split(';') if dc.strip()]
        if s and dcs: seqs.append(s); lbs.append(dcs)

# Top 10 classes for speed
lc = Counter()
for l in lbs:
    for x in l: lc[x] += 1
top = [c for c,n in lc.most_common(10) if n>=20]
print(f'Top 10 classes: {top}', flush=True)

fl = [[x for x in s if x in top] for s in lbs]
vi = [i for i,s in enumerate(fl) if s]
seqs = [seqs[i] for i in vi]; fl = [fl[i] for i in vi]
mlb = MultiLabelBinarizer(); y = mlb.fit_transform(fl)
print(f'{len(seqs)} seqs, 10 classes', flush=True)

print("Dipeptide...", flush=True)
X = np.array([dipeptide(s) for s in seqs])

lr = OneVsRestClassifier(LogisticRegression(max_iter=1000, C=0.1, random_state=42), n_jobs=-1)

# RANDOM
print("\n=== RANDOM ===", flush=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
sc = StandardScaler(); X_tr = sc.fit_transform(X_tr); X_te = sc.transform(X_te)
t0=time.time(); lr.fit(X_tr, y_tr); t1=time.time()
mi, ma, au = evaluate(lr, X_te, y_te)
print(f'Micro F1={mi:.4f} Macro F1={ma:.4f} AUC={au:.4f} Time={t1-t0:.1f}s', flush=True)
r_mi, r_ma, r_au = mi, ma, au

# CLUSTERED
print("\n=== CLUSTERED ===", flush=True)
nc = max(2, len(seqs)//50)
cl = AgglomerativeClustering(n_clusters=nc, metric='cosine', linkage='average')
labels = cl.fit_predict(X)

uc = list(set(labels)); np.random.seed(42); np.random.shuffle(uc)
si = int(len(uc)*0.8)
ti = [i for i,c in enumerate(labels) if c in set(uc[:si])]
ei = [i for i,c in enumerate(labels) if c in set(uc[si:])]
print(f'Train: {len(ti)} Test: {len(ei)}', flush=True)

X_tr, X_te = X[ti], X[ei]; y_tr, y_te = y[ti], y[ei]
sc = StandardScaler(); X_tr = sc.fit_transform(X_tr); X_te = sc.transform(X_te)
t0=time.time(); lr.fit(X_tr, y_tr); t1=time.time()
mi, ma, au = evaluate(lr, X_te, y_te)
print(f'Micro F1={mi:.4f} Macro F1={ma:.4f} AUC={au:.4f} Time={t1-t0:.1f}s', flush=True)
c_mi, c_ma, c_au = mi, ma, au

# RESULT
drop = (r_mi - c_mi) / r_mi * 100
print(f'\n============================================', flush=True)
print(f'RANDOM  Micro F1: {r_mi:.4f}  AUC: {r_au:.4f}', flush=True)
print(f'CLUSTER Micro F1: {c_mi:.4f}  AUC: {c_au:.4f}', flush=True)
print(f'Data leakage inflation: {drop:.1f}%', flush=True)
print(f'TRAIN ON CLUSTERED SPLIT FOR REAL GENERALIZATION SCORES', flush=True)
print(f'============================================', flush=True)
