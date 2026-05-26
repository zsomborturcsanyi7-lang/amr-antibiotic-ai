"""ESM-2 benchmark: protein language model embeddings for AMR prediction.

Compares ESM-2 embeddings vs dipeptide on clustered split.
"""
import csv, time, warnings, json
from collections import Counter
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.cluster import AgglomerativeClustering
import xgboost as xgb
import torch
from transformers import AutoTokenizer, AutoModel
warnings.filterwarnings('ignore')

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
MODEL_NAME = "facebook/esm2_t6_8M_UR50D"

def dipeptide(seq):
    v = np.zeros(400)
    for i in range(len(seq)-1):
        a1,a2 = seq[i],seq[i+1]
        if a1 in AA_TO_IDX and a2 in AA_TO_IDX:
            v[AA_TO_IDX[a1]*20+AA_TO_IDX[a2]] += 1
    t = v.sum()
    return v/t if t>0 else v

def esm_embeddings(sequences, batch_size=16):
    """Extract per-protein embeddings from ESM-2 using mean pooling."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Loading {MODEL_NAME} on {device}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    
    all_embs = []
    n = len(sequences)
    for i in range(0, n, batch_size):
        batch = sequences[i:i+batch_size]
        encoded = tokenizer(batch, return_tensors='pt', padding=True, 
                          truncation=True, max_length=512)
        encoded = {k: v.to(device) for k, v in encoded.items()}
        
        with torch.no_grad():
            outputs = model(**encoded)
            mask = encoded['attention_mask'].unsqueeze(-1)
            embs = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
        
        all_embs.append(embs.cpu().numpy())
        
        if (i // batch_size) % 20 == 0:
            print(f"    {i}/{n}", flush=True)
    
    del model, tokenizer
    torch.cuda.empty_cache()
    
    return np.concatenate(all_embs, axis=0)

def evaluate(model, X_te, y_te):
    yp = model.predict(X_te)
    probas = np.column_stack([
        e.predict_proba(X_te)[:,1] if e.predict_proba(X_te).shape[1]>1 
        else np.zeros(len(X_te)) for e in model.estimators_
    ])
    aucs = []
    for i in range(y_te.shape[1]):
        try: aucs.append(float(roc_auc_score(y_te[:,i], probas[:,i])))
        except: aucs.append(0.5)
    return {
        'micro_f1': float(f1_score(y_te, yp, average='micro')),
        'macro_f1': float(f1_score(y_te, yp, average='macro')),
        'mean_auc': float(np.mean(aucs)),
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
n_classes = len(mlb.classes_)
print(f'{len(seqs)} seqs, {n_classes} classes', flush=True)

# CLUSTERED SPLIT (compute once, reuse for both feature sets)
print("Clustering...", flush=True)
X_di = np.array([dipeptide(s) for s in seqs])
nc = max(2, len(seqs)//50)
cl = AgglomerativeClustering(n_clusters=nc, metric='cosine', linkage='average')
labels = cl.fit_predict(X_di)

uc = list(set(labels)); np.random.seed(42); np.random.shuffle(uc)
si = int(len(uc)*0.8)
ti = [i for i,c in enumerate(labels) if c in set(uc[:si])]
ei = [i for i,c in enumerate(labels) if c in set(uc[si:])]
y_tr = y[ti]; y_te = y[ei]
print(f'Train: {len(ti)} Test: {len(ei)}', flush=True)

# ============================================
# 1. DIPEPTIDE BASELINE (clustered)
# ============================================
print("\n=== DIPEPTIDE (clustered) ===", flush=True)
X_tr = X_di[ti]; X_te = X_di[ei]
sc = StandardScaler(); X_tr = sc.fit_transform(X_tr); X_te = sc.transform(X_te)

m = MultiOutputClassifier(xgb.XGBClassifier(
    n_estimators=50, max_depth=5, learning_rate=0.2,
    random_state=42, n_jobs=4, verbosity=0, eval_metric='logloss', tree_method='hist'))
t0 = time.time(); m.fit(X_tr, y_tr); t1 = time.time()
res_di = evaluate(m, X_te, y_te)
print(f'Micro F1={res_di["micro_f1"]:.4f} Macro F1={res_di["macro_f1"]:.4f} '
      f'AUC={res_di["mean_auc"]:.4f} Time={t1-t0:.1f}s', flush=True)

# ============================================
# 2. ESM-2 EMBEDDINGS
# ============================================
print("\n=== ESM-2 EMBEDDINGS ===", flush=True)
X_esm = esm_embeddings(seqs, batch_size=16)
print(f'  Embeddings: {X_esm.shape}', flush=True)

# Same clustered split
X_tr_esm = X_esm[ti]; X_te_esm = X_esm[ei]
sc = StandardScaler(); X_tr_esm = sc.fit_transform(X_tr_esm); X_te_esm = sc.transform(X_te_esm)

print("\n=== ESM-2 XGB (clustered) ===", flush=True)
m = MultiOutputClassifier(xgb.XGBClassifier(
    n_estimators=50, max_depth=5, learning_rate=0.2,
    random_state=42, n_jobs=4, verbosity=0, eval_metric='logloss', tree_method='hist'))
t0 = time.time(); m.fit(X_tr_esm, y_tr); t1 = time.time()
res_esm = evaluate(m, X_te_esm, y_te)
print(f'Micro F1={res_esm["micro_f1"]:.4f} Macro F1={res_esm["macro_f1"]:.4f} '
      f'AUC={res_esm["mean_auc"]:.4f} Time={t1-t0:.1f}s', flush=True)

# ============================================
# 3. ESM-2 + DIPEPTIDE
# ============================================
print("\n=== ESM-2 + DIPEPTIDE ===", flush=True)
X_comb = np.column_stack([X_esm, X_di])
X_tr_comb = X_comb[ti]; X_te_comb = X_comb[ei]
sc = StandardScaler(); X_tr_comb = sc.fit_transform(X_tr_comb); X_te_comb = sc.transform(X_te_comb)

m = MultiOutputClassifier(xgb.XGBClassifier(
    n_estimators=50, max_depth=5, learning_rate=0.2,
    random_state=42, n_jobs=4, verbosity=0, eval_metric='logloss', tree_method='hist'))
t0 = time.time(); m.fit(X_tr_comb, y_tr); t1 = time.time()
res_comb = evaluate(m, X_te_comb, y_te)
print(f'Micro F1={res_comb["micro_f1"]:.4f} Macro F1={res_comb["macro_f1"]:.4f} '
      f'AUC={res_comb["mean_auc"]:.4f} Time={t1-t0:.1f}s', flush=True)

# ============================================
# SUMMARY
# ============================================
print(f'\n{"="*60}', flush=True)
print(f'RESULTS (clustered split, {len(ti)} train / {len(ei)} test):', flush=True)
print(f'{"Feature":<25s} {"Micro F1":<10s} {"Macro F1":<10s} {"AUC":<10s}', flush=True)
print(f'{"-"*25} {"-"*10} {"-"*10} {"-"*10}', flush=True)
print(f'{"Dipeptide":<25s} {res_di["micro_f1"]:<10.4f} {res_di["macro_f1"]:<10.4f} {res_di["mean_auc"]:<10.4f}', flush=True)
print(f'{"ESM-2":<25s} {res_esm["micro_f1"]:<10.4f} {res_esm["macro_f1"]:<10.4f} {res_esm["mean_auc"]:<10.4f}', flush=True)
print(f'{"ESM-2 + Dipeptide":<25s} {res_comb["micro_f1"]:<10.4f} {res_comb["macro_f1"]:<10.4f} {res_comb["mean_auc"]:<10.4f}', flush=True)

improvement = (res_esm["micro_f1"] - res_di["micro_f1"]) / res_di["micro_f1"] * 100 if res_di["micro_f1"] > 0 else 0
print(f'\nESM-2 improvement: +{improvement:.0f}% over dipeptide', flush=True)
print(f'{"="*60}', flush=True)
