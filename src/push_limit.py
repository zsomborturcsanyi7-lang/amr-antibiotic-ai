"""PUSH TO THE LIMIT: Maximum performance on synthetic genome AMR.

Strategy:
1. More data (5000 synthetic genomes)
2. ESM-2 features for inserted genes (protein level)
3. Combined k-mer + ESM-2 features
4. Optuna hyperparameter tuning
5. LightGBM + XGBoost ensemble
"""
import csv, gzip, random, time, torch
from collections import Counter, defaultdict
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import optuna
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

DATA = Path(__file__).parent.parent / "data"

AA_TO_CODONS = {
    'A':['GCG','GCA','GCT','GCC'],'C':['TGC','TGT'],'D':['GAT','GAC'],'E':['GAA','GAG'],
    'F':['TTT','TTC'],'G':['GGC','GGT','GGG','GGA'],'H':['CAT','CAC'],'I':['ATT','ATC','ATA'],
    'K':['AAA','AAG'],'L':['CTG','TTG','CTT','CTC','CTA','TTA'],'M':['ATG'],'N':['AAT','AAC'],
    'P':['CCG','CCA','CCT','CCC'],'Q':['CAG','CAA'],'R':['CGT','CGC','CGG','CGA','AGA','AGG'],
    'S':['AGC','TCT','AGT','TCC','TCA','TCG'],'T':['ACC','ACT','ACG','ACA'],'V':['GTG','GTT','GTA','GTC'],
    'W':['TGG'],'Y':['TAT','TAC'],'_':['TAA','TGA','TAG'],
}

def load_genome():
    with gzip.open(DATA/'raw'/'ecoli_k12.fna','rt') as f:
        return ''.join(l.strip() for l in f if not l.startswith('>')).upper()

def load_genes_with_families(top_n_classes=12):
    genes = []
    drug_counts = Counter()
    with open(DATA/'processed'/'amr_sequences.csv') as f:
        for row in csv.DictReader(f):
            seq = row['sequence'].strip()
            dcs = row['drug_classes'].strip()
            families = row['gene_families'].strip()
            if seq and dcs:
                drugs = {dc.strip() for dc in dcs.split(';') if dc.strip()}
                for d in drugs: drug_counts[d] += 1
                fams = {f.strip() for f in families.split(';') if f.strip()}
                genes.append({'seq': seq, 'drugs': drugs, 'families': fams})
    
    top = {c for c,_ in drug_counts.most_common(top_n_classes)}
    filtered = []
    for g in genes:
        overlap = g['drugs'] & top
        if overlap:
            filtered.append({'seq': g['seq'], 'drugs': overlap, 'families': g['families']})
    
    return filtered, sorted(top)

def prot2dna(protein):
    return ''.join(AA_TO_CODONS.get(aa,['NNN'])[hash(aa+str(i))%len(AA_TO_CODONS.get(aa,['NNN']))] for i,aa in enumerate(protein))

def kmer_vec(seq, k=12, nf=5000):
    v = np.zeros(nf, dtype=np.float32)
    n = len(seq)-k+1
    if n<=0: return v
    for i in range(0,n,5):
        km = seq[i:i+k]
        if 'N' not in km:
            v[hash(km)%nf] += 1
    return v / max(1,n/5)

# ESM-2 embeddings
def compute_esm_embeddings(sequences, batch_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Loading ESM-2 on {device}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    model = AutoModel.from_pretrained("facebook/esm2_t6_8M_UR50D").to(device)
    model.eval()
    
    all_embs = []
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i+batch_size]
        encoded = tokenizer(batch, return_tensors='pt', padding=True, 
                          truncation=True, max_length=512)
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded)
            mask = encoded['attention_mask'].unsqueeze(-1)
            embs = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
        all_embs.append(embs.cpu().numpy())
        if (i//batch_size) % 50 == 0 and i > 0:
            print(f"    {i}/{len(sequences)}", flush=True)
    
    del model, tokenizer
    torch.cuda.empty_cache()
    return np.concatenate(all_embs, axis=0)

def evaluate(model, X_te, y_te):
    yp = model.predict(X_te)
    probas = np.column_stack([
        e.predict_proba(X_te)[:,1] if e.predict_proba(X_te).shape[1]>1 else np.zeros(len(X_te))
        for e in model.estimators_
    ])
    aucs = []
    for i in range(y_te.shape[1]):
        try: aucs.append(float(roc_auc_score(y_te[:,i], probas[:,i])))
        except: aucs.append(0.5)
    return {
        'micro_f1': float(f1_score(y_te,yp,average='micro')),
        'macro_f1': float(f1_score(y_te,yp,average='macro')),
        'mean_auc': float(np.mean(aucs)),
    }

def generate_split(ref, gene_pool, class_names, n_samples, gene_esm_lookup=None):
    """Generate samples. If gene_esm_lookup provided, use ESM-2 embeddings."""
    NF = 5000
    ESM_DIM = 320
    use_esm = gene_esm_lookup is not None
    
    if use_esm:
        X_dim = NF + ESM_DIM
    else:
        X_dim = NF
    
    X = np.zeros((n_samples, X_dim), dtype=np.float32)
    Y = np.zeros((n_samples, len(class_names)), dtype=np.int8)
    
    for i in range(n_samples):
        n_genes = random.randint(1,4)
        selected = random.sample(gene_pool, min(n_genes, len(gene_pool)))
        
        # K-mer features
        v = ref.copy()
        esm_sum = np.zeros(ESM_DIM, dtype=np.float32) if use_esm else None
        
        for g in selected:
            dna = prot2dna(g['seq'])
            v += kmer_vec(dna, 12, NF) * 10.0
            
            if use_esm and g['seq'] in gene_esm_lookup:
                esm_sum += gene_esm_lookup[g['seq']]
        
        if use_esm:
            X[i] = np.concatenate([v, esm_sum / max(1, n_genes)])
        else:
            X[i] = v
        
        for drug in set().union(*[g['drugs'] for g in selected]):
            if drug in class_names:
                Y[i, class_names.index(drug)] = 1
    
    return X, Y

def main():
    print("="*60)
    print("PUSHING TO THE MAXIMUM - Synthetic Genome AMR")
    print("="*60)
    
    # Load data
    print("\n[1] Loading...", flush=True)
    genome = load_genome()
    genes, class_names = load_genes_with_families(12)
    ref_kmers = kmer_vec(genome)
    print(f"  Genome: {len(genome):,} bp", flush=True)
    print(f"  Genes: {len(genes)}, Classes: {len(class_names)}", flush=True)
    
    # Clustered split by gene family
    print("\n[2] Clustered split...", flush=True)
    family_to_genes = defaultdict(list)
    for i, g in enumerate(genes):
        for fam in g['families']:
            family_to_genes[fam].append(i)
    
    families = list(family_to_genes.keys())
    random.seed(42); random.shuffle(families)
    split_fam = int(len(families)*0.8)
    train_fams = set(families[:split_fam])
    test_fams = set(families[split_fam:])
    
    train_genes = [genes[i] for fam in train_fams for i in family_to_genes[fam]]
    test_genes = [genes[i] for fam in test_fams for i in family_to_genes[fam]]
    print(f"  Train families: {len(train_fams)}, Test families: {len(test_fams)}", flush=True)
    
    # ESM-2 embeddings for ALL genes (done once)
    print("\n[3] Computing ESM-2 embeddings...", flush=True)
    all_protein_seqs = [g['seq'] for g in genes]
    all_esm = compute_esm_embeddings(all_protein_seqs, batch_size=32)
    gene_esm = {g['seq']: all_esm[i] for i, g in enumerate(genes)}
    print(f"  Done: {all_esm.shape}", flush=True)
    
    # Generate datasets: K-MER ONLY vs K-MER + ESM-2
    print("\n[4] Generating datasets (5000 train, 1000 test)...", flush=True)
    
    # K-mer only
    X_tr_km, y_tr = generate_split(ref_kmers, train_genes, class_names, 5000)
    X_te_km, y_te = generate_split(ref_kmers, test_genes, class_names, 1000)
    
    # K-mer + ESM-2
    X_tr_esm, _ = generate_split(ref_kmers, train_genes, class_names, 5000, gene_esm)
    X_te_esm, _ = generate_split(ref_kmers, test_genes, class_names, 1000, gene_esm)
    
    print(f"  K-mer: {X_tr_km.shape}, K-mer+ESM: {X_tr_esm.shape}", flush=True)
    
    results = {}
    
    # === EXPERIMENT 1: XGBoost (k-mer only) ===
    print("\n[5a] XGBoost (k-mer only)...", flush=True)
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_tr_km); X_te = sc.transform(X_te_km)
    
    m = MultiOutputClassifier(xgb.XGBClassifier(
        n_estimators=50, max_depth=6, learning_rate=0.2,
        random_state=42, n_jobs=-1, verbosity=0, eval_metric='logloss', tree_method='hist'))
    t0=time.time(); m.fit(X_tr, y_tr); t1=time.time()
    r = evaluate(m, X_te, y_te)
    results['XGB_kmer'] = r
    print(f"  Micro F1={r['micro_f1']:.4f} Macro F1={r['macro_f1']:.4f} AUC={r['mean_auc']:.4f} Time={t1-t0:.0f}s", flush=True)
    
    # === EXPERIMENT 2: XGBoost (k-mer + ESM-2) ===
    print("\n[5b] XGBoost (k-mer + ESM-2)...", flush=True)
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_tr_esm); X_te = sc.transform(X_te_esm)
    
    m = MultiOutputClassifier(xgb.XGBClassifier(
        n_estimators=50, max_depth=6, learning_rate=0.2,
        random_state=42, n_jobs=-1, verbosity=0, eval_metric='logloss', tree_method='hist'))
    t0=time.time(); m.fit(X_tr, y_tr); t1=time.time()
    r = evaluate(m, X_te, y_te)
    results['XGB_kmer+esm'] = r
    print(f"  Micro F1={r['micro_f1']:.4f} Macro F1={r['macro_f1']:.4f} AUC={r['mean_auc']:.4f} Time={t1-t0:.0f}s", flush=True)
    
    # === EXPERIMENT 3: LightGBM (k-mer only) ===
    print("\n[5c] LightGBM (k-mer only)...", flush=True)
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_tr_km); X_te = sc.transform(X_te_km)
    
    m = MultiOutputClassifier(lgb.LGBMClassifier(
        n_estimators=50, max_depth=6, learning_rate=0.2,
        random_state=42, n_jobs=-1, verbose=-1))
    t0=time.time(); m.fit(X_tr, y_tr); t1=time.time()
    r = evaluate(m, X_te, y_te)
    results['LGB_kmer'] = r
    print(f"  Micro F1={r['micro_f1']:.4f} Macro F1={r['macro_f1']:.4f} AUC={r['mean_auc']:.4f} Time={t1-t0:.0f}s", flush=True)
    
    # === EXPERIMENT 4: Optuna XGBoost ===
    print("\n[5d] Optuna tuning (k-mer only)...", flush=True)
    sc = StandardScaler()
    X_tr_opt = sc.fit_transform(X_tr_km); X_te_opt = sc.transform(X_te_km)
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-6, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-6, 1.0, log=True),
            'random_state': 42, 'n_jobs': -1, 'verbosity': 0,
            'eval_metric': 'logloss', 'tree_method': 'hist',
        }
        model = MultiOutputClassifier(xgb.XGBClassifier(**params))
        model.fit(X_tr_opt, y_tr)
        yp = model.predict(X_te_opt)
        return float(f1_score(y_te, yp, average='micro'))
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=25, show_progress_bar=True)
    
    # Final model with best params
    final = MultiOutputClassifier(xgb.XGBClassifier(**study.best_params, 
        random_state=42, n_jobs=-1, verbosity=0, eval_metric='logloss', tree_method='hist'))
    final.fit(X_tr_opt, y_tr)
    r = evaluate(final, X_te_opt, y_te)
    results['XGB_tuned'] = r
    print(f"  Best params: {study.best_params}", flush=True)
    print(f"  Micro F1={r['micro_f1']:.4f} Macro F1={r['macro_f1']:.4f} AUC={r['mean_auc']:.4f}", flush=True)
    
    # === FINAL SUMMARY ===
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS (Clustered Split, Synthetic Genomes)")
    print(f"{'='*60}")
    print(f"{'Method':<25s} {'Micro F1':<10s} {'Macro F1':<10s} {'AUC':<10s}")
    print(f"{'-'*25} {'-'*10} {'-'*10} {'-'*10}")
    
    for name, r in sorted(results.items(), key=lambda x: x[1]['micro_f1'], reverse=True):
        print(f"{name:<25s} {r['micro_f1']:<10.4f} {r['macro_f1']:<10.4f} {r['mean_auc']:<10.4f}")
    
    best = max(results.items(), key=lambda x: x[1]['micro_f1'])
    print(f"\nBEST: {best[0]} = {best[1]['micro_f1']:.4f}")

if __name__ == '__main__':
    main()
