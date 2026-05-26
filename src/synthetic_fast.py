"""FAST Synthetic Genome AMR Dataset - optimized k-mer pipeline."""
import csv, gzip, random, time
from collections import Counter
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score, roc_auc_score
import xgboost as xgb

DATA_DIR = __import__('pathlib').Path(__file__).parent.parent / "data"

# Pre-computed codon table (frequency-weighted for E. coli)
AA_TO_CODONS = {
    'A': ['GCG','GCA','GCT','GCC'], 'C': ['TGC','TGT'],
    'D': ['GAT','GAC'], 'E': ['GAA','GAG'], 'F': ['TTT','TTC'],
    'G': ['GGC','GGT','GGG','GGA'], 'H': ['CAT','CAC'],
    'I': ['ATT','ATC','ATA'], 'K': ['AAA','AAG'],
    'L': ['CTG','TTG','CTT','CTC','CTA','TTA'],
    'M': ['ATG'], 'N': ['AAT','AAC'],
    'P': ['CCG','CCA','CCT','CCC'], 'Q': ['CAG','CAA'],
    'R': ['CGT','CGC','CGG','CGA','AGA','AGG'],
    'S': ['AGC','TCT','AGT','TCC','TCA','TCG'],
    'T': ['ACC','ACT','ACG','ACA'], 'V': ['GTG','GTT','GTA','GTC'],
    'W': ['TGG'], 'Y': ['TAT','TAC'], '_': ['TAA','TGA','TAG'],
}

def load_genome(path):
    with gzip.open(path, 'rt') as f:
        return ''.join(l.strip() for l in f if not l.startswith('>')).upper()

def load_genes():
    genes = []
    with open(DATA_DIR / "processed" / "amr_sequences.csv") as f:
        for row in csv.DictReader(f):
            seq = row['sequence'].strip()
            dcs = row['drug_classes'].strip()
            if seq and dcs:
                genes.append({
                    'dna': protein_to_dna_fast(seq),
                    'drugs': set(dc.strip() for dc in dcs.split(';') if dc.strip())
                })
    return genes

def protein_to_dna_fast(protein):
    """Fast protein -> DNA conversion."""
    result = []
    for aa in protein:
        codons = AA_TO_CODONS.get(aa, ['NNN'])
        result.append(codons[hash(aa + str(len(result))) % len(codons)])
    return ''.join(result)

def compute_kmers_sparse(sequence, k=12, n_features=20000, seed=42):
    """Compute k-mer frequencies using sparse feature hashing.
    
    Uses hash trick: h(kmer) % n_features -> count.
    Returns normalized feature vector.
    """
    features = np.zeros(n_features, dtype=np.float32)
    n_kmers = len(sequence) - k + 1
    if n_kmers <= 0:
        return features
    
    for i in range(0, n_kmers, 3):  # Step of 3 for speed (sampling)
        if i + k > len(sequence):
            break
        kmer = sequence[i:i+k]
        if 'N' in kmer:
            continue
        h = hash(kmer) % n_features
        features[h] += 1
    
    # Normalize
    features /= max(1, n_kmers / 3)
    return features

def compute_ref_kmers(genome, k=12, n_features=20000):
    """Pre-compute reference genome k-mers (done once)."""
    return compute_kmers_sparse(genome, k, n_features)

def generate_sample(ref_kmers, genes, n_genes, k=12, n_features=20000):
    """Generate one synthetic genome's k-mer profile."""
    selected = random.sample(genes, min(n_genes, len(genes)))
    
    # Start with reference k-mers
    features = ref_kmers.copy()
    all_drugs = set()
    
    for gene in selected:
        # Add gene's k-mers (they'll be detected against the background)
        gene_kmer = compute_kmers_sparse(gene['dna'], k, n_features)
        features += gene_kmer * 2.0  # Amplify gene signal
        all_drugs.update(gene['drugs'])
    
    return features, all_drugs

def generate_dataset(ref_kmers, genes, n_samples, genes_per_sample=(1, 8),
                     k=12, n_features=20000):
    X = np.zeros((n_samples, n_features), dtype=np.float32)
    Y_sets = []
    
    for i in range(n_samples):
        n_genes = random.randint(*genes_per_sample)
        X[i], drugs = generate_sample(ref_kmers, genes, n_genes, k, n_features)
        Y_sets.append(drugs)
        if i % 1000 == 0:
            print(f"  {i}/{n_samples}", flush=True)
    
    # Multi-label binarizer
    all_classes = sorted(set().union(*Y_sets))
    mlb = MultiLabelBinarizer(classes=all_classes)
    Y = mlb.fit_transform(Y_sets)
    
    return X, Y, mlb

def main():
    print("="*60)
    print("FAST SYNTHETIC GENOME AMR")
    print("="*60)
    
    # Load
    print("Loading genome...", flush=True)
    genome = load_genome(DATA_DIR / "raw" / "ecoli_k12.fna")
    print(f"  {len(genome):,} bp", flush=True)
    
    print("Loading genes...", flush=True)
    genes = load_genes()
    print(f"  {len(genes)} genes", flush=True)
    
    # Pre-compute reference k-mers ONCE
    print("Pre-computing reference k-mers...", flush=True)
    t0 = time.time()
    ref_kmers = compute_ref_kmers(genome)
    print(f"  {time.time()-t0:.1f}s", flush=True)
    
    # Generate dataset
    N_TRAIN, N_TEST = 2000, 500
    K = 12
    NF = 20000
    
    print(f"\nGenerating {N_TRAIN} train samples...", flush=True)
    t0 = time.time()
    X_train, y_train, mlb = generate_dataset(ref_kmers, genes, N_TRAIN, (1, 6), K, NF)
    print(f"  {time.time()-t0:.1f}s", flush=True)
    
    print(f"Generating {N_TEST} test samples...", flush=True)
    X_test, y_test, _ = generate_dataset(ref_kmers, genes, N_TEST, (1, 6), K, NF)
    
    print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}, {y_train.shape[1]} classes")
    print(f"Classes: {list(mlb.classes_)}")
    
    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    # Train
    print("\nTraining XGBoost...", flush=True)
    model = MultiOutputClassifier(xgb.XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        random_state=42, n_jobs=-1, verbosity=0, eval_metric='logloss',
        tree_method='hist'
    ))
    t0 = time.time()
    model.fit(X_train_s, y_train)
    print(f"  {time.time()-t0:.1f}s", flush=True)
    
    # Evaluate
    y_pred = model.predict(X_test_s)
    probas = np.column_stack([
        e.predict_proba(X_test_s)[:,1] if e.predict_proba(X_test_s).shape[1]>1
        else np.zeros(len(X_test_s)) for e in model.estimators_
    ])
    
    micro_f1 = f1_score(y_test, y_pred, average='micro')
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    aucs = []; 
    for i in range(y_test.shape[1]):
        try: aucs.append(roc_auc_score(y_test[:,i], probas[:,i]))
        except: aucs.append(0.5)
    
    print(f"\nRESULTS:")
    print(f"  Micro F1: {micro_f1:.4f}")
    print(f"  Macro F1: {macro_f1:.4f}")
    print(f"  Mean AUC: {np.mean(aucs):.4f}")
    
    # Per-class
    print(f"\nPer-class AUC:")
    for i, cls in enumerate(mlb.classes_):
        sup = y_test[:,i].sum()
        if sup > 0:
            print(f"  {cls:<35s} AUC={aucs[i]:.4f} n={sup:.0f}")

if __name__ == '__main__':
    main()
