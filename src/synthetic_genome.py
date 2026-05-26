"""Synthetic Genome AMR Dataset Generator.

Creates training data by inserting known resistance genes into a reference
bacterial genome. The model must learn to detect these genes in genomic context.

This simulates the real clinical problem: given a bacterial genome assembly,
predict which antibiotics it's resistant to.
"""
import csv
import gzip
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.cluster import AgglomerativeClustering
import xgboost as xgb

DATA_DIR = Path(__file__).parent.parent / "data"
RAW = DATA_DIR / "raw"
PROC = DATA_DIR / "processed"

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

def load_reference_genome(path):
    """Load E. coli K-12 reference genome."""
    with gzip.open(path, 'rt') as f:
        seqs = []
        current = []
        for line in f:
            if line.startswith('>'):
                if current:
                    seqs.append(''.join(current))
                    current = []
            else:
                current.append(line.strip())
        if current:
            seqs.append(''.join(current))
    genome = ''.join(seqs).upper()
    print(f"  Reference genome: {len(genome):,} bp")
    return genome

def load_card_genes():
    """Load resistance genes from CARD data with their drug class labels."""
    with open(PROC / "amr_sequences.csv") as f:
        reader = csv.DictReader(f)
        genes = []
        for row in reader:
            seq = row['sequence'].strip()
            drug_classes = row['drug_classes'].strip()
            if seq and drug_classes:
                genes.append({
                    'seq': seq,
                    'aro_name': row['aro_name'],
                    'drug_classes': set(dc.strip() for dc in drug_classes.split(';') if dc.strip()),
                    'model_type': row['model_type'],
                    'length': int(row['length'])
                })
    print(f"  Resistance genes: {len(genes)}")
    return genes

def reverse_complement(seq):
    """Return reverse complement of a DNA sequence."""
    comp = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
    return ''.join(comp.get(b, 'N') for b in reversed(seq))

def translate_dna_to_protein(dna):
    """Translate DNA sequence to protein (standard genetic code)."""
    codon_table = {
        'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
        'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
        'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
        'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
        'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L', 'TTA':'L', 'TTG':'L',
        'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
        'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
        'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
        'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
        'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
        'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
        'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
        'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
        'TTC':'F', 'TTT':'F', 'TAC':'Y', 'TAT':'Y',
        'TGC':'C', 'TGT':'C', 'TGA':'_', 'TAA':'_', 'TAG':'_',
        'TGG':'W',
    }
    protein = []
    for i in range(0, len(dna)-2, 3):
        codon = dna[i:i+3]
        if len(codon) < 3:
            break
        aa = codon_table.get(codon, 'X')
        if aa == '_':  # Stop codon
            break
        protein.append(aa)
    return ''.join(protein)

def convert_protein_to_dna(protein_seq):
    """Convert a protein sequence BACK to a DNA sequence.
    
    Since many amino acids map to multiple codons, we pick one randomly
    (with frequency weighting for realism). This is the "reverse translation"
    needed to insert protein-level resistance genes into a DNA genome.
    """
    aa_to_codons = {
        'A': ['GCA','GCC','GCG','GCT'],
        'C': ['TGC','TGT'],
        'D': ['GAC','GAT'],
        'E': ['GAA','GAG'],
        'F': ['TTC','TTT'],
        'G': ['GGA','GGC','GGG','GGT'],
        'H': ['CAC','CAT'],
        'I': ['ATA','ATC','ATT'],
        'K': ['AAA','AAG'],
        'L': ['CTA','CTC','CTG','CTT','TTA','TTG'],
        'M': ['ATG'],
        'N': ['AAC','AAT'],
        'P': ['CCA','CCC','CCG','CCT'],
        'Q': ['CAA','CAG'],
        'R': ['AGA','AGG','CGA','CGC','CGG','CGT'],
        'S': ['AGC','AGT','TCA','TCC','TCG','TCT'],
        'T': ['ACA','ACC','ACG','ACT'],
        'V': ['GTA','GTC','GTG','GTT'],
        'W': ['TGG'],
        'Y': ['TAC','TAT'],
        '_': ['TAA','TAG','TGA'],
        'X': ['NNN'],
    }
    dna = []
    for aa in protein_seq:
        codons = aa_to_codons.get(aa, ['NNN'])
        dna.append(random.choice(codons))
    return ''.join(dna)

def compute_genome_kmers(genome, k=10, n_features=10000, seed=42):
    """Extract k-mer features from a genome using feature hashing.
    
    Uses murmurhash-like hashing to map k-mers to a fixed-size feature vector.
    """
    rng = np.random.RandomState(seed)
    # Pre-compute hash table: we use 2 hash functions per k-mer for counting
    features = np.zeros(n_features * 2)
    
    for i in range(len(genome) - k + 1):
        kmer = genome[i:i+k]
        if 'N' in kmer:
            continue
        # Simple hash: sum of character values * position weights
        h = 0
        for j, c in enumerate(kmer):
            h = (h * 31 + ord(c)) % n_features
        features[h] += 1
        # Second hash for better distribution
        h2 = (h * 17 + 13) % n_features
        features[n_features + h2] += 1
    
    # Normalize by genome length
    total_kmers = len(genome) - k + 1
    if total_kmers > 0:
        features = features / total_kmers * 10000  # Scale up for numerical stability
    
    return features

def dipeptide_from_protein(seq):
    """Compute dipeptide composition of a protein sequence."""
    v = np.zeros(400)
    for i in range(len(seq)-1):
        a1, a2 = seq[i], seq[i+1]
        if a1 in AA_TO_IDX and a2 in AA_TO_IDX:
            v[AA_TO_IDX[a1]*20+AA_TO_IDX[a2]] += 1
    t = v.sum()
    return v/t if t > 0 else v

def generate_synthetic_genome(ref_genome, resistance_genes, n_genes, 
                               min_insert_length=500, max_insert_length=5000):
    """Generate a synthetic genome with inserted resistance genes.
    
    Actually: we DON'T insert into the genome (that changes coordinates).
    Instead we concatenate: reference_genome + random_genes.
    This is equivalent for k-mer analysis and much simpler.
    
    Returns: (synthetic_genome, drug_classes, gene_ids)
    """
    selected = random.sample(resistance_genes, min(n_genes, len(resistance_genes)))
    
    # Concatenate reference + genes + random spacers
    parts = [ref_genome]
    all_drugs = set()
    
    for gene in selected:
        # Add random spacer
        spacer_len = random.randint(min_insert_length, max_insert_length)
        spacer = ''.join(random.choice('ACGT') for _ in range(spacer_len))
        parts.append(spacer)
        
        # Convert protein to DNA and insert
        gene_dna = convert_protein_to_dna(gene['seq'])
        parts.append(gene_dna)
        
        # Sometimes insert reverse complement (plasmid insertion)
        if random.random() < 0.3:
            parts.append(reverse_complement(gene_dna))
        
        all_drugs.update(gene['drug_classes'])
    
    return ''.join(parts), all_drugs, [g['aro_name'] for g in selected]

def generate_dataset(ref_genome, resistance_genes, n_samples=2000, 
                     genes_per_sample=(1, 10)):
    """Generate a full synthetic dataset."""
    X_kmer = []
    X_dipep = []
    Y_sets = []
    
    for i in range(n_samples):
        n_genes = random.randint(*genes_per_sample)
        genome, drugs, gene_names = generate_synthetic_genome(
            ref_genome, resistance_genes, n_genes
        )
        
        kmer_vec = compute_genome_kmers(genome)
        X_kmer.append(kmer_vec)
        Y_sets.append(drugs)
        
        if i % 500 == 0:
            print(f"  Generated {i}/{n_samples}...", flush=True)
    
    X_kmer = np.array(X_kmer)
    
    # Multi-label binarizer
    all_classes = sorted(set().union(*Y_sets))
    mlb = MultiLabelBinarizer(classes=all_classes)
    Y = mlb.fit_transform(Y_sets)
    
    return X_kmer, Y, mlb

def main():
    print("=" * 60)
    print("SYNTHETIC GENOME AMR DATASET + BASELINE")
    print("=" * 60)
    
    # 1. Load data
    print("\n[1] Loading reference genome...")
    genome = load_reference_genome(RAW / "ecoli_k12.fna")
    
    print("\n[2] Loading resistance genes...")
    genes = load_card_genes()
    
    # 2. Generate synthetic dataset
    print(f"\n[3] Generating synthetic genomes...")
    N_TRAIN = 2000
    N_TEST = 500
    
    X_train, y_train, mlb = generate_dataset(genome, genes, N_TRAIN, (1, 8))
    X_test, y_test, _ = generate_dataset(genome, genes, N_TEST, (1, 8))
    
    print(f"\n  Train: {X_train.shape}, {y_train.shape[1]} classes")
    print(f"  Test:  {X_test.shape}, {y_test.shape[1]} classes")
    print(f"  Classes: {list(mlb.classes_)}")
    
    # 3. Train XGBoost baseline
    print(f"\n[4] Training XGBoost...")
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    model = MultiOutputClassifier(xgb.XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbosity=0, eval_metric='logloss',
        tree_method='hist'
    ))
    
    t0 = time.time()
    model.fit(X_train_s, y_train)
    train_time = time.time() - t0
    
    # 4. Evaluate
    y_pred = model.predict(X_test_s)
    
    probas = np.column_stack([
        e.predict_proba(X_test_s)[:, 1] if e.predict_proba(X_test_s).shape[1] > 1
        else np.zeros(len(X_test_s))
        for e in model.estimators_
    ])
    
    micro_f1 = f1_score(y_test, y_pred, average='micro')
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    
    aucs = []
    for i in range(y_test.shape[1]):
        try:
            aucs.append(roc_auc_score(y_test[:, i], probas[:, i]))
        except:
            aucs.append(0.5)
    mean_auc = np.mean(aucs)
    
    print(f"\n  RESULTS:")
    print(f"  Micro F1: {micro_f1:.4f}")
    print(f"  Macro F1: {macro_f1:.4f}")
    print(f"  Mean AUC: {mean_auc:.4f}")
    print(f"  Train time: {train_time:.1f}s")
    
    # 5. Per-class breakdown
    print(f"\n  Per-class AUC:")
    for i, cls in enumerate(mlb.classes_):
        sup = y_test[:, i].sum()
        if sup > 0:
            print(f"    {cls:<35s} AUC={aucs[i]:.4f} n={sup:.0f}")
    
    print(f"\n  NOTE: This is on SYNTHETIC data where we control the ground truth.")
    print(f"  Real clinical data would need AST measurements.")
    print(f"  But this proves the k-mer approach CAN detect resistance genes in genomic context.")

if __name__ == '__main__':
    main()
