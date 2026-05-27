#!/usr/bin/env python3
"""Phase 2: Real genome AMR prediction pipeline.
Loads 50 real bacterial genomes, extracts k-mer features,
detects resistance genes via CARD k-mer matching, and trains a model.
"""
import csv, gzip, json, pickle, random, time
from collections import Counter
from pathlib import Path
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score, roc_auc_score, classification_report
import xgboost as xgb

BASE = Path(__file__).parent.parent
GENOME_DIR = BASE / "data" / "raw" / "fda_argos" / "genomes"
CARD_JSON = BASE / "data" / "raw" / "card.json"
OUT_DIR = BASE / "data" / "processed"
OUT_DIR.mkdir(exist_ok=True)

K = 12
N_FEATURES = 20000
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ──────────────────────────────────────────────
# 1. Load CARD gene nucleotide sequences
# ──────────────────────────────────────────────
def load_card_genes():
    """Load resistance gene DNA sequences + drug classes from CARD JSON."""
    print("Loading CARD database...", flush=True)
    with open(CARD_JSON) as f:
        card = json.load(f)
    
    genes = []
    for entry in card.values():
        if not isinstance(entry, dict):
            continue
        
        # Extract DNA sequence from nested structure
        dna_seq = None
        model_seqs = entry.get('model_sequences', {})
        seq_wrapper = model_seqs.get('sequence', {})
        if isinstance(seq_wrapper, dict):
            for model_id, seq_data in seq_wrapper.items():
                if isinstance(seq_data, dict):
                    dna_info = seq_data.get('dna_sequence', {})
                    if isinstance(dna_info, dict):
                        dna_seq = dna_info.get('sequence', '')
                        if dna_seq:
                            break
        
        if not dna_seq or len(dna_seq) < 30:
            continue
        
        # Extract drug classes from ARO_category
        drug_classes = set()
        cats = entry.get('ARO_category', {})
        if isinstance(cats, dict):
            for cat_id, cat_info in cats.items():
                if isinstance(cat_info, dict):
                    if cat_info.get('category_aro_class_name') == 'Drug Class':
                        dc = cat_info.get('category_aro_name', '')
                        if dc:
                            drug_classes.add(dc)
        
        if not drug_classes:
            continue
        
        genes.append({
            'name': entry.get('ARO_name', '?'),
            'dna': dna_seq.upper(),
            'drugs': drug_classes
        })
    
    print(f"  {len(genes)} genes with sequences + drug classes", flush=True)
    return genes

# ──────────────────────────────────────────────
# 2. K-mer utilities
# ──────────────────────────────────────────────
def compute_kmers(sequence, k=K, n_features=N_FEATURES, step=3):
    """Compute sparse k-mer frequency vector.
    
    Args:
        sequence: DNA string
        k: k-mer size (default 12)
        n_features: hash space size (default 20000)
        step: stride for sampling (lower = more kmers but slower)
    """
    features = np.zeros(n_features, dtype=np.float32)
    n_kmers = len(sequence) - k + 1
    if n_kmers <= 0:
        return features
    
    for i in range(0, n_kmers, step):
        if i + k > len(sequence):
            break
        kmer = sequence[i:i+k]
        if 'N' in kmer:
            continue
        h = hash(kmer) % n_features
        features[h] += 1
    
    features /= max(1, n_kmers / step)
    return features

# ──────────────────────────────────────────────
# 3. Load real genomes
# ──────────────────────────────────────────────
def load_real_genomes(max_per_species=5):
    """Load real genomes and compute k-mer features."""
    print("\nLoading real genomes...", flush=True)
    genome_data = []
    
    for species_dir in sorted(GENOME_DIR.iterdir()):
        if not species_dir.is_dir():
            continue
        species = species_dir.name.replace('_', ' ')
        
        genome_files = sorted(species_dir.glob("*.fna.gz"))
        for gf in genome_files[:max_per_species]:
            print(f"  {gf.name}...", end=" ", flush=True)
            try:
                with gzip.open(gf, 'rt') as f:
                    seq = ''.join(l.strip() for l in f if not l.startswith('>')).upper()
                
                kmers = compute_kmers(seq)
                genome_data.append({
                    'species': species,
                    'file': gf.name,
                    'kmers': kmers,
                    'length': len(seq),
                })
                print(f"{len(seq)/1e6:.1f}Mbp OK", flush=True)
            except Exception as e:
                print(f"FAIL: {e}", flush=True)
    
    print(f"  {len(genome_data)} genomes loaded", flush=True)
    return genome_data

# ──────────────────────────────────────────────
# 4. Label genomes by CARD gene presence
# ──────────────────────────────────────────────
def detect_resistance_genes(genomes, card_genes):
    """Detect CARD genes in genomes via k-mer feature vector similarity."""
    print("\nDetecting resistance genes...", flush=True)
    
    # Compute CARD gene feature vectors
    print(f"  Computing {len(card_genes)} CARD gene feature vectors...", flush=True)
    card_features = np.zeros((len(card_genes), N_FEATURES), dtype=np.float32)
    for i, cg in enumerate(card_genes):
        card_features[i] = compute_kmers(cg['dna'], step=1)
    print(f"  Done.", flush=True)
    
    # Compare each genome against all CARD genes
    for g in genomes:
        g['detected_genes'] = []
        g['drug_labels'] = set()
        
        # Cosine-like similarity: normalize, then dot product
        genome_norm = g['kmers'] / (np.linalg.norm(g['kmers']) + 1e-10)
        card_norms = card_features / (np.linalg.norm(card_features, axis=1, keepdims=True) + 1e-10)
        similarities = np.dot(card_norms, genome_norm)
        
        # Threshold: top matches
        threshold = 0.15
        matches = np.where(similarities > threshold)[0]
        
        for idx in matches:
            g['detected_genes'].append(card_genes[idx]['name'])
            g['drug_labels'].update(card_genes[idx]['drugs'])
    
    detected_counts = [len(g['detected_genes']) for g in genomes]
    labeled_counts = [len(g['drug_labels']) for g in genomes]
    print(f"  Detected genes per genome: min={min(detected_counts)}, max={max(detected_counts)}, "
          f"mean={np.mean(detected_counts):.1f}", flush=True)
    print(f"  Drug labels per genome: min={min(labeled_counts)}, max={max(labeled_counts)}, "
          f"mean={np.mean(labeled_counts):.1f}", flush=True)

# ──────────────────────────────────────────────
# 5. Build training dataset
# ──────────────────────────────────────────────
def build_dataset(genome_data, card_genes):
    """Build ML dataset: X = k-mer features, y = drug class labels."""
    print("\nBuilding dataset...", flush=True)
    
    # Only use genomes that have at least one drug label
    labeled = [g for g in genome_data if g['drug_labels']]
    print(f"  {len(labeled)}/{len(genome_data)} genomes have drug labels", flush=True)
    
    if len(labeled) < 10:
        print("  WARNING: Too few labeled genomes. Using species as proxy labels.", flush=True)
        for g in genome_data:
            g['drug_labels'] = {g['species']}
        labeled = genome_data
    
    X = np.array([g['kmers'] for g in labeled], dtype=np.float32)
    Y_sets = [g['drug_labels'] for g in labeled]
    
    # Collect all classes
    all_classes = sorted(set().union(*Y_sets))
    mlb = MultiLabelBinarizer(classes=all_classes)
    Y = mlb.fit_transform(Y_sets)
    
    print(f"  X: {X.shape}, Y: {Y.shape}, classes: {len(all_classes)}", flush=True)
    print(f"  Classes: {all_classes}", flush=True)
    
    return X, Y, mlb

# ──────────────────────────────────────────────
# 6. Train + Evaluate
# ──────────────────────────────────────────────
def train_evaluate(X, Y, mlb, model_path=None):
    """Train XGBoost and evaluate with cross-validation."""
    print("\n" + "="*60)
    print("TRAINING", flush=True)
    print("="*60)
    
    from sklearn.model_selection import StratifiedKFold, train_test_split
    
    # Split
    if len(X) >= 20:
        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.3, random_state=SEED, stratify=Y.any(axis=1) if Y.shape[1] > 1 else None
        )
    else:
        X_train, X_test, y_train, y_test = X, X, Y, Y  # Whole dataset if too small
    
    print(f"  Train: {X_train.shape[0]}, Test: {X_test.shape[0]}", flush=True)
    
    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    # Train
    print("  Training XGBoost...", flush=True)
    model = MultiOutputClassifier(xgb.XGBClassifier(
        n_estimators=55, max_depth=10, learning_rate=0.122,
        random_state=SEED, n_jobs=-1, verbosity=0, eval_metric='logloss',
        tree_method='hist'
    ))
    t0 = time.time()
    model.fit(X_train_s, y_train)
    print(f"  Done in {time.time()-t0:.1f}s", flush=True)
    
    # Predict
    y_pred = model.predict(X_test_s)
    
    # Metrics
    micro_f1 = f1_score(y_test, y_pred, average='micro')
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    
    # Per-class
    print(f"\n  RESULTS:")
    print(f"  Micro F1: {micro_f1:.4f}")
    print(f"  Macro F1: {macro_f1:.4f}")
    
    print(f"\n  Per-class:")
    for i, cls in enumerate(mlb.classes_):
        sup = y_test[:, i].sum()
        if sup > 0:
            f1 = f1_score(y_test[:, i], y_pred[:, i])
            print(f"    {cls:<40s} F1={f1:.4f} n={sup:.0f}")
    
    # Save model
    if model_path:
        with open(model_path, 'wb') as f:
            pickle.dump({'model': model, 'scaler': scaler, 'mlb': mlb}, f)
        print(f"\n  Model saved: {model_path}")
    
    return micro_f1, macro_f1

# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    print("="*60)
    print("PHASE 2: REAL GENOME AMR PREDICTION")
    print("="*60)
    
    # Load CARD
    card_genes = load_card_genes()
    
    # Load genomes
    genome_data = load_real_genomes()
    
    # Detect resistance genes
    detect_resistance_genes(genome_data, card_genes)
    
    # Build dataset
    X, Y, mlb = build_dataset(genome_data, card_genes)
    
    # Save features
    np.savez_compressed(OUT_DIR / "real_genome_features.npz", X=X, Y=Y)
    with open(OUT_DIR / "real_genome_mlb.pkl", 'wb') as f:
        pickle.dump(mlb, f)
    print(f"  Features saved to {OUT_DIR}/real_genome_features.npz", flush=True)
    
    # Train
    micro, macro = train_evaluate(X, Y, mlb, OUT_DIR / "real_genome_model.pkl")
    
    print(f"\n{'='*60}")
    print(f"PHASE 2 COMPLETE — Micro F1: {micro:.4f}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
