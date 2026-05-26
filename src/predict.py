"""AMR Prediction CLI - Predict antibiotic resistance from protein sequence.

Usage:
    python -m src.predict <sequence_or_fasta> [--model MODEL] [--top-k 5]
    
Examples:
    # Single protein sequence
    python -m src.predict MALWMRLLPLLALLALWGPDPAAA...
    
    # From FASTA file
    python -m src.predict resistance_genes.fasta
    
    # From stdin
    cat gene.fasta | python -m src.predict --stdin
"""
import sys
import pickle
import argparse
from pathlib import Path

import numpy as np

# Import feature extraction from training module
from train_baseline import compute_combined_features, AMINO_ACIDS

def parse_fasta(path):
    """Parse FASTA file, return list of (header, sequence)."""
    sequences = []
    with open(path) as f:
        header = None
        seq_lines = []
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if header:
                    sequences.append((header, ''.join(seq_lines)))
                header = line[1:]
                seq_lines = []
            else:
                seq_lines.append(line)
        if header:
            sequences.append((header, ''.join(seq_lines)))
    return sequences

MODELS_DIR = Path(__file__).parent.parent / "models"

def load_model(model_name='xgb'):
    """Load trained model, scaler, and multi-label binarizer."""
    with open(MODELS_DIR / f'{model_name}_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open(MODELS_DIR / 'scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open(MODELS_DIR / 'mlb.pkl', 'rb') as f:
        mlb = pickle.load(f)
    return model, scaler, mlb

def predict_sequence(seq, model, scaler, mlb, threshold=0.5):
    """Predict resistance profile for a single protein sequence."""
    features = compute_combined_features(seq).reshape(1, -1)
    features_scaled = scaler.transform(features)
    
    # Get probabilities from each estimator
    probas = []
    for est in model.estimators_:
        proba = est.predict_proba(features_scaled)[0]
        probas.append(proba[1] if len(proba) > 1 else 0.0)
    
    predictions = []
    for i, (cls, prob) in enumerate(zip(mlb.classes_, probas)):
        predictions.append({
            'class': cls,
            'probability': float(prob),
            'resistant': prob >= threshold
        })
    
    # Sort by probability
    predictions.sort(key=lambda x: x['probability'], reverse=True)
    return predictions

def is_valid_protein(seq):
    """Check if sequence looks like a valid protein (amino acids)."""
    valid = set(AMINO_ACIDS)
    seq_clean = seq.upper().replace('*', '')  # Remove stop codon
    if len(seq_clean) < 20:  # Too short
        return False
    invalid = set(seq_clean) - valid
    if invalid:
        return False
    return True

def predict_fasta(fasta_path, model, scaler, mlb, threshold=0.5, top_k=5):
    """Predict resistance profile for all sequences in a FASTA file."""
    seqs = parse_fasta(fasta_path) if isinstance(fasta_path, str) else fasta_path
    
    results = []
    for header, seq in seqs:
        if not is_valid_protein(seq):
            print(f"  WARNING: Skipping {header[:60]} - not a valid protein sequence", 
                  file=sys.stderr)
            continue
        
        preds = predict_sequence(seq, model, scaler, mlb, threshold)
        resistant = [p for p in preds if p['resistant']]
        
        # Extract gene name from header
        gene_name = header.split()[0][:50]
        
        results.append({
            'gene': gene_name,
            'full_header': header[:100],
            'seq_length': len(seq),
            'resistant_to': [p['class'] for p in resistant],
            'all_predictions': preds[:top_k]
        })
    
    return results

def format_output(results, show_all=True):
    """Format prediction results for display."""
    for r in results:
        print(f"\n{'='*60}")
        print(f"  Gene: {r['gene']}")
        print(f"  Length: {r['seq_length']} aa")
        print(f"{'='*60}")
        
        if r['resistant_to']:
            print(f"\n  RESISTANT TO ({len(r['resistant_to'])} classes):")
            for cls in r['resistant_to']:
                print(f"    * {cls}")
        else:
            print(f"\n  No resistance predicted.")
        
        if show_all:
            print(f"\n  ALL PREDICTIONS (top {len(r['all_predictions'])}):")
            for p in r['all_predictions']:
                marker = ">>>" if p['resistant'] else "   "
                print(f"    {marker} {p['class']:<40s} {p['probability']:.3f}")
    
    # Summary
    total = len(results)
    with_resistance = sum(1 for r in results if r['resistant_to'])
    print(f"\n{'='*60}")
    print(f"  SUMMARY: {with_resistance}/{total} sequences predicted resistant")
    
    # Count resistance classes
    from collections import Counter
    class_counts = Counter()
    for r in results:
        for cls in r['resistant_to']:
            class_counts[cls] += 1
    
    if class_counts:
        print(f"\n  Resistance class distribution:")
        for cls, count in class_counts.most_common():
            print(f"    {cls}: {count}")

def main():
    parser = argparse.ArgumentParser(
        description='Predict antibiotic resistance from protein sequences'
    )
    parser.add_argument('input', nargs='?', 
                       help='Protein sequence or FASTA file path')
    parser.add_argument('--model', default='xgb', choices=['xgb', 'rf'],
                       help='Model to use (default: xgb)')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Probability threshold for resistance (default: 0.5)')
    parser.add_argument('--top-k', type=int, default=5,
                       help='Show top K predictions (default: 5)')
    parser.add_argument('--stdin', action='store_true',
                       help='Read FASTA from stdin')
    parser.add_argument('--json', action='store_true',
                       help='Output as JSON')
    
    args = parser.parse_args()
    
    # Load model
    model, scaler, mlb = load_model(args.model)
    
    # Get input
    if args.stdin:
        # Read FASTA from stdin
        fasta_content = sys.stdin.read()
        # Write to temp file for parsing
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
            f.write(fasta_content)
            tmp_path = f.name
        results = predict_fasta(tmp_path, model, scaler, mlb, args.threshold, args.top_k)
        Path(tmp_path).unlink()
    
    elif args.input is None:
        parser.print_help()
        return
    
    elif Path(args.input).exists():
        # FASTA file
        results = predict_fasta(args.input, model, scaler, mlb, args.threshold, args.top_k)
    
    else:
        # Raw sequence
        seq = args.input.strip().upper()
        if not is_valid_protein(seq):
            print(f"ERROR: Input doesn't look like a valid protein sequence and isn't a file.",
                  file=sys.stderr)
            print(f"       Valid amino acids: {AMINO_ACIDS}", file=sys.stderr)
            sys.exit(1)
        
        preds = predict_sequence(seq, model, scaler, mlb, args.threshold)
        resistant = [p for p in preds if p['resistant']]
        
        results = [{
            'gene': 'input_sequence',
            'full_header': 'Direct input',
            'seq_length': len(seq),
            'resistant_to': [p['class'] for p in resistant],
            'all_predictions': preds[:args.top_k]
        }]
    
    # Output
    if args.json:
        import json
        # Convert numpy types
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            return obj
        print(json.dumps(results, indent=2, default=convert))
    else:
        format_output(results, show_all=True)

if __name__ == '__main__':
    main()
