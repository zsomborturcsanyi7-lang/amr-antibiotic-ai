"""Build AMR dataset from CARD data.

Extracts:
1. Gene sequences (nucleotide + protein) from CARD FASTA files
2. Gene -> Drug Class mappings from aro_index.tsv
3. Gene -> Resistance Mechanism mappings

Then builds a labeled dataset: sequence -> list of antibiotic classes resisted.
"""
import json
import csv
from collections import Counter, defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

def load_aro_index():
    """Load aro_index.tsv: maps Protein Accession -> Drug Class, Resistance Mechanism."""
    mapping = defaultdict(lambda: {"drug_classes": set(), "mechanisms": set(), "gene_family": set(), "aro_name": ""})
    with open(DATA_DIR / "aro_index.tsv") as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            prot_acc = row.get('Protein Accession', '').strip()
            if not prot_acc:
                continue
            dc = row.get('Drug Class', '').strip()
            mech = row.get('Resistance Mechanism', '').strip()
            family = row.get('AMR Gene Family', '').strip()
            aro_name = row.get('ARO Name', '').strip()
            
            entry = mapping[prot_acc]
            entry["aro_name"] = aro_name  # last one wins
            if dc:
                for d in dc.split(';'):
                    entry["drug_classes"].add(d.strip())
            if mech:
                entry["mechanisms"].add(mech.strip())
            if family:
                entry["gene_family"].add(family.strip())
    return mapping

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

def extract_protein_id(header):
    """Extract protein accession from FASTA header.
    
    Headers look like:
    >gb|AAB60941.1|+|0-733|ARO:3005099|Erm(A)
    """
    parts = header.split('|')
    if len(parts) >= 2:
        return parts[1]  # e.g. AAB60941.1
    return header.split()[0]

def main():
    print("Building AMR Dataset")
    print("=" * 60)
    
    # 1. Load gene->drug mapping
    print("\n[1/4] Loading ARO index...")
    aro_index = load_aro_index()
    print(f"  {len(aro_index)} unique protein accessions mapped")
    
    # Show some examples
    for prot, info in list(aro_index.items())[:5]:
        print(f"  {prot}: {info['aro_name']} -> {info['drug_classes']}")
    
    # 2. Parse FASTA files
    print("\n[2/4] Parsing FASTA files...")
    fasta_files = {
        'protein_homolog': 'protein_fasta_protein_homolog_model.fasta',
        'protein_variant': 'protein_fasta_protein_variant_model.fasta',
        'protein_knockout': 'protein_fasta_protein_knockout_model.fasta',
        'protein_overexpression': 'protein_fasta_protein_overexpression_model.fasta',
    }
    
    all_sequences = []
    for model_type, filename in fasta_files.items():
        path = DATA_DIR / filename
        if not path.exists():
            print(f"  SKIP {filename} - not found")
            continue
        seqs = parse_fasta(path)
        print(f"  {filename}: {len(seqs)} sequences")
        for header, seq in seqs[:5]:  # first 5
            prot_id = extract_protein_id(header)
            if prot_id in aro_index:
                info = aro_index[prot_id]
                print(f"    {prot_id}: {info['aro_name'][:50]} -> {info['drug_classes']}")
        all_sequences.extend([(model_type, h, s) for h, s in seqs])
    
    print(f"\n  Total sequences: {len(all_sequences)}")
    
    # 3. Build labeled dataset
    print("\n[3/4] Building labeled dataset...")
    
    drug_classes_all = Counter()
    mechanism_all = Counter()
    labeled = []
    unlabeled = []
    
    for model_type, header, seq in all_sequences:
        prot_id = extract_protein_id(header)
        if prot_id in aro_index:
            info = aro_index[prot_id]
            labeled.append({
                'protein_id': prot_id,
                'aro_name': info['aro_name'],
                'drug_classes': list(info['drug_classes']),
                'mechanisms': list(info['mechanisms']),
                'gene_families': list(info['gene_family']),
                'model_type': model_type,
                'sequence': seq,
                'length': len(seq)
            })
            for dc in info['drug_classes']:
                drug_classes_all[dc] += 1
            for m in info['mechanisms']:
                mechanism_all[m] += 1
        else:
            unlabeled.append((prot_id, model_type, header))
    
    print(f"  Labeled: {len(labeled)}")
    print(f"  Unlabeled: {len(unlabeled)}")
    
    print(f"\n  Drug class distribution:")
    for dc, count in drug_classes_all.most_common(15):
        print(f"    {dc}: {count}")
    
    print(f"\n  Resistance mechanisms:")
    for m, count in mechanism_all.most_common(10):
        print(f"    {m}: {count}")
    
    # 4. Save processed data
    print(f"\n[4/4] Saving processed data...")
    PROCESSED_DIR.mkdir(exist_ok=True)
    
    # Save as simplified CSV for ML
    with open(PROCESSED_DIR / 'amr_sequences.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['protein_id', 'aro_name', 'drug_classes', 'mechanisms', 
                         'gene_families', 'model_type', 'sequence', 'length'])
        for item in labeled:
            writer.writerow([
                item['protein_id'],
                item['aro_name'],
                ';'.join(item['drug_classes']),
                ';'.join(item['mechanisms']),
                ';'.join(item['gene_families']),
                item['model_type'],
                item['sequence'],
                item['length']
            ])
    
    print(f"  Saved {len(labeled)} sequences to {PROCESSED_DIR / 'amr_sequences.csv'}")
    
    # Save drug class list
    with open(PROCESSED_DIR / 'drug_classes.txt', 'w') as f:
        for dc, count in drug_classes_all.most_common():
            f.write(f"{dc}\t{count}\n")
    print(f"  Saved {len(drug_classes_all)} drug classes")

if __name__ == '__main__':
    main()
