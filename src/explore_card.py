"""Explore CARD data to understand structure and extract gene-antibiotic mappings."""
import json
import csv
from collections import Counter
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"

def load_card():
    with open(DATA_DIR / "card.json") as f:
        return json.load(f)

def load_aro_index():
    """Load ARO index: maps ARO accession to gene info."""
    aro = {}
    with open(DATA_DIR / "aro_index.tsv") as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            aro[row['ARO Accession']] = row
    return aro

def load_aro_categories():
    """Load ARO categories: maps ARO accession to categories (drug class, mechanism)."""
    cats = {}
    with open(DATA_DIR / "aro_categories.tsv") as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            aro_acc = row['ARO Accession']
            if aro_acc not in cats:
                cats[aro_acc] = []
            cats[aro_acc].append(row)
    return cats

def load_shortname_antibiotics():
    """Load antibiotic short name mapping."""
    abx = {}
    with open(DATA_DIR / "shortname_antibiotics.tsv") as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            abx[row['short_name']] = row['long_name']
    return abx

def main():
    print("=" * 60)
    print("CARD DATABASE ANALYSIS")
    print("=" * 60)

    card = load_card()
    print(f"\nTotal AMR detection models: {len(card)}")

    # Model types
    model_types = Counter()
    for k, v in card.items():
        if isinstance(v, dict):
            model_types[v.get('model_type', 'unknown')] += 1
    print(f"\nModel types:")
    for mt, count in model_types.most_common():
        print(f"  {mt}: {count}")

    # Sample a few models
    print(f"\nSample models:")
    for i, (k, v) in enumerate(card.items()):
        if i >= 3:
            break
        if isinstance(v, dict):
            print(f"  [{k}] {v.get('model_name')} - {v.get('model_type')}")
            seq = v.get('model_sequences', {})
            if isinstance(seq, dict):
                print(f"    Sequences: {len(seq)}")
            elif isinstance(seq, list):
                print(f"    Sequences: {len(seq)} entries")

    # ARO categories
    cats = load_aro_categories()
    print(f"\nARO Categories: {len(cats)} unique ARO accessions")
    
    # Drug class distribution
    drug_classes = Counter()
    mechanisms = Counter()
    for aro_acc, entries in cats.items():
        for e in entries:
            dc = e.get('Drug Class', '')
            if dc:
                drug_classes[dc] += 1
            mech = e.get('Resistance Mechanism', '')
            if mech:
                mechanisms[mech] += 1
    
    print(f"\nTop Drug Classes:")
    for dc, count in drug_classes.most_common(10):
        print(f"  {dc}: {count}")
    
    print(f"\nResistance Mechanisms:")
    for mech, count in mechanisms.most_common():
        print(f"  {mech}: {count}")

    # Map ARO categories to models
    print(f"\nCat entries for first 3 models:")
    for i, (k, v) in enumerate(card.items()):
        if i >= 3:
            break
        aro_acc = v.get('ARO_accession', '')
        if aro_acc in cats:
            for cat in cats[aro_acc]:
                print(f"  [{k}] {v.get('model_name')}: Drug={cat.get('Drug Class','?')}, "
                      f"Mechanism={cat.get('Resistance Mechanism','?')}, "
                      f"Family={cat.get('AMR Gene Family','?')}")

    # Antibiotics
    abx = load_shortname_antibiotics()
    print(f"\nAntibiotic abbreviations: {len(abx)} entries")
    for sn, ln in list(abx.items())[:10]:
        print(f"  {sn} -> {ln}")

if __name__ == '__main__':
    main()
