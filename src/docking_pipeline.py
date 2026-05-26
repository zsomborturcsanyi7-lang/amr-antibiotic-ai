"""Docking Integration Pipeline for AMR Prediction.

For each bacterial genome + antibiotic pair:
1. Extract target protein sequence (e.g., gyrA for fluoroquinolones)
2. Map known mutations or use the full sequence
3. Dock the antibiotic to the target protein
4. Compute binding energy (ΔG) as a feature

This captures structural resistance mechanisms that sequence-based
models cannot see (point mutations that disrupt drug binding).

Requires: AutoDock Vina (vina.exe), OpenBabel (obabel.exe)
"""
import os
import csv
import subprocess
import tempfile
from pathlib import Path
from collections import defaultdict
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================

# Path to tools (adjust for your system)
VINA_EXE = "vina"  # or full path to vina.exe on Windows
OBABEL_EXE = "obabel"  # or full path to obabel.exe

# Antibiotic → Target Protein mappings
# Format: (gene_name, reference_pdb, drug_smiles_or_pdbqt)
DRUG_TARGET_MAP = {
    'ciprofloxacin': {
        'gene': 'gyrA',
        'pdb': '2Y3P',  # E. coli gyrA + ciprofloxacin
        'drug_smiles': 'C1CC1C(=O)NC2=C(C=C3C(=O)C(=CN(C4=C3C(=O)C(=O)C(=C4)N5CC5)F)C(=O)O)F',
        'active_site_center': (20, 30, 40),  # approximate, will auto-detect
    },
    'levofloxacin': {
        'gene': 'gyrA',
        'pdb': '2Y3P',
        'drug_smiles': 'CC1COC2=C3N1C=C(C(=O)C3=CC(=C2N4CCN(CC4)C)F)C(=O)O',
    },
    'rifampicin': {
        'gene': 'rpoB',
        'pdb': '5UAC',
        'drug_smiles': 'CC1C=CC=C(C(=O)NC2=C(C3=C(C(=C2O)C)OC(C4=C3C(=O)C(=O)C5=C4C(=O)C=C(C5(O6)C)C)O)C)C',
    },
    'meropenem': {
        'gene': 'ftsI',  # PBP3
        'pdb': '4BJP',
        'drug_smiles': 'CC1C2C(C(=O)N2C(=C1SC3CC(NC3)C(=O)N(C)C)C(=O)O)C(O)C',
    },
    'ampicillin': {
        'gene': 'mrcA',  # PBP2
        'pdb': '3UDI',
        'drug_smiles': 'CC1(C(N2C(S1)C(C2=O)NC(=O)C(C3=CC=CC=C3)N)C(=O)O)C',
    },
    'sulfamethoxazole': {
        'gene': 'folP',
        'pdb': '1AJ0',
        'drug_smiles': 'CC1=CC(=NO1)NS(=O)(=O)C2=CC=C(C=C2)N',
    },
    'gentamicin': {
        'gene': 'aac_genes',  # Aminoglycoside-modifying enzymes
        'method': 'gene_presence',  # Not docking, gene presence based
    },
    'tetracycline': {
        'gene': 'tet_resistance_genes',  # tetA, tetM, etc.
        'method': 'gene_presence',
    },
}

# Reference protein sequences (from E. coli K-12, used for alignment)
REFERENCE_SEQUENCES = {
    'gyrA': 'MSDLAREITPVNIEEELKSSYLDYAMSVIVGRALPDVRDGLKPVHRRILYAMNVLGNDWNKAYKKSARVVGDVIGKYHPHGDTAVYDTIVRMAQPFSLRYMLVDGQGNFGSIDGDSAAAMRYTEIRMAKIAHELMADLEKETVDFVDNYDGTERIPDVMPTKIPNLLVNGSSGIAVGMATNIPPHNLTEVINGCLAYIDDEDISIEGLMEHIPGPDFPTAAIINGRRGIEEAYRTGRGKVYIRARAEVEVDAKTGRETIIVHEIPYQVNKARLIEKIAELVKEKRVEGISALRDESDKDGMRIVIEVKRDAVGEVVLNNLYSQTQLQVSFGINMVALHHGQPKIMNLKDIIAAFVRHRREVVTRRTIFELRKARDRAHILEALAVALANIDPIIELIRHAPTPAEAKTALVANPWQLGNVAAMLERAGDDAARPEWLEPEFGVRDGLYYLTEQQAQAILDLRLQKLTGLEHEKLLDEYKELLEQIAELLHILGSADRLMEVIREELEAVREQFGDARRTEITANSADINIEDLINQEDVVVTLSHQGYVKYQPLSEYEAQRRGGKGKSAARIKEEDFIDRLLVANTHDTILCFSSRGRLYWMKVYQLPEASRGARGRPIVNLLPLEQDERITAILPVREFEEGVKVFMATANGTVKKTVLTEFNRLRTAGKVAIKLDDGDELIGVDLTSGEDEILFSAEGKVVRFKESSVRAMGCNTTGVRGIRLGEGDKVVSLIVPRGDGAILTVTQNGYGKRTAAAEYPTKSRATKGVISIKVTERNGLVVGAVQVDDCDQIMMITDAGTLVRTRVSEISIVGRNTQGVILIRTAEDENVVGLQRVAEPVDEEDLDTIDGSAAEGDDEIAPEVDVDDEPEEE',
    'rpoB': 'MVYSVTEKKRIRKDFGKRPHVPTLLGGLQEHRSIGLASNLALERFGYGEVIEALREIKRAIDIEAEYINLFNDLESEVTAAKSDGGSLYLDSDQSLILAGGEETLFDGGTPLDIYNSEYRLDIHATRVSVGDRVVVTVRERVIVQNRLLRQRISDLKMNEERDLLKELVYRRQLGEPFDKTLANKLKEGRIGVSDDLFNQAVYSHIATVAAEPVVIADGSEMDVYLILSVEDGHVVTRRSPITIRIVDDDDDDLEEAVRGNVVLRNLTMYQTKVTYPDTIIGDSHAGVIVSRTAVEGLRDVCDILGRHVIVVLDKAYQEDEYSQLVESLSYPVIEGGIVDIIDFDNIERLRDVILESRFGGKYALCNVGELVYRSNDSLVKLRLLDARLRGSFVINLAQLGVDVDNVPVEAGVKEVKDYLLQHGATPKDILFVPKDERISRVRAELEDKLAEIGVNRVSTQVDDIISNLFLSDALAADSSGSASVDIDSHINRFLQIASKRMDTDAALTRSIKLKELRNDLFGALLEQIEHSAKDITIAQVVAAVSSMSSSIVSDAEKVRAQIKQQLDHLTDHIRLRNKLGDLASRLEPIGGNNIETCARILDELFSHNQKNLENLAQELRERITAKLSELERLGESRFLPQLNEQLARVREDLRAALIDKQVIQLDQHSIHEDLRNNVAAILADRLGVGKEAFARLFSKLWDMLEGSASELMDIRDKKFKPTENALKLVSDGIELRRVQYDDEVKSYFETQKGRPIWLPAWVDAGDEALVEALLMHGIADNPQDIQKILQTLKGAVAIEYDGLVEGGLRFQHRYALTGNVLEGDTGVHAISSGAQFLSMKMLARVSRIARELQSALDEDNPMGRKVRSLAGSLGDTQVIPNGAETPLAERERGNILEFVDLVRGVIRVALDGRVVTVGEERHILADGDLIKTSVTDLVRFLNLKEQLDMIEKDKGERISTSEILDEVIDEIISELEKVLSDEKLRQRIHDLGDRIRKIRALVQAEEELDIEEEEIDEYLDAIEDELRIDVNDDESGSDGKPATVALSTEDNVDIEGDISDESDSMGSVDGIDLNPEDLAVIDEEDEFEENA'
}

def find_target_gene(genome_sequences, gene_name):
    """Find target gene in genome by searching for the protein sequence.
    
    In practice, you'd use Prodigal to predict genes, then BLAST/HMM
    to find the specific target gene. For now, we do a simple search.
    """
    # This is a stub - you'd use Prodigal + BLAST in production
    # For now, return the reference sequence (will be replaced with actual)
    return REFERENCE_SEQUENCES.get(gene_name, "")

def run_docking(protein_seq, drug_smiles, protein_name, drug_name, 
                output_dir=None):
    """Run AutoDock Vina docking for a protein + drug pair.
    
    Args:
        protein_seq: Amino acid sequence of the target protein
        drug_smiles: SMILES string of the antibiotic
        protein_name: Name for output files
        drug_name: Name for output files
        output_dir: Directory for output files
    
    Returns:
        dict with 'binding_energy' (kcal/mol) and 'rmsd'
    """
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp())
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    protein_pdb = output_dir / f"{protein_name}.pdb"
    drug_pdbqt = output_dir / f"{drug_name}.pdbqt"
    protein_pdbqt = output_dir / f"{protein_name}.pdbqt"
    output_pdbqt = output_dir / f"{protein_name}_{drug_name}_out.pdbqt"
    log_file = output_dir / f"{protein_name}_{drug_name}_log.txt"
    
    # Step 1: Convert SMILES to 3D PDBQT
    print(f"  Converting {drug_name} (SMILES) to 3D...")
    cmd = [
        OBABEL_EXE, f"-:{drug_smiles}", "-O", str(drug_pdbqt),
        "--gen3d", "--partialcharge", "gasteiger", "-h"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    WARNING: obabel failed for {drug_name}: {result.stderr[:200]}")
        return {'binding_energy': 0.0, 'error': 'obabel_failed'}
    
    # Step 2: Convert protein sequence to PDBQT
    # For real use: ESMFold predicted structure → PDB → PDBQT
    # For now: use a pre-computed reference structure
    print(f"  Preparing {protein_name} structure...")
    
    # Simplified: write sequence to FASTA, then use obabel
    fasta_file = output_dir / f"{protein_name}.fasta"
    with open(fasta_file, 'w') as f:
        f.write(f">{protein_name}\n{protein_seq}\n")
    
    # Note: Full pipeline would use ESMFold for 3D structure prediction
    # For now, we use the reference PDB from CARD/drug repurposing project
    # This is a placeholder - replace with actual structure
    
    # For demonstration: return mock binding energy
    # In production, this would run actual Vina docking
    return {
        'binding_energy': -8.5,  # kcal/mol (mock)
        'protein_name': protein_name,
        'drug_name': drug_name,
        'status': 'mock',
        'note': 'Replace with actual Vina docking using ESMFold structures'
    }

def extract_docking_features(genome_path, antibiotics, output_dir=None):
    """Extract docking-based features for a single genome.
    
    Args:
        genome_path: Path to genome FASTA file
        antibiotics: List of antibiotic names to test
        output_dir: Directory for docking outputs
    
    Returns:
        dict mapping antibiotic -> binding energy (kcal/mol)
    """
    # Load genome
    with open(genome_path) as f:
        genome = ''.join(l.strip() for l in f if not l.startswith('>'))
    
    features = {}
    
    for abx in antibiotics:
        if abx not in DRUG_TARGET_MAP:
            features[abx] = 0.0
            continue
        
        target_info = DRUG_TARGET_MAP[abx]
        
        # For gene-presence based antibiotics
        if target_info.get('method') == 'gene_presence':
            features[f"{abx}_gene"] = -1.0  # placeholder
            continue
        
        # Find target gene in genome
        protein_seq = find_target_gene(genome, target_info['gene'])
        
        if not protein_seq:
            features[f"{abx}_dG"] = 0.0
            continue
        
        # Run docking
        result = run_docking(
            protein_seq,
            target_info['drug_smiles'],
            f"{target_info['gene']}_{Path(genome_path).stem}",
            abx,
            output_dir
        )
        
        features[f"{abx}_dG"] = result.get('binding_energy', 0.0)
    
    return features

def build_docking_matrix(genome_list, antibiotics, output_dir, parallel=5):
    """Build docking feature matrix for multiple genomes.
    
    This is the main entry point for batch processing.
    
    Args:
        genome_list: List of paths to genome FASTA files
        antibiotics: List of antibiotic names
        output_dir: Directory for all outputs
        parallel: Number of parallel Vina workers
    
    Returns:
        X (n_genomes x n_features) numpy array
    """
    from concurrent.futures import ProcessPoolExecutor
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"Docking {len(genome_list)} genomes × {len(antibiotics)} antibiotics...")
    print(f"Total: {len(genome_list) * len(antibiotics)} docking runs")
    print(f"Parallel workers: {parallel}")
    
    # For now: sequential (can parallelize later)
    all_features = []
    for i, genome_path in enumerate(genome_list):
        if i % 100 == 0:
            print(f"  {i}/{len(genome_list)} genomes processed...")
        
        genome_dir = output_dir / Path(genome_path).stem
        features = extract_docking_features(genome_path, antibiotics, genome_dir)
        all_features.append(features)
    
    # Build matrix
    feature_names = []
    for abx in antibiotics:
        feature_names.append(f"{abx}_dG")
    
    X = np.zeros((len(genome_list), len(feature_names)))
    for i, feat in enumerate(all_features):
        for j, name in enumerate(feature_names):
            X[i, j] = feat.get(name, 0.0)
    
    return X, feature_names

if __name__ == '__main__':
    print("Docking Pipeline Ready")
    print("=" * 40)
    print(f"Supported antibiotics: {list(DRUG_TARGET_MAP.keys())}")
    print(f"Reference proteins: {list(REFERENCE_SEQUENCES.keys())}")
    print()
    print("Next steps:")
    print("1. Install ESMFold for protein structure prediction")
    print("2. Run Prodigal on genomes to find target genes")
    print("3. Run docking with AutoDock Vina")
    print("4. Build feature matrix")
