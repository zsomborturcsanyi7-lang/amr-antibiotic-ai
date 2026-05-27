#!/usr/bin/env python3
"""Download AMR bacterial genomes via NCBI E-utilities (reliable, tested)."""
import urllib.request, urllib.parse, json, time
from pathlib import Path

OUT_DIR = Path("data/raw/fda_argos/genomes")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPECIES = [
    "Escherichia coli",
    "Klebsiella pneumoniae",
    "Pseudomonas aeruginosa",
    "Acinetobacter baumannii",
    "Staphylococcus aureus",
    "Enterococcus faecium",
    "Enterococcus faecalis",
    "Salmonella enterica",
    "Streptococcus pneumoniae",
    "Enterobacter cloacae",
]

PER_SPECIES = 5  # genomes per species

total = 0
for species in SPECIES:
    safe = species.replace(" ", "_")
    species_dir = OUT_DIR / safe
    species_dir.mkdir(exist_ok=True)
    
    existing = list(species_dir.glob("*.fna.gz"))
    if len(existing) >= PER_SPECIES:
        print(f"{species}: {len(existing)} already downloaded", flush=True)
        total += len(existing)
        continue
    
    # Search NCBI Assembly
    query = f'"{species}"[Organism] AND "complete genome"[Assembly Level] AND "latest refseq"[filter]'
    params = urllib.parse.urlencode({
        'db': 'assembly', 'term': query, 'retmax': PER_SPECIES,
        'retmode': 'json', 'sort': 'relevance',
    })
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?{params}"
    
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            ids = json.loads(r.read())['esearchresult']['idlist']
    except Exception as e:
        print(f"{species}: search failed: {e}", flush=True)
        continue
    
    if not ids:
        print(f"{species}: no complete genomes found", flush=True)
        continue
    
    print(f"{species}: {len(ids)} found, downloading...", flush=True)
    time.sleep(0.4)
    
    for uid in ids[:PER_SPECIES]:
        # Get FTP path via esummary
        sum_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=assembly&id={uid}&retmode=json"
        try:
            with urllib.request.urlopen(sum_url, timeout=30) as r:
                info = json.loads(r.read())['result'][uid]
        except Exception as e:
            print(f"  [{uid}] summary failed: {e}", flush=True)
            continue
        
        ftp = info.get('ftppath_refseq') or info.get('ftppath_genbank')
        if not ftp:
            continue
        
        https_base = ftp.replace('ftp://', 'https://')
        acc = ftp.rstrip('/').split('/')[-1]
        fna_url = f"{https_base}/{acc}_genomic.fna.gz"
        out_file = species_dir / f"{acc}.fna.gz"
        
        if out_file.exists():
            print(f"  {acc}: exists", flush=True)
            total += 1
            continue
        
        try:
            urllib.request.urlretrieve(fna_url, out_file)
            mb = out_file.stat().st_size / 1e6
            print(f"  {acc}: {mb:.1f} MB", flush=True)
            total += 1
        except Exception as e:
            print(f"  {acc}: FAIL ({e})", flush=True)
        
        time.sleep(0.5)
    
    time.sleep(0.4)

print(f"\nDone. {total} genomes in {OUT_DIR}", flush=True)
