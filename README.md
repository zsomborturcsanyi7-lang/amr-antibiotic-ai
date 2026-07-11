# amr-prediction — Antimicrobial Resistance Prediction

Antimicrobial resistance prediction from protein/genomic sequences using machine learning.

## Features
- CARD database integration
- Protein sequence analysis
- AMR gene detection
- Resistance profile prediction

## Data
- 50+ bacterial genomes
- CARD (Comprehensive Antibiotic Resistance Database)

## Usage
```bash
# Run AMR prediction
python src/predict_amr.py --input genome.fasta

# Results
cat results/predictions.csv
```

## Dependencies
- Python 3.8+
- Biopython, scikit-learn
- pandas, numpy
