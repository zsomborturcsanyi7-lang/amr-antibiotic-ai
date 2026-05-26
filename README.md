# AMR Prediction

**Antimicrobial Resistance Prediction from Protein Sequences**

Machine learning system that predicts antibiotic resistance profiles directly from protein sequences. Built on CARD (Comprehensive Antibiotic Resistance Database) data with dipeptide composition features and XGBoost.

## Performance

| Metric | Value |
|--------|-------|
| Micro F1 | 0.927 |
| Subset Accuracy | 85.6% |
| Hamming Loss | 0.016 |
| Macro F1 | 0.621 |

Per-class AUC for major drug classes:
- Carbapenem: 0.995
- Cephalosporin: 0.997
- Penicillin beta-lactam: 0.991
- Monobactam: 0.995

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Download CARD data
cd data/raw
curl -O https://card.mcmaster.ca/latest/data
tar -xjf card-data.tar.bz2
cd ../..

# Build dataset
python src/build_dataset.py

# Train model
python src/train_baseline.py

# Predict
python src/predict.py MALWMRLLPLLALLALWGPDPAAA...
python src/predict.py resistance_genes.fasta
```

## How It Works

1. **Feature Extraction**: Protein sequences → amino acid composition (20) + dipeptide composition (400) = 420 features
2. **Model**: Multi-label XGBoost (binary relevance), 17 drug classes
3. **Data**: 6,318 resistance genes from CARD 4.0.1

## Project Structure

```
amr-prediction/
├── data/
│   ├── raw/          # CARD data (downloaded)
│   └── processed/    # Built datasets
├── src/
│   ├── build_dataset.py   # Data pipeline
│   ├── explore_card.py    # CARD data exploration
│   ├── train_baseline.py  # Model training
│   └── predict.py         # CLI prediction tool
├── models/           # Trained models (pickle)
├── results/          # Evaluation results
├── requirements.txt
└── README.md
```

## Limitations

- Current model uses protein sequences (not genomes). For genome-level prediction, preprocessing with gene-finding tools (Prodigal) + RGI is needed.
- Train/test split is random; sequence clustering (CD-HIT) needed for true generalization benchmark.
- Classes with < 50 samples have poor recall. Need more data.

## Roadmap

- [x] Protein-level AMR prediction (baseline)
- [ ] Genome-level prediction (k-mer + gene detection)
- [ ] DNA transformer model (Nucleotide Transformer / DNABERT)
- [ ] CD-HIT clustered validation
- [ ] Real AST (Antibiotic Susceptibility Testing) data integration
- [ ] CARB-X grant application

## Data Source

CARD 2023: Alcock et al. "CARD 2023: expanded curation, support for machine learning, and resistome prediction at the Comprehensive Antibiotic Resistance Database" *Nucleic Acids Research*, 51, D690-D699.

## Authors

Turcsanyi Zsombor & Hermes AI

## License

MIT
