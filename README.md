# amr-antibiotic-ai

Antimicrobial resistance prediction from bacterial genomic sequences.

## Overview & Purpose
amr-antibiotic-ai trains machine learning models on CARD (Comprehensive Antibiotic Resistance Database) genomic markers to predict antimicrobial resistance profiles from raw bacterial DNA sequences.

## Key Features
- CARD database gene alignment parser.
- Feature extraction from genomic FASTA files.
- Machine learning classification pipeline for resistance prediction.

## Tech Stack & Dependencies
- **Language**: Python 3.9+
- **Bioinformatics**: Biopython
- **Machine Learning**: Scikit-Learn, Pandas

## Project Structure
```text
amr-antibiotic-ai/
├── predict.py
├── train_model.py
├── requirements.txt
└── README.md
```

## Installation & Setup

### Prerequisites
- Python 3.9+

### Steps
```bash
git clone https://github.com/zsomborturcsanyi7-lang/amr-antibiotic-ai.git
cd amr-antibiotic-ai
pip install -r requirements.txt
python predict.py
```

## Usage Examples
```bash
python predict.py --fasta sample.fasta
```

## Status & License
Status: Bioinformatics Research Code.
License: MIT
