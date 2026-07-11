# amr-prediction — Antimicrobial Resistance Prediction

Antimikrobiális rezisztencia predikció protein/genomi szekvenciákból gépi tanulással.

## Funkciók
- CARD adatbázis integráció
- Protein szekvencia elemzés
- AMR gén detektálás
- Rezisztencia profil predikció

## Adatok
- 50+ bakteriális genom
- CARD (Comprehensive Antibiotic Resistance Database)

## Használat
```bash
# AMR predikció futtatása
python src/predict_amr.py --input genome.fasta

# Eredmények
cat results/predictions.csv
```

## Függőségek
- Python 3.8+
- Biopython, scikit-learn
- pandas, numpy
