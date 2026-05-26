# AMR Prediction — VÉGLEGES TERV (v5)

## Cél: 0.80 Micro F1 clustered split-en

## A diagnózis

A 0.42 → 0.80 út NEM módszertani, hanem ADAT probléma.

| Amink van | Ami kell | Miért |
|---|---|---|
| 6236 gén (CARD) | 2000+ genom | A rezisztencia a teljes genom tulajdonsága |
| CARD annotációk | Labor AST (MIC) | A valós fenotípus nem azonos az annotációval |
| Protein szekvencia | Teljes genom assembly | Mutációk, promoterek, kópiaszám — mind a genomban vannak |

## A TERV

### Fázis 1: FDA-ARGOS adatok (közvetlenül elérhető)

**Forrás:** NCBI BioProject PRJNA231221
**Tartalom:** ~2000 klinikai izolátum (E. coli, K. pneumoniae, S. aureus, stb.)
**AST:** MIC értékek (mg/L) 10-20 antibiotikumra
**Genomok:** Assembly-k FASTA formátumban

**Letöltés:**
```
# NCBI Datasets CLI-vel:
datasets download genome accession --inputfile acc_list.txt

# Vagy közvetlen FTP:
ftp://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/...
```

**Feldolgozás:**
1. Genom assembly-k → 10-mer extraction
2. Feature hashing: MurmurHash a 10-mer-ekre → 10,000 dimenziós vektor
3. AST adatok → S/R bináris label (EUCAST breakpointok alapján)
4. Clustered split: faj (species) alapján — különböző fajok külön clusterbe

**Első baseline:** XGBoost a k-mer mátrixon
**Várható Micro F1:** 0.60-0.75 (a BV-BRC hasonló modelljei ezt érik el)

### Fázis 2: ESM-2 genome embedding (Kaggle T4 GPU)

**Megközelítés:**
1. Prodigal: gén predikció a genomokban
2. Fordítás: nukleotid → aminosav szekvencia
3. ESM-2 (esm2_t33_650M, 1280 dim): embedding minden prediktált fehérjére
4. Mean pooling: genom-szintű embedding (1280 dim)
5. Ez a feature a k-mer-ek MELLETT

**Várható javulás:** +5-10 pp Micro F1

### Fázis 3: Ensemble

1. Genom k-mer XGBoost → predikciók (valószínűségek)
2. ESM-2 genome embedding XGBoost → predikciók
3. Logistic regression meta-learner → végső predikció

**Várható Micro F1:** 0.78-0.85

---

## MI A TITOK?

**A genom k-mer mindent lát.** Egy 10 mert hosszú szekvencia univerzális jel — felfogja:
- A rezisztencia gének jelenlétét (a gén szekvenciája benne van)
- A pontmutációkat (SNP → új k-mer)
- A promoter régiók variációit (overexpresszió → több k-mer találat)
- A plazmid kópiaszámot (több azonos k-mer)
- Faj-specifikus háttérjellemzőket

Ezért működik a BV-BRC XGBoost modellje 0.61-0.97 W1 score-al. Mi ugyanezt csináljuk + ESM-2 embeddinget adunk hozzá.

---

## IMPLEMENTÁCIÓS SORREND

1. **Most:** FDA-ARGOS letöltés → első baseline (XGBoost k-mer-en) → várt: 0.65-0.75
2. **Ha < 0.70:** Hyperparameter tuning (Optuna) → várt: +3-5 pp
3. **Következő:** ESM-2 genome embedding Kaggle-en → várt: +5-10 pp
4. **Végül:** Ensemble → várt: +3-5 pp

**Teljes várható Micro F1: 0.76-0.85**

A 0.80 elérhető.
