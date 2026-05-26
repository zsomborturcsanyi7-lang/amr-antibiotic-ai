# AMR Prediction — Hogyan érjük el a 0.80 Micro F1-et? (v3)

## Kiindulás: 0.42 → Cél: 0.80

## FELISMERÉS: A genom-szint elkerülhetetlen

A rezisztencia NEM egyetlen gén tulajdonsága. Hanem:
1. **Gén jelenlét** (pl. CTX-M = cephalosporin rezisztens) — ezt tudjuk most
2. **Célfehérje mutációk** (pl. gyrA mutáció = fluoroquinolone rezisztens)
3. **Promoter mutációk** → gén overexpresszió → rezisztencia
4. **Efflux pump overexpresszió** → multidrug rezisztencia
5. **Kópiaszám variáció** (pl. plazmidon többszörözött gén)

A mi ESM-2 modellünk **csak az 1. pont 50%-át fedi le** (a protein szekvenciát, de a szabályozó régiókat nem).

**A 0.80-hoz genom-szintű feature-ök KELLENEK.**

---

## ÚJ STRATÉGIA: 4 komponensű ensemble

### Komponens 1: ResFinder/RGI output (rule-based)
- ResFinder: ismert rezisztencia gének keresése BLAST/HMM-mel
- RGI (CARD): homology-based AMR gén detekció
- Kimenet: gén jelenlét → gyógyszerosztály predikció (bináris vektor)
- **Micro F1 önmagában: ~0.70** (ez a "gold standard" klinikumban)

### Komponens 2: Genom k-mer + XGBoost (statisztikai)
- 8-12 mert extraction a teljes genom assembly-ből
- Feature hashing (mert 4^10 = 1M+ dimenzió)
- XGBoost classifier
- **Becsült Micro F1: 0.65-0.75**

### Komponens 3: ESM-2 protein embedding + MLP (mély tanulás)
- A mi jelenlegi modellünk, de:
  - Fine-tuningolt ESM-2 (nem frozen)
  - MLP classification head (nem XGBoost)
  - Nagyobb modell: esm2_t33_650M (1280 dimenzió)
- **Becsült Micro F1: 0.50-0.60**

### Komponens 4: Szintetikus genom data augmentation
- Alap genom (E. coli K-12) + véletlenszerűen beszúrt rezisztencia gének
- A modell megtanulja detektálni a géneket genom kontextusban
- **Becsült Micro F1: 0.55-0.65**

### Meta-ensemble
- A 4 komponens predikcióinak súlyozott kombinációja
- Logistic regression mint meta-learner
- **Cél Micro F1: 0.78-0.85**

---

## DE! Hol szerezünk valós AST adatokat?

### Elsődleges forrás: FDA-ARGOS (BioProject PRJNA231221)
- 2000+ klinikai izolátum
- AST (MIC) adatok több antibiotikumra
- Genom assembly-k elérhetők
- Ingyenes, publikus

### Másodlagos forrás: BV-BRC genomok valós AST-vel
- A BV-BRC-ben az "evidence" mező lehet "Laboratory", "AST", "Experimental" — nem csak "Computational Method"
- Ki kell szűrni a valós méréseket
- Az antibiogram API megfelelő szűrésével

### Harmadlagos: Szintetikus adat
- A CARD gének + alap genom = végtelen mennyiségű training adat
- Gyengeség: nem valós, de jó pretrainingnek

---

## KONKRÉT MEGVALÓSÍTÁSI TERV

### 1. lépés (most): FDA-ARGOS letöltése
- 2000 genom + AST → azonnali baseline
- Ha itt elérjük a 0.70-et, akkor jó úton vagyunk

### 2. lépés: Genom k-mer pipeline
- Genom assembly → k-mer extraction → feature hashing → XGBoost
- Ez a BV-BRC által is használt módszer
- Ha W1 score > 0.85, akkor első komponens kész

### 3. lépés: ESM-2 fine-tuning Kaggle-en
- T4 GPU (16 GB) → esm2_t33_650M befér
- Fine-tuning a CARD adatokon (clustered split train része)
- MLP classification head

### 4. lépés: Ensemble
- 3-4 modell predikciójának kombinálása
- Logistic regression meta-learner
- Calibrated probabilities

---

## MIÉRT ÉRHETJÜK EL A 0.80-at?

1. **A ResFinder/RGI már most 0.70 körül van** a klinikai használatban (ismert génekre)
2. **A genom k-mer felfogja az ismeretlen mintázatokat** amiket a rule-based eszközök nem
3. **Az ESM-2 a távoli homológokat is detektálja** amiket a BLAST nem
4. **Az ensemble kombinálja a három megközelítés erősségeit**
5. **A szintetikus adat pretraininggel** a modell látott elég variációt

---

Tovább gondolkodva: van-e GYENGE PONT?

**GYENGE PONT: A ResFinder/RGI nem API-kompatibilis a pipeline-unkkal**
- A ResFinder egy webes eszköz + CLI
- Az RGI (CARD tool) Pythonban van, de 6 GB adatbázis kell hozzá
- A Docker konténerben futnak
- Megoldás: mi implementáljuk a legegyszerűbb detekciót (BLAST a CARD FASTA ellen)

Tovább iterálok...
