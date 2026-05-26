# AMR Prediction — Hogyan érjük el a 0.80-at? (v4 — VÉGSŐ)

## A FELISMERÉS: Nem módszer, hanem ADAT probléma

A 0.42 → 0.80 út 80%-ban ADAT kérdés, 20%-ban MÓDSZER.

| Ami van most | Ami kell |
|---|---|
| 6236 rezisztencia GÉN (CARD) | 2000+ teljes GENOM + AST |
| CARD annotációk (számított) | Labor MIC mérések (mért) |
| Protein szekvencia | Teljes genom assembly |
| Egy gén → rezisztencia | Teljes genom → rezisztencia |

**A BV-BRC ezt a módszert használja (genom k-mer + XGBoost) és 0.61-0.97 W1 score-t ér el.** Ha mi ugyanezt csináljuk valós AST adatokkal, elérjük a 0.80-at.

---

## A LEHETSÉGES LEGEGYSZERŰBB ÚT A 0.80-HOZ

### 1. Valós genom + AST adat (FDA-ARGOS)
```
Forrás: NCBI BioProject PRJNA231221
Tartalom: ~2000 klinikai izolátum, több faj
AST adatok: MIC értékek 10-20 antibiotikumra
Genomok: assembly-k (FASTA), letölthetők
```

### 2. Genom k-mer pipeline
```
Genom FASTA → 10-mer extraction → feature hashing → XGBoost
                                    (100K dim → 10K dim)
```

### 3. ESM-2 fine-tuning Kaggle T4-en (opcionális, +5-10 pp)
```
esm2_t33_650M (1280 dim) → fine-tuning → MLP head
```

### 4. Ensemble a kettőből
```
k-mer XGBoost predikciók + ESM-2 predikciók → logistic regression
```

---

## MIÉRT MŰKÖDIK EZ?

A genom k-mer felfog MINDENT:
- ✅ Gén jelenlét/hiány (a rezisztencia gén szekvenciája benne van a k-mer-ben)
- ✅ Mutációk (SNP-k megjelennek mint új k-mer-ek)
- ✅ Promoter régiók (upstream szekvencia is benne van)
- ✅ Kópiaszám (többször előforduló k-mer-ek)
- ✅ Plazmidok (a teljes assembly tartalmazza)
- ✅ Faj-specifikus mintázatok

A k-mer + XGBoost kombó a BV-BRC-nél 0.61-0.97 között teljesít. A mi hozzáadott értékünk:
- ESM-2 embedding a génekhez (nem csak nyers k-mer)
- Jobb validáció (clustered split faj szerint)
- Kontrasztív tanulás a ritka osztályokhoz (opcionális)

---

## KONKRÉT KÖVETKEZŐ LÉPÉS

1. **FDA-ARGOS adatok letöltése** (SRA toolkit vagy közvetlen FTP)
   - Idő: ~1 óra (letöltés + feldolgozás)
   - Eredmény: 2000 genom + AST CSV

2. **K-mer pipeline megépítése**
   - Idő: ~30 perc
   - Eredmény: X_train (2000 x 10K), y_train (2000 x N_antibiotics)

3. **Első baseline futtatás**
   - Idő: ~10 perc
   - Várható: Micro F1 = 0.60-0.75

4. **Iteratív javítás**
   - ESM-2 embedding hozzáadása
   - Hyperparameter tuning
   - Ensemble

---

## A CÉL?

**0.80 Micro F1 clustered split-en, valós klinikai adatokon.**

Ez egy Nature Biotechnology / Lancet Digital Health szintű eredmény.

---

## VÉGSŐ GONDOLAT

A jelenlegi 0.42-es eredményünk egy GÉN-szintű modell. Ha ezt egy az egyben próbáljuk 0.80-ra húzni, lehetetlen — mert a biológiai valóság ennél összetettebb. A rezisztencia emergens tulajdonsága a teljes genomnak, nem egyetlen génnek.

**A helyes következő lépés: hagyjuk el a gén-szintű predikciót, és térjünk át genom-szintre.**
