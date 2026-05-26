# AMR Prediction — Hogyan érjük el a 0.80 Micro F1-et? (v2)

## Kiindulási állapot

| Módszer | Split | Micro F1 |
|---|---|---|
| Dipeptid + XGBoost | Random (átverés) | 0.93 |
| ESM-2 embedding + XGBoost | Clustered (valós) | **0.42** |

Cél: 0.80.

## Probléma diagnózis

1. **CARD = géncsaládok, nem független minták** — CTX-M-1...CTX-M-200 mind ugyanaz. Clustered split szétválasztja őket.
2. **Egyedi gének, nem teljes genomok** — egy baktériumban 3000-6000 gén, a rezisztencia ezek együttes hatása.
3. **Nincs valós AST adat** — CARD annotációk, nem laborban mért MIC értékek.

## ÚT A 0.80-HOZ (három támadási pont)

### A) Genom-szintű predikció k-merrel + valós AST

Ez a legnagyobb ugrás. A rezisztencia nem egy gén, hanem az egész genom tulajdonsága.

**Adatforrások valós AST-hoz:**
- **BV-BRC genomok** — az antibiogram API visszaad AST-t, de "Computational Method" jelöléssel. Külön kell választani a valós labor adatokat. A BV-BRC-ben van >500000 genom, kb. 5-10%-nak van AST-ja.
- **NCBI Pathogen Detection** — az Isolates Browserben vannak AMR fenotípusok. API nehézkes, de a bulk download működik.
- **FDA-ARGOS** (BioProject PRJNA231221) — ~2000 klinikai izolátum AST adatokkal. Kicsi, de jó minőség.
- **Egyetemi repozitóriumok** — pl. MD Anderson, Wellcome Sanger Institute.

**Megvalósítás:**
1. Genom assembly letöltése (FASTA) + AST fenotípus (MIC vagy S/R)
2. K-mer extraction (8-12 mert használunk, nem 4-et — túl nagy lenne)
3. Feature hashing / dimensionality reduction (Minhash, Random projection)
4. XGBoost / deep learning classifier
5. Clustered validáció faj (species) szinten — a különböző fajok külön clusterbe

**Becsült Micro F1: 0.65-0.75**

### B) ESM-2 fine-tuning AMR feladatra

A jelenlegi ESM-2 embedding FROZEN — csak kivesszük a 320 dimenziós vektort, nem tanítjuk tovább. Ha fine-tuningoljuk az AMR predikciós feladatra:

**Megvalósítás:**
1. ESM-2 (esm2_t33_650M_UR50D — 650M paraméteres, 1280 dimenziós embedding) betöltése
2. Classification head hozzáadása (2 rétegű MLP)
3. Fine-tuning a CARD adatokon a clustered split train részén
4. Eval a test-en

**Miért jobb:**
- A 320 dimenziós embedding helyett 1280 dimenzió
- A modell megtanulja hogy mely aminosav mintázatok relevánsak a rezisztencia szempontjából
- A transformer attention mechanizmusa megtalálja a konzervált régiókat
- A nagyobb modell jobb reprezentációt ad a ritka géncsaládokra is

**Becsült Micro F1: 0.50-0.60**

### C) Kontrasztív tanulás + prototípus hálózat

Ahelyett hogy minden osztályt függetlenül prediktálnánk, használjunk metrikus tanulást:

**Megvalósítás:**
1. ESM-2 embeddingeket vetítsük át egy tanulható MLP-vel
2. Kontrasztív loss: azonos gyógyszerosztályú gének legyenek közel, különbözőek távol
3. Prototípus hálózat: minden gyógyszerosztályhoz számolunk egy prototípus vektort
4. Új szekvencia → legközelebbi prototípus → rezisztencia predikció

**Miért jobb:**
- Jól kezeli a multi-label esetet (egy gén több gyógyszerre is rezisztens lehet)
- A ritka osztályok is kapnak prototípust, nem csak a gyakoriak
- A kontrasztív tanulás erősebb reprezentációt ad mint a binary relevance

**Becsült Micro F1: 0.55-0.65**

---

## KOMBINÁLT STRATÉGIA A 0.80-HOZ

A három módszer kombinálva:

1. **Adat:** Valós AST adatok a BV-BRC-ből + genom assembly-k (A terv)
2. **Feature:** ESM-2 fine-tuningolt embedding + genom k-mer (B terv)
3. **Modell:** Kontrasztív tanulás + prototípus hálózat + XGBoost ensemble (C terv)

**Becsült Micro F1: 0.78-0.85** — itt már benne van a 0.80!

## KÖVETKEZŐ LÉPÉS (azonnal)

1. Valós AST adatok letöltése a BV-BRC-ből
2. Genom assembly-k letöltése
3. Genom k-mer pipeline építése
4. ESM-2 fine-tuning (ha van elég GPU memória — Kaggle T4 16GB)

---

Most gondolkodom tovább: mi a GYENGE pontja ennek a tervnek?

A GYENGE PONT: a BV-BRC AST adatok valódisága. Ha a "Computational Method" jelölésűek, akkor már egy meglévő modell predikcióit használjuk training adatnak → körkörös hiba.

ALTERNATÍV ADATFORRÁS: Mi lenne ha közvetlenül az NCBI SRA-ból töltenénk le izolátumokat amikhez van AST metadata? Vagy használjuk a European Committee on Antimicrobial Susceptibility Testing (EUCAST) adatbázisát?

Tovább gondolkodom...
