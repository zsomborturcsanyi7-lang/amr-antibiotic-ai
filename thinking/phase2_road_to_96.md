# PHASE 2: ÚT A 0.96-HOZ

## Hol tartunk: 0.84 (szintetikus genom, k-mer + XGBoost)

## A 0.96 négy komponense

A rezisztencia NEM egyetlen jelenség. Négy szinten kell detektálni:

| Szint | Mit detektál | Módszer | Eddig? |
|---|---|---|---|
| 1. Gén jelenlét | Ott van-e a CTX-M gén? | k-mer + XGBoost | ✅ 0.84 |
| 2. Gén expresszió | Túl sok efflux pump? | k-mer promoter régió | ⬜ k-mer tanulja |
| 3. Célfehérje mutáció | gyrA S83L → fluorokinolon rezisztens? | Docking | ⬜ ÚJ! |
| 4. Ismeretlen mechanizmus | Teljesen új rezisztencia | DNA Transformer | ⬜ ÚJ! |

---

## FÁZIS 2A: Valós klinikai adatok (FDA-ARGOS)

**Mit csinál:** Letöltjük a 2000+ klinikai izolátumot AST adatokkal.

**Pipeline:**
```
NCBI FTP → genome FASTA → k-mer extraction → XGBoost → validate
```

**Várható Micro F1:** 0.82-0.88

**Idő:** 2-3 óra

---

## FÁZIS 2B: DNA Foundation Model (Kaggle T4 GPU)

**Mit csinál:** Nucleotide Transformer v2 (500M paraméter) — a genomot "értő" transformer modell.

**Miért jobb mint a k-mer:**
- A k-mer elveszíti a sorrendet. Egy CTX-M gén és egy véletlen AT-gazdag régió hasonló k-mer profilú lehet.
- A transformer figyelem (attention) megtalálja a rezisztencia génekre jellemző MOTÍVUMOKAT — promoter szekvenciákat, riboszóma kötőhelyeket, start kodontól való távolságot.
- A modell megtanulja: "ez a 500 bp hosszú szakasz egy CTX-M gén eleje, amit egy promoter előz meg".

**Pipeline:**
```
Genom FASTA → 500 bp ablakokra bontás → NT v2 embedding → mean pooling → LightGBM
```

**Várható Micro F1:** 0.87-0.91 (önmagában)

**Idő:** 4-6 óra Kaggle T4-en

---

## FÁZIS 2C: DOCKING INTEGRÁCIÓ (a titkos fegyverünk)

### A probléma amit a k-mer és transformer NEM lát:

Van egy pontmutáció a gyrA génben (S83L). A DNS szinten ez csak 1 nukleotid változás a 4.6 millióból. A k-mer-ben ez 12 db új k-mer a 800 ezerből. Gyakorlatilag láthatatlan.

DE: ez a mutáció megváltoztatja a fehérje 3D szerkezetét → a fluorokinolon antibiotikum nem tud kötődni → a baktérium rezisztens.

A docking PONTOSAN ezt méri: "mennyire erősen kötődik a gyógyszer a mutáns fehérjéhez?"

### Hogyan csináljuk:

Minden genomból:
1. **Kiválasztjuk a gyógyszer-célfehérjéket** (gyrA, parC, rpoB, folP, PBP-k, stb.)
2. **Kinyerjük a fehérje szekvenciákat** (a genomból prediktálva vagy RGI-vel)
3. **Prediktáljuk a 3D szerkezetet** (ESMFold — gyors GPU-n, 10 mp/fehérje)
4. **Dockingoljuk a gyógyszert** (AutoDock Vina — 5 mp/docking, már megvan a tapasztalat)
5. **Kötődési energia (ΔG)** → numerikus feature a modellnek

### Antibiotikum → Célfehérje párok:

| Antibiotikum | Célfehérje | PDB azonosító |
|---|---|---|
| Ciprofloxacin (fluorokinolon) | gyrA | 2Y3P |
| Rifampicin | rpoB | 5UAC |
| Sulfamethoxazole | folP | 1AJ0 |
| Ampicillin (β-laktám) | PBP2 | 3UDI |
| Meropenem (carbapenem) | PBP3 | 4BJP |
| Tetraciklin | 16S rRNA* | (RNA, nehezebb) |
| Gentamicin (aminoglikozid) | 16S rRNA* | (RNA, nehezebb) |

*: A 16S rRNA célpontokhoz nem docking kell hanem RNS másodlagos szerkezet predikció + metilációs hely detekció. Ezt inkább szekvencia-alapon csináljuk.

### Feature mátrix (2000 genom × 8 gyógyszer):

| Genom | gyrA ΔG (Cipro) | rpoB ΔG (Rifamp) | folP mutációk | PBP ΔG (Ampi) | ... |
|---|---|---|---|---|
| FDA_001 | -8.2 | -9.1 | 0 | -7.8 |
| FDA_002 | -5.1 ⚠️ | -9.0 | 2 ⚠️ | -7.5 |
| FDA_003 | -8.3 | -9.0 | 0 | -4.2 ⚠️ |

Ahol ⚠️ = rezisztencia prediktálva.

### Számítási igény:

2000 genom × 6 fehérje/genom × 1 docking/fehérje = 12 000 docking
5 másodperc/docking (Vina, 5 parallel workerrel, ahogy a drug repurposingnál csináltuk)
= ~3.3 óra a laptopodon

**Várható Micro F1:** 0.88-0.94 (a docking feature-ökkel kombinálva)

---

## FÁZIS 2D: MULTI-MODÁLIS ENSEMBLE

### A teljes pipeline:

```
Genom FASTA
    │
    ├─→ k-mer (5000 dim) ──────────────┐
    ├─→ NT v2 embedding (1280 dim) ─────┤
    ├─→ Docking energiák (6-8 dim) ─────┤
    └─→ RGI gén detekció (bináris) ─────┘
                                          │
                                    ┌─────▼──────┐
                                    │  Ensemble  │
                                    │ (LightGBM) │
                                    └─────┬──────┘
                                          │
                                    Rezisztencia
                                    predikció
```

### Az ensemble architektúra:

1. **Level 1 modellek:** Mind a 4 feature-szetten külön XGBoost/LightGBM
2. **Level 2 meta-learner:** Logistic regression az L1 predikciókon
3. **Calibráció:** Platt scaling a valószínűségekre

### Miért működik ez a kombó:

- **k-mer** fogja a bruttó génjelenlétet és a promoter mintázatokat
- **NT v2** fogja a finom motívumokat és a hosszú távú kontextust
- **Docking** fogja a strukturális mutációkat amiket a szekvencia modellek nem látnak
- **RGI** adja a "biztos" predikciókat (ismert génekre 95%+ pontos)

A három információs csatorna KOMPLEMENTER — amit az egyik nem lát, látja a másik.

**Várható Micro F1: 0.93-0.96**

---

## IMPLEMENTÁCIÓS TERV

### Holnap (2-3 óra):
1. FDA-ARGOS letöltés és pipeline (Fázis 2A) → baseline valós adaton
2. Docking pipeline vázlat (Fázis 2C) → első 50 genomon teszt

### Holnapután (4-6 óra Kaggle):
3. NT v2 fine-tuning (Fázis 2B) → Kaggle T4-en
4. Docking batch futtatás (Fázis 2C) → 2000 genom × 6 fehérje

### Harmadik nap:
5. Ensemble építés (Fázis 2D) → végső modell
6. Whitepaper írás

### Negyedik nap:
7. CARB-X grant application

---

## A CÉL

**0.96 Micro F1 clustered split-en, valós klinikai AST adatokon.**

Ez Nature / Science szintű eredmény.
