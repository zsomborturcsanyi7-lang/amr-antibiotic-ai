# MEKKORA AZ ABSZOLÚT MAXIMUM?

## Honnan indulunk: 0.42

## A plafon keresése — 6 szint

---

### SZINT 1: Genom k-mer + XGBoost (BV-BRC szint)

A BV-BRC ezt használja. W1 score: 0.61-0.97 antibiotikumtól függően.

**Elméleti plafon:** 0.85  
**Mit limitál:** A k-mer elveszíti a hosszú távú kontextust. Egy 10 mert nem tudja hogy két gén együtt ad-e rezisztenciát (epistasis).

---

### SZINT 2: DNA foundation model (Nucleotide Transformer / HyenaDNA / Evo 2)

Transformer ami közvetlenül a DNS szekvencián tanul:
- Nucleotide Transformer v2: 500M paraméter, 650B nukleotidon tanítva
- HyenaDNA: 1M bp kontextus ablak — teljes plazmidok, operonok beférnek
- Evo 2 (Arc Institute): 40B paraméter, 9.3T nukleotidon tanítva

Ezek a modellek "értik" a genomot — promoter régiókat, operon szerkezetet, kromoszómális vs plazmid kontextust.

**Elméleti plafon:** 0.90  
**Mit limitál:** 1M bp kontextus is kevés egy teljes 5M bp bakteriális genomhoz. Kell a hierarchical pooling.

---

### SZINT 3: Multi-modális ensemble

Kombinálva:
- DNA foundation model embedding (genom-szintű)
- ESM-2 protein embedding (mean pooling a genom összes fehérjéjére)
- Genom k-mer statisztikai jellemzők
- RGI/ResFinder output mint bináris feature-ök (melyik ismert gének vannak jelen)
- Célfehérje mutációk (SNP-k a gyrA, parC, stb.-ben)

**Elméleti plafon:** 0.92  
**Mit limitál:** A modellek közötti redundancia nehezen kezelhető. A multi-modal fusion művészet, nem tudomány.

---

### SZINT 4: Self-supervised pretraining + kontrasztív tanulás

1. Vedd az ÖSSZES elérhető bakteriális genomot (NCBI: >2M genom assembly)
2. Kontrasztív pretraining: azonos faj → közel, különböző faj → távol
3. Masked language modeling: tanuld meg prediktálni a kitakart nukleotidokat
4. Fine-tuning az AMR feladatra (a kevés címkézett adaton)

A pretrainingből a modell megtanulja a bakteriális genomok "nyelvtanát":
- Hol vannak a gének (start/stop kodon mintázat)
- Mi a kromoszómális vs plazmid eredet
- Milyen faj-specifikus mintázatok vannak

**Elméleti plafon:** 0.94  
**Mit limitál:** Számítási kapacitás. 2M genom pretrainingje >1000 GPU-óra.

---

### SZINT 5: Strukturális biológia + docking integrálása

A rezisztencia végső soron fizikai-kémiai: a gyógyszer nem tud kötődni a célfehérjéhez.

1. AlphaFold3 / ESMFold: prediktáld a mutáns célfehérje 3D szerkezetét
2. AutoDock Vina: számold ki a kötődési energiát a gyógyszer és a mutáns fehérje között
3. Ez a ΔΔG érték egy ERŐS feature a rezisztencia predikcióhoz — mert pontosan azt méri amit a rezisztencia okoz

Ezt a megközelítést MÉG SENKI nem csinálta meg scale-ben (mert lassú). De ha 2000 izolátum x 10 antibiotikum x 5 célfehérje = 100 ezer docking — ez már párhuzamosítható.

**Elméleti plafon:** 0.96  
**Mit limitál:** A docking pontossága (Vina korrelációja a valós kötődési affinitással R²=0.5-0.7). Ez a feature-ök zajosságát adja.

---

### SZINT 6: Az AST teszt reprodukálhatósága (fizikai plafon)

Ugyanazt a baktérium törzset kétszer letesztelve az AST ±1 higítási lépés (azaz MIC 8 vs MIC 16). Ez azt jelenti hogy az emberi annotációk között is van 3-5% zaj.

Továbbá: a rezisztencia NEM teljesen determinisztikus a genotípusból:
- Epigenetikai faktorok (DNS metiláció)
- Gén expresszió sztochasztikus variabilitása
- Környezeti faktorok (pH, hőmérséklet, tápanyag)
- Heterorezisztencia (a populáció egy része rezisztens, más része nem)

**Abszolút fizikai plafon:** 0.95-0.98

---

## ÖSSZEFOGLALÁS

| Szint | Módszer | Plafon |
|---|---|---|
| 1 | Genom k-mer + XGBoost | 0.85 |
| 2 | DNA foundation model | 0.90 |
| 3 | Multi-modális ensemble | 0.92 |
| 4 | Self-supervised pretraining | 0.94 |
| 5 | + Docking integrálás | 0.96 |
| 6 | Fizikai plafon (AST zaj) | 0.98 |

---

## MIT TUDUNK MI ELÉRNI REALISZTIKUSAN?

A szint 1 + 2 + 3 kombója elérhető most:
- Genom k-mer + XGBoost (szint 1)
- ESM-2 protein embedding (a szint 2 protein megfelelője)
- Ensemble (szint 3)

**Realistikus maximumunk: 0.88-0.92**

A szint 5 (docking integrálás) pedig a MI egyedi erősségünk — a drug repurposing projektből ez a kompetenciánk. Ha ezt is beletesszük, a 0.92-0.96 sáv is elérhető.

**A válasz: 0.95 körül van az abszolút maximum.**
