# AMR Prediction — Hogyan érjük el a 0.80 Micro F1-et?

## Kiindulási állapot

| Módszer | Split | Micro F1 |
|---|---|---|
| Dipeptid + XGBoost | Random (átverés) | 0.93 |
| Dipeptid + XGBoost | Clustered (valós) | 0.17 |
| ESM-2 embedding + XGBoost | Clustered (valós) | **0.42** |

Cél: 0.80 Micro F1 clustered split-en.

## A probléma gyökere

1. **A CARD adatbázis géncsaládok gyűjteménye, nem független minták.** CTX-M-1-től CTX-M-200-ig mind ugyanazt a rezisztenciát adják. A clustered split ezeket szétválasztja → a modell nem tud általánosítani.

2. **Egyedi géneket prediktálunk, nem teljes genomokat.** Egy valódi baktériumban 3000-6000 gén van. A rezisztencia ezek kombinációjából adódik. Mi csak 1 gént nézünk egyszerre.

3. **Nincs valós AST (Antibiotic Susceptibility Testing) adatunk.** A CARD annotációk azt mondják hogy "ez a gén carbapenem rezisztenciát ad", de a valóságban a rezisztencia függ a gén expressziójától, kópiaszámától, promoter régiótól, és más génekkel való interakciótól.

## ÖTLET #1: Genom-szintű predikció k-merrel + valós AST adatokkal

**Hogyan működik:**
1. Letöltjük a BV-BRC/PATRIC adatbázisból a teljes genomokat + AST (MIC) adatokat
2. K-mer-eket számolunk a teljes genomra (nem csak rezisztencia génekre)
3. Betanítjuk a modellt: teljes genom k-mer → MIC érték / S/R
4. Ez a BV-BRC saját megközelítése, de ők csak XGBoost-ot használnak

**Miért lehet jobb:**
- Valós fenotípus adatok (laborban mérve, nem annotációból)
- Teljes genom kontextus (promoter régiók, kópiaszám, interakciók is benne vannak)
- A BV-BRC XGBoost modellek W1 score-ja 0.61-0.97 között van (láttuk a korábbi API hívásoknál)

**Becsült Micro F1: 0.65-0.75** (ha elérjük a BV-BRC szintjét)

**Probléma:** A BV-BRC API-ból az AST adatok "Computational Method" címkéjűek (predikciók, nem valós labor adatok). Meg kell találni a valós AST adatokat.

---

Következő iteráció: ezt fejlesztem tovább.
