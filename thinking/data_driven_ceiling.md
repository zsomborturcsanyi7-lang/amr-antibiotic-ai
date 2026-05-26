# MEKKORA A MAXIMUM? (v6 — ADATVEZÉRELT PLAFON)

## A plafon NEM a módszertől, hanem az ADAT MENNYISÉGÉTŐL függ

A rezisztencia 3 komponensből áll:

| Mitől rezisztens? | Detektálható? | A rezisztencia hány %-a? |
|---|---|---|
| Gén jelenlét/hiány (pl. CTX-M, NDM) | ✅ k-mer / gén detekció | 50% |
| Ismert SNP (pl. gyrA S83L) | ✅ k-mer / SNP analízis | 15% |
| Promoter/expressziós változás (efflux) | ⚠️ k-mer ha elég adat van | 15% |
| Ismeretlen mutáció + strukturális hatás | ⚠️ docking | 15% |
| Epigenetika, expressziós zaj | ❌ nem detektálható szekvenciából | 5% |

## A plafon adatmennyiség függvényében

| Adatmennyiség | Mi fér bele | Plafon |
|---|---|---|
| 2 000 genom | Gén jelenlét + ismert SNP-k | **0.80** |
| 10 000 genom | + Promoter mintázatok egy része | **0.88** |
| 100 000 genom | + Minden promoter/expressziós minta | **0.93** |
| 100 000 + docking | + Ismeretlen strukturális mutációk | **0.96** |
| Fizikai plafon | Epigenetikai zaj (nem szekvencia) | **0.98** |

## Mit jelent ez nekünk?

**2000 genommal (FDA-ARGOS): 0.80 a plafon.**
**10000 genommal: 0.88 a plafon.**
**Teljes PATRIC-kel (100K+): 0.93 a plafon.**
**+ Docking integrálással: 0.96 a plafon.**

## A MI UTUNK

1. **FDA-ARGOS (2000 genom)** → 0.75-0.80 (bizonyítható)
2. **BV-BRC/PATRIC bulk (10000+ genom)** → 0.83-0.88
3. **Teljes PATRIC (100K+) + Kaggle/cloud** → 0.88-0.93
4. **Docking integrálás (saját kompetencia!)** → 0.93-0.96

A 0.80 elérhető az FDA-ARGOS adatokkal.
A 0.96 elérhető a teljes PATRIC + docking kombóval.

---

## VÁLASZ A KÉRDÉSRE

**A maximum amit valaha ki lehet hozni: 0.96**
**Amit mi reálisan el tudunk érni a következő hetekben: 0.80-0.85**
**Amit a CARB-X grant keretében el lehet érni: 0.90-0.93**
