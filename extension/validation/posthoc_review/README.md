# Other methods given the same atlas, and calibration of the readout (validation test, after scoring)

Two analyses asked for in review, specified in `PLAN.md` after the validation test (`../PLAN.md`) and its
biological readout (`../readout/PLAN.md`) had been scored, and hashed with `rrun.py`, every module they import,
Champollion's worker and settings, the data, the validation's hashed predictions and the readout's results at
16:31 EDT on 29 September 2026 (`results/freeze.json`), before any quantity below had been computed.

* **A. Other methods given the same atlas.** SemiCCA and reference ridge regression (the benchmark's
  implementations, each pooled, through the atlas's condition maps and per cell type) and Champollion (first draw
  of every fold and budget) were given the estimator's paired draws, contrasts, panels and atlas, with tuning from
  the paired cells and the atlas only. This separates "external unpaired data help" from "this estimator is an
  effective way to exploit them". K1 and K2: the paired cells SemiCCA and reference regression need to reach the
  validation's primary target over those the estimator needs, 95% lower bound above 1.
* **B. Calibration of the readout.** Truth sets at P < 0.05, 0.01 (the readout's) and 0.001 in both halves and at
  a 5% false discovery rate (within each held-out sample and cell type, or over all of them); majority-sign
  baselines; the readout aggregated with every unit, cell type or held-out sample weighted alike; intervals of the
  headline differences, one-sided bootstrap P-values, Bonferroni-simultaneous intervals and Holm's adjustment of
  B1-B3.

## Results (`results/review.json`)

Integrity: every regenerated mean prediction of the validation's arms equals the hashed one (largest absolute
difference 1.2e-7 over 11,220 predictions; squared norms within 1e-12); the readout is reproduced (largest difference 3.3e-6, in one
paired-only method at 800 paired cells, from the number of computing threads) and so are the validation's recovered
fractions (1.7e-14).

**A. Alternatives** (paired cells to the validation's primary target, 28.3%; 95% intervals from 2,000 resamples of
donors, target fixed):

| Arm (best form at each budget) | Recovered with 100 (%) | Paired cells | Relative to the estimator | Direction, 100 (%) | Top 100, 100 (%) | Cognate, 100 (%) |
|---|---|---|---|---|---|---|
| Estimator with the atlas | 22.5 | 147 (125-185) | 1 | 86.1 | 47.2 | 38.8 |
| Reference regression (six forms) | 22.4 | 169 (131-236) | 1.15 (1.00-1.36); K2 met | 86.2 | 50.0 | 27.5 |
| SemiCCA (three forms) | 17.5 | 496 (308->800) | 3.4 (2.4-4.6); K1 met | 76.4 | 31.8 | 12.9 |
| Paired cells only (eight methods) | 3.5 | >800 | at least 5.4 | 74.8 | 27.9 | 24.7 |
| Champollion (first draw of every fold) | 20.2 | 180 (165-198) | 1.23 (0.98-1.44) against the estimator on the same draws (147) | 87.8 | 42.2 | - |
* Every alternative gained from the atlas (savings over paired-only estimation at least 1.6 for SemiCCA and 4.7
  for reference regression), through its per-cell-type or mapped forms: pooled over cell types, none reached the
  target. The estimator was more efficient than SemiCCA by a wide margin and than reference regression by a small
  one (lower bound 1.003), and reference regression made more of its top pairs reproducible.
* Champollion, fitted on the paired contrasts with the external test's settings and transporting 2,000 RNA and
  2,000 protein atlas cells of each type, was run on the first draw of every fold and budget (fits took seconds to
  minutes; about 50 s on average including a period when two runs shared the cores). On those draws it needed
  180 paired cells against 147 for the estimator (1.23, 0.98-1.44), recovered at most 39.9% against 48.4%, gave
  the direction of 87.8% of associations with 100 paired cells against 86.6%, and made 42.2% of its top pairs
  reproducible against 47.0%.

**B. Calibration of the readout** (with 100 paired cells; atlas minus paired-only estimation, percentage points):

* Truth sets: P < 0.01 in both halves (the readout's; 120,649 associations), P < 0.05 (189,472), P < 0.001
  (79,414), 5% FDR within units (83,863) and over all units (80,156; P <= 0.00106). Weightings: pooled (the
  readout's), every unit, cell type or held-out sample alike. Under every combination the direction advantage was
  8.3 to 11.8 points (95% lower bounds at least 7.1) and the top-100 advantage 16.5 to 21.0 points (lower bounds at
  least 15.2).
* Signs: 57.1% of the readout's associations positive; the oracle majority-sign baseline gave 57.1% of directions
  pooled (58.1% with cell types alike), against 86.1% (atlas) and 74.8% (paired-only).
* Cell types alike: 80.3% against 70.4%. By type (direction, 100 paired cells): CD4 T +15.2 (13.5-16.1), CD8 T
  +12.8 (11.6-13.9), NK +9.5 (8.4-10.8), B +6.6 (5.5-7.7), CD14 monocytes +1.3 (0.5-1.9).
* Headline differences: direction 86.1% against 74.8%, +11.2 (10.3-12.0); top 100 47.2% against 27.9%, +19.3
  (18.0-20.5). All eight planned comparisons: no resampled difference at most zero in 20,000 resamples (one-sided
  P <= 5e-5), Bonferroni-simultaneous intervals above zero; B1-B3 Holm-adjusted P <= 1.5e-4.


## Run history

`run --part 0` and `run --part 1` ran side by side (16:31-17:39). Champollion ran first as one process with the
other two (`champ.out`, folds 0-3), was stopped between folds and restarted as two processes side by side
(`champ0.out`, `champ1.out`, folds 4-8), which slowed every fit several-fold because the two processes' threads
oversubscribed the two cores, and then ran one process at a time (even folds, then odd folds; about 150 s per
fold). A fold's record depends only on its fold, budgets and seeds, not on the process that computed it.
`score` ran once, after every record existed (`score.out`).

## Files

| File | Role |
|---|---|
| `PLAN.md` | The plan, written after scoring and before any quantity was computed |
| `rrun.py` | `smoke`, `timing` (before the freeze), `run --part 0/1`, `champ`, `score` |
| `freeze.py` | Writes `results/freeze.json` |
| `results/freeze.json` | Hashes of the plan, code, modules, Champollion's worker and settings, data and inputs |
| `results/timing.json` | Champollion's fit times on one real paired draw (before the freeze; no held-out cell read) |
| `results/folds/fold*.npz` | Per fold: per-unit numerators and denominators of E1-E3 under every truth set, recovered-fraction terms of every arm, integrity check |
| `results/champ/fold*.npz` | Per fold: the same for Champollion and the estimator on the first draw |
| `results/review.json` | Everything the manuscript reports from these analyses |
| `results_smoke/` | The smoke run on synthetic values (hashed in the freeze) |

## Run and verify

```bash
python rrun.py smoke; python rrun.py timing          # before the freeze
python freeze.py
python rrun.py run --part 0 & python rrun.py run --part 1    # about 70 minutes each on one core
python rrun.py champ                                  # Champollion (its own environment; see ../../ENVIRONMENT.md)
python rrun.py score                                  # results/review.json
cd ../.. && python verify_validation.py               # checks the freeze, the order of steps, the integrity and
                                                      # reproduction records, and recomputes review.json from the
                                                      # per-fold records
```
