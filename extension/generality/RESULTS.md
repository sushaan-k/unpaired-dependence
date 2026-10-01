# Generality benchmark: results

Frozen plan: `PLAN.md` (hashes in `results/freeze.json`); amendments and deviations: `DEVIATIONS.md`. Savings are cells the comparator envelope needs divided by cells the proposed estimator needs, with 95% intervals from 2,000 bootstrap resamples of held-out units.

## Prespecified rule (targets relative to the fully paired reference)

* G1: geometric mean 1.50 (95% CI 1.40-1.55); per data set {"hao": 1.0, "stephenson": 1.0, "bmmc_cite": 2.48, "colon": 1.0, "bmmc_multiome": 1.0, "scala_m1": 1.0, "gouwens_visp": 1.39, "banc": 3.66, "malecns": 3.07}
* G2: geometric mean 1.34 (95% CI 1.05-1.39); per data set {"hao": 1.0, "stephenson": 1.0, "bmmc_cite": 2.48, "colon": 1.0, "bmmc_multiome": 1.0, "scala_m1": 1.0, "gouwens_visp": 1.27, "banc": 2.26, "malecns": 2.0}
* G3: geometric mean 0.77 (95% CI 0.75-0.82); per data set {"hao": 1.0, "stephenson": 1.0, "bmmc_cite": 0.43, "colon": 1.0, "bmmc_multiome": 1.0, "scala_m1": 1.0, "gouwens_visp": 0.59, "banc": 0.61, "malecns": 0.62}
* Verdicts: {"G1_met": true, "G2_met": true}
* G4: Spearman -0.09, one-sided p = 0.6271

## Amendments 1-2 (targets relative to the best recovery any method reached)

* Informative data sets: hao, stephenson, bmmc_cite, colon, bmmc_multiome, gouwens_visp, banc, malecns; not informative: scala_m1
* G1' (informative): geometric mean 2.90 (95% CI 2.26-3.03)
* G2' (informative): geometric mean 2.29 (95% CI 1.82-2.38)
* G3' (informative): geometric mean 0.58 (95% CI 0.55-0.69)
* G1' (all): geometric mean 2.21 (95% CI 1.79-2.75)
* G2' (all): geometric mean 2.09 (95% CI 1.63-2.17)
* G3' (all): geometric mean 0.61 (95% CI 0.59-0.71)
* Verdicts: {"G1_amended_met": true, "G2_amended_met": true, "G1_amended_all_met": true, "G2_amended_all_met": true}
* G4': Spearman 0.42 over 11 data sets, one-sided p = 0.1040

| Data set | Units | Cells | Reference (%) | Best (%) by | Proposed cells | Paired-only cells | Saving (95% CI) | vs SemiCCA (95% CI) | Self-tuned vs fixed | Prespecified-rule saving |
|---|---|---|---|---|---|---|---|---|---|---|
| hao | 8 | 72000 | 70.0 | 35.4 ridge_p/mapped | =408 | >800 | 1.96 (1.00-2.07) | 1.96 (1.00-2.07) | 0.54 (0.51-1.00) | 1.00 |
| stephenson | 118 | 70800 | -1042.4 | 12.8 ridge_u/percond | =390 | >800 | 2.05 (1.00-2.60) | 1.02 (0.53-1.33) | 0.49 (0.38-1.00) | 1.00 |
| bmmc_cite | 9 | 36000 | 65.8 | 46.1 bjs2/mapped | =168 | >800 | 4.75 (1.00-5.99) | 3.71 (1.00-4.62) | 0.39 (0.36-1.00) | 2.48 |
| colon | 12 | 15953 | -237.2 | 25.3 bjs2/mapped | =126 | >800 | 6.36 (3.30-7.70) | 6.36 (1.95-7.70) | 0.59 (0.42-0.72) | 1.00 |
| bmmc_multiome | 10 | 36450 | -148.1 | 25.1 bjs2/mapped | =285 | >800 | 2.81 (1.71-3.02) | 2.81 (1.71-3.02) | 0.81 (0.75-1.04) | 1.00 |
| scala_m1 | 263 | 1203 | -273.7 | 0.0 lowrank/pooled | >100 | <=25 | 0.25 (0.25-2.73) | 1.00 (0.25-1.29) | 1.00 (0.85-1.00) | 1.00 |
| gouwens_visp | 362 | 3395 | 43.0 | 41.5 ridge_u/percond | =157 | =219 | 1.39 (1.20-1.59) | 1.28 (1.10-1.45) | 0.59 (0.56-0.63) | 1.39 |
| banc | 913 | 2487 | 75.8 | 68.1 bjs2/mapped | =25 | =90 | 3.59 (3.14-3.85) | 2.29 (2.02-2.42) | 0.62 (0.57-0.68) | 3.66 |
| malecns | 1010 | 3009 | 77.0 | 68.5 ridge_p/mapped | <=25 | =73 | 2.93 (2.63-3.13) | 1.94 (1.68-2.07) | 0.69 (0.59-0.80) | 3.07 |
| banc_crossanimal | 913 | 2487 | 54.9 | 48.8 ridge_p/percond | >400 | >400 | 1.00 (1.00-1.00) | 0.09 (0.08-0.10) | 1.00 (1.00-1.00) | 1.00 |

Recovered fraction (%) by budget:

* hao (budgets 25, 50, 100, 200, 400, 800): proposed -7.1 / -9.1 / 4.3 / 9.9 / 17.4 / 30.3; self-tuning -31.4 / -24.9 / -10.7 / -3.2 / 3.1 / 18.9; paired only 0.3 / 0.5 / 0.7 / 1.8 / 1.7 / 2.1; SemiCCA 1.8 / 4.2 / 6.5 / 8.4 / 10.1 / 15.0; bases from paired cells -18.5 / -9.2 / -6.6 / -2.5 / -1.1 / 0.1
* stephenson (budgets 25, 50, 100, 200, 400, 800): proposed -14.1 / -20.5 / -0.7 / 0.7 / 6.6 / 9.3; self-tuning -20.9 / -29.7 / -5.9 / -7.2 / -2.9 / 2.1; paired only 0.0 / 0.0 / 0.4 / 0.7 / 2.1 / 3.7; SemiCCA 0.0 / 0.0 / 1.6 / 3.3 / 6.4 / 9.7; bases from paired cells -18.7 / -29.5 / -4.3 / -5.8 / 0.0 / 0.7
* bmmc_cite (budgets 25, 50, 100, 200, 400, 800): proposed -6.8 / 3.3 / 15.9 / 25.4 / 36.3 / 46.1; self-tuning -3.2 / -3.8 / 4.3 / 12.5 / 21.9 / 33.9; paired only 0.3 / 0.5 / 1.1 / 2.0 / 2.7 / 5.9; SemiCCA 0.7 / 4.5 / 7.7 / 16.1 / 16.7 / 26.6; bases from paired cells -18.9 / -10.9 / -9.2 / -0.7 / 1.6 / 4.6
* colon (budgets 25, 50, 100, 200, 400, 800): proposed -2.7 / -2.9 / 11.1 / 15.9 / 19.7 / 25.3; self-tuning -5.0 / -0.8 / 8.4 / 12.3 / 16.1 / 22.0; paired only 0.3 / 0.5 / 1.2 / 2.7 / 4.9 / 7.7; SemiCCA 0.1 / 0.2 / 1.6 / 5.1 / 8.8 / 12.4; bases from paired cells -11.8 / -12.7 / -1.7 / 2.6 / 6.9 / 11.9
* bmmc_multiome (budgets 25, 50, 100, 200, 400, 800): proposed -18.1 / -4.8 / -3.7 / 6.8 / 18.1 / 25.1; self-tuning -22.7 / -6.4 / -5.6 / 3.5 / 14.6 / 22.3; paired only 0.0 / 0.0 / 0.0 / 0.1 / 0.3 / 0.6; SemiCCA -0.6 / -2.4 / -1.9 / -1.3 / 5.6 / 7.1; bases from paired cells -52.9 / -21.4 / -9.9 / -0.8 / 5.7 / 9.9
* scala_m1 (budgets 25, 50, 100): proposed -6.9 / -28.5 / -11.3; self-tuning -7.3 / -19.6 / -8.0; paired only 0.0 / 0.0 / 0.0; SemiCCA -0.8 / -2.3 / -3.6; bases from paired cells -12.1 / -42.6 / -15.0
* gouwens_visp (budgets 25, 50, 100, 200, 400): proposed 3.6 / 6.5 / 14.5 / 24.1 / 31.6; self-tuning -4.0 / 1.9 / 8.0 / 17.3 / 25.9; paired only 3.0 / 5.7 / 12.4 / 19.5 / 29.5; SemiCCA 2.8 / 4.8 / 11.3 / 20.6 / 31.9; bases from paired cells -9.1 / -8.2 / -4.4 / 1.1 / 3.6
* banc (budgets 25, 50, 100, 200, 400): proposed 34.0 / 46.0 / 56.2 / 63.9 / 68.1; self-tuning 26.0 / 37.6 / 47.7 / 56.7 / 63.4; paired only 13.5 / 23.4 / 35.9 / 45.9 / 55.5; SemiCCA 15.8 / 31.5 / 44.2 / 51.1 / 56.9; bases from paired cells 2.0 / 22.7 / 39.8 / 51.6 / 57.6
* malecns (budgets 25, 50, 100, 200, 400): proposed 35.5 / 45.8 / 54.0 / 62.0 / 67.1; self-tuning 28.9 / 38.7 / 46.4 / 54.6 / 61.3; paired only 15.9 / 27.8 / 39.5 / 49.1 / 56.4; SemiCCA 20.8 / 34.8 / 47.3 / 54.1 / 56.9; bases from paired cells 4.8 / 24.9 / 41.1 / 52.9 / 59.2
* banc_crossanimal (budgets 25, 50, 100, 200, 400): proposed -30.3 / -112.8 / -77.6 / -101.3 / -109.0; self-tuning -19.6 / -36.2 / -9.6 / -6.6 / -7.1; paired only 0.0 / -68.7 / -51.5 / -65.8 / -69.2; SemiCCA 16.1 / 30.1 / 27.9 / -61.6 / -92.8; bases from paired cells -93.0 / -157.1 / -98.7 / -94.2 / -83.8
