# Predictability study: results

Frozen plan: `PLAN.md` (`results/freeze.json`); data record `results/data_freeze.json`; deviations `DEVIATIONS.md`. Savings are cells needed by one-block James-Stein divided by cells needed by block James-Stein in the unpaired bases, at a recovered fraction of the population's dependence.

* Verdicts: {"evaluable": true, "H1_law_calibrated": true, "H2_law_ranks": true, "H3_random_control_no_saving": true, "H4_pilot_calibrated": true, "H5_transfer_calibrated": true, "H6_within_unpaired_bounds": true}
* Target 0.5: 10 of 29 problems estimable; law median |log2| 0.276 (CI 0.151-0.365), bias -0.276; pilot 0.164; transfer 0.486; Spearman 0.976; random control 0.996
* Target 0.3: 8 of 29 problems estimable; law median |log2| 0.801 (CI 0.624-1.289), bias -0.801; pilot 0.972; transfer 1.008; Spearman 0.952; random control 0.991
* Target 0.7: 5 of 29 problems estimable; law median |log2| 0.062 (CI 0.027-0.076), bias -0.062; pilot 0.026; transfer 0.364; Spearman 1.000; random control 0.997
* Secondary: {"0.5": {"boot_median_abs_log2": 0.09029785700742259, "boot_median_log2_bias": 0.030741284490058758, "boot_n": 10, "law_cells_js_median_log2": 0.057811441839678956, "law_cells_blocks_median_log2": 0.3029671945365451, "censored_problems": 19, "censoring_checks": {"blocks_above_at_start": [9, 9], "blocks_cells_within_2x": [2, 4], "blocks_not_reached": [5, 6], "js_above_at_start": [8, 8], "js_not_reached": [10, 10]}, "censoring_failures": ["snare_cortex/p100:blocks_cells_within_2x", "snare_cortex/p200:blocks_cells_within_2x", "human_gaba/p100:blocks_not_reached"]}, "0.3": {"boot_median_abs_log2": 0.40770247611292587, "boot_median_log2_bias": -0.1633667290882798, "boot_n": 7, "law_cells_js_median_log2": 0.0005878513438135054, "law_cells_blocks_median_log2": 0.838235833816732, "censored_problems": 21, "censoring_checks": {"blocks_above_at_start": [14, 14], "blocks_cells_within_2x": [3, 6], "blocks_not_reached": [0, 1], "js_above_at_start": [9, 9], "js_not_reached": [7, 7]}, "censoring_failures": ["snare_cortex/p100:blocks_cells_within_2x", "snare_cortex/p200:blocks_cells_within_2x", "snare_cortex/p400:blocks_cells_within_2x", "human_gaba/p400:blocks_not_reached"]}, "0.7": {"boot_median_abs_log2": 0.10633204840226526, "boot_median_log2_bias": 0.10633204840226526, "boot_n": 5, "law_cells_js_median_log2": 0.11311383871399543, "law_cells_blocks_median_log2": 0.15817529282495868, "censored_problems": 24, "censoring_checks": {"blocks_above_at_start": [9, 9], "blocks_cells_within_2x": [6, 6], "blocks_not_reached": [9, 9], "js_not_reached": [15, 15]}, "censoring_failures": []}, "curve_mean_abs_err": {"bjs2": 0.04215132511384208, "js": 0.008044626225761657}}

| Data set | Problem | Target | Observed | Estimable | Law | Pilot | Transfer | Bound | Random observed |
|---|---|---|---|---|---|---|---|---|---|
| sln111 | p100 | 0.5 | 5.38 (=/=) | True | 6.76 | 7.57 | 7.41 | 92.5 | 1.00 |
| sln111 | p100 | 0.3 | 7.41 (=/=) | True | 10.07 | 10.53 | 12.44 | 92.5 | 0.99 |
| sln111 | p100 | 0.7 | 2.75 (>/=) | False | 3.70 | 4.81 | 4.24 | 92.5 | 1.00 |
| sln111 | p200 | 0.5 | 8.33 (=/=) | True | 9.70 | 8.77 | 12.09 | 151.2 | 1.00 |
| sln111 | p200 | 0.3 | 7.82 (=/=) | True | 15.53 | 15.02 | 20.33 | 151.2 | 0.99 |
| sln111 | p200 | 0.7 | 2.35 (>/=) | False | 4.73 | 3.95 | 6.86 | 151.2 | 1.00 |
| sln111 | p400 | 0.5 | 11.57 (=/=) | True | 14.55 | 13.40 | 18.85 | 233.5 | 1.00 |
| sln111 | p400 | 0.3 | 15.17 (=/=) | True | 25.47 | 25.80 | 31.53 | 233.5 | 1.00 |
| sln111 | p400 | 0.7 | 2.25 (>/=) | False | 5.95 | 4.83 | 10.74 | 233.5 | 1.00 |
| sln206 | p100 | 0.5 | 7.06 (=/=) | True | 10.03 | 17.44 | 9.23 | 144.7 | 0.99 |
| sln206 | p100 | 0.3 | 8.12 (=/=) | True | 14.67 | 26.13 | 18.00 | 144.7 | 0.99 |
| sln206 | p100 | 0.7 | 1.97 (>/=) | False | 4.81 | 7.16 | 4.59 | 144.7 | 1.00 |
| sln206 | p200 | 0.5 | 11.43 (=/=) | True | 14.72 | 29.64 | 15.03 | 234.7 | 1.00 |
| sln206 | p200 | 0.3 | 14.43 (=/=) | True | 22.23 | 44.94 | 29.27 | 234.7 | 0.99 |
| sln206 | p200 | 0.7 | 1.85 (>/=) | False | 6.41 | 9.69 | 7.41 | 234.7 | 1.00 |
| sln206 | p400 | 0.5 | 11.48 (>/=) | False | 20.16 | 54.45 | 23.57 | 367.4 | 1.00 |
| sln206 | p400 | 0.3 | 23.01 (=/=) | True | 34.39 | 85.19 | 45.87 | 367.4 | 1.00 |
| sln206 | p400 | 0.7 | 1.81 (>/=) | False | 7.08 | 11.69 | 11.61 | 367.4 | 1.00 |
| pbmc10k | p100 | 0.5 | 1.00 (<=/<=) | False | 2.39 | 2.46 | 1.26 | 21.0 | 1.00 |
| pbmc10k | p100 | 0.3 | 1.00 (<=/<=) | False | 2.49 | 2.56 | 1.30 | 21.0 | 1.00 |
| pbmc10k | p100 | 0.7 | 1.12 (=/<=) | False | 2.23 | 2.30 | 1.21 | 21.0 | 0.99 |
| pbmc10k | p200 | 0.5 | 1.00 (<=/<=) | False | 3.35 | 3.43 | 1.62 | 28.4 | 1.00 |
| pbmc10k | p200 | 0.3 | 1.00 (<=/<=) | False | 3.49 | 3.56 | 1.69 | 28.4 | 1.00 |
| pbmc10k | p200 | 0.7 | 1.39 (=/<=) | False | 3.10 | 3.18 | 1.53 | 28.4 | 0.99 |
| pbmc10k | p400 | 0.5 | 1.00 (<=/<=) | False | 4.88 | 5.06 | 2.19 | 37.4 | 1.00 |
| pbmc10k | p400 | 0.3 | 1.00 (<=/<=) | False | 5.10 | 5.26 | 2.28 | 37.4 | 1.00 |
| pbmc10k | p400 | 0.7 | 1.88 (=/<=) | False | 4.48 | 4.67 | 2.04 | 37.4 | 1.00 |
| malt10k | p100 | 0.5 | 1.00 (<=/<=) | False | 4.99 | 4.68 | 2.07 | 23.8 | 1.00 |
| malt10k | p100 | 0.3 | 1.00 (<=/<=) | False | 5.04 | 4.73 | 2.23 | 23.8 | 1.00 |
| malt10k | p100 | 0.7 | 1.44 (=/<=) | False | 4.88 | 4.55 | 1.86 | 23.8 | 0.99 |
| malt10k | p200 | 0.5 | 1.00 (<=/<=) | False | 5.90 | 5.34 | 2.46 | 35.7 | 1.00 |
| malt10k | p200 | 0.3 | 1.00 (<=/<=) | False | 5.96 | 5.42 | 2.65 | 35.7 | 1.00 |
| malt10k | p200 | 0.7 | 1.31 (=/<=) | False | 5.77 | 5.17 | 2.21 | 35.7 | 1.08 |
| malt10k | p400 | 0.5 | 1.00 (<=/<=) | False | 8.10 | 7.17 | 3.37 | 48.0 | 1.00 |
| malt10k | p400 | 0.3 | 1.00 (<=/<=) | False | 8.22 | 7.31 | 3.64 | 48.0 | 1.00 |
| malt10k | p400 | 0.7 | 2.05 (=/<=) | False | 7.84 | 6.84 | 3.01 | 48.0 | 1.00 |
| bmcite | p100 | 0.5 | 1.00 (<=/<=) | False | 4.77 | 4.63 | 1.65 | 14.4 | 1.00 |
| bmcite | p100 | 0.3 | 1.00 (<=/<=) | False | 4.91 | 4.76 | 1.71 | 14.4 | 1.00 |
| bmcite | p100 | 0.7 | 1.56 (=/<=) | False | 4.45 | 4.34 | 1.54 | 14.4 | 0.99 |
| bmcite | p200 | 0.5 | 1.00 (<=/<=) | False | 6.89 | 6.64 | 2.29 | 22.5 | 1.00 |
| bmcite | p200 | 0.3 | 1.00 (<=/<=) | False | 7.07 | 6.82 | 2.38 | 22.5 | 1.00 |
| bmcite | p200 | 0.7 | 1.91 (=/<=) | False | 6.48 | 6.24 | 2.12 | 22.5 | 0.99 |
| bmcite | p400 | 0.5 | 1.05 (=/<=) | False | 10.36 | 9.92 | 3.26 | 34.0 | 0.99 |
| bmcite | p400 | 0.3 | 1.00 (<=/<=) | False | 10.61 | 10.15 | 3.39 | 34.0 | 1.00 |
| bmcite | p400 | 0.7 | 2.61 (=/<=) | False | 9.80 | 9.41 | 3.02 | 34.0 | 1.00 |
| fetal_cortex | p100 | 0.5 | 1.00 (>/>) | False | 3.29 | 49.15 | 9.57 | 187.0 | 1.00 |
| fetal_cortex | p100 | 0.3 | 1.12 (>/=) | False | 4.03 | 51.36 | 22.35 | 187.0 | 1.00 |
| fetal_cortex | p100 | 0.7 | 1.00 (>/>) | False | 2.47 | 44.93 | 3.83 | 187.0 | 1.00 |
| fetal_cortex | p200 | 0.5 | 1.00 (>/>) | False | 4.46 | 17.14 | 12.81 | 240.3 | 1.00 |
| fetal_cortex | p200 | 0.3 | 1.58 (>/=) | False | 5.79 | 24.39 | 29.32 | 240.3 | 1.00 |
| fetal_cortex | p200 | 0.7 | 1.00 (>/>) | False | 3.13 | 5.95 | 5.10 | 240.3 | 1.00 |
| fetal_cortex | p400 | 0.5 | 1.00 (>/>) | False | 5.81 | 32.96 | 17.96 | 353.6 | 1.00 |
| fetal_cortex | p400 | 0.3 | 1.72 (>/=) | False | 8.24 | 35.80 | 40.74 | 353.6 | 1.00 |
| fetal_cortex | p400 | 0.7 | 1.00 (>/>) | False | 3.69 | 27.27 | 7.12 | 353.6 | 1.00 |
| snare_cortex | p100 | 0.5 | 1.18 (>/=) | False | 28.25 | 5.37 | 11.03 | 226.5 | 1.00 |
| snare_cortex | p100 | 0.3 | 5.92 (>/=) | False | 48.69 | 12.75 | 26.78 | 226.5 | 1.00 |
| snare_cortex | p100 | 0.7 | 1.00 (>/>) | False | 9.51 | 2.71 | 4.22 | 226.5 | 1.00 |
| snare_cortex | p200 | 0.5 | 1.54 (>/=) | False | 40.71 | 2.36 | 17.52 | 362.3 | 1.00 |
| snare_cortex | p200 | 0.3 | 7.37 (>/=) | False | 79.66 | 9.47 | 42.78 | 362.3 | 1.00 |
| snare_cortex | p200 | 0.7 | 1.00 (>/>) | False | 12.19 | 1.47 | 6.59 | 362.3 | 1.00 |
| snare_cortex | p400 | 0.5 | 2.04 (>/=) | False | 51.91 | 19.37 | 26.41 | 540.8 | 1.00 |
| snare_cortex | p400 | 0.3 | 9.26 (>/=) | False | 110.41 | 48.23 | 64.16 | 540.8 | 1.00 |
| snare_cortex | p400 | 0.7 | 1.00 (>/>) | False | 16.01 | 9.00 | 9.89 | 540.8 | 1.00 |
| fafb_vpn | p100 | 0.5 | 1.87 (=/=) | True | 2.34 | 2.17 | 1.57 | 56.1 | 0.99 |
| fafb_vpn | p100 | 0.3 | 1.27 (=/<=) | False | 2.95 | 2.68 | 1.84 | 56.1 | 0.99 |
| fafb_vpn | p100 | 0.7 | 1.65 (=/=) | True | 1.74 | 1.68 | 1.34 | 56.1 | 1.00 |
| fafb_vpn | p200 | 0.5 | 2.49 (=/=) | True | 2.93 | 2.71 | 1.85 | 21.9 | 1.00 |
| fafb_vpn | p200 | 0.3 | 1.34 (=/<=) | False | 3.79 | 3.52 | 2.18 | 21.9 | 0.99 |
| fafb_vpn | p200 | 0.7 | 1.97 (=/=) | True | 2.06 | 1.94 | 1.53 | 21.9 | 1.00 |
| fafb_vpn | p400 | 0.5 | 3.09 (=/=) | True | 3.48 | 3.17 | 2.13 | 24.4 | 1.00 |
| fafb_vpn | p400 | 0.3 | 1.47 (=/<=) | False | 4.52 | 4.16 | 2.53 | 24.4 | 0.99 |
| fafb_vpn | p400 | 0.7 | 2.25 (=/=) | True | 2.39 | 2.23 | 1.75 | 24.4 | 1.00 |
| banc_vpn | p100 | 0.5 | 3.26 (=/=) | True | 3.54 | 3.15 | 2.23 | 29.4 | 0.99 |
| banc_vpn | p100 | 0.3 | 2.46 (=/<=) | False | 4.59 | 3.94 | 2.75 | 29.4 | 0.98 |
| banc_vpn | p100 | 0.7 | 2.48 (=/=) | True | 2.46 | 2.34 | 1.76 | 29.4 | 1.00 |
| banc_vpn | p200 | 0.5 | 3.85 (=/=) | True | 4.27 | 3.90 | 2.70 | 26.6 | 1.01 |
| banc_vpn | p200 | 0.3 | 2.64 (=/<=) | False | 5.59 | 4.98 | 3.31 | 26.6 | 1.00 |
| banc_vpn | p200 | 0.7 | 2.95 (=/=) | True | 2.87 | 2.77 | 2.13 | 26.6 | 1.00 |
| human_gaba | p100 | 0.5 | 1.00 (>/>) | False | 1.89 | 1.29 | 1.38 | 73.6 | 1.00 |
| human_gaba | p100 | 0.3 | 0.82 (=/=) | True | 2.17 | 1.37 | 1.46 | 73.6 | 0.96 |
| human_gaba | p100 | 0.7 | 1.00 (>/>) | False | 1.55 | 1.22 | 1.28 | 73.6 | 1.00 |
| human_gaba | p200 | 0.5 | 1.00 (>/>) | False | 1.79 | 1.66 | 1.49 | 26.3 | 1.00 |
| human_gaba | p200 | 0.3 | 0.92 (=/=) | True | 2.07 | 1.84 | 1.59 | 26.3 | 0.94 |
| human_gaba | p200 | 0.7 | 1.00 (>/>) | False | 1.48 | 1.49 | 1.36 | 26.3 | 1.00 |
| human_gaba | p400 | 0.5 | 1.00 (>/>) | False | 1.77 | 1.61 | 1.61 | 17.5 | 1.00 |
| human_gaba | p400 | 0.3 | 1.00 (>/>) | False | 2.06 | 1.85 | 1.73 | 17.5 | 1.00 |
| human_gaba | p400 | 0.7 | 1.00 (>/>) | False | 1.47 | 1.37 | 1.46 | 17.5 | 1.00 |
