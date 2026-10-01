# Deviations from the frozen plan (recorded after unsealing)

`PLAN.md` and `results/freeze.json` are kept exactly as frozen (14:49:37 EDT, 27 September 2026). Deviations are
recorded here, with times read from the system clock.

## Deviation 1 (recorded 15:08 EDT, 27 September 2026, after `evaluate` had been run)

**What.** `PLAN.md` names the proposed estimator `bjs2/mapped` (two-sided block James-Stein in the unpaired RNA
and protein eigenbases, with the condition map). The frozen `run_external.py`, however, kept the line
`PROPOSED = "bjs/mapped"` from its draft, written before the two-sided bases were adopted in development, so
`results/overcite.json` (written 15:07:01 EDT by `run_external.py evaluate`) reports the savings, the
differences from the best comparator and the verdicts for the one-sided arm `bjs/mapped`. The error was noticed
when the stored "proposed" curve was compared with the `bjs2/mapped` curve in the same file.

**What was done.** Nothing that was frozen or hashed was changed. The predictions of both arms had been computed
and hashed with all others before any held-out file was opened (`results/manifest.json`, 15:06:29 EDT).
`rescore.py` applies the frozen scoring function, unchanged, to the same hashed predictions and held-out
statistics with the plan's arm, and writes `results/overcite_plan_estimator.json` (15:08:11 EDT). Both files are
kept and reported.

**Consequence.** The verdicts are the same for both arms: H1, H2 and H3 are met.

| Comparison (recovered fraction 0.3) | Plan's estimator `bjs2/mapped` | Frozen script's arm `bjs/mapped` |
|---|---|---|
| Paired cells the estimator needs | 73 | 101 |
| H1: paired-only (357 cells) | 4.88 (3.22-6.03), met | 3.55 (2.61-4.12), met |
| H2: SemiCCA (122 cells) | 1.66 (1.32-2.05), met | 1.21 (1.09-1.34), met |
| H3: Champollion (131 cells; non-inferiority, lower bound > 0.8) | 1.79 (1.22-2.24), met | 1.30 (0.96-1.54), met |
| Every comparator in its original form (115 cells) | 1.57 (1.12-1.93) | 1.14 (0.92-1.27) |

Ratios are (cells the comparator needs) / (cells the estimator needs), with 95% bootstrap intervals over
held-out ORFs. The manuscript reports the plan's estimator as the proposed one and this deviation with the
frozen script's figures.
