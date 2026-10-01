# Analyses added in revision

A referee's report on the submitted manuscript asked for checks and analyses that the published tests did not
contain. They were specified in `PLAN.md` after every test had been scored, and `freeze.py` hashed the plan, the
scripts, every module they import from `extension/` and the data files into `results/freeze.json` at 04:30:00 EDT
on 30 September 2026, before any of their quantities was computed. Nothing here changes a published estimate. The
manuscript reports them in Supplementary Note 11 and cites them where the claims they qualify are made.

| Plan | Question | Script | Result record |
|---|---|---|---|
| A | Is the noise estimate that sets each block's shrinkage calibrated after contrasts, with the scaling estimated from the same cells? | `noise_calibration.py` | `results/noise_calibration.json` |
| B | How much of the atlas's benefit comes from its axes and how much from its condition maps, at every budget? | `validation_review.py` | `results/validation_review.json` |
| C | Does reweighting the atlas to the paired cells' mix of cell types rescue atlases with a skewed mix? | `validation_review.py` | as B |
| D1 | How uncertain is the saving for one study, which draws its paired cells once? | `validation_review.py` | as B |
| D2 | And when the atlas's patients are resampled? | `atlas_bootstrap.py` | `results/atlas_bootstrap.json` |
| E | Which named within-type relationships become recoverable, how consistently across donors, and against reference regression with the atlas? | `validation_review.py` | as B |
| F | Is the Fisher-z test that defines the readout's reproducible associations calibrated, down to 20 cells per half? | `readout_checks.py` | `results/readout_checks.json` |
| G | Do the readout's gains hold within finer cell states and after technical covariates are removed? | `validation_review.py` | as B |
| H | What remains of the law's validation with data sets, not problems, as the units? | `law_review.py inference` | `results/law_review.json` |
| I | Does a budget chosen from a pilot, pilot included, reach the chosen accuracy? | `law_review.py pilot` | as H |
| J | What do comparisons at fixed accuracy show, with unresolved comparisons left unresolved? | `benchmark_review.py` | `results/benchmark_review.json` |

The draw-scoring check (`../checks/draw_scoring.py`), which verifies that every published curve is the mean of
per-draw scores, is not part of this plan: it checks published results rather than computing new ones.

## Running

`run_all.sh` lists every run in order (about twelve hours on two cores; the runs need the public data, `../DATA.md`)
and then the scoring. Each run checks `results/freeze.json` first. The per-fold, per-replicate and per-data-set
records are in `results/noise/`, `results/validation/`, `results/atlas_bootstrap/`, `results/law/` and
`results/benchmark/`, and the logs of the runs in `results/logs/`. The scoring commands need no data:

```bash
python noise_calibration.py score
python validation_review.py score
python atlas_bootstrap.py score
python law_review.py score
python benchmark_review.py score
python verify.py      # hashes, order, and every scored record recomputed from the stored records
```

`results_smoke/` holds the smoke runs made before the freeze on synthetic values or development data, and is hashed
in `results/freeze.json`. `DEVIATIONS.md` records how the runs departed from the plan's order of execution (none
changed a script, a setting or a value).

## Results

Every number below is read from the scored records in `results/` (Supplementary Note 11 reports them with their intervals and tables).

- **A, validation test:** the implemented noise estimate was 1.005 to 1.016 times the variance across 500 independent resamples (blocks 1.00 to 1.07); the jackknife 1.03 to 1.48.
- **A, bone-marrow test:** the implemented noise estimate was 1.005 to 1.029 times the variance across 500 independent resamples (blocks 1.00 to 1.31); the jackknife 1.05 to 1.54.
- **B:** with 100 paired cells, axes and maps 22.5%, axes only 7.8%, maps only 4.7%, neither 2.4%; paired cells to the target: 145, more than 800, more than 800, more than 800.
- **C:** paired cells to the target with published and reweighted maps: full atlas 145 and 131; B cells and CD14 monocytes from 15 patients more than 800 and 148; CD4 and CD8 T cells from 15 patients more than 800 and 311.
- **D1:** per draw, saving over paired-only at least 5.2 to 6.0; K2 0.95 to 1.17 (resampled 95% interval 0.85 to 1.45).
- **D2:** over 10 atlas replicates, the estimator needed 141 to 170 paired cells; K2 0.99 to 1.27.
- **E:** with 100 paired cells the estimator recovered more of the relationship than reference regression in 7 of 8 and than paired-only estimation in 7 of 8.
- **F:** under permutation, 4.90% of Fisher-z tests gave P<0.05 and 1.07% P<0.01; false reproducible associations about 0.21% of those counted.
- **G:** reproducible associations 120,649 (broad types), 31,509 (finer states), 98,739 (technical covariates removed); gain in direction over paired-only with 100 cells 11.2, 2.8 and 10.4 points.
- **H:** at half the dependence, 10 problems in 4 data sets; data-set means ranked in order (exact P 0.042); within-data-set permutation P 0.007; slope with data-set effects 1.08.
- **I:** at half the dependence, a pilot of 100 cells reached the target in 65% of replicates; the law's budget from every reservoir cell reached it in 26%.
- **J:** at 20% recovery, read from the raw curves, 2 of 10 comparisons could be made (geometric mean 2.46); 6 were unresolved because a curve never reached the level and 2 because a curve already exceeded it with 25 paired cells; the running maximum changed no count.

