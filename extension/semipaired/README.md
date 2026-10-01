# Unpaired measurements reduce the paired cells needed (semi-paired study)

How many cells measured in both assays are needed to estimate within-condition RNA-protein dependence when many
cells are measured in only one assay, and how much do the unpaired cells save? Development on the Frangieh and
Papalexi perturbation screens (opened by earlier analyses), then one external test on OverCITE-seq frozen before
any statistic of its held-out cells.

## Results

* Development (`RESULTS.md`): with 100 paired cells the estimator recovered 48% (Frangieh) and 37% (Papalexi) of
  within-condition dependence in held-out targets, against 16% and 21% for the best paired-only estimator.
  Paired-only estimation needed 4.4-5.9 and 1.9-2.5 times as many paired cells to reach recovered fractions of
  0.3-0.6. Champollion, with its lasso weight chosen on the evaluation groups, needed 1.4-1.7 times as many in
  Frangieh; in 7 Papalexi targets it needed fewer at small budgets. Bases estimated from the paired cells, or
  blocks in marker coordinates, lost most of the saving.
* External test (`external/RESULTS.md`, sealed): the estimator reached a recovered fraction of 0.3 with 73 paired
  cells, against 357 for paired-only estimation (ratio 4.9, 95% CI 3.2-6.0), 122 for SemiCCA (1.7, 1.3-2.0) and
  131 for Champollion (1.8, 1.2-2.2); H1-H3 were met (H3 is a non-inferiority test, margin 0.8). The cell numbers
  are log-linear interpolations between budgets on each curve's running maximum (73 between 50 and 100 cells, 357
  between 200 and 400); intervals resample the 11 held-out ORFs, the independent units (draws of paired cells are
  averaged, not replicates; the deposited data do not identify donors). One deviation, a scoring-script line naming
  the one-sided arm, is recorded in `external/DEVIATIONS.md`; its provenance (plan names `bjs2/mapped`; its 200
  prediction arrays were hashed before unsealing and are reproduced exactly by the frozen code) is in
  `external/results/provenance.json` (`external/provenance_check.py`).
* Theory checks (`bound_check.py`, `THEORY.md`): the risk bound held numerically for the implemented estimator at
  every budget in all three data sets; the bound extends to the condition map up to a budget-independent map error.
* Self-tuning variant (`selftune.py`, `DEV_SELFTUNE.md`, `dev_selftune.py`): preconditioned shrinkage (V2) was chosen
  on the development screens by a rule written first (recovered fraction averaged over budgets 61.9 against 60.8 for
  the fixed estimator; +4.0 and +3.1 points at 50 paired cells); it is the prespecified second arm of the generality
  benchmark (`../generality`).

## Estimator (`sp_estimators.py`, `run_study.py`)

`bjs2/mapped`: the mean cross-product of B paired cells, expressed in the eigenbases of the unpaired pool's latent
(count-split) RNA correlation and protein correlation; positive-part James-Stein shrinkage of each block (RNA
eigen-blocks 1-5, 6-20, 21-60, rest x protein eigen-blocks 1-3, rest) with the noise trace estimated from the
per-cell products; mapped to each condition through its unpaired latent RNA covariance (leading 120
eigendirections, shrunk towards the identity by split-half reliability). No tuning. Theory: `THEORY.md`
(risk bound for correlated, unequal noise; condition map); manuscript Propositions S10-S11.

## Files

| File | Role |
|---|---|
| `common.py` | Pool/reservoir split (25% paired reservoir, 75% unpaired pool; RNA and protein from different pool cells), unpaired summaries, endpoint terms |
| `sp_estimators.py` | Eigenbases, block James-Stein (one- and two-sided), noise terms |
| `competitors.py` | SCOSE, FCOSE, SemiCCA, reference ridge regression, low rank |
| `run_study.py` | Development comparison: every denoiser pooled, per condition and mapped; ablations (bases from paired cells, marker blocks, maps from paired cells or measured covariance) and unpaired dose |
| `champ_worker.py`, `champ_tune.py`, `champ_dev.py` | Champollion (inverse optimal transport; run in its own environment), its tuning and its development runs on the same paired draws |
| `summarize.py` | Recovered fractions, uncorrected loss, savings at accuracy targets with bootstrap intervals over held-out targets (`RESULTS.md`, `results/summary_<dataset>.json`) |
| `effective_dimension.py` | Checks the condition of the risk bound (effective noise dimension of each block) |
| `bound_check.py` | Numerical check of the risk bound for the implemented estimator (reservoir as population; `logs/bound_check_<dataset>.json`) |
| `selftune.py`, `dev_selftune.py`, `DEV_SELFTUNE.md` | Self-tuning variants (SURE-chosen partition; preconditioned shrinkage), the development rule written before they were run, and their development comparison (`results/selftune_<dataset>.json`) |
| `explore*.py` | Development explorations (map variants, shrinkage profiles, per-protein blocks, lasso, two-sided bases); logs in `logs/` |
| `make_assets.py` | Manuscript macros, figure and table (`revision/source/semi_*.tex`) |
| `external/` | The external test: `PLAN.md` (frozen), `extract.py` (seal), `odata.py`, `targets.py`, `pilot.py` and `pilot_champ.py` (training ORFs only), `freeze.py`, `run_external.py` (panel, predict, evaluate), `rescore.py` and `DEVIATIONS.md` (deviation 1), `RESULTS.md` |
| `results/superseded/` | Development results of the one-sided estimator, replaced when the two-sided bases were adopted (still in development) |

## Reproduce

```bash
python run_study.py frangieh && python run_study.py papalexi          # about 20 and 30 minutes
python champ_dev.py frangieh --draws 2 --budgets 50 100 200 400 --cs 1.0 1.5 2.25   # Champollion environment
python champ_dev.py papalexi --folds 7 --draws 1 --budgets 50 100 200 400 --cs 1.0 1.5 2.25
python summarize.py frangieh papalexi
python effective_dimension.py frangieh && python effective_dimension.py papalexi
python bound_check.py overcite && python bound_check.py frangieh && python bound_check.py papalexi
python dev_selftune.py papalexi && python dev_selftune.py frangieh
cd external && python provenance_check.py && cd ..                  # training file only
cd external && python run_external.py predict && python run_external.py evaluate   # after freeze.py
python rescore.py                                                   # deviation 1: the plan's estimator
cd .. && python make_assets.py --out ../../revision/source
```

Champollion (github.com/cantinilab/champollion) needs Python >= 3.12 and PyTorch; the worker runs in a separate
virtual environment whose path is set in `champ_tune.py` (`VENV`). Raw counts are not redistributed; the
loaders check the SHA-256 of the extracted files.
