# Biological readout of the validation test: within-cell-type gene-protein associations

Written 2026-09-29, after the validation test (`../PLAN.md`) had been scored, and before any statistic of the
endpoints below was computed. What was known when it was written: the validation's pooled recovered fractions,
paired cells needed, verdicts (V1-V3 met, V4 not met), audit and calibration (`../results/summary.json`). No
held-out correlation of any individual gene-protein pair, no count of associations and no value of any endpoint
below had been computed or looked at. `freeze.py` hashes this plan, `brun.py`, every module they import, the data
files and the validation's hashed predictions into `results/freeze.json` before `brun.py run` reads any held-out
cell for this analysis.

## Question

The validation measured the fraction of within-cell-type gene-protein cross-correlation recovered in a held-out
sample. A laboratory would use such an estimate to say which genes and proteins co-vary within a cell type, and in
which direction. Here the question is whether the paired cells that the atlas saves translate into that readout:
calling within-cell-type gene-protein associations that replicate in independent held-out cells, relative to the
same paired budget without the atlas.

## Predictions

The validation stored, for every fold, arm and budget, the mean of the five draws' predictions. The endpoints below
are computed per draw, as a single study would obtain them, so the per-draw predictions are regenerated with the
validation's frozen code path (`vrun.py`: contrasts, `rs.denoise` for the paired-only methods,
`es.two_sided_js` and `drun.mapped` for the references), the same folds, panels, references and seeds
(`[20261202, fold, budget, draw]`; paired-only methods `[20261203, fold, budget, draw]`).

* Arms: the eight paired-only methods (James-Stein, SCOSE, FCOSE, cross-validated low rank; pooled or per cell
  type), and the estimator with the atlas, with the other-pool reference and with the own-sample reference.
* Budgets: 25, 50, 100, 150, 200, 400 and 800 paired cells, five draws each. 150 is added to the validation's
  budgets (new draws with the same seed rule), so that the readout is available just above the 147 paired cells
  with which the atlas arm reached the validation's primary target.
* Integrity check: at the validation's six budgets, the mean over draws of every regenerated prediction must equal
  the hashed stored mean (largest absolute difference at most 1e-5) and the mean squared norm must equal the stored
  one (relative difference at most 1e-6). If not, the analysis stops and reports the failure.

## Truth: reproducible associations in held-out cells

For each fold (held-out sample) and cell type, the held-out cells are split into the validation's two sub-halves
(`validation-half-v1`). In each sub-half s, the correlation r_s of every gene with every protein across the cells of
the type (RNA log-normalized, protein centred log-ratio over the panel; `vrun.heldout`) gives
z_s = atanh(r_s) * sqrt(n_s - 3), with n_s the cells of the type in the sub-half. A gene-protein pair is a
**reproducible association** if |z_A| >= 2.576 and |z_B| >= 2.576 (two-sided p < 0.01 in each sub-half) with the
same sign; that sign is its direction. For a pair without any association the chance of passing is below 5e-5.
A cell type of a held-out sample is scored only if it has at least 20 held-out cells in each sub-half (a scored
unit). Neither the paired cells nor any reference contains a held-out cell.

## Endpoints

Each endpoint is computed for every draw and averaged over the five draws, then pooled over scored units.

* **E1, direction:** the fraction of reproducible associations whose sign the estimate gets right; an estimate of
  exactly zero counts one half (chance 50%).
* **E2, nomination:** in each scored unit, the 100 gene-protein pairs with the largest absolute estimate (ties at
  the boundary resolved by their expectation under random tie-breaking); the fraction of them that are reproducible
  associations with the estimate's sign (a zero estimate counts one half). Reported with two references: random
  nomination (half the unit's fraction of reproducible pairs) and perfect nomination (min(100, reproducible
  pairs) / 100).
* **E3, cognate pairs:** the 16 panel antibodies whose target gene (the feature reference's `target_gene_name`)
  is a panel gene: CD11b-ITGAM, CD8-CD8A, CD14-CD14, CD16-FCGR3A, CD31-PECAM1, CD161-KLRB1, KLRG1-KLRG1,
  HLA-DR-HLA-DRA, CD314-KLRK1, CD79b-CD79B, CD122-IL2RB, CD83-CD83, CD124-IL4R, CD127-IL7R, CD94-KLRD1,
  CD85j-LILRB1. In each scored unit, the correlation between each gene's RNA and its protein within the cell type:
  the noise-unbiased recovered fraction over these 16 entries (the validation's formula,
  (2<C,T> - ||C||^2) / <T_A,T_B>, pooled over units), and, descriptively, E1 restricted to cognate pairs that are
  reproducible associations.

Paired-only estimation is, for each endpoint and budget, the best of the eight paired-only methods (the envelope
used by the validation, chosen on the held-out endpoint itself, which favours paired-only estimation).

## Hypotheses

Intervals: 2,000 bootstrap resamples of donors (a donor carries its held-out samples), seed 20261204; the envelope
is recomputed in every resample.

* **B1:** E1 with the atlas minus E1 of paired-only estimation, 95% lower bound above zero at each of 50, 100 and
  150 paired cells.
* **B2:** the same for E2.
* **B3:** the same for E3 (recovered fraction on cognate pairs), at 100 and 150 paired cells.

Every hypothesis is reported whatever its outcome.

## Descriptive analyses

* E1-E3 for every arm and budget with 95% intervals; E1 and E2 per cell type; the number of scored units and of
  reproducible associations per cell type.
* Paired cells needed to reach the midpoint of E1 (50% plus half of the best E1 of any arm minus 50%) and half of
  the best E2 of any arm, for the atlas, other-pool and own-sample arms and for paired-only estimation, read from
  the running maximum by log-linear interpolation over the budgets above (a curve that does not reach the target
  counts as needing 800), with the atlas's saving over paired-only estimation and its interval.
* The largest differences of the integrity check.

## Checks before the freeze

`brun.py smoke` runs the whole analysis on synthetic values with the real design (two folds, budgets 25-100, two
draws), with the integrity check against the validation's smoke predictions (`../results_smoke/`).
