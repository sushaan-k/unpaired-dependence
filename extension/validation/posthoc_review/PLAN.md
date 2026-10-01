# Semi-paired alternatives and calibration of the readout (validation test, after scoring)

Written on 29 September 2026, after the validation test (`../PLAN.md`) and its biological readout
(`../readout/PLAN.md`) had been scored, in answer to review, and before any quantity below was computed. What was
known when it was written: the validation's results (`../results/summary.json`), the readout's results
(`../readout/results/readout.json`: E1-E3 of every arm and budget pooled over units, with intervals; E1 and E2 per
cell type; the number of reproducible associations per cell type; B1-B3 met), the robustness analyses
(`../results_posthoc/`) and the donor allocation (`../posthoc_allocation.py`, from sample labels only: every fold
is donor-disjoint). Not known: any value of the alternatives below in this data set (SemiCCA and reference
regression had been run in the development comparison, the external test and the generality benchmark, never
here); any count or endpoint under the other truth sets; any value per held-out sample and cell type; the shares of
positive and negative associations. `freeze.py` hashes this plan, `rrun.py`, every module they import, the data,
the validation's hashed predictions and the smoke results into `results/freeze.json` before `rrun.py run` or
`rrun.py champ` reads any held-out cell.

## A. Semi-paired alternatives in the atlas setting

**Question.** The validation showed that an atlas of another study reduces the paired cells needed, with this
estimator. Do established semi-paired methods, given the same information, exploit the atlas as well? The answer
separates two claims: external unpaired data help, and this estimator is an effective way to exploit them.

**The same information.**

* Paired cells: the validation's paired draws (same folds, budgets 25, 50, 100, 150, 200, 400 and 800 paired cells,
  five draws, seeds `[20261202, fold, budget, draw]`), centred within their populations by the same Helmert
  contrasts, on the same 200 genes and 111 proteins.
* Reference: the atlas's unpaired summaries that the estimator uses (pooled RNA and protein correlation matrices and
  their cell counts, the latent RNA eigenbasis, the per-cell-type summaries and the condition maps), or, for
  Champollion, which works on cells, the atlas's unpaired RNA and protein cells.
* Tuning: only the paired cells and the reference. SemiCCA's weight (0.2, 0.5, 0.8) and rank (1, 2, 4, 8, 111)
  and the ridge penalty (10^-3 to 10^2 in half-decades) are chosen by two-fold cross-validation within the paired
  contrasts (generator `[20261203, fold, budget, draw]`, as for the paired-only methods), as in the benchmark.
  Champollion's settings are those fixed for the external test by its pilot on another data set
  (`../../semipaired/logs/champ_choice.json`): epsilon 1, 2,000 iterations, lasso weight c_B / sqrt(rows) with
  c_B = 3.4, 2.25, 2.25, 1.5 and 1.5 at 25, 50, 100, 200 and 400 paired cells (800 takes the value at 400; rows is
  the number of contrast rows). The estimator has no tuning parameter.

**Arms.**

* SemiCCA (the paired cross-covariance projected on the SemiCCA subspaces, with the atlas's covariances), in the
  benchmark's three forms: pooled; through the atlas's condition map; refitted on each cell type's contrasts with
  the atlas's summaries of that type (at least 6 rows, else zero).
* Reference regression: protein on RNA by ridge regression in the paired contrasts, with the Gram matrix from the
  paired contrasts or from the atlas, the prediction R_x W; in the same three forms (six arms).
* Champollion: fitted on the paired contrasts of the first draw of every fold at 25, 50, 100, 200, 400 and 800
  paired cells; transport between up to 2,000 RNA-half and 2,000 protein-half atlas cells of each cell type,
  centred on their population means and scaled by the atlas's pooled standard deviations (`champ_tune.condition_pools`);
  the plan's cross-correlation within the type is the prediction. One draw only: a fit took 19 s with 100 and
  68 s with 800 paired cells on two cores (`rrun.py timing`), so five draws would take about ten hours. The
  estimator with the atlas is scored on the same first draws alongside it.
* The validation's arms, regenerated as the readout regenerated them: the estimator with the atlas, with the
  other-pool reference and with the own-sample reference; the eight paired-only methods.

**Endpoints.**

* The validation's recovered fraction, pooled over the 34 folds; the paired cells needed to reach the validation's
  primary target (28.34%, fixed), read from the running maximum by log-linear interpolation over the validation's
  budgets (25-800; 150 is not used, so that the validation's values are reproduced); a curve that never reaches the
  target counts as needing 800 (reported as more than 800).
* The readout's E1, E2 and E3 (primary truth), pooled over scored units.
* Families: SemiCCA is the best of its three forms and reference regression the best of its six at each budget,
  chosen on the held-out endpoint (as the paired-only envelope; this favours the alternatives).

**Comparisons (each reported whatever its outcome).** Intervals: 2,000 resamples of donors (a donor carries its
held-out samples), seed 20261207, families recomputed in every resample, target fixed.

* **K1:** the paired cells SemiCCA needs over those the estimator with the atlas needs; the estimator exploits the
  atlas more efficiently than SemiCCA if the lower bound is above 1.
* **K2:** the same for reference regression.
* Descriptive: every form; each family against paired-only estimation (does the atlas help it?) and against the
  own-sample reference; E1-E3 of the estimator minus each family at every budget; for Champollion, its recovered
  fraction, paired cells needed and their ratio to the estimator's on the same draws, and E1 and E2.

## B. Calibration of the readout

**Question.** Do the readout's conclusions depend on the significance threshold, on control of false
discoveries, on an imbalance of signs, or on how samples and cell types are weighted, and how precise are the
headline differences?

**Truth sets** (within scored units, as in the readout; direction: the sign in the first sub-half):

* `p01` (the readout's): |z| >= 2.576 in both sub-halves with the same sign.
* `p05` and `p001`: the same at 1.960 and 3.291.
* `fdr05_unit`: Benjamini-Hochberg at 5% within each scored unit (a held-out sample and cell type, 22,200 pairs)
  on the replicability P-value p = 2 (1 - Phi(min(|z_A|, |z_B|))) if the two signs agree and 1 otherwise, which is
  at most a level exactly when both sub-halves pass at that level with the same sign.
* `fdr05_all`: the same over every pair of every scored unit.

**Weighting** (E1 and E2): pooled (the readout's: units weighted by their reproducible associations for E1 and
equally for E2); every unit alike; every cell type alike (units pooled within a type); every held-out sample alike
(units pooled within a sample).

**Baselines for E1:** a random sign (50%); each unit's majority sign among its reproducible associations (an oracle
that knows the majority of every sample and cell type); one sign for all units. For E2: random nomination and
perfect nomination, as in the readout.

**Intervals and multiplicity.**

* The estimator with the atlas minus paired-only estimation (envelope recomputed in every resample) at 50, 100 and
  150 paired cells, with 95% intervals from the readout's 2,000 resamples of donors (seed 20261204, reproduced),
  for every truth set and weighting, and per cell type under the readout's truth set.
* The eight planned comparisons (B1: E1 at 50, 100 and 150; B2: E2 at the same budgets; B3: E3 at 100 and 150):
  one-sided bootstrap P-values (1 + number of resampled differences at most 0) / 20,001 from 20,000 resamples of
  donors (seed 20261206); Bonferroni-simultaneous intervals over the eight comparisons (level 1 - 0.05/8); for
  each hypothesis, the intersection-union P-value (the largest over its budgets; a hypothesis requires every one of
  its budgets, so no adjustment is needed within it), Holm-adjusted over B1-B3.

## Checks

* Integrity: every regenerated mean prediction of the validation's arms must equal the hashed stored mean (largest
  absolute difference at most 1e-5; mean squared norms within 1e-6 relative); otherwise the run stops.
* Reproduction: under the readout's truth set and pooling, E1-E3 of the readout's arms must reproduce
  `readout.json`, and the recovered fractions of the validation's arms `summary.json` (differences reported).
* Before the freeze: `rrun.py smoke` runs everything on synthetic values with the real design, and `rrun.py timing`
  measured Champollion's fits on real paired cells without reading any held-out cell.

The runs use two processes (even and odd folds) with one BLAS thread each, then Champollion with its own two
threads; the integrity tolerance covers differences in the last bits from the thread count.
