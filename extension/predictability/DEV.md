# Development of the saving predictor (post hoc; recorded before the prospective plan was frozen)

Everything in this file used the nine data sets of the generality benchmark, which had been analysed before, so it
is development, not a test. It fixed the choices written into PLAN.md.

## Design used in development (`dev.py`, `dev2.py`, `ptools.py`)

Cells were split by hash into truth cells (25%, used only to score), a reservoir of paired cells (30%) and an
unpaired pool (45%). The estimand was the pooled within-population cross-covariance in pool standard-deviation units,
the quantity the proposed estimator targets before its condition map. Block James-Stein in the unpaired bases (the
proposed estimator, pooled) and one-block James-Stein (paired-only) were fitted to 20 draws of B reservoir cells at
budgets 25, 50, ..., 3,200 (up to the reservoir), and scored by the noise-unbiased recovered fraction of the truth
cells' dependence. The fast implementation reproduces `sp_estimators.block_js2` and `js_matrix` to 1e-15
(`dev.py selfcheck`). A random orthonormal basis pair with the same block sizes served as a control.

## Findings (results/dev/, `dev_summary.py`, `dev2_summary.py`)

* The oracle-linear law of Proposition S14 with block moments of the reservoir predicted the observed saving
  closely. Over 14 problems with an estimable saving at a recovered fraction of 0.5 (nine data sets and panels of
  50, 200 and 800 genes), the median absolute log2 ratio of observed to predicted saving was 0.09 (largest 0.29),
  every prediction was within 25%, and the law overpredicted slightly (median log2 ratio -0.09). Observed savings
  ranged from 1.4 to 14.7, and widening the gene panel increased them as the law predicted (for example colon 2.5,
  6.5 and 14.7 observed against 2.5, 6.3 and 18.0 predicted at 50, 200 and 800 genes).
* At a recovered fraction of 0.7 the median absolute log2 ratio was 0.06 (11 problems). At 0.3 it was 0.31
  (13 problems), because the proposed estimator reaches low targets at budgets of 25-50 cells, where the implemented
  rule falls short of the oracle-linear risk (small leading blocks, plug-in noise trace; Proposition S13).
* In random bases the observed saving was 1.00 (median over 15 problems) and the law predicted 0.9-1.0.
* A Gaussian spherical-equivalent refinement (`ptools.sph_risk`) predicted worse than the oracle-linear law and was
  dropped. A bootstrap law from the reservoir (`ptools.boot_curves`) corrected the low-target bias partly
  (median 0.11 at 0.3) and is kept as a secondary prediction.
* Pilots were noisier, as Proposition S15(c) predicts. With 200 pilot cells the median absolute log2 error at 0.5
  was 0.17 over five base problems, with one twofold miss (colon). Thresholded or shrunk pilot signals were worse.
* Unpaired noise shares (independence formula, Proposition S15a) agreed with the paired ones (for example hao,
  0.023 against 0.014 for the leading block and 0.50 against 0.49 for the rest), but combining them with paired
  signal shares was less accurate than the paired noise shares (median 0.13 at 0.5).
* Signal shares were strikingly similar across data sets and modality pairs: 22-62% of the dependence in the
  leading block (RNA eigenvectors 1-5 x protein 1-3), 14-53% in RNA 1-5 x protein rest, and little beyond RNA
  eigenvector 20. Their mean over the eight data sets with recoverable dependence (results/transfer_profile.json),
  combined with each problem's unpaired noise shares, predicted the saving with median absolute log2 error 0.27 at
  0.5 (in-sample). This zero-pair transfer is tested prospectively as H5; Proposition S15(b) shows it cannot be
  guaranteed.
* Motor-cortex Patch-seq had no recoverable dependence (every curve below 0.3 at 100 cells); its savings were
  censored.

## Choices fixed from development

Primary: the closed-form law (S14a) with block moments of all reservoir cells, at a recovered fraction of 0.5.
Secondary: targets 0.3 and 0.7; a pilot of min(200, 20% of the reservoir) cells; the zero-pair transfer profile; the
bootstrap law; budgets 25 x 2^k up to 12,800. Panels of 100, 200 and 400 X features.
