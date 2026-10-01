# Recoverability: measurement noise, channel mismatch and a sealed test in bone marrow

Added in response to a review of an earlier version of the manuscript. The review
found that the exact identified set of the frozen pairing-free channel was empty
for every recipient, that the "identified subspace" of the analyses is a
penalty-dependent effective supported subspace, that the RNA-only share of
variance on it was not a validated predictor of recoverability, and that the
conclusions overstated the necessity of pairing. It asked for one focused repair:
separate insufficient population variation from incompatible between-population
and within-cell relationships, test the resulting diagnostics on unseen data
against simple controls, and show a practical use for choosing a small paired
reference. The revision of 29 September 2026 keeps only the noise validation (Methods "Measurement
noise", Proposition S5 and Supplementary Note 2); the other analyses here were reported in an earlier
version (Results "Measurement noise and channel mismatch" and "A sealed test in bone marrow").

## Results in brief

* **Why the identified set was empty (corrected later).** Population
  means average out RNA sampling noise, so the channel they identify acts on
  latent expression, and noise in measured covariances can only inflate the
  compatibility statistic ||F|| (Proposition S5, part 3). RNA-only count splitting
  estimates each gene's noise share. Simulations with known latent covariance at
  realistic depths and the exact preprocessing (`simulate_noise.py`) showed that
  the corrected statistic understates the latent value (median ratio 0.22-0.93)
  and the measured one overstates it (1.33-3.03), bracketing it in 431 of 432
  cases. The frozen channel's latent statistic in bone marrow therefore lies
  between 0.52 and 2.80 and is not determined; the earlier reading
  (compatible in all 36 analyses) is withdrawn. The channel trained on
  donor-by-type groups (P1) is incompatible, because its corrected statistic,
  which is biased low, exceeded one in every bone-marrow analysis.
* **Identified versus modelled.** The sharp entrywise bounds of the compatible set
  determined the sign of no gene-protein correlation in any analysis. Closed-form
  transfer's implied channel along S differed from the population estimate by a
  median of 0.82 (relative), so its predictions are model-based.
* **Type means.** Cell-type means measured separately in each assay predicted total
  cross-correlations better than every pairing-free channel (bone marrow 0.21
  against 0.52) and better than the paired reference in 16 of 20 earlier recipients.
* **Failure prediction (sealed).** The prespecified score (coverage, compatibility,
  agreement with type means, composition) had AUC 0.87 over 1,476 pairs. It beat
  composition and sample-size controls but not coverage alone (difference 0.030;
  95% CI -0.014 to 0.059) or covariance shift (0.045; -0.012 to 0.086). Among
  251 high-coverage pairs it recognized the 19 failures (AUC 0.99; 0.98-1.00),
  18 of them from P1 within cell types.
* **Paired reference (sealed).** Neyman allocation by unpaired within-type variance
  beat random selection in all 12 samples at every budget (mean within-type error
  difference 0.034; 0.026-0.041) and balanced selection (0.068). Post hoc controls
  (`posthoc_reference.py`, added later): random selection needed 243 cells to
  match 200 allocated cells, a saving of 18% (14-20%), and unpaired type means
  alone matched the total error of 800 random paired cells, so the earlier
  statement crediting 50 paired cells with that match is withdrawn.
* **Diagnostic, post hoc (`posthoc_diagnostic.py`, `posthoc_labels.py`).** Channel
  family and analysis kind alone reached AUC 0.96 at high coverage (the score
  added 0.029; 0.002-0.082). Refitted without the donor-by-type family the score
  kept AUC 0.99 but separated failures within it weakly (0.77). Labels assigned
  separately in each assay left both AUCs unchanged.

## Provenance

1. Development on already-scored data: `dev_compatibility.py`, `dev_units2.py`,
   `dev_bootstrap.py`, `dev_typecheck.py`, `dev_bounds.py`, `dev_reference.py`
   (the earlier `dev_retrospective.py` and `dev_noise.py` are superseded; `dev_noise.py`
   was stopped before writing output). All exploratory.
2. `extract_bmmc.py` (22:50 EDT, 26 September 2026) split GSE194122 into adaptation
   and scoring files and recorded their SHA-256 in `results/bmmc_seal.json`;
   `define_bmmc_units.py` defined the analyses from cell metadata only;
   `fit_bmmc_channels.py` fitted the 63 channels from training data only;
   `fit_failure_model.py` fitted the failure model on development pairs.
3. `freeze.py` recorded the SHA-256 of `PLAN.md` and the code at 23:28:09 EDT
   (`results/freeze.json`). Amendment 1 (`results/freeze_amendment_1.json`, 23:30:19,
   before unsealing): a fallback for a singular value decomposition that did not
   converge; see "Deviations" in `PLAN.md`. The log of the stopped run is
   `logs/predict_bmmc_run1.log`.
4. `predict_bmmc.py` and `bootstrap_bmmc.py` used adaptation cells only and wrote
   `results/bmmc_manifest.json` (SHA-256 of every result file and of every
   prediction matrix via `bmmc_digests.json`); `results/bmmc_bootstrap_record.json`
   holds the bootstrap output's hash, recorded before unsealing.
5. `evaluate_bmmc.py` checked the manifest and the freeze, opened the scoring file,
   recomputed every prediction and checked its digest, and wrote
   `results/bmmc_evaluation.json` and `results/bmmc_errors.json`.
6. `make_assets.py --out ../../revision/source` writes the manuscript macros (`recov_macros.tex`, used in
   Methods), the figure and tables of the earlier version, and `results/RESULTS.md`.

## Files

| File | Purpose |
|---|---|
| `PLAN.md` | Frozen protocol, hypotheses, decision rules, deviations |
| `noise.py` | Binomial count splitting and latent RNA correlation |
| `compat.py` | Compatibility statistic, completion bounds, channel departure |
| `training.py`, `fit_bmmc_channels.py` | Channels refitted on the bone-marrow panel |
| `bmmc.py`, `extract_bmmc.py`, `define_bmmc_units.py` | Data split, seal and units |
| `reference.py` | Paired-reference allocation and stratified James-Stein estimator |
| `stats.py` | AUC, Spearman, logistic regression, donor bootstrap |
| `predict_bmmc.py`, `bootstrap_bmmc.py`, `evaluate_bmmc.py` | Sealed test |
| `diagnostics.py` | Earlier diagnostic helpers (coverage, per-protein compatibility, controls) |
| `results/` | All outputs; `RESULTS.md` lists every manuscript number |

Raw counts are not redistributed; the bone-marrow data are public (GEO GSE194122).

## Reproducing

```
python extract_bmmc.py && python define_bmmc_units.py      # needs the GEO h5ad
python fit_bmmc_channels.py && python fit_failure_model.py  # needs cross_study training data
python predict_bmmc.py && python bootstrap_bmmc.py          # refuse to run if the freeze does not match
python evaluate_bmmc.py
python make_assets.py --out ../../revision/source
```

`python verify_recoverability.py` checks the freeze, the pre-unsealing manifest, the
bootstrap record, the prespecified verdicts and the manuscript macros without raw
data. The lite archive omits four large result files (`bmmc_channels.npz`,
`bmmc_predictions.npz`, `bmmc_bounds.npz`, `bmmc_selections.npz`); the verifier
lists them as not checked. The dev scripts need the scored cohorts of `extension/cross_study`. With two cores,
prediction takes about a minute, the bootstrap about ten minutes and the
evaluation about twenty.
