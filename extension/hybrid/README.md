# Unpaired perturbation programs plus a small paired calibration set (development)

A bounded development comparison on the two perturbation screens already
opened by earlier analyses (Frangieh et al. 2021; Papalexi et al. 2021). It asks
whether programs learned from unpaired perturbations, combined with a small
and strictly separated set of paired calibration cells, recover
within-condition RNA-protein dependence with fewer paired cells than methods
without perturbation programs. Nothing here is confirmatory.

* `PLAN.md`: arms, budgets, splits, endpoint and the decision rule, hashed in
  `results/plan.json` before any arm was computed; deviations (one, after a
  smoke test) and post hoc analyses are listed at its end.
* `hmethods.py`: estimators. Unpaired: the frozen estimator (`current`), a
  SplitUP-style GMM with cross-fold debiased moments (`splitup`, after Schur et
  al. 2026), and a noise-aware estimator whose programs must reproduce across
  independent guides (`noise_aware`). Paired: mean cross-product, James-Stein,
  cross-validated low rank, Gaussian entropic transfer of the paired estimate
  (the Gaussian analogue of bridge-learned entropic transport such as
  Champollion). With unpaired information: block James-Stein in the unpaired
  eigenbases (`block_js`), a single-weight combination with the transfer
  (`combine_global`), and the program-wise hybrid (`hybrid`), with its
  variants and a principal-component control (`hybrid_pc`).
* `run.py`: `python run.py frangieh|papalexi` writes `results/<dataset>.json`
  (per-group endpoint terms for every arm, budget and block).
* `posthoc_baselines.py`: reference-regression baselines and pooled-covariance
  variants of the hybrid, added after scoring.
* `summarize.py`: applies the decision rule, computes saving factors and
  writes `results/summary.json` and `RESULTS.md`.
* `logs/`: smoke tests and full runs.

Raw counts are not redistributed; the loaders check the SHA-256 of the
extracted files recorded by `extension/perturbation` and
`extension/perturbation_replication`.
