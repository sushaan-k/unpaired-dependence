# Bone-marrow test: unpaired cells from other sites and from another assay

The manuscript calls this the bone-marrow test; file names call it the deployment test. In every earlier test of the
estimator the unpaired cells came from the same samples as the paired cells, with one modality withheld. This folder
tests whether the saving of paired cells survives when the unpaired profiles come from other sites and donors, or,
for RNA, from another assay (single-nucleus multiome), with the paired cells centred and scaled on their own
statistics so that the unpaired cells supply only correlation structure.

Frozen plan: `PLAN.md`. `freeze.py` hashed the plan, `drun.py`, every module it imports and the two data files at
19:35:44 EDT on 28 September 2026 (`results/freeze.json`), before any estimator of this test was run on the data
under its design. Predictions of all 12 folds were hashed at 19:46:50 EDT (`results/manifest.json`) before
`drun.py evaluate` read any held-out cell. The bone-marrow CITE-seq and multiome extracts (NeurIPS 2021, GSE194122)
are those of the benchmark, which had analysed them with unpaired cells of the same samples.

## Prespecified results (`results/summary.json`)

Best recovery of any arm 35.9%; primary target half of it (17.9%). Paired cells needed to reach it (pooled over the
12 held-out site-by-donor batches; 95% intervals from 2,000 resamples of held-out batches stratified by site):

| Arm | Paired cells |
|---|---|
| Paired cells only (best of JS, SCOSE, FCOSE, low rank; pooled or per cell type), own centring | >800 |
| Proposed, unpaired cells of the same samples, centred on their means (earlier design; secondary) | 169 |
| Proposed, own-site unpaired cells, own centring | 235 |
| Proposed, other-site unpaired cells, own centring | 205 |
| Proposed, other-site nuclear RNA, own centring | >800 |
| Proposed, other-site unpaired cells, centred on their cell-type means (secondary) | >800 (every budget worse than independence) |

* **D1 (primary) met**: saving over paired-only estimation with other-site unpaired cells at least 3.91 (3.29-4.45).
* **D2 not met**: with nuclear RNA the estimator recovered at most 7.3% and never reached the target.
* **D3 met**: other-site / own-site paired cells 0.87 (0.81-0.96).
* Secondary targets: savings 5.28 (4.70-5.90) at a quarter and 2.29 (2.11-2.59) at three quarters of the best
  recovery; every site gave a saving (1.97-4.41, all lower bounds).
* Below 200 paired cells the own-centred estimates were worse than independence (-535% at 25 paired cells with
  other-site unpaired cells).

## Post hoc analyses (after scoring; `results_posthoc/`)

| Script | Output | Finding |
|---|---|---|
| `posthoc_contrasts.py predict` / `evaluate` | `contrasts_folds/` (per-fold predictions), `contrasts_summary.json` | Own centring on the mean of two or three cells makes their cross-products dependent, so every noise term estimated from the spread of per-cell products is too small (by the factor n/(n-1)). Replacing each population's drawn cells by its n-1 Helmert contrasts (independent for Gaussian cells, same cross-products) removes this: with other-site unpaired cells 0.6% at 25 paired cells, 7.4% at 50, 16.6% at 100; target reached with 125 paired cells (1.64x fewer than the prespecified centring, 1.18-1.91); paired-only with contrasts >800 (saving at least 6.41, 3.70-8.32); other-site / own-site 0.68 (0.61-0.93). From 400 paired cells on the two centrings agree. Paired-only estimation is no worse with contrasts: 0.3%, 0.9% and 2.1% at 25, 50 and 100 paired cells against -16%, 0.0% and 0.9% with own centring, and within 0.1 percentage points at 400 and 800. The frozen own-centred arm of fold 0 is reproduced exactly before anything else runs. |
| `posthoc_calibration.py` | `calibration.json` | Calibration of the noise term (run after the validation test was scored): 20 fresh draws per budget from each fold's frozen reservoir, other-site bases and blocks; mean estimated noise trace of each block mean over its variance across draws (divided by 1 - B/N), summed over blocks and folds. Contrasts 1.00 at 25-200 paired cells (0.996-1.005); the prespecified own centring 0.64, 0.75, 0.84 and 0.91 at 25, 50, 100 and 200. Held-out cells are not read. |
| `posthoc_popsizes.py` | `population_sizes.json` | At 25 paired cells 60% of drawn cells lie in populations of at most three cells; at 50, 29%. |
| `posthoc_map.py` | `map_by_source.json` | With contrasts and 800 paired cells, estimates without condition maps recovered 7.0% (other sites) and 6.9% (nuclear); with their maps 35.4% and 7.3%; nuclear bases with the other-site maps 25.6%. Nuclear RNA fails mainly through the condition maps. |
| `posthoc_law.py` | `law_by_source.json` | The prospectively tested law (extension/predictability) in each pool's bases, with reservoir moments centred by contrasts: leading five RNA eigenvectors carry 77% (own site), 68% (other sites) and 45% (nuclear) of the dependence; predicted savings of block over one-block shrinkage 6.9, 4.3 and 3.0 (geometric means over folds). |
| `posthoc_assay.py` | `assay_structure.json` | Principal-angle overlap of the leading five latent RNA eigenvectors with the own-site pool: 0.74 (other sites), 0.53 (nuclear); off-diagonal correlation 0.80 and 0.59. |
| `posthoc_fly.py` | `fly_own_standardization.json` | Cross-animal fly analysis (unpaired FAFB and MANC neurons; the benchmark's prespecified secondary analysis centred the paired BANC neurons on the other animals' means and failed) with own centring as in `drun.own_std`: 60.0% at 100 paired neurons. |
| `posthoc_fly_contrasts.py` | `fly_contrasts.json` | The same with contrasts (the version in the manuscript): 60.0% at 100 paired neurons, against 36.5% paired-only and 56.2% with the same animal's unpaired neurons; within 0.5 percentage points of own centring for the estimator. |

## Files

| File | Role |
|---|---|
| `PLAN.md` | Frozen plan: data, folds, pools, panels, arms, endpoint, hypotheses D1-D3 |
| `drun.py` | `smoke` (synthetic values, real design), `predict` (checks the freeze; hashes predictions), `evaluate` (checks the hashes; held-out statistics, bootstrap) |
| `freeze.py` | Writes `results/freeze.json` |
| `results/` | `freeze.json`, `manifest.json`, `predictions.npz` (640 MB, not in the lite package), `predictions_info.json`, `summary.json`, logs |
| `results_smoke/` | Smoke-test run on synthetic values (not in the lite package) |
| `results_posthoc/` | Post hoc records above; `contrasts_folds/` (about 480 MB) is not in the lite package |

## Run and verify

```bash
python drun.py smoke            # synthetic values, two folds
python drun.py predict          # about 11 minutes on two cores
python drun.py evaluate
python posthoc_contrasts.py predict && python posthoc_contrasts.py evaluate
python posthoc_map.py; python posthoc_law.py; python posthoc_fly_contrasts.py; python posthoc_popsizes.py
python posthoc_calibration.py   # about a minute on two cores
cd .. && python verify_deployment.py [--data]
```

`verify_deployment.py` checks the freeze, the manifest, the verdicts and the post hoc records; `--data` recomputes
`summary.json` and `contrasts_summary.json` from the stored predictions in a temporary folder. The manuscript's
numbers, figure (Fig. 3) and tables (Supplementary Tables 3-6 and 8, as numbered in `../../revision/source/si_refs.tex`) are generated by `../synthesis/make_assets.py`,
which `../verify_manuscript_assets.py` regenerates and compares.

`drun.py` and `posthoc_contrasts.py` are imported by the validation test (`../validation`), whose freeze hashed them at
21:54:31 EDT on 28 September 2026; they must not change.
