# Validation test: contrast-centred paired cells with an external atlas, in a study not analysed before

The bone-marrow test (`../deployment`, the deployment test of the file names) found that correlation structure from unpaired cells of other sites keeps the
saving of paired cells, but its prespecified centring of the paired cells on their own means was worse than
independence below 200 paired cells; centring by within-population contrasts, defined after that test was scored,
removed the failure. This folder tests the resulting workflow, fixed in advance, in a study this project had not
analysed: a few paired cells of a small study, centred within their populations by Helmert contrasts, with the
unpaired correlation structure taken from another study's atlas, at budgets of 25 to 800 paired cells.

Frozen plan: `PLAN.md`. `freeze.py` hashed the plan, `vrun.py`, `vdata.py`, every module they import (including
`../deployment/drun.py` and `../deployment/posthoc_contrasts.py`), the two data files and the raw downloads at
21:54:31 EDT on 28 September 2026 (`results/freeze.json`). Before the freeze the data had been extracted (counts and
metadata; cell counts, library sizes and the antibody name match), the pipeline had been run on synthetic values with
the real design (`results_smoke/`) and the atlas and two other-pool references had been summarized without any
paired cell (`results_check/pools_check.json`); both check files are hashed in the freeze. Predictions of all 34
folds were hashed at 22:16:09 EDT (`results/manifest.json`) before `vrun.py evaluate` read any held-out cell.

## Data

* GEO GSE314416 (ImmunoMicrobiome study of healthy adults; 10x 5' CITE-seq, TotalSeq-C panel of 140 antibodies):
  the seven pools with antibody counts, 34 samples of 18 donors. Raw counts are not redistributed; `vdata.py
  download` and `vdata.py extract` rebuild `gse314416.npz` from GEO.
* Atlas: the Stephenson et al. 2021 blood CITE-seq extract of the benchmark (E-MTAB-10026; 118 patients,
  600 cells per patient, 192 antibodies).
* Five cell types (B, CD4 T, CD8 T, NK, CD14 monocytes); 200 genes chosen from the atlas's RNA-half cells; the 111
  antibodies matched between the panels by name or target gene and detected in at least 1% of atlas protein-half
  cells.

## Prespecified results (`results/summary.json`)

Best recovery of any arm 57%; primary target half of it (28%). Recovered fraction (%) of within-cell-type
gene-protein dependence of the held-out sample, pooled over the 34 folds, and paired cells needed to reach the
target (95% intervals from 2,000 resamples of donors):

| Arm | 25 | 50 | 100 | 200 | 400 | 800 | Paired cells |
|---|---|---|---|---|---|---|---|
| Paired cells only (contrasts) | 0.5 | 2.4 | 3.5 | 7.7 | 13.9 | 12.6 | >800 |
| Atlas (primary) | 0.5 | 11.2 | 22.5 | 32.9 | 42.0 | 48.1 | 147 |
| Other pools and donors | 8.8 | 23.0 | 33.9 | 42.6 | 50.4 | 56.7 | 70 |
| Own samples | 5.6 | 19.1 | 27.5 | 34.1 | 40.3 | 45.1 | 110 |
| Own samples, their averages (secondary) | 8.2 | 18.7 | 27.0 | 33.7 | 39.7 | 43.8 | 114 |
| Atlas, centred as in bone marrow (secondary) | -408 | -60 | 12.8 | 31.4 | 41.7 | 48.1 | 178 |

* **V1 (primary) met:** saving over paired-only estimation with the atlas at least 5.43 (4.82-5.96).
* **V2 met:** the atlas arm recovered 11% (8.8-14%) with 50 and 23% (20-25%) with 100 paired cells.
* **V3 met:** saving with the other-pool reference at least 11.4 (10.3-12.8).
* **V4 not met:** the atlas needed 1.34 (1.15-1.56) times the paired cells needed with the own samples' unpaired
  cells; the non-inferiority margin was 1.25.
* Every pool gave a saving with the atlas (lower bounds 4.5 to 8.2).
* Degrees of freedom: with 25, 50 and 100 paired cells, 16, 40 and 90 contrast rows on average.
* Calibration of the noise term (atlas bases, estimator's blocks, 25-200 paired cells): estimated noise trace over
  the across-draw variance of the block means 1.00 under contrasts at every budget, and 0.67-0.95 under the
  bone-marrow test's own centring.

## Biological readout (specified after scoring; `readout/`)

Written and hashed at 07:06 EDT on 29 September 2026, after the test had been scored and before any of its endpoints
had been computed. With 50 and 100 paired cells, the atlas arm gave the direction of 80% and 86% of 120,649
reproducible within-cell-type gene-protein associations (P < 0.01 with the same sign in both halves of the held-out
cells), against 67% and 75% for paired-only estimation; of its 100 strongest pairs per sample and cell type, 36% and
47% were reproducible associations of the estimated sign, against 15% and 28% (random 1.6%); and it recovered 39% of
the within-type correlation of 16 cognate RNA-protein pairs with 100 paired cells, against 25%. B1-B3 met. See
`readout/README.md`, including where it did not help (B cells, CD14 monocytes, the direction of cognate pairs).

## Robustness (after scoring; `posthoc_robustness.py`, `posthoc_composition_map.py`)

* From the hashed predictions: every pool and every donor gave a saving (donors 1.08 to 8.1, most of them lower
  bounds); with any one pool left out the saving was at least 5.2 to 5.6, and the atlas still needed 1.24 to 1.42
  times the paired cells of the own samples' unpaired cells.
* Atlas arm recomputed with the reference changed (the unchanged atlas reproduces the hashed predictions exactly):
  cells of 59, 30, 15 and 7 of the 118 patients: 174, 177, 185 and 385 paired cells to the target (147 with all);
  4 patients (2,132 cells): target not reached within 800. Without B cells, CD4 T cells, NK cells or CD14 monocytes:
  174, 159, 184 and 209; without CD8 T cells (47% of the readout's reproducible associations): not reached. B cells
  and CD14 monocytes, or CD4 and CD8 T cells, from 15 patients only while the other types come from all 118: not
  reached (at most 20% and 14%; with T cells underrepresented, recovery falls to -2.7% at 800 paired cells).
* Scored without condition maps, the unchanged and the two skewed atlases recover alike (14%, 14% and 13% with 800
  paired cells): the bases carry over, and the maps, which convert the pooled estimate through the atlas's pooled
  covariance, fail when the atlas's composition departs from the paired cells'.

## Donor separation (`posthoc_allocation.py`, from sample labels)

Every fold is donor-disjoint (`results_posthoc/allocation.json`). Each pool holds at most one sample of a donor, and
the 16 donors sampled twice have their two samples in different pools, so the two paired samples of a fold come
from two donors other than the held-out donor; the other-pool reference excludes every donor of the held-out
sample's pool (and with it the held-out donor's other sample); the own-sample reference is the paired samples'
other cells; the atlas is another study. The held-out cells behind every endpoint, including both halves used by
the readout, come from a donor that contributed no paired and no reference cell to its fold. Supplementary Table
"Sample, donor and pool allocation" lists every fold.

## Other methods given the same atlas, and calibration of the readout (`posthoc_review/`)

Specified in `posthoc_review/PLAN.md` after the test and its readout had been scored, and hashed before any of its
quantities was computed (16:31 EDT on 29 September). See `posthoc_review/README.md`.

* Given the same paired cells, contrasts, panels, atlas and tuning information, SemiCCA needed 496 paired cells to
  reach the primary target (3.4 times the estimator's 147; K1 met), reference ridge regression 169 (1.15,
  1.003-1.36; K2 met, narrowly) and Champollion, on the first draw of every fold, 180 against 147 on the same draws
  (1.23, 0.98-1.44). Every method gained from the atlas.
* The readout's advantage held at P < 0.05 and P < 0.001 in both halves, under 5% FDR control and with units,
  cell types or samples weighted alike (8.3-11.8 points in direction with 100 paired cells); 57% of associations
  are positive and the oracle majority-sign baseline gives 57%; all eight planned comparisons hold with
  Bonferroni-simultaneous intervals (Holm-adjusted P for B1-B3 at most 2e-4).

## After scoring

`../synthesis/make_assets.py` recomputes the saving with the budget axis counted in contrast rows (mean rows at each
budget, same interpolation): 137 rows against more than 790 for paired-only estimation, a saving of at least 5.8
(5.4 counted in drawn cells). The bone-marrow calibration of the noise term is `../deployment/posthoc_calibration.py`.

## Files

| File | Role |
|---|---|
| `PLAN.md` | Frozen plan: data, folds, references, arms, endpoint, hypotheses V1-V4, descriptive analyses |
| `vdata.py` | `download` (GEO files), `extract` (counts and metadata into `gse314416.npz`) |
| `vrun.py` | `smoke`, `pools` (checks before the freeze), `predict` (checks the freeze; hashes predictions), `evaluate` (checks the hashes; held-out statistics, bootstrap over donors, audit) |
| `freeze.py` | Writes `results/freeze.json` |
| `results/` | `freeze.json`, `manifest.json`, `predictions.npz` (948 MB, not in the lite package), `predictions_info.json`, `audit.json`, `summary.json`, logs |
| `results_smoke/`, `results_check/` | Checks before the freeze (the smoke predictions are not in the lite package) |
| `readout/` | Biological readout: plan, runner, freeze, results (`readout/README.md`) |
| `posthoc_robustness.py`, `results_posthoc/robustness.json` | After scoring: per donor and pool, each pool left out, atlas size and composition |
| `posthoc_composition_map.py`, `results_posthoc/composition_map.json` | After scoring: the skewed atlases scored without condition maps |
| `posthoc_allocation.py`, `results_posthoc/allocation.json` | After scoring, from sample labels: the sample-donor-pool allocation of every fold and its donor separation |
| `posthoc_review/` | Specified after scoring: other methods given the same atlas (K1, K2; Champollion) and calibration of the readout |

## Run and verify

```bash
python vdata.py download && python vdata.py extract
python vrun.py smoke; python vrun.py pools
python vrun.py predict          # about 21 minutes on two cores
python vrun.py evaluate         # under a minute
python posthoc_robustness.py    # after scoring; about 53 minutes
python posthoc_composition_map.py   # about 12 minutes
python posthoc_allocation.py        # seconds; sample labels only
cd posthoc_review && python rrun.py run --part 0 & python rrun.py run --part 1   # then champ --part 0/1, score
cd .. && python verify_validation.py [--data]
```

`verify_validation.py` checks the hashes of the frozen code and data, the order freeze < predictions < scoring, the
manifest and the verdicts; `--data` recomputes `summary.json` from the hashed predictions in a temporary folder and
compares it with the stored one, and checks the readout's freeze, order of steps and integrity record and the
robustness record. The manuscript's numbers, Fig. 2 and Supplementary Tables 7-11 (as numbered in
`../../revision/source/si_refs.tex`) are generated by `../synthesis/make_assets.py`.
