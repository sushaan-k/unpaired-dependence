# Validation test: contrast-centred semi-paired estimation with an external reference, in untouched data

Written 2026-09-28, before any paired-cell estimate and any statistic of held-out cells of the validation data was
computed. `freeze.py` hashes this plan, `vrun.py`, `vdata.py`, the modules they import and the data files into
`results/freeze.json`; predictions are hashed into `results/manifest.json` before `vrun.py evaluate` reads any
held-out cell.

## Question

The deployment test (extension/deployment) found that correlation structure from unpaired cells of other sites and
donors keeps the saving of paired cells when the paired cells are centred on their own population means, but its
prespecified centring was worse than independence below 200 paired cells. After scoring, centring each population's
paired cells by Helmert contrasts (`deployment/posthoc_contrasts.py`) removed that failure. This test evaluates the
resulting workflow, fixed in advance, on data this project has never analysed: a few paired cells of a small study,
centred within their populations by contrasts, and unpaired correlation structure from an external reference.

## Checks before the freeze

* `vrun.py smoke`: the whole pipeline (two folds, budgets 25-100, two draws, evaluation) on synthetic values with
  the real design (`results_smoke/`).
* `vrun.py pools`: on the real data, only the unpaired summaries of the references (the atlas and the other-pool
  references of two folds): cells, populations, map reliabilities, map norms and the leading latent eigenvalues
  (`results_check/pools_check.json`). No paired-cell estimate and no statistic of held-out cells was computed.
* `vdata.py extract` read counts and metadata and computed cell counts, library sizes and the antibody name match.

## Data

* **Validation data (untouched):** GSE314416, ImmunoMicrobiome study of healthy adults, 10x 5' CITE-seq with the
  TotalSeq-C panel (140 antibodies). Only the seven pools with antibody counts (DB8-DB14) are used: 34 samples of
  18 donors (16 donors at two time points, in different pools), 2-4 GEM wells per pool. Cell types are the authors'
  labels (`predicted.manual.celltype`) mapped by name to coarse types (`vdata.py`).
* **Atlas (external reference):** the Stephenson et al. 2021 COVID-19 PBMC CITE-seq extract of the generality
  benchmark (E-MTAB-10026; 118 patients from three UK sites; 600 cells per patient; TotalSeq-C panel of 192
  antibodies). Its labels (`initial_clustering`) are mapped to the same coarse types.
* **Cell types:** B, CD4 T (including regulatory T cells), CD8 T, NK and CD14 monocytes, the five types that the
  atlas holds with enough cells per patient for its populations (at least 10 cells in each assay half); CD16
  monocytes and dendritic cells average 15 and 7 atlas cells per patient and are left out of both data sets.
  Cells of other or unlabelled types are left out.
* **Features:** 200 genes chosen from the atlas's RNA-half cells by the benchmark's binned-dispersion rule (the
  genes of the atlas extract, all measured in the validation data); proteins: the 111 validation antibodies matched
  to an atlas antibody by name or by target gene (`vdata.match_antibodies`; isotype controls never matched) that are
  detected in at least 1% of the atlas's protein-half cells. RNA: log(1 + 10^4 counts / library); protein:
  centred log-ratio over the panel. The panel is fixed for every fold and uses no statistic of the validation data.

## Folds and roles

Every validation sample is held out once (34 folds). Within its pool, samples are ordered by
`validation-order-v1|<sample>`; the held-out sample at position i has paired samples at positions i+1 and i+2
(cyclic), so each fold is a small paired study of two samples of one processing pool. Cells are assigned by hash:

* **Paired reservoir:** the paired samples' cells selected by the development barcode hash (`hybrid-reservoir-v1`,
  25%; about 1,500-2,200 cells). Paired draws come from it.
* **Held-out cells:** all cells of the held-out sample, split into two sub-halves by `validation-half-v1|<cell>`.
* **Unpaired references** (RNA from part-0 cells and protein from part-1 cells, `validation-part-v1|<cell>`; a cell
  contributes one modality):
  * *atlas* (primary): all atlas cells of the five types (63,796 cells, 566 patient-by-type populations);
  * *other pools*: every validation cell of samples in other pools whose donor does not appear in the held-out
    sample's pool (other donors processed in other batches; 81,772-105,238 cells);
  * *own samples*: the non-reservoir cells of the two paired samples (the same samples as the paired cells).

No held-out cell is in any reference, and no cell of a donor of the held-out sample's pool is in the atlas or
other-pool references.

## Estimation

Populations are cell type x sample. In every draw of B paired cells, each population's drawn cells are replaced by
their n-1 orthonormal Helmert contrasts (`posthoc_contrasts.contrasts`; populations with one drawn cell contribute
none), scaled by the contrasts' pooled standard deviations. The estimator (two-sided block James-Stein in the
reference's latent RNA and protein eigenbases, blocks RNA 1-5, 6-20, 21-60, rest x protein 1-3, rest) and its
condition map (k = 120, shrunk by the split-half reliability of the reference) are those of the benchmark,
unchanged, with the reference's unpaired summaries.

Arms (all draws of a budget use the same paired cells):

| Arm | Paired cells | Reference |
|---|---|---|
| `paired_only` | contrasts | none: best at each budget of James-Stein, SCOSE, FCOSE, cross-validated low rank, pooled or per cell type (at least 6 contrasts of the type) |
| `atlas` (primary) | contrasts | atlas |
| `other` | contrasts | other pools |
| `own` | contrasts | own samples |
| `own_poolstd` (secondary) | centred on the own-sample reference's population means, scaled by its SDs (the design of the earlier tests) | own samples |
| `atlas_dfcentre` (secondary) | centred on their own population means with the factor sqrt(n/(n-1)) (the deployment test's prespecified centring) | atlas |

Budgets 25, 50, 100, 200, 400 and 800 paired cells; five draws each (seed `[20261202, fold, budget, draw]`).

## Endpoint and hypotheses

The endpoint is the noise-unbiased recovered fraction of within-cell-type gene-protein cross-correlation of the
held-out sample, pooled over cell types and folds (as in the deployment test). Paired cells needed to reach a target
are read from the running maximum of each curve by log-linear interpolation; a curve that does not reach the target
counts as needing 800 paired cells, so a ratio with such a numerator is a lower bound. The primary target is half of
the best recovery that any arm reaches; secondary targets a quarter and three quarters of it.

Intervals: 2,000 bootstrap resamples of donors (18; a donor carries its held-out samples), seed 20261203.

* **V1 (primary):** paired cells needed by paired-only estimation / by the atlas arm at the primary target; 95%
  lower bound above 1.
* **V2:** the atlas arm's recovered fraction has 95% lower bounds above zero at 50 and at 100 paired cells.
* **V3:** as V1 with the other-pool reference.
* **V4 (non-inferiority):** paired cells needed with the atlas / with the own-sample reference; 95% upper bound
  below 1.25.

Every hypothesis is reported whatever its outcome.

## Descriptive analyses (prespecified, no thresholds)

* Recovered fractions of every arm at every budget with 95% intervals; the ratios above at the secondary targets;
  the ratios of the secondary arms; per-pool results.
* **Degrees of freedom of the paired draws:** for every budget, drawn cells, populations with at least one cell,
  singletons, contrast rows (drawn cells minus populations) and rows kept by the df-corrected centring.
* **Calibration of the noise term** at 25, 50, 100 and 200 paired cells, in the atlas bases: for each block, the
  mean over draws of the estimated noise trace of the block mean, against the variance of the block mean across the
  five draws (corrected by the finite-reservoir factor 1 - B/N), for contrasts and for the df-corrected centring.
