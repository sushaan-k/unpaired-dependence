# Deployment test: unpaired cells from other sites and from another assay

Written 2026-09-28, before any estimator of this study was run on the bone-marrow data under this design and before
any statistic of held-out cells was computed. `freeze.py` hashes this plan, `drun.py`, the modules it imports and
the two data files into `results/freeze.json`; predictions are hashed into `results/manifest.json` before
`drun.py evaluate` reads any held-out cell.

## Checks before the freeze

The pipeline was run end to end on synthetic values with the real design (`drun.py smoke`: random counts, 400
cells per batch, two folds). On the real data, only the unpaired summaries of two folds were computed, to check
that the pools are large enough for stable condition maps (pool cells, reliabilities, map norms and the latent
eigenvalue spectrum); no paired-cell estimate and no statistic of held-out cells was computed.

## Question

In every earlier test of the semi-paired estimator, the unpaired cells came from the same samples as the paired
cells, with one modality withheld. A study that pairs a few cells of its own and borrows unpaired profiles from
other laboratories differs in three ways: the unpaired cells come from other donors, from other sites (processing,
batch) and possibly from another assay. Does the estimator keep its saving of paired cells then?

The estimator uses unpaired cells for two different things: population means and standard deviations, which centre
and scale the paired cells, and correlation matrices, which give the eigenbases, the blocks' noise structure and
the condition map. Means and scales can shift between sites and assays; correlation structure within cell types
need not. The primary arms therefore take means and scales from the paired cells themselves ("own
standardization") and only correlation structure from the unpaired cells. Borrowing the other sites' means is
scored as a secondary, descriptive arm.

## Data

NeurIPS 2021 bone-marrow data (GSE194122), as extracted for the generality benchmark (3,000 cells per batch;
SHA-256 in the freeze record):

* `bmmc_cite.npz`: CITE-seq, RNA counts of 3,000 candidate genes and 134 antibody counts (centred log-ratios),
  12 batches (site x donor) from four sites and nine donors.
* `bmmc_multiome.npz`: single-nucleus multiome, RNA counts of 3,000 candidate genes (the ATAC part is not used),
  13 batches from four sites and ten donors.

Conditions are the eight coarse cell types present in both assays (B, CD4 T, CD8 T, DC, Mono, NK, erythroid,
progenitor); cells of the CITE-seq type `other T` are left out. Donors are identified by the donor identifiers,
which are shared between the two assays.

## Folds and roles

Every CITE-seq batch is held out once (12 folds, batches in sorted order). For a held-out batch at site S:

* **Paired cells**: the other two CITE-seq batches of site S; the reservoir is their cells selected by the
  development barcode hash (`hybrid-reservoir-v1`, 25%). Paired draws come from this reservoir.
* **Held-out cells**: all cells of the held-out batch, split into two sub-halves by `deployment-half-v1|<cell>`
  for the noise-unbiased endpoint.
* **Unpaired pools** (cells never paired; RNA from part 0 and protein from part 1 by `deployment-part-v1|<cell>`):
  * *same site*: the non-reservoir cells of the two paired batches (the design of every earlier test);
  * *other sites*: all CITE-seq cells of batches at the other three sites whose donor does not appear at site S;
  * *other assay*: RNA from all multiome nuclei of batches at the other three sites whose donor does not appear at
    site S; protein from the part-1 cells of the *other sites* pool.

No held-out cell, and no cell of a donor of site S, is in the *other sites* or *other assay* pools.

## Features

Per fold, from the *other sites* pool only: 200 genes by the binned-dispersion rule of the external test among the
candidate genes of both assays (log(1 + 10^4 count / library), detected in at least 5% of cells); proteins detected
in at least 1% of cells (at most 200), as centred log-ratios. The same panel is used by every arm of the fold.

## Standardization of paired cells

* **Own** (primary): in every draw, paired cells are centred on the means of their own populations (batch x
  condition) among the drawn cells and scaled by their own pooled within-population standard deviations; centred
  values are multiplied by sqrt(n/(n-1)) for a population of n >= 2 drawn cells, and populations with one drawn cell
  are dropped.
* **Pool** (secondary): the earlier rule. Paired cells are centred on the pool's population means and scaled by its
  pooled standard deviations. For the *same site* pool the populations are the same; for the *other sites* pool the
  pool's condition means are used.

## Arms

All arms of a draw use the same paired cells.

* `paired_only`: James-Stein shrinkage, SCOSE, FCOSE and cross-validated low rank, each pooled and refitted per
  condition (at least 6 paired cells), own standardization; the envelope is the best of the eight at each budget.
* `proposed_same`, `proposed_other`, `proposed_assay`: the proposed estimator (two-sided block James-Stein in the
  unpaired eigenbases with the condition map), own standardization, unpaired summaries from the *same site*,
  *other sites* and *other assay* pools.
* `proposed_other_poolstd` (secondary): the proposed estimator with the *other sites* pool's condition means and
  standard deviations.
* `proposed_same_poolstd` (secondary): the earlier design, pool standardization with the *same site* pool.

The estimator, blocks, map rank and comparators are those of `extension/semipaired`, unchanged.

## Budgets, endpoint and targets

Budgets 25, 50, 100, 200, 400 and 800 paired cells, five draws each per fold. The endpoint is the noise-unbiased
recovered fraction of within-population gene-protein cross-correlation of held-out cells, pooled within condition
and summed over conditions and folds, as in the generality benchmark. The target is a fraction of the largest
recovered fraction that any arm reaches at any budget: 0.5 (primary), 0.25 and 0.75. Paired cells needed are
interpolated log-linearly on the running maximum of each curve; a curve that never reaches the target counts as
needing the largest budget, and a saving computed from it is a lower bound.

## Hypotheses

* **D1 (primary, other sites).** Paired cells needed by the paired-only envelope divided by those needed by
  `proposed_other`: 95% bootstrap lower bound above one.
* **D2 (other assay).** The same for `proposed_assay`.
* **D3 (non-inferiority of borrowed correlation structure).** Paired cells needed by `proposed_other` divided by
  those needed by `proposed_same`: 95% bootstrap upper bound below 1.25.

The two secondary arms, the savings at the other targets and the per-site savings are reported without a claim.

## Uncertainty

2,000 bootstrap resamples of held-out batches, stratified by site (three batches with replacement within each of
the four sites), recomputing the curves, the level, the targets and the paired cells needed. Draws of paired cells
are averaged and are not replicates.
