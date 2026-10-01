# Perturbation test: pairing-free recovery of within-condition RNA-protein dependence

Hypothesis (from review): intervention-induced population variation exposes
directions that enable pairing-free recovery of within-condition RNA-protein
dependence.

Written after development on training targets only (`dev.py`,
`results/dev.json`, `results/dev_cf.json`) and before any statistic of a
held-out or non-targeting cell other than its library size was computed.
`freeze.py` records the SHA-256 of this file, the code, the shared code of the
other studies and the training fits in `results/freeze.json`.

## Data

Frangieh et al. 2021 Perturb-CITE-seq screen (scPerturb release, Zenodo record
7041849; SHA-256 of both files in `pdata.py`): 218,331 patient-derived melanoma
cells with CRISPR knockouts of 248 genes in three conditions (cultured alone,
with interferon-gamma, with autologous tumour-infiltrating lymphocytes), RNA and
24 antibody-derived tags. The release has no replicate or batch annotation.

Targets are read from the detected guides. A cell is *single* if all its guides
target one gene (111,453 cells) and *non-targeting* (NT) if all its guides are
non-targeting (16,319); cells with guides for several genes, with targeting and
non-targeting guides, or with no detected guide are not used. (The release's
"perturbation" column names only the first of several targeted genes and labels
mixed and guide-less cells as controls.)

## Split and seal (`extract.py`, `results/seal.json`)

* Held-out targets: every fourth of the 248 targets in SHA-256 order (salt
  `perturb-heldout-v1`): 62 targets. Every cell whose guides target a held-out
  gene is outside training and tuning. The other 186 targets are training targets.
* Within each training population (target x condition), cells in SHA-256 order
  of their names: first half part 0 (RNA moments), second half part 1 (protein
  moments). Pairing-free fits never use a cell's pairing.
* Within each held-out group (target x condition) and each NT condition: first
  half adaptation cells (inputs), second half scoring cells (endpoint), whose
  cells alternate between sub-halves A and B.
* Gene panel from training cells only: the 19 genes encoding target antibodies
  that are detected in at least 1% of training cells, then 181 genes of highest
  dispersion (detected in at least 5%, not mitochondrial or ribosomal, log
  dispersion z-scored within 20 bins of mean log expression). Proteins: the 20
  target antibodies (isotype controls excluded).
* Preprocessing as in the other cohorts: RNA log1p(1e4 x count / library), the
  library summed over all genes; protein log1p(count) minus the cell's mean over
  the 20 antibodies.
* Before this plan, held-out and NT cells were read only to write them to the
  adaptation and scoring files (library sizes were computed for all cells).

## Units and eligibility

* Training population: target x condition with at least 20 cells in each part.
* Held-out group: held-out target x condition with at least 40 adaptation cells
  (which implies at least 20 cells in each scoring sub-half).
* NT groups: the three conditions.

## Arms (`pmethods.py`, `fit.py`, `predict.py`)

Each arm predicts a group's gene-protein cross-correlation with cells centred
within the group, from the group's adaptation cells (each assay separately) and
the training fits.

Pairing-free channels use the frozen rules of `cross_study`
(`estimators.fit_pf_means`): pooled within-population SD units, weighted ridge
regression of protein means on RNA means, penalty scale chosen by three-fold
cross-validation over training targets (all conditions of a target in one fold)
from the grid 1e-4 to 1 with the edge rule; B = W' Psi^-1; closed-form transfer
with the group's shrunk RNA and protein correlation matrices.

1. **Primary: condition-stratified pairing-free channel** (`pf_within_measured`).
   Population means are centred on their condition's cell-weighted mean before
   the regression, so the channel is learned from perturbation-induced
   variation within conditions; transfer with the group's measured RNA
   correlation. Rationale: the endpoint is dependence within conditions, and
   pooling conditions applies between-condition slopes within conditions (the
   ecological fallacy of Proposition S8; the analogue of the type-centred channel
   P1 of the bone-marrow test). Measured rather than noise-corrected RNA
   correlation is primary because the correction's accuracy is under review; the
   corrected version is secondary.
2. **Matched regression controls** for the primary channel, same estimator,
   training cells and stratification, ten draws each (seed 20261011):
   *pseudo-populations* (within each condition the population labels of the
   training cells are permuted, so populations keep their number and approximate
   size but differ only by sampling) and *derangements* (every population's
   protein moments are replaced by another population's of the same condition).
3. **Paired comparators** (paired training cells of the same training
   populations): **paired closed form** (primary paired comparator: the same
   operator with an interaction fitted by penalized profile likelihood over
   training populations, penalty by three-fold cross-validation over targets);
   reference regression (ridge regression of protein on RNA pooled within
   populations, applied to the group's RNA correlation); transferred correlation
   (pooled within-population cross-correlation of training populations of the
   same condition).
4. **Independence** (zero cross-correlation).
5. Secondary: unstratified channel (`pf_measured`, `pf_latent`) with its own
   unstratified controls; primary channel with noise-corrected RNA correlation
   (`pf_within_latent`), regression-implied cross-correlation
   (`pf_within_regression`) and training condition marginals instead of the
   group's (`pf_within_condition`); paired closed form with corrected RNA
   correlation; the group's own adaptation pairs (tuned ridge regression,
   `benchmark`; not permitted in the setting).

## Endpoint

For group g with scoring cross-correlation T_g (all scoring cells) and sub-half
estimates T_A,g and T_B,g, the terms 2<C_g, T_g> - ||C_g||^2 and <T_A,g, T_B,g>
are unbiased (to first order) for ||T*||^2 - ||C_g - T*||^2 and ||T*||^2, where
T* is the group's true cross-correlation. The **recovered fraction**
RF = sum_g (2<C_g, T_g> - ||C_g||^2) / sum_g <T_A,g, T_B,g> over eligible groups
is 0 for independence, 1 for a perfect prediction and negative for predictions
worse than independence. Scoring noise is removed from both terms, so arms are
compared on the true within-group dependence. Secondary: RF by condition; mean
relative Frobenius error ||C - T|| / ||T|| (inflated by scoring noise).

Uncertainty: 2,000 bootstrap resamples of held-out targets (seed 20261013),
all conditions of a resampled target together; controls are averaged over their
ten draws inside every replicate. NT cells: 2,000 resamples of guide sets
(cells with the same detected guides) within each condition's scoring cells,
predictions fixed. There are no biological replicates.

## Hypotheses and decision rules

* **H1 (recovery).** RF of the primary channel in held-out groups has a 95%
  lower bound above 0.
* **H2 (population variation is the source).** RF(primary) minus RF(matched
  control) has a 95% lower bound above 0, separately for pseudo-populations and
  for derangements.
* **H3 (non-targeting cells).** RF of the primary channel pooled over the three
  NT conditions has a 95% lower bound above 0.
* **H4 (most of what pairing buys).** RF(primary) / RF(paired closed form) has
  a 95% lower bound above 0.5 (the criterion of the cross-study test).

The hypothesis is supported if H1 and both parts of H2 hold, and confirmed in
cells without a perturbation if H3 also holds. H4 decides whether pairing-free
recovery is comparable to recovery with paired training cells.

Descriptive: every arm's RF and interval; RF by condition; the supported
dimension, the share of the group's RNA correlation inside the supported
subspace and the compatibility statistic of both channels.

## Development (training targets only; exploratory)

Every fourth training target (salt `perturb-dev-v1`, 47 targets, 118 groups)
played the held-out role, with part-0 cells as adaptation and part-1 cells as
scoring cells, and 374 populations of the other training targets trained the
channels. RF (95% interval over targets): unstratified channel -0.30
(-0.37 to -0.24; two supported directions, both between-condition contrasts);
condition-stratified channel 0.10 (0.08 to 0.13; one supported direction);
stratified pseudo-populations -0.00 (no supported direction); stratified
derangements -0.03; paired closed form 0.38 (0.31 to 0.45); transferred
correlation 0.97 (0.92 to 1.02); reference regression 0.55; own adaptation pairs
0.24 (`results/dev_cf.json`). The stratified channel was chosen as primary after
this analysis, for the reason given above. The code was then run end to end on
the development split (all arms, the endpoint and both bootstraps, with three
development targets standing in for non-targeting cells) to check it before
the freeze.

## Deviations

None.
