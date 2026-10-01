# Analyses for the second review (after scoring)

Written on 30 September 2026, in answer to a referee report on the submitted manuscript, before any quantity below
was computed. Every test in the paper had been scored; every analysis here is therefore post hoc, specified before
its own quantities were computed. `freeze.py` hashes this plan, the scripts, every module they import from
`extension/` and the data files into `results/freeze.json` before any script reads a held-out cell, a truth cell
or a result of this plan; smoke runs use synthetic values or development data only.

What was known when this plan was written: every published result, including the validation test
(`validation/results/summary.json`), its readout, the alternatives and calibration of the readout
(`validation/posthoc_review/results/review.json`), the robustness analyses (`validation/results_posthoc/`: the
skewed atlases lost the saving with condition maps, and at 800 paired cells the unmapped estimates of the full and
skewed atlases recovered similar amounts), the bone-marrow test and its reanalyses, the benchmark, and the
prospective test of the law (10 of 29 problems estimable, from four data sets). Not known: any quantity defined
below. The draw-scoring check (`extension/checks/draw_scoring.py`) verifies published curves and is not part of this
plan.

Common settings: the validation's folds, panels (200 genes, 111 proteins), references, budgets and seeds; five
cell types (B, CD4 T, CD8 T, NK, CD14 monocytes); a held-out unit is one cell type in one held-out sample, scored
when each of its hash sub-halves holds at least 20 cells (the readout's rule). Intervals are percentile intervals
from resampling donors (validation), held-out batches (bone marrow) or data sets (prospective test), with the seeds
given in the scripts. Nothing below changes a published estimate.

## A. Calibration of the noise estimate after contrasts (`noise_calibration.py`)

**Question.** Proposition S9 assumes independent product rows. The validation and the bone-marrow reanalysis centre
paired cells by Helmert contrasts and scale them by standard deviations estimated from the same contrasts. Contrasts
are uncorrelated, but their products can be dependent. Is the noise estimate that sets each block's shrinkage
calibrated for this pipeline?

**Design.** For every fold of the validation test (34; atlas axes) and of the bone-marrow test (12; other-site axes,
the reanalysis with contrasts), and each budget B in 25, 50, 100, 200, 400, 800: 500 independent resamples of B
cells drawn with replacement from the fold's reservoir of paired cells (independent draws from the reservoir's
cells; generator `[20261301, study, fold, B, r]`). Each resample is processed by the complete pipeline: population
keys, Helmert contrasts in draw order, scaling by the contrasts' pooled standard deviations, rotation into the
reference's RNA and protein eigenbases, and the eight blocks (RNA eigen-coordinates 1-5, 6-20, 21-60, the rest;
protein 1-3, the rest). Per block b: the block mean m_b and noise estimates of tr Cov(m_b):

1. `helmert`: the implemented estimate, from Helmert contrasts in draw order;
2. `helmert_shuffled`: the same after a random reordering of the cells within each population;
3. `orthonormal`: from a random orthonormal contrast basis of each population (Haar on the sum-zero subspace);
4. `jackknife`: delete-one-cell jackknife of m_b, removing a cell from its population's centred cross-product sum
   (scaling held fixed), a dependence-aware alternative;
5. `own_centring`: rows = cells centred on their population means and multiplied by sqrt(n/(n-1)), the bone-marrow
   test's original centring (for reference).

m_b does not depend on the order or the contrast basis.

**Endpoints.** Per study, budget and block: the calibration ratio, the mean over resamples of each estimate divided by
the variance of m_b across resamples, pooled over folds (sums of numerators over sums of denominators); the same
summed over blocks. Unrounded values with 95% intervals from 2,000 two-level resamples (donors or held-out batches,
then resamples within folds; generator 20261302). Sensitivity to ordering, for the first 100 resamples of each fold
and budget: 20 random within-population orderings (generator `[20261303, ...]`); the median over resamples of the
coefficient of variation of `helmert` across orderings, and the median range of the block's shrinkage factor
(1 - t_b/||m_b||^2)_+ across orderings. Consequence for recovery: the recovered fraction of the validation's atlas
estimate (condition maps applied) with `helmert` and with `jackknife` noise, on the same resamples, pooled over
folds, scored against each fold's held-out truth.

**Theory.** Proposition S12 (Supplementary Note 13) gives the exact expectation of the noise estimate after
orthonormal contrasts for independent cells with finite fourth moments: E t - Var(mean) = [sum over populations of
m_p (mu_p - mu)^2 - sum of kappa_p (m_p^2/n_p - F_p)] / (N(N-1)), with m_p = n_p - 1 contrasts of population p,
N = sum of m_p, mu_p the population's cross-covariance, kappa_p the fourth cumulant of the pair and F_p the sum of
fourth powers of the contrast coefficients; for Helmert contrasts m_p^2/n_p - F_p = O(log m_p). The endpoints above
are its empirical counterpart, including the sample-derived scaling that the proposition holds fixed.

## B. Reference axes and condition maps at small budgets (`validation_review.py`)

**Question.** How much of the atlas's benefit comes from its axes (denoising coordinates) and how much from its
condition maps (transfer from the pooled estimate to each cell type), at the budgets where the saving was measured?

**Arms** (validation folds, draws and seeds; budgets 25, 50, 100, 150, 200, 400, 800), for the atlas, the other
pools and the own samples: `mapped` (the published arms, regenerated and checked against the stored means) and
`axes` (the same block estimate along the reference's axes, given to every cell type without a map). For the atlas
also `map_only`: paired-only single-block James-Stein pooled over types, passed through the atlas's condition maps.
With the published paired-only arm `js/pooled` (neither axes nor maps) these form a two-by-two design.

**Endpoints.** Recovered fraction pooled over folds at every budget with donor-bootstrap intervals (2,000
resamples; generator 20261304); paired cells to reach the validation's primary target (28.34%); savings over the
paired-only envelope.

## C. Reference reweighting under composition shift (`validation_review.py`)

**Question.** A condition map passes the pooled paired estimate through the reference's pooled covariance, which
weights cell types as the reference does, whereas the paired contrast estimate weights them by their contrast rows.
Does matching the weights rescue the atlases whose composition was skewed?

**References.** The full atlas and the two skewed atlases of the robustness analysis (`few_B_mono`: B cells and
CD14 monocytes from 15 patients; `few_T`: CD4 and CD8 T cells from 15 patients; other types from all 118).

**Maps.** For type weights w_c, the reference's pooled latent RNA covariance S is moved to S + sum_c (w_c - w0_c)
S_c, with S_c the reference's per-type latent covariances (pooled-SD units) and w0_c the reference's own share of
RNA cells of type c; it is rescaled to the reweighted pooled standard deviations (the square root of
sum_c w_c r_c^2 / sum_c w0_c r_c^2, r_c the type's standard deviation over the pooled one, for RNA and protein).
Its leading 120 eigenpairs give the maps exactly as the published ones are built (`run_study.Context.set_maps`:
G_c = S'_c P_k + Q_k, shrunk by the reference's split-half reliability, with the type's scale factors); with
w = w0 they are the published maps, which the script checks. `reweighted`: w_c = the draw's contrast rows of type c
(sum over its populations of n_p - 1) over all contrast rows, the weighting of the pooled contrast estimate. Per
reference: `mapped` (published maps), `axes` and `reweighted`.

**Endpoints.** As in B, for each reference and map; the full atlas's published curve and paired cells are the
benchmark. A skewed atlas is rescued at a budget if the reweighted map's recovered fraction lies within the full
atlas's 95% interval; paired cells to the primary target with intervals.

## D. One study, one draw (`validation_review.py`, `atlas_bootstrap.py`)

**Question.** The validation's intervals resample held-out donors with the fitted predictions averaged over five
draws of paired cells and one atlas. How uncertain is the saving for one new study, which draws paired cells once?

**D1. Draw variation.** From the per-draw scores of B and E: curves and paired cells needed for each draw index
separately (1-5), and 2,000 two-level resamples (donors; then one draw index per fold, shared by all arms;
generator 20261305). Ratios: paired-only envelope over atlas (V1) and reference-regression family (the best of its
six forms at each budget, as in `review.json`) over atlas (K2).

**D2. Atlas variation.** 10 bootstrap replicates of the atlas's patients (drawn with replacement, a patient drawn
twice contributing two populations per type; generator `[20261306, replicate]`); for each, the atlas context is
rebuilt and the atlas estimator and the four non-pooled reference-regression forms (ridge with paired or atlas Gram,
mapped or refitted per type) are refitted on the first draw of every fold at 50, 100, 200, 400 and 800 paired cells;
the paired-only envelope is the regenerated first-draw one (section B). K2 and V1 per replicate, and the
distribution over replicates combined with donor resampling (200 donor resamples per replicate). A baseline
replicate takes every patient once through the same construction. Budgets and the number of replicates are fixed by
run time (about six minutes per replicate).

**Conditioning.** Reported with what remains fixed: the validation study, the atlas study, the panel and the fold
design (paired samples reused as other folds' held-out samples).

## E. Named relationships (`validation_review.py`)

**Question.** Beyond counts of reproducible associations, what named within-type relationships become recoverable at
low paired budgets, how consistently across donors, and does the atlas-assisted estimate beat atlas-assisted
reference regression on them?

**Relationships** (fixed from prior knowledge of blood cell states; genes and proteins in the panels; expected sign):

| | Cell type | Genes (program) | Protein | Sign |
|---|---|---|---|---|
| R1 | CD14 monocytes | interferon-stimulated: ISG15, IFI6, IFI44L, MX1, OAS1, OAS2, HERC5, XAF1, EPSTI1, IFITM3, STAT2, IRF9 | CD169 (SIGLEC1) | + |
| R2 | CD14 monocytes | MHC class II: HLA-DRA, HLA-DRB1, HLA-DRB5, HLA-DQA1, HLA-DQB1, HLA-DPA1, HLA-DPB1, HLA-DMA, HLA-DMB | HLA-DR | + |
| R3 | B cells | MHC class II (as R2) | HLA-DR | + |
| R4 | CD8 T cells | cytotoxic: GZMB, GZMH, PRF1, GNLY, NKG7, FGFBP2, CST7, KLRD1, ADGRG1 | CD57 | + |
| R5 | CD8 T cells | cytotoxic (as R4) | CD27 | - |
| R6 | NK cells | cytotoxic: GZMB, PRF1, FGFBP2, SPON2, GNLY, NKG7 | CD16 | + |
| R7 | NK cells | cytotoxic (as R6) | CD56 | - |
| R8 | CD4 T cells | LEF1 | CD45RA | + |

A relationship's value in a held-out unit is the mean over its genes of the within-type correlation with the
protein; a prediction's value is the same mean of the predicted matrix.

**Endpoints** per relationship, arm and budget (primary budgets 50, 100 and 150): the recovered fraction of the
relationship, sum over units of mean over draws of (2 c t - c^2) over the sum of t_A t_B (sub-halves); the fraction of
units in which the prediction has the sign of the held-out value, among units whose two sub-halves agree in sign;
the mean absolute error; and, per donor (averaging its units), the difference in absolute error between the atlas
estimator and the reference-regression family and between the atlas estimator and the paired-only envelope, with
the number of donors favouring each and donor-bootstrap intervals (generator 20261307). Families (paired-only,
regression) take their best member per relationship and budget on the pooled error, which favours the comparators.

## F. Calibration of the association test (`readout_checks.py`)

**Question.** Is the Fisher-z test that defines reproducible associations calibrated under the actual
distributions, down to 20 cells per sub-half?

**Design.** In every scored unit and sub-half, protein vectors are permuted across cells 20 times (generator
`[20261308, fold, type, half, k]`), which keeps both marginals and destroys the pairing. Endpoints: the rate of
|z| >= 2.576 (nominal 1%) and >= 1.960 (5%) over all gene-protein pairs, overall and by cells per sub-half (20-39,
40-79, 80-159, 160 or more); the rate at which both sub-halves, permuted independently, pass with the same sign
(nominal 0.005%), against the observed rate of reproducible associations; quantiles of the null P-values.

## G. Finer cell states and technical covariates (`validation_review.py`)

**Question.** Does the atlas's gain in the readout remain when the held-out truth is freed of structure within the
broad types?

**Truth variants** (held-out cells only; predictions unchanged): `primary` (published: centred within the broad type);
`fine` (centred within the authors' finer states, e.g. naive and memory B cells; states with fewer than two cells
dropped); `technical` (within the broad type, RNA and protein residualized on log RNA library size, log protein
library size and 10x lane indicators); `both`. Centring and residualization are done separately in all cells of the
unit and in each sub-half, as for the published truth. Reproducible associations are redefined under each variant by
the readout's rule, with Fisher's z scaled by sqrt(n - 3 - k), k the number of terms removed beyond the mean.

**Endpoints.** E1 (direction of reproducible associations), E2 (reproducible associations among the 100 largest
predictions) and the recovered fraction, for the atlas estimator, the paired-only envelope and the regression family,
at every budget, with donor-bootstrap intervals (generator 20261309); the atlas's differences from each comparator
at 50, 100 and 150 paired cells.

## H. Dataset-aware inference for the law (`law_review.py`)

**Question.** The prospective test's rank correlation was tested by permuting problems, although panels of 100,
200 and 400 features from one data set are nested. What remains with data sets as the units?

**Analyses** at recovered fractions 0.5 (primary), 0.3 and 0.7, from the published predictions and observations:
(i) per data set, the median of |log2(observed/law)| and of log2(observed/law); (ii) leave one data set out: the
median error without each data set; (iii) the rank correlation of data-set means of log predicted and log observed
saving, with its exact permutation P-value over data sets; (iv) a permutation test of the problem-level Spearman
correlation that permutes predictions only within data sets (all within-data-set permutations when at most 10,000,
else 10,000 random ones; generator 20261310); (v) the slope of log observed on log predicted with data-set fixed
effects and its interval from resampling data sets (2,000; generator 20261311); (vi) the median error with intervals
from resampling studies instead of data sets (mouse lymphoid CITE-seq: sln111 and sln206; fly visual neurons:
fafb_vpn and banc_vpn; every other data set its own study). Non-estimable problems are tabulated with the reason
(target below both curves at the smallest budget; a curve not reaching the target).

## I. End-to-end pilot design (`law_review.py`)

**Question.** Does a pilot-chosen total budget, counting the pilot, achieve the chosen accuracy?

**Design.** For every one of the 29 problems, target a in 0.3, 0.5, 0.7 and pilot size m in 50, 100, 200 (m at most
a fifth of the reservoir): 100 replicates (generator `[20261312, data set, panel, a, m, k]`). A replicate draws m
pilot cells at random from the reservoir, estimates each block's signal and noise from them (Proposition S11c), and
sets the total budget B* = max(m, ceil(law's paired cells for block shrinkage at accuracy a)); if the pilot's total
signal estimate is not positive, or B* exceeds the reservoir, the budget is the whole reservoir and the replicate is
flagged. It then draws B* - m further reservoir cells, fits block James-Stein along the unpaired axes to all B* cells
(pilot included) and scores the recovered fraction against the truth cells. Reference design: the law with block
moments of all reservoir cells, fitted and scored the same way. Endpoints per target and pilot size, over problems
(each problem weighted equally) and per data set: the median achieved recovered fraction, the fraction of
replicates reaching the target and reaching it within 0.05, the median ratio of the chosen total to the paired cells
the observed curve needs (where it crosses), the flagged fraction, and the median total budget.

## J. Fixed-accuracy comparisons in the benchmark (`benchmark_review.py`)

**Question.** The summary across data sets combines dataset-specific targets, censored ratios and the running
maximum. What do comparisons at fixed accuracy show, with non-crossing cases left unresolved?

**Design.** From the published curves of the nine benchmark data sets and the external test, at recovered fractions
0.05, 0.10, 0.15, 0.20, 0.30 and 0.40: paired cells for the estimator (`bjs2/mapped`; the external test's
prespecified arm) and the paired-only envelope, (i) on the raw curves (first budget reaching the level, log-linear
interpolation from the budget before; no running maximum) and (ii) with the running maximum. Each comparison is
estimable (both reach the level between two budgets), unresolved above (a curve never reaches it), or unresolved below
(both at or above it at the smallest budget). Ratios with 95% intervals from 2,000 resamples of held-out units
(the benchmark's generator) for estimable comparisons; at each level, the geometric mean over estimable ratios and
the counts of each kind of unresolved comparison.
