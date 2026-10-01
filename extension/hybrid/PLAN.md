# Unpaired perturbation programs plus a small paired calibration set (development)

Written on 27 September 2026, before any arm of this comparison was computed.
Both screens (Frangieh et al. 2021, Papalexi et al. 2021) have been opened by
earlier analyses and serve here as development data. Nothing in this
comparison is confirmatory. It decides whether a hybrid estimator merits a
new frozen external test.

## Question

Can within-condition RNA-protein dependence be recovered with far fewer
paired cells by learning programs from unpaired perturbations and using a
small, strictly separated set of paired calibration cells to choose how much
of each program to transfer and to estimate the dependence that unpaired data
leave unresolved?

## Data and splits

Features, preprocessing, eligibility and evaluation groups are those of
`extension/perturbation` (Frangieh: 173 training targets, 141 held-out
target-by-condition groups from 51 targets, non-targeting cells of 3
conditions) and `extension/perturbation_replication` (Papalexi:
leave-one-target-out over 21 test targets, replicate in the role of the
condition, non-targeting cells of 3 replicates).

Training cells of eligible populations (the earlier rule, applied to all
training cells) are divided by a fixed hash of the cell barcode (salt
`hybrid-reservoir-v1`): 25% form the paired calibration reservoir and 75% the
unpaired pool; populations with fewer than 10 pool cells in either assay half
are dropped from all arms. Pool cells keep their original assay
half (RNA from part 0, protein from part 1), so no pool cell contributes both
assays. Calibration cells never enter unpaired summaries, and held-out cells
enter neither. In Papalexi the reservoir of each fold contains only cells of
the fold's training targets.

Paired budgets: B = 50, 100, 200, 400, 800, 1600, 3200 cells (Frangieh; 20
draws per budget) and B = 50, 100, 200, 400, 800, 1600 (Papalexi; 10 draws per
budget and fold). A draw is a uniform sample without replacement from the
reservoir, and every arm that uses paired cells uses the same cells in a draw.

## Arms

All arms estimate the within-group gene-protein cross-correlation of each
evaluation group. Units: features divided by pooled within-population standard
deviations of the unpaired pool. Recipient inputs are the group's adaptation
cells, each assay separately, as before.

Unpaired only (no paired cells):

* `current`: the frozen estimator (condition-centred ridge ecological
  regression, penalty by cross-validated prediction of population means),
  transferred in closed form.
* `splitup`: SplitUP-style estimation (Schur et al. 2026): cross-fold
  (infinite-split) debiased Gram matrix of population RNA means, cross-moment
  with protein means from disjoint cells, l1-penalized GMM per protein with the
  penalty chosen by three-fold cross-validation over targets on the held-out
  moment residual, post-selection refit; transferred in closed form.
* `noise_aware`: population RNA means computed separately for each guide; the
  between-population Gram matrix is formed only from products of different
  guides (unbiased for noise and blind to guide-specific effects);
  programs are its eigendirections whose eigenvalues exceed the 95th
  percentile of the largest eigenvalue under random sign flips of guide
  deviations (50 flips); the channel is the moment regression restricted to
  these programs; transferred in closed form.

Paired calibration cells only (paired cells centred on unpaired population
means):

* `paired_corr`: mean cross-product of the B cells.
* `paired_js`: the same, positive-part James-Stein shrinkage towards zero with
  the noise variance estimated from per-cell products.
* `paired_lowrank`: truncated singular value decomposition, rank by two-fold
  cross-validation within the B cells.
* `paired_eot`: the shrunk estimate converted to a Gaussian entropic
  interaction on the pooled unpaired marginals and transferred in closed form
  to each recipient (the Gaussian analogue of bridge-learned entropic
  transport, as in Champollion).

Paired cells with unpaired information (coefficients in the eigenbases of the
pooled unpaired RNA and protein correlation matrices; positive-part
James-Stein shrinkage with noise variances from per-cell products):

* `block_js`: shrinkage towards zero separately in four blocks of RNA
  eigen-coordinates (1-5, 6-20, 21-60, the rest), each with all protein
  coordinates.
* `combine_global`: the `noise_aware` transfer plus the paired deviation from
  it, shrunk with a single weight (the established way to combine a
  prediction with a paired correction).
* `hybrid` (proposed): each `noise_aware` program defines an RNA score; the
  protein slopes on these scores are estimated from the paired cells and
  shrunk, program by program, towards the slopes learned without pairs
  (positive-part James-Stein), so a program whose unpaired slope agrees with
  the paired cells is taken from unpaired data and one that disagrees is
  re-estimated; the remaining cross-covariance is estimated as in
  `block_js`. The program part is transferred to each recipient through its
  own RNA covariance with the program scores.
* `hybrid_current` (secondary): the same with the `current` channel's
  supported directions and slopes.
* `hybrid_pc` (control): the same with the leading eigenvectors of the pooled
  unpaired RNA correlation, as many as `noise_aware` programs, and slopes
  shrunk towards zero, i.e. programs without perturbation information.

Context: `paired_all` (the pooled within-population cross-correlation of all
paired training cells, per condition, as the earlier "transferred
correlation").

## Endpoint

Recovered fraction pooled over evaluation groups (noise-unbiased; independence
0), averaged over draws, for all genes and proteins and for the prespecified
blocks of `extension/perturbation/PLAN_programs.md` (interferon block, its
complement, and each other Hallmark program with at least 5 panel genes x all
proteins). Uncertainty: 1,000 bootstrap resamples of held-out targets (seed
20261027).

## Decision rule

The hybrid is pursued (and a frozen external test proposed) only if, in both
screens:

1. at B = 100, 200 and 400, its recovered fraction over all genes and proteins
   exceeds that of the best comparator at the same budget (highest point
   estimate among all arms that use no perturbation-derived program: the
   unpaired-only, paired-only, `block_js`, `combine_global` and `hybrid_pc`
   arms), with the lower 95% bootstrap bound of the difference above 0; and
2. at B = 100 and 200, the comparators need at least twice as many paired
   cells to match it (log-linear interpolation of the upper envelope of the
   comparators over budgets).

Otherwise the hybrid is not pursued and no confirmation screen is proposed.
Reported regardless: recovery by program, non-targeting cells, the number of
programs selected, the unpaired-only arms against each other, and the value of
unpaired marginal information (`eb_marginal` against paired-only arms).

## Deviations

1. After a smoke test (two Papalexi folds and the non-targeting cells, budgets
   50 and 100, two draws; logs/smoke_papalexi.log), the empirical-Bayes arms
   were replaced by the James-Stein arms above, before any full run. In the
   smoke test, every unpaired transfer was far worse than independence in
   Papalexi (noise_aware -287%, current -129% over six test groups), the
   marginal-likelihood prior scale was zero at both budgets, so eb_marginal
   predicted zero, and the transfer's error spread from the program
   coordinates into the others, where a shared prior scale kept it
   (eb_global -207% to -381%, hybrid -146% to -261%). The revised hybrid takes
   only the program directions and a prior for their slopes from the unpaired
   data, and `hybrid_pc` was added as its control. Paired-only arms and
   unpaired arms are unchanged. The decision rule now names its comparators
   explicitly.

## Post hoc analyses (added after the Frangieh comparison had been scored)

`posthoc_baselines.py`, same folds, reservoir and draws: reference-regression
baselines that use the recipient's RNA covariance (`ridge_paired`, ridge on the
paired cells; `ridge_unpaired`, the same with the pooled unpaired RNA
correlation in the normal equations), and diagnostic variants of the hybrid
and of its control whose program part uses the pooled unpaired RNA
correlation instead of each recipient's (`hybrid_pooled`,
`hybrid_pc_pooled`). They do not enter the decision rule.
