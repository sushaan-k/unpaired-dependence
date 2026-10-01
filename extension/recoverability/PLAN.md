# Recoverability test on an unseen tissue: protocol

This protocol is frozen by `freeze.py`, which records the SHA-256 of this file
and of the code and the system time in `results/freeze.json`. Nothing computed
from bone-marrow counts precedes that record. Times below come from the system
clock of the machine that ran the analysis.

## Why this test

A review of the manuscript found that (i) the exact identified set of the
frozen channel was empty for every recipient, so the reported predictions came
from a repaired model rather than from the identified set; (ii) the "identified
subspace" of the analyses is a penalty-dependent effective supported subspace,
not the span of population means in the theorem; (iii) the RNA-only share of
variance on that subspace was not a validated predictor of recoverability,
because its correlation with error mainly separated cohorts and analysis types;
and (iv) coverage and channel compatibility are separate requirements. The
review asked for one focused repair: separate insufficient population
variation from incompatible between-population and within-cell relationships,
test whether these diagnostics explain failures better than coverage alone and
than simple composition, sample-size and distribution-shift controls on data
not yet seen, and show a practical use for choosing a small paired reference.

## Development (already-scored data; exploratory)

All development analyses used the untouched blood cohort and colon, which were
scored in earlier sessions, and are labelled exploratory:

* `dev_compatibility.py`: the empty identified set is not caused by the protein
  log-ratio constraint or by covariance shrinkage (largest singular value of the
  whitened identified block, median 2.8 for total analyses, 2.7 after removing
  the log-ratio null direction, 2.8 without shrinkage).
* `dev_units2.py` (uncertainty in `dev_bootstrap.py`): population means average out RNA sampling
  noise, so the channel they identify acts on latent expression. Replacing the
  recipient RNA correlation matrix by its latent part (count splitting,
  `noise.py`) lowered the statistic from 3.5 to 1.5 (untouched cohort, total),
  from 3.8 to 0.7-0.9 (within types) and from 2.6 to 0.2 (colon). Errors of
  closed-form transfer were unchanged for total analyses and fell within types.
* `dev_typecheck.py`: a recipient's cell types are populations whose RNA and
  protein means are measured separately; a channel's prediction of the
  recipient's between-type cross-covariance can therefore be checked against the
  label-based estimate without paired cells.
* `dev_units2.py`: the label-based between-type estimator (type means in each
  assay) had lower error than every pairing-free channel for total analyses
  (untouched cohort 0.10-0.16; colon 0.38-0.95).
* `dev_reference.py`: a Neyman allocation of paired cells over types gave lower
  within-type error than random or balanced selection, and adding unpaired
  between-type means to paired within-type estimates gave lower total error
  than paired cells alone.
* `fit_failure_model.py`: the failure model below was fitted on 1,776
  development pairs (analysis x channel with a nonempty supported subspace).

## Data

GSE194122 (NeurIPS 2021 multimodal benchmark), bone-marrow mononuclear cells
profiled by CITE-seq: 90,261 cells in 12 samples from 9 donors at 4 sites (donor
15078 was profiled at all 4 sites). Bone marrow has not been used in this
project. `extract_bmmc.py` split every sample's cells into an adaptation and a
scoring half by SHA-256 of `bmmc-recoverability-v1|batch|barcode` and wrote the
two halves to separate files at 22:50 on 26 September 2026 (EDT); their SHA-256
values are in `results/bmmc_seal.json`. To write them it read the count matrix
and computed library sizes and an integer check, nothing else. Otherwise only
cell metadata (annotations, donors, batches) and feature names were read before
the freeze (`define_bmmc_units.py`).

Features: the 201 of the 206 frozen genes present among the deposited features
and the 88 of the 122 frozen protein targets whose antibody name in this panel
equals the target name. Preprocessing is unchanged: RNA log1p(1e4 x count /
library), library = sum over deposited genes; protein log1p(count) minus its mean
over the 88 targets of the cell.

## Units

Per sample (`define_bmmc_units.py`, `results/bmmc_units.json`): the total
analysis (all cells); the within-coarse analysis (cells centred on their type in
both assays, 9 coarse types defined from the deposited names in
`extract_bmmc.py`); and the within-fine analysis (45 deposited types). Types
enter an analysis if they have at least 10 cells in each half. 36 analyses.

## Channels

`training.py`, `fit_bmmc_channels.py`; every channel is refitted on the 201 x 88
panel by the frozen pairing-free rules (disjoint RNA and protein halves of each
training population; pooled within-population SD units; ridge scale from
{1e-4, ..., 1} with patient-grouped three-fold cross-validation and the edge
rule):

* `frozen`: 118 blood patients;
* `P1`: blood patient x cell-type groups, centred within type;
* `lc00`-`lc39`: the 40 learning-curve subsets of 8, 16, 32 and 64 patients;
* `perm00`-`perm19`: donor-aware derangements of protein summaries across
  patients (a wrong channel by construction);
* `colon`: 21 colon biopsies (colon adaptation cells), folds grouped by donor.

The effective supported subspace S of a channel is spanned by the eigenvectors
of its between-population RNA-mean covariance G whose ridge factor
g/(g + lambda) is at least 1/2. A channel with empty S predicts zero and
abstains; abstentions are reported by family and excluded from failure analyses.

## Predictions

For each analysis and channel (`predict_bmmc.py`), closed-form transfer of the
channel's interaction with the recipient's adaptation-cell protein correlation
matrix and (primary) the latent RNA correlation matrix, or (secondary, as in the
earlier frozen protocol) the measured RNA correlation matrix. Latent RNA
correlation: each RNA count is split into two binomial halves (seed 20260926);
the Spearman-Brown full-depth reliability of each gene's two half-depth
log-normalized values (centred within type for within analyses) gives its noise
share nu, and the latent matrix is R_x - diag(nu R_x,jj), projected onto the
positive semidefinite cone. Reference arms: the label-based between-type
estimator for total analyses (coarse-type means in each assay), the recipient's
own adaptation pairs (tuned ridge, a ceiling that the setting does not permit)
and independence. SHA-256 digests of all prediction matrices are recorded
before the scoring file is opened.

## Diagnostics and controls

Computed from adaptation cells without any cross-assay moment:

* coverage: tr(P_S R P_S) / tr(R) with the latent RNA correlation matrix R;
* compatibility: ||F||, the largest singular value of
  F = Q_N^T R^1/2 W^T R_y^-1/2 (Q_N an orthonormal basis of R^-1/2 S); the set
  of cross-covariances that agree with W on S and give a valid joint law is
  nonempty if and only if ||F|| <= 1;
* type agreement: cosine similarity between the channel's prediction of the
  recipient's between-type cross-covariance (recipient between-type RNA
  covariance times the channel on S) and the label-based between-type
  cross-covariance (coarse types, all adaptation cells of the sample);
* controls: composition (between-type share of standardized variance, coarse
  types, averaged over assays; 0 for within analyses), sample size (log cells
  in the analysis; log training populations), covariance shift (one minus the
  cosine similarity of off-diagonal RNA correlations between the recipient and
  the channel's mean training population).

## Failure model (frozen)

Failure: relative error of the primary prediction >= 1 (no better than
independence). Scores (`results/failure_model.json`, fitted on development
pairs only): the primary model is a logistic regression on coverage, log(||F||
+ 0.001), type agreement and composition; the label-free model uses coverage
and log(||F|| + 0.001). Coefficients, centring and scaling are applied
unchanged.

## Endpoints and decision rules

Pairs are all analysis x channel combinations with nonempty S. AUCs are for
failure; differences between AUCs have 95% intervals from 2,000 bootstrap
resamples of donors (9 clusters).

* H1 (the two failure modes add to coverage): AUC(primary) - AUC(coverage alone)
  has a lower 95% bound above 0. Secondary: the same for the label-free model.
* H2 (controls): AUC(primary) exceeds the AUC of composition, log cells, log
  training populations and covariance shift, each with a lower 95% bound above 0.
  Each control is reported separately.
* H3 (high coverage, wrong channel): among pairs with coverage >= 0.4, the AUC of
  the primary score has a lower 95% bound above 0.5. Evaluated only if there are
  at least 5 failures and 5 successes; the AUCs of type disagreement and of
  ||F|| alone are reported.
* Secondary: Spearman correlations with error; concordance of each score with
  the ordering of channels within an analysis (pairs of channels whose errors
  differ by at least 0.05); abstention rate by family; failure rate by family and
  analysis.

## Compatibility under sampling uncertainty

`bootstrap_bmmc.py`: for the frozen, P1 and colon channels and every analysis,
200 replicates resampling training populations (patients; P1 patients with all
their groups, re-centred within type; colon donors with all their biopsies) with
the channel's penalty scale fixed, and the analysis's cells. Reported: ||F||
with measured and latent RNA covariance, and the number of analyses whose basic
bootstrap lower bound 2 ||F|| - q_0.95 exceeds 1 (incompatibility beyond
sampling uncertainty). Descriptive.

## Association bounds

For the frozen channel and every analysis with ||F|| <= 1 (latent covariance):
entrywise sharp bounds of the compatible set (`compat.completion_bounds`); after
scoring, the share of target entries inside the bounds, the share whose sign the
bounds determine and the accuracy of those signs. Descriptive; the bounds assume
that the channel on S is shared by training populations and recipient cells and
leave the channel off S free.

## Paired-reference selection

Setting (`reference.py`): each sample's adaptation cells of kept coarse types are
the unpaired study and the pool from which N cells could be profiled with paired
RNA and protein; N in {50, 100, 200, 400, 800}. Strategies: random, balanced
(equal numbers per type) and guided (Neyman allocation n_t proportional to
pi_t sqrt(V_t), V_t from unpaired within-type variances). Estimator for every
strategy: stratified within-type cross-covariance centred on unpaired type
means, weighted by unpaired type shares, scaled by unpaired pooled within-type
SDs and shrunk by the positive-part James-Stein factor with the noise energy
sum_t pi_t^2 V_t / n_t. 50 draws per sample, budget and strategy (seed
20261002), recorded before unsealing. Targets: the scoring half's within-coarse
cross-correlation (primary) and its total cross-correlation over kept-type cells
(secondary; hybrid = unpaired between-type means + paired within-type estimate,
against paired cells alone with the same shrinkage).

* H6: guided has lower mean within-type error than random (and, separately, than
  balanced) at every budget, and the 95% donor-bootstrap interval of the
  sample-level difference averaged over budgets lies below 0.
* Secondary: the budget at which random selection matches guided selection; the
  hybrid's total error against paired cells alone.

## Interpretation

If H1 or H2 fails, the manuscript states that the diagnostics did not outperform
the corresponding controls on unseen data and does not claim that they predict
recoverability better. Results are reported whatever their direction. No
threshold, model, unit, channel or estimator is changed after the freeze; any
deviation is recorded below with its reason and time.

## Order of operations

1. `freeze.py` (this file and the code); 2. `predict_bmmc.py` and
`bootstrap_bmmc.py` (adaptation file only; manifest with SHA-256 of every result
file); 3. `evaluate_bmmc.py` (checks the manifest and the freeze, opens the
scoring file, recomputes every prediction and checks its digest).

## Deviations

Amendment 1 (recorded in `results/freeze_amendment_1.json`; before the scoring
file was opened). The first run of `predict_bmmc.py` stopped in sample s2d1,
within-fine analysis, because numpy's singular value decomposition (LAPACK
gesdd) did not converge for the interaction of channel lc34, a finite matrix.
Every singular value decomposition in the test now falls back to LAPACK gesvd
when gesdd does not converge (`compat.svd`); results are identical whenever
gesdd converges. `freeze.py` gained the amendment record used here. No unit,
channel, estimator, diagnostic, model, threshold or endpoint changed. Outputs of
the stopped run (the log only; no result file had been written) are kept in
`logs/predict_bmmc_run1.log`.
