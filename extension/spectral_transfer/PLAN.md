# Closed-form transfer at scale: analysis plan

Written 2026-09-26 before running the full analysis. One donor (XAUT1-HS1) was
run once while developing the code; its scoring R^2 values were seen. Every
analysis here is post hoc relative to the released paper and uses public data
(Mennillo et al. colon CITE-seq, Figshare 21919356 v3, MD5-verified).

## Question

The released method reconstructs full binary pattern distributions (2^p
states per assay), which limits it to about ten markers. The response theorem
says transferred dependence is governed to first order by within-assay
covariances. Does a closed-form covariance operator transfer RNA-protein
dependence at realistic panel size, and does the paper's resolution result
(within-assay coexpression beats individual summaries) hold at that scale?

## Method under test

Continuous analogue of Eq. (1): joint density exp(x^T B y + a_k(x) + b_k(y))
with Gaussian within-assay laws. B is shared; recipients supply only separate
assays. Transfer: C = Sx^{1/2} U g(beta) V^T Sy^{1/2}. B is fitted on paired
reference donors by maximizing the concave profile likelihood with a ridge
penalty chosen by inner 3-fold leave-donors-out likelihood within the
reference (grid 1e-3, 1e-2, 1e-1 per cell). Covariances are shrunk 10% toward
their diagonals.

## Data and features

All 15,953 paired (GEX_CITE) cells of 12 donors. RNA: log1p of counts per
10,000; 200 genes with the highest variance-to-mean ratio among genes with
mean > 0.05 across all paired cells (RNA only), plus the paper's nine panel
genes. Protein: all 177 ADTs, centred log ratio per cell. Each donor's cells
are split by SHA-256 of `spectral-transfer-v1|donor|barcode` into an adaptation
half (used only as two separate assays) and a scoring half (paired endpoint).

## Designs

D1 (primary). Leave one donor out over all 12 donors: reference = the other 11
donors' paired cells.
D2 (secondary). Disease shift: reference = the 4 healthy-control donors;
recipients = the 8 ulcerative-colitis donors.

## Arms

independence; reference_cov (pooled within-donor reference cross-covariance);
reference_corr (reference cross-correlation rescaled by recipient SDs);
regression (reference RNA-to-protein regression applied to the recipient's RNA
covariance; standard imputation); first_order (Sx B Sy); variances_only
(closed form with diagonal recipient covariances); closed_form (full recipient
covariances); oracle (recipient's own paired adaptation cross-covariance;
uses pairing, reported only as a ceiling).

## Endpoints

Primary: relative Frobenius error of the predicted cross-covariance against
the scoring-half paired cross-covariance, averaged equally over recipients.
Primary contrasts: closed_form vs variances_only (resolution), and closed_form
vs regression, reference_corr and reference_cov (what is transferred).
Secondary: correlation between predicted and held-out cross-covariance
entries; R^2 of protein predicted from RNA in scoring cells, using the
recipient adaptation RNA covariance for the regression step.
Uncertainty: paired donor bootstrap (20,000 draws, seed 20260927) of mean
differences; Bonferroni across the four primary contrasts. All arms, donors
and metrics are reported regardless of direction. No arm, feature set, or
shrinkage is changed after seeing D1 results; any later addition is labelled.

## Amendment 1 (2026-09-26, before any D1 result beyond XAUT1-HS1 was seen): pairing-free design D3

Question. Can the cross-assay coupling be learned with no paired cells at all,
from how separately measured RNA and protein distributions co-vary across
samples?

Identification. Under the fixed-interaction model with free within-assay laws
(the paper's assumption), unpaired data carry no information about B: any B
reproduces any pair of marginals. Under the stronger shared-channel model
y | x ~ N(W x + b, Psi), with W, b, Psi shared across samples and the RNA law
sample-specific, the protein moments satisfy mu_y,s = W mu_x,s + b and
Sigma_y,s = W Sigma_x,s W^T + Psi, so variation across samples identifies W
(the covariance equations up to sign; the mean equations fix the sign within
their span). The shared-channel model is a special case of the fixed-interaction
model with B = W^T Psi^{-1}.

Units and data. Biopsy samples (CoLabs_sample; 21 samples of 12 donors). For
each recipient donor, W, b and diagonal Psi are estimated from the samples of
the other 11 donors using only within-assay moments of their paired cells
(pairing never used). Estimator: maximize the Gaussian likelihood of each
sample's protein moments given its RNA moments, with ridge penalty on W chosen
by inner 3-fold leave-samples-out likelihood of held-out samples' protein
moments (grid 1e-2, 1e-1, 1, 10 per cell), initialized at the ridge regression
of sample protein means on sample RNA means. (Changed from leave-one-sample-out
before any D3 computation, for run time: about 4.6 h versus under 1 h.)

Arms. pf_regression: C = Sx_k W^T. pf_closed_form: closed-form transfer with
B = W^T Psi^{-1} and the recipient's separate-assay covariances. Both are
compared with the paired-reference arms of D1 on the same recipients, halves,
features and endpoints. The result is reported whether or not it succeeds.

## Amendment 2 (2026-09-26; D1 had crashed at the second donor; only XAUT1-HS1 results seen)

- Numerical fix: donor XAUT1-HS10 has genes with zero variance in its
  adaptation half, making the RNA covariance singular. All within-assay
  covariances now add 1e-3 times their mean diagonal to the diagonal, after
  the 10% diagonal shrinkage. D1 is rerun from the start for all donors.
- Added arm cell_coupling: exact entropic coupling (log-domain Sinkhorn) of the
  recipient's adaptation RNA profiles with its adaptation protein profiles,
  treated as two unpaired clouds with uniform weights, using the same fitted B
  and kernel exp((x - mu_x)^T B (y - mu_y)). Its cross-covariance is the
  coupling-weighted cross-moment. This is the full-pattern analogue at scale
  and tests the resolution ordering cells > covariances > variances with one
  interaction. It uses the same B as the closed form (fitted by the Gaussian
  profile likelihood); an interaction refitted for cell-level coupling
  (inverse optimal transport) is not tested.

## Amendment 3 (2026-09-26, before any D3 computation)

The full-rank D3 likelihood (W with 177 x 209 entries, 19 training samples per
recipient) would take about 7 hours here. D3 now has two estimators, both
using only within-assay moments of the training biopsies:
- pf_means (primary): ecological regression of sample protein means on sample
  RNA means, weighted by cells, with ridge penalty chosen by inner 3-fold
  leave-samples-out prediction error of held-out sample protein means (grid
  1e-3, 1e-2, 1e-1, 1 times the trace of the weighted RNA mean covariance).
  Psi is the cell-weighted mean of diag(Sy_s - W Sx_s W^T), floored at 5% of
  the mean protein variance.
- pf_lowrank (secondary): the Gaussian moment likelihood of amendment 1 with
  W = U V^T of rank 8, initialized at the rank-8 truncation of pf_means, ridge
  on U and V fixed at the value chosen for pf_means.
Each estimator gives two arms: regression (C = Sx_k W^T) and closed form
(B = W^T Psi^{-1}).
