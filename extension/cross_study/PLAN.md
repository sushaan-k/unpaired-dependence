# Cross-study pairing-free transfer: frozen protocol

Written 2026-09-26, before any prediction for the recipients below was computed
and before any recipient scoring cell was read. Post hoc relative to the
released paper and to `extension/spectral_transfer`.

Prior use of the data, stated up front:

- **Colon** (Figshare 54027674) informed method development. D1-D3 of
  `spectral_transfer` used every colon cell, so the colon analysis here is a
  secondary, touched cohort. Its new adaptation/scoring split (salt
  `blood-colon-v1`, `seal.json`) was made before this plan and before any
  analysis of the split.
- **Stephenson et al. 2021** (E-MTAB-10026) supplied the released paper's
  nine-marker source fit and recipients. It is used here only as the training
  population, never as a recipient.
- **Hao et al. 2021** 3' CITE-seq (GEO GSE164378) has not been used anywhere in
  this project. It is the confirmatory, untouched cohort.

## Question

Can the RNA-protein dependence within a population be learned without any
paired cell, from populations whose RNA and protein were measured on
different cells, and transferred to another study? Which parts of it do the
available populations identify?

## Data and units of analysis

- **Training populations (Stephenson).** Samples without LPS stimulation; cells
  with `initial_clustering` in {Platelets, RBC} removed. Unit = patient (118
  patients; the 13 with two samples are pooled). At most 2,000 cells per
  patient: the first 2,000 in ascending SHA-256 of
  `stephenson-reference-v1|patient|barcode`.
- **Disjoint halves for pairing-free training.** Within each patient, cells
  sorted by SHA-256 of `pf-halves-v1|patient|barcode`; the first half supplies
  RNA moments only and the second half protein moments only. No cell
  contributes both assays to any pairing-free estimator.
- **Paired reference arms** use all selected cells of each patient, paired.
- **Recipient A: Hao (confirmatory).** 8 donors; time points pooled within
  donor. Each donor's cells are sorted by SHA-256 of `blood-hao-v1|donor|barcode`;
  the first half (floor) is adaptation, the rest scoring. Scoring cells are
  written to a separate file whose SHA-256 is recorded at split time.
- **Recipient B: colon (touched).** 12 donors, split of `seal.json`.
- Recipient adaptation cells are used only as two separate assays: within-assay
  moments; their cross-moments are used only by the recipient-paired benchmark.

## Features (frozen rules; `features.py`)

- **Proteins.** Targets measured in all three panels, matched by specificity
  in `adt_matching.py` (122 targets). Isotype controls and ambiguous targets are
  excluded (reasons in that file). Values: `log1p(count)` minus its mean over the
  122 matched targets in the same cell (centred log ratio over the matched set),
  identically in every study.
- **Genes.** Values `log1p(1e4 * count / total RNA count of the cell)`.
  Candidates: symbols present in all three studies. Eligible: detected (count
  > 0) in at least 1% of the adaptation cells of each recipient study (RNA
  only). Ranking: variance/mean of the log-normalised values over the selected
  Stephenson cells, among genes with mean > 0.05. Take the top 200 eligible
  genes and add the nine panel genes (CD4, CD7, CD14, CD19, CD33, CD38, CD44,
  CD47, CD52) when present in all three studies.
- No protein value, pairing, recipient scoring cell or prediction score enters
  feature selection.

## Scales

- **Pairing-free training scale.** Each feature is divided by its pooled
  within-patient standard deviation in Stephenson (genes from RNA halves,
  proteins from protein halves), so fitted interactions are in within-population
  SD units.
- **Paired reference.** Per-patient standardized moments (within-patient
  correlation scale).
- **Recipients.** Within-assay correlation matrices of the adaptation cells.
  Within-assay matrices everywhere get the shrinkage of `colon_scale.moments`
  (10% towards the diagonal, ridge floor 1e-3 x mean diagonal).
- **Target.** The cross-correlation matrix T (genes x proteins) of the
  recipient's scoring cells.

## Arms (each predicts the recipient's cross-correlation matrix)

Tuned arms share one budget: a five-value grid, three patient-grouped folds
(patients assigned by ascending SHA-256 of `folds-v1|patient`, dealt in
turn), and the edge rule: if the chosen value is at a grid edge, extend the grid
by factors of 10 in that direction until it is interior (at most three
extensions; if still at the edge, report it).

Pairing-free (no paired cell anywhere in training or tuning):

1. **pf_means (primary).** Shared linear channel y = Wx + b + e. W is the
   weighted ridge regression of patient protein means on patient RNA means
   (weights = cells per half), penalty scale in {1e-4, 1e-3, 1e-2, 1e-1, 1}
   times trace of the weighted between-patient RNA covariance, chosen by
   held-out protein-mean error. Residual variances psi as in `pairing_free.py`
   (average of diag(S_y - W S_x W^T), floor 0.05 x mean protein variance).
   Interaction B = W^T diag(psi)^-1, transferred by the closed form with the
   recipient's two correlation matrices.
2. **pf_moment (secondary).** Gaussian likelihood of each patient's protein
   moments given its RNA moments, S_y = W S_x W^T + diag(psi), W of rank r
   (`pairing_free.lowrank_negloglik`), penalty 1e-3 x the pf_means penalty
   scale; r in {2, 4, 8, 16, 32} chosen by held-out patients' moment
   likelihood; initialised from pf_means; L-BFGS-B with gtol 1e-7, ftol
   1e-13, up to 20,000 iterations; convergence reported. Transferred as in 1.

Paired reference (Stephenson paired cells):

3. **paired_closed_form.** Concave profile likelihood of B on per-patient
   standardized moments (`gaussian_transfer.fit_interaction`, up to 2,000
   iterations), ridge in {1e-4, 1e-3, 1e-2, 1e-1, 1}, chosen by held-out profile
   likelihood; closed-form transfer with the recipient's correlation matrices.
4. **reference_regression.** Ridge coefficients from pooled within-patient
   standardized moments, penalty in {1e-4, ..., 1} x mean diagonal, chosen by
   held-out patients' prediction error computed from their moments;
   prediction R_x W.
5. **transferred_correlation.** Pooled (cell-weighted) within-patient
   cross-correlation.
6. **transferred_covariance.** Pooled within-patient cross-covariance in
   original units, divided by the recipient adaptation cells' standard
   deviations.
7. **variances_only.** Arm 3's B transferred with diagonal (identity)
   recipient marginals.

Recipient information only:

8. **recipient_benchmark.** The recipient's own adaptation pairs: ridge
   regression with the same grid and three folds of adaptation cells (by
   ascending SHA-256 of `bench-v1|barcode`); prediction R_x W. It uses
   recipient pairs and is a reference point, not a competitor.
9. **independence.** Zero.

## Endpoints

- Primary endpoint per recipient donor: E = ||C_hat - T||_F / ||T||_F.
- Share of paired improvement recovered: s = (1 - E_pf_means) / (1 -
  E_paired_closed_form), per donor, summarized by the donor mean.
- Secondary: correlation between the entries of C_hat and T.

## Hypotheses and success criteria

Confirmatory cohort Hao; colon reported with identical rules as a touched
cohort. Donor bootstrap: 10,000 resamples of donors, seed 20260926.

- **H1 (primary): recovery beyond chance, donor-aware.** Permutation null:
  pairing-free training in which each patient's protein moments are paired with
  another patient's RNA moments from the same site (within-site
  derangement, no fixed points; 200 permutations, seed 20260927), refitted with
  the same tuning rule. Success: the donor-mean E of pf_means is below that of
  the permutations, p = (1 + #{perm <= observed}) / 201 < 0.05.
- **H2 (primary): most of what pairing buys.** Success: donor-mean share s of
  pf_means has a bootstrap 95% lower bound above 0.5. The pairing-free claim is
  supported in a cohort only when H1 and H2 both succeed (intersection-union
  test; no multiplicity adjustment needed).
- **H3 (composition control).** Within-type analysis (below) with Hao
  `celltype.l1` and colon `coarse_annotations_MK`. Success: the donor-mean
  within-type E of pf_means has a bootstrap 95% upper bound below 1, i.e. it
  recovers dependence within matched cell types. The same quantity is reported
  for paired_closed_form.
- **H4 (closed form vs matched baselines).** paired_closed_form against
  reference_regression, transferred_correlation, transferred_covariance and
  variances_only: donor-mean difference in E with Bonferroni-adjusted
  bootstrap intervals (8 comparisons: 4 baselines x 2 cohorts); success for
  a baseline when the adjusted interval excludes zero in favour of the closed
  form. Donors favouring each arm are reported.
- Descriptive (no criterion): pf_moment vs pf_means; every arm vs
  recipient_benchmark; Hao `celltype.l2` within-type analysis; entry
  correlations.

## Composition-controlled (within-type) analysis

Within each recipient donor and half, every cell is centred by the mean of its
annotated cell type, in both assays; types with fewer than 10 cells in either
half of that donor are dropped. Target: cross-correlation of type-centred
scoring cells. Predictions: arms 1-3 transfer their frozen B with the
type-centred adaptation correlation matrices; arm 4 applies its frozen
coefficients to them; arms 5 and 7 are unchanged; arm 6 divides by the
type-centred adaptation standard deviations; arm 8 is refitted on
type-centred adaptation cells with its rule; arm 9 is zero. Type means
removed, differences in composition cannot contribute to the target. Labels
come from the original authors (Hao used both modalities to annotate); they are
used only to stratify evaluation and adaptation moments, never in training.

## Identifiability analyses (descriptive; Proposition S7)

- Spectrum of the weighted between-patient RNA-mean covariance G and the
  pf_means ridge operator H = G (G + lambda I)^-1. Effective dimension tr(H);
  identified subspace M = eigenvectors with H-eigenvalue >= 1/2.
- Pairing-free recoverability diagnostic per recipient (RNA only): share of the
  recipient's standardized RNA variance in M.
- After scoring: the recipient's truth split into the part identified through M
  and the remainder, using the scoring cells' own regression; the remainder's
  relative size is the error floor for any estimator that uses population
  means only.
- Identified-set radius: the Chebyshev radius, in the recipient's units, of the
  set of cross-correlations compatible with the identified part and a valid
  joint law (contraction completion), relative to ||T||_F.
- Learning curve: pf_means refitted on random subsets of 8, 16, 32 and 64
  training patients (10 subsets each, seed 20260928) and on all patients.

## Pairing-free rigor re-analysis of colon D3 (post hoc, not confirmatory)

Same recipients and halves as D3 (salt `spectral-transfer-v1`). Changes:
disjoint RNA/protein cell halves within each training biopsy
(`pf-halves-v1`); donor-grouped folds; the grids and edge rule above;
pf_moment rank chosen by donor-grouped held-out likelihood and run to
convergence; donor-aware permutation (biopsy-level derangements with no
within-donor assignment; 200 for pf_means, 20 for pf_moment); identifiability
diagnostics.

## Sealing and order of operations

1. Extract features from the training data and the recipients' adaptation
   files. 2. Fit all reference-side estimators. 3. Compute every prediction
   (total and within-type) and write `results/predictions_manifest.json` with
   SHA-256 of the prediction files. 4. Only then read the scoring files
   (checking their SHA-256 against the seals) in `evaluate.py`. Permutation,
   identifiability and learning-curve refits use only rules fixed here.

## Amendment 1 (2026-09-26 12:50 EDT, before any recipient prediction was computed)

Reason: computational cost. In the cross-study fit, the pf_moment rank search
chose the upper edge of its grid (32), so the edge rule required three
cross-validation fits at full rank (122 = min(genes, proteins)); each was
estimated at over an hour on the two available cores (the first had run for
14 minutes when stopped). The colon re-analysis would have needed the same at
rank 177 for each of 12 recipients, plus 20 permuted likelihood fits per
recipient. A full-rank channel is also not the low-rank model this secondary
estimator is meant to test.

Changes:

1. pf_moment ranks are chosen from {2, 4, 8, 16, 32} without the edge
   extension. A choice at 32 is reported as being at the edge of the grid.
2. Colon re-analysis: pf_moment keeps the rank of D3 (8) for every recipient
   instead of re-selecting it, so that each fit runs to convergence and the 20
   donor-aware permutation refits per recipient remain feasible. Its penalty
   stays 1e-3 times the recipient's pf_means penalty scale.

Nothing else changes. Information available when this amendment was written:
the cross-study pf_means penalty scale (0.01, interior), the paired closed
form ridge (0.01, interior), the reference-regression penalty (0.1, interior),
the penalty scales chosen in the 200 permutation refits (all 0.01), and the
fact that the pf_moment rank search reached the upper edge of its grid (its
cross-validation losses were not inspected and were lost when the run was
stopped). No recipient prediction or score, and no result of the colon
re-analysis, had been computed.

## Post hoc analysis P1 (written 2026-09-26 15:40 EDT, after scoring; exploratory)

H3 failed: channels learned from patient-level populations, which vary mainly in
cell-type proportions, did not predict dependence within cell types.
Proposition S7 predicts that populations identify the directions along which
they vary. P1 tests the constructive side of that prediction: if the training
populations vary within cell types, within-type dependence should become
identifiable.

- Training units: Stephenson groups defined by patient x `initial_clustering`
  type with at least 50 cells in each hash half (RNA half and protein half, as
  in the frozen design). RNA moments come from the RNA half and protein moments
  from the protein half of each group.
- Within-type centring: each unit's RNA and protein means minus the
  cell-weighted mean of its type across patients, so that only within-type,
  between-patient variation remains.
- Estimator and tuning: pf_means with the frozen grid, edge rule and
  patient-grouped folds; features scaled by pooled within-unit standard
  deviations; B = W^T diag(psi)^-1 transferred by the closed form.
- Evaluation: the frozen scoring targets of every recipient and analysis (total,
  Hao level-1 and level-2 types, colon coarse types), and the identifiability
  diagnostics of the within-type channel.
- The cell-type labels are the original authors', who used both assays. P1
  therefore assumes that each assay carries its own cell-type labels, as
  annotated atlases do; it uses no paired cell otherwise. It is reported as
  exploratory whatever the outcome.

## Post hoc analysis P2 (written 2026-09-26 16:05 EDT, after P1; exploratory)

P1 enlarged the identified space (86 directions) but made within-type
prediction worse, which suggests that identification is not what failed within
types; the shared-channel condition is. A mixture argument (Additional file 1)
predicts that when populations differ only in the proportions of cell types
with fixed type-specific laws, population means identify the between-type part
of the cross-covariance exactly and carry no information about the within-type
part. P2 tests this directly.

- For each recipient donor, the scoring cells' cross-covariance over the kept
  level-1 (Hao) or coarse (colon) types is split exactly into a between-type
  part (type means, weighted by type proportions) and a within-type part
  (pooled within-type cross-covariance), both divided by the scoring cells'
  total standard deviations.
- The frozen pf_means channel W predicts each part from the adaptation cells:
  between-type RNA covariance times W^T, and within-type RNA covariance times
  W^T (direct channel, no saturation), in the same units.
- Reported: the relative size of the two parts of the truth, and the relative
  error of each predicted part. Exploratory whatever the outcome.
