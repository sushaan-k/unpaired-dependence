# Cross-study pairing-free transfer (extension, 2026-09-26)

## Question

Can RNA-protein dependence be recovered without any paired cell, from
populations whose RNA and protein were measured on different cells, and
transferred to another study? Which parts of it do the available populations
identify?

## Design (frozen in `PLAN.md` before any recipient prediction)

- **Training:** 118 patients of the Stephenson et al. 2021 blood CITE-seq atlas.
  Within each patient a hash splits the cells into an RNA half and a protein
  half; pairing-free estimators see only RNA moments of the first and protein
  moments of the second. No cell contributes both assays.
- **Confirmatory recipients:** 8 donors of Hao et al. 2021 (GEO GSE164378), a
  study never used in this project. **Secondary recipients:** the 12 colon
  donors, which informed method development.
- **Sealing:** each recipient donor's cells are split into adaptation and
  scoring halves; scoring halves are written to separate files whose SHA-256
  is recorded (`seal.json`, `hao_seal.json`). Predictions are hashed
  (`results/predictions_manifest.json`) before `evaluate.py` opens them.
- **Features:** 122 antibody targets present in all three panels
  (`adt_matching.py`), 206 genes chosen by an unsupervised rule
  (`features.py`). No protein value, pairing or scoring cell enters selection.
- **Arms:** pairing-free means (primary) and moments; paired closed form,
  reference regression, transferred correlation, transferred covariance,
  variances only; the recipient's own paired half (benchmark, not permitted);
  independence. One tuning budget for every tuned arm.
- **Criteria:** H1 donor-aware permutation null; H2 more than half of the paired
  improvement; H3 recovery within matched cell types; H4 closed form beats the
  four baselines. See `PLAN.md`.

Also here: the identifiability analyses of the Supplementary Information, Proposition S7
(`identifiability.py`, `learning_curve.py`), and a rigor re-analysis of the
earlier within-colon pairing-free design D3 (`colon_pf_rigor.py`): disjoint
halves, donor-grouped folds, extended grids, rank-8 fits run for up to 20,000
iterations (with a post hoc continuation of those that stopped at the limit)
and a donor-aware permutation control.

## Results

`results/RESULTS.md` is generated from the result files and lists every number.
In brief (relative error of the predicted gene-protein cross-correlation matrix;
independence = 1):

- **Untouched blood cohort (confirmatory).** Pairing-free, means: 0.292
  (donors 0.24-0.32), against a donor-aware permutation null of 0.949
  (p = 1/201 overall and per donor; H1 met). It recovered 82% (80-84%) of the
  paired closed form's improvement over independence (H2 met), and was more
  accurate than paired-reference regression (0.522) and transferred
  correlations (0.630) in 8/8 donors. The paired closed form reached 0.134,
  against 0.120 for the recipient's own paired half, and beat all four
  baselines in 8/8 donors (H4 met).
- **Within matched cell types (H3 not met).** Pairing-free 1.082 (1.014-1.162)
  across 8 level-1 types, worse than independence; paired closed form 0.528.
  The pairing-free recovery is the between-cell-type (mixture) component.
- **Colon (touched).** Pairing-free 0.988, no better than independence (it beat
  only its permutation null); paired blood closed form 0.719, better than all
  four baselines in 12/12 donors.
- **Identifiability.** From 118 donors the ridge keeps 11 gene directions by at
  least one half (effective dimension 16.9 of 206). They hold 38% of the
  untouched recipients' RNA variance but 12% of the within-type variance; the
  unidentified share of the truth is 0.16 in total and 0.72 within types. The
  RNA-only share tracks error across all 48 recipient analyses (r = -0.94).
  Error falls from 0.637 with 8 training donors to 0.292 with 118.
- The secondary moment estimator (rank 32, at the edge of an amended grid) was
  less accurate than the means estimator in the untouched cohort (0.389).
- **Within-colon re-analysis of D3** (`colon_pf_rigor.py`, post hoc; relative
  error of the predicted cross-covariance). Learned from the other donors'
  biopsies without paired cells, the means estimator reached 0.566 across the
  12 recipients (65% of the paired reference's improvement over independence),
  against a donor-aware permutation null of 1.086 (p = 1/201 for every
  recipient); the rank-8 estimator reached 0.680. No penalty sat at a grid
  edge. Nine of 12 rank-8 fits converged within 20,000 iterations; continuing
  the other three to convergence (`colon_rigor_continuation.py`) changed their
  errors by at most 0.0015. Biopsy means identified 11-12 of 209 gene
  directions.

### Post hoc analyses (exploratory; appended to `PLAN.md` after scoring)

- **P2, between and within cell types** (`posthoc_mixture.py`). The scoring
  cells' cross-covariance over the kept level-1 types splits into a
  between-type part (97% of the norm of the total in the untouched cohort) and
  a within-type part (11%). The frozen channel, applied as a direct linear
  channel, predicted the between-type part with relative error 0.71 (entry
  correlation 0.75) and the within-type part with error 1.92 (entry correlation
  0.15; 1.8 times the norm of the truth). In colon: 1.14 and 1.38.
  Descriptive addition computed afterwards (`mixture_diagnostics.py`, not part
  of the P2 plan): the identified subspace holds 88% of the recipients'
  between-type RNA covariance but 12% of their within-type RNA covariance
  (colon 57% and 9%). Supplementary Information, Proposition S8, explains
  the pattern: populations that differ only in cell-type composition identify
  the between-type slope and nothing about within-type slopes.
- **P1, training on within-type variation** (`posthoc_within.py`). With 512
  patient-by-type units centred within type, the identified space grew from 11
  to 86 directions and the unidentified within-type fraction fell from 0.72 to
  0.28, yet within-type error rose to 1.287.
- **Gene-protein pairs** (descriptive, in `make_assets.py`). Within level-1
  types, the four largest held-out correlations between a gene and its own
  protein were FCGR3A-CD16 (0.63), KLRB1-CD161 (0.51), IGHD-IgD (0.51) and
  IL7R-CD127 (0.48). The paired closed form predicted 0.52, 0.30, 0.36 and 0.39;
  the pairing-free channel 0.12, 0.07, 0.10 and 0.20.

## Reproduce

Raw data are not redistributed. Download (checksums in `PLAN.md` and the
seals): the Stephenson h5ad (E-MTAB-10026, `covid_portal_210320_with_raw.h5ad`),
GSE164378 files `GSM5008737_RNA_3P-*`, `GSM5008738_ADT_3P-*` and
`GSE164378_sc.meta.data_3P.csv.gz`, and the colon count matrix (see
`../spectral_transfer/README.md`). Paths are set at the top of each script.

```bash
python ../spectral_transfer/extract_colon.py --h5ad colon_counts.h5ad --out colon_paired.npz
python seal_colon.py                                  # colon adaptation/scoring files
python extract_stephenson.py stats && python features.py candidates
python seal_hao.py                                    # Hao adaptation/scoring files
python features.py select && python extract_stephenson.py extract
python fit_reference.py pf & python fit_reference.py paired; python fit_reference.py merge
python predict.py                                     # hashes predictions
python evaluate.py                                    # opens the scoring files
python permutation.py fit && python permutation.py evaluate
python identifiability.py && python learning_curve.py
python sensitivity_moment.py                          # post hoc: restart check of the rank-32 fit
python colon_pf_rigor.py && python colon_pf_rigor.py --merge
python colon_rigor_continuation.py                    # post hoc: continues rank-8 fits stopped at the limit
python posthoc_within.py && python posthoc_mixture.py   # post hoc P1 and P2
python mixture_diagnostics.py                         # descriptive addition to P2
python analyze.py && python make_assets.py --out ../../revision/source
python -m unittest -v test_cross_study.py
python verify_cross_study.py [--data]
```

## Files

| File | Content |
|---|---|
| `PLAN.md` | Frozen protocol, amendment 1 and the post hoc analyses P1 and P2 (SHA-256 of each version in `results/plan_sha256.txt`) |
| `adt_matching.py`, `features.py` | Antibody matching; gene rule |
| `extract_stephenson.py`, `seal_hao.py`, `seal_colon.py`, `data.py` | Extraction, sealed splits, preprocessing |
| `estimators.py`, `fit_reference.py` | All reference-side arms and tuning rules |
| `predict.py`, `evaluate.py` | Frozen predictions; sealed scoring |
| `permutation.py`, `analyze.py` | H1 null; prespecified summaries |
| `identifiability.py`, `learning_curve.py` | Proposition S7 diagnostics; per-gene analysis (post hoc) |
| `sensitivity_moment.py` | Post hoc restart check of the moment estimator |
| `colon_pf_rigor.py`, `colon_rigor_continuation.py` | Rigor re-analysis of D3; post hoc continuation of its unconverged rank-8 fits |
| `posthoc_within.py`, `posthoc_mixture.py`, `mixture_diagnostics.py` | Post hoc P1 and P2; descriptive addition to P2 |
| `make_assets.py`, `test_cross_study.py`, `verify_cross_study.py` | Manuscript assets; tests; verification |

## Caveats

- One untouched cohort: 8 blood donors, whose three vaccination time points
  were pooled. The protocol was written after the within-colon results were
  known, and colon itself is a touched cohort.
- Amendment 1 (before any prediction) capped the moment estimator's rank at 32
  for computational cost; its chosen rank is at that edge. Its final fit
  stopped on a line-search failure; restarting changed no endpoint by more
  than 2e-6 (`results/sensitivity_moment.json`).
- The identified block from population means exceeded a valid joint law in
  every total analysis: the linear shared channel is violated between
  populations, and the closed form's saturation keeps the predictions valid.
  The reported identified-set radius caps that block at one.
- Hao et al. annotated cell types with both modalities; the labels stratify the
  evaluation only. Colon within-type analyses have few cells per type.
- Cross-study transfer assumes the interaction carries over in
  within-population standard-deviation units between different antibody
  panels and chemistries.
- The per-gene identifiability analysis, the restart check, the gene-protein
  pair summaries and the P1/P2 analyses were added after scoring. P1 and P2
  were appended to `PLAN.md` after scoring; they are exploratory.
