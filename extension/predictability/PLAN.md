# Prospective test: is the saving of paired cells predictable? (frozen plan)

Written after the development in DEV.md (post hoc, on the nine benchmark data sets) and before any of the data sets
below was extracted or any statistic of dependence between their modalities was computed. The plan, the code
(`ptools.py`, `prun.py`) and the development records are hashed in `results/freeze.json` before extraction. The
extraction code (`pdata.py`) is written after the freeze, because it depends on the file formats; it may use
cell and feature metadata and marginal statistics of one modality at a time (library sizes, detection rates,
missing values) and nothing else. The extracted files are hashed in `results/data_freeze.json` before any
prediction.

## Question

Proposition S14 says that the saving of paired cells bought by the unpaired block structure is a function of where
the dependence lies (signal shares w_b) relative to where the per-cell noise lies (noise shares pi_b). The primary
question is whether this law, with w and pi measured from the paired reservoir, predicts the saving observed in data
sets that were not used to develop it. Secondary questions: how well a pilot of paired cells, and unpaired cells
alone, predict it.

## Data sets (none analysed before in this study)

| Name | Pair | Source | Cells | Unit / condition / population |
|---|---|---|---|---|
| sln111 | RNA, 111 proteins | spleen_lymph_111.h5ad, github.com/YosefLab/totalVI_reproducibility (GEO GSE150599) | cell types without "doublet" or "Low quality", hash_id not Negative | mouse (batch_indices) / cell type / mouse x tissue |
| sln206 | RNA, 206 proteins | spleen_lymph_206.h5ad, same repository | same | same |
| pbmc10k | RNA, 14 proteins | pbmc_10k_protein_v3.h5ad, same repository (10x Genomics) | all | one donor / all / all |
| malt10k | RNA, 14 proteins | malt_10k_protein_v3.h5ad, same repository (10x Genomics) | all | one donor / all / all |
| bmcite | RNA, 25 proteins | GEO GSE128639 (GSM3681518-9) | all cells with RNA and ADT | all / all / all |
| fetal_cortex | RNA, chromatin peaks | GEO GSE162170 multiome (RNA and ATAC counts, cell metadata) | DF_classification Singlet | sample / RNA cluster name / sample |
| snare_cortex | RNA, chromatin peaks | GEO GSE126074 adult mouse cortex SNARE-seq | barcodes in both matrices | library (barcode prefix) / all / library |
| fafb_vpn | optic-lobe wiring, central-brain wiring | FlyWire FAFB v783 compiled data (fly connectome tutorial) | visual projection and visual centrifugal neurons with a cell type and >= 50 synapses on each side | cell type / super class / all |
| banc_vpn | the same | BANC v888 compiled data | the same | the same |
| human_gaba | RNA, electrophysiological features | github.com/AllenInstitute/human_patchseq_gaba (Lee, Dalley et al.) | cells with expression and complete selected features | donor / subclass / all |

The FAFB and BANC files were used before for other neurons (descending and ascending neurons, and FAFB as an
unpaired pool); no visual projection or visual centrifugal neuron was analysed. The other files were downloaded
after development and only their structure (keys, shapes, metadata columns) was inspected before this plan.

Extraction rules. X candidates: features detected in at least 1% of cells, at most 3,000 by detection (ties by
name); X library = total counts over all features of the file. Proteins: every antibody except isotype controls,
raw counts, centred log-ratio (`y_kind` clr). Chromatin: peaks detected in at least 5% of cells, at most 3,000 by
detection; Y library = total over all peaks; log counts per 10,000 (`lognorm`). Connectomes: X = synapse counts with
optic-lobe partners grouped by the partner's cell type (else cell class), Y = synapse counts with central-brain
partners grouped by super class : hemilineage (else cell class), inputs and outputs separately; partners of the
visual projection, visual centrifugal, descending, ascending, glial and unannotated classes are excluded; X
candidates as above, Y groups detected in at least 1%; `lognorm`. Patch-seq: expression as provided (log
normalized, no counts; latent correlation equals the measured one, as for gouwens_visp); electrophysiological
features with at most 10% missing values, cells with all of them present; raw values. Droplet data are capped at
3,000 cells per population in SHA-256 order of "predictability-cap-v1|<name>|<cell>".

## Problems, roles and estimators (`ptools.py`, `prun.py`)

* A problem is a data set with one X panel of 100, 200 or 400 features, chosen by the dispersion rule of the
  benchmark from unpaired pool cells only (Y panel as in the benchmark: proteins detected in at least 1% of cells,
  at most 200; peaks or partner groups by the dispersion rule, at most 200; every electrophysiological feature).
  A panel identical to a smaller one is dropped.
* Roles by SHA-256 of the cell identifier: truth cells 25%, reservoir 30%, pool 45%; pool cells contribute X if a
  second hash is below one half and Y otherwise. Populations need at least 10 pool cells in each half.
* Estimand: the pooled within-population cross-covariance in pool standard-deviation units. Unpaired summaries,
  bases and blocks are those of the proposed estimator (latent RNA correlation by count splitting where counts
  exist; eigen-blocks RNA 1-5/6-20/21-60/rest x Y 1-3/rest).
* Estimators: block James-Stein in the unpaired bases (the proposed estimator, pooled), one-block James-Stein
  (paired-only), and block James-Stein in a random orthonormal basis pair (control).
* Pilot: the first m = min(200, 20% of the reservoir) reservoir cells in SHA-256 order. Budgets: 25 x 2^k up to
  min(12,800, reservoir - m); 20 draws per budget from the reservoir cells after the pilot.
* Observed curves: noise-unbiased recovered fraction of the truth cells' dependence (the two truth-cell halves give
  the denominator), averaged over draws. Cells needed to reach a target: log-linear interpolation on the running
  maximum of the curve (`grun.needed`). Observed saving = cells needed by one-block James-Stein / cells needed by
  block James-Stein. It is estimable when both curves cross the target within the budgets and the block curve is
  below it at the smallest budget; otherwise it is censored and excluded from calibration statistics.

## Predictions (hashed before any statistic of truth cells)

* LAW (primary): per block, the unbiased signal r_b^2 = ||m_b||^2 - t_b (negative values set to zero) and the
  per-cell noise trace tau_b, from all reservoir cells; saving S(eps) and cells needed from Proposition S14(a) in
  closed form, at eps = 1 - target.
* PILOT: the same from the m pilot cells.
* TRANSFER (zero pairs): the development signal-share profile (`results/transfer_profile.json`), collapsed to the
  problem's blocks, with noise shares from the unpaired pool's marginal covariances (Proposition S15a).
* UNPAIRED BOUND: 1 / min_b pi_b with the unpaired noise shares.
* RANDOM: the law with moments in the random bases.
* Secondary: the law's predicted curves; the bootstrap law from the reservoir.

## Hypotheses (targets are recovered fractions; primary 0.5)

The unit of analysis is a problem. Statistics use problems with an estimable observed saving at the target; H1-H6
are evaluated only if at least 8 problems are estimable at 0.5, and are otherwise reported as not evaluable.

* H1 (primary, the law is calibrated): the median over problems of |log2(observed / LAW)| at 0.5 is below
  log2(1.25).
* H2 (the law ranks problems): Spearman correlation of LAW with the observed saving at 0.5 is positive with
  one-sided permutation P < 0.05 (10,000 permutations of the predictions).
* H3 (no saving without unpaired structure): the median observed saving of the random-basis control at 0.5 lies in
  [0.8, 1.25].
* H4 (pilot): the median |log2(observed / PILOT)| at 0.5 is below log2(1.5).
* H5 (zero pairs, transfer): the median |log2(observed / TRANSFER)| at 0.5 is below log2(1.5).
* H6 (unpaired bounds): every estimable observed saving at 0.5 lies in [0.8, 1.25 / min_b pi_b].

Reported without thresholds: the same statistics at 0.3 and 0.7; 95% intervals of the medians from 2,000
resamples of data sets with their problems; the fraction of problems within 25%; the median signed log2 ratio
(bias); cells needed by the proposed estimator predicted by LAW and PILOT against observed; the bootstrap law.

## What would count against the law

H1 or H2 failing. A systematic bias beyond 25% at 0.5 would mean that the oracle-linear approximation (Proposition
S13) does not carry the implemented estimator's saving in data unlike the development sets.
