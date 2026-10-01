# Generality benchmark: frozen plan

Written 27 September 2026 and hashed with all code and the extracted data files (`results/freeze.json`) before any
statistic of held-out cells was computed and before the estimators were run on any of these data sets. Deviations
after the freeze go to `DEVIATIONS.md`; this file is not edited.

## Question

The semi-paired estimator was developed on two perturbation screens and confirmed once, prospectively, in
OverCITE-seq. None of those three data sets replicates across donors. Does its saving of paired cells hold across
tissues, species, modality pairs and nervous systems when independent replication units (donors, patients, mice,
recording days, neuron types) are held out, and does its size follow from how the dependence is concentrated in
the unpaired eigen-blocks?

## Data sets

Nine primary data sets, extracted by `gdata.py` (files in `/home/claude/cbio/rawdata/generality`, hashed in the
freeze record; the extraction used cell counts and detection rates of one modality at a time, never a statistic of
dependence between modalities, and no estimator had been run on them):

| name | modality pair (X, Y) | tissue / system | replication unit held out | condition | centring population |
|---|---|---|---|---|---|
| hao | RNA, 228 surface proteins | human blood (PBMC; Hao et al. 2021), 3,000 cells per sample | donor (8) | celltype.l1 | sample (donor x time) |
| stephenson | RNA, 192 surface proteins | human blood, COVID-19 (Stephenson et al. 2021), 600 cells per patient | patient (118) | initial clustering | sample |
| bmmc_cite | RNA, 134 surface proteins | human bone marrow (NeurIPS 2021), 3,000 cells per batch | donor (9) | coarse type | batch |
| colon | RNA, 177 surface proteins | human colon (Mennillo et al. 2024) | patient (12) | coarse annotation | sample |
| bmmc_multiome | RNA, chromatin accessibility (peaks) | human bone marrow multiome (NeurIPS 2021), 3,000 cells per batch | donor (10) | coarse type | batch |
| scala_m1 | RNA, 29 electrophysiological features | mouse motor cortex Patch-seq (Scala et al. 2021) | mouse (264) | RNA family | condition |
| gouwens_visp | RNA (log CPM, no counts), 68 electrophysiological features | mouse visual-cortex interneurons, Patch-seq (Gouwens et al. 2020) | recording day (362) | subclass | condition |
| banc | brain-side wiring, nerve-cord-side wiring | fly brain-and-nerve-cord connectome (BANC, female): descending and ascending neurons | cell type (913) | descending / ascending | condition |
| malecns | brain-side wiring, nerve-cord-side wiring | male fly CNS connectome: descending and ascending neurons | cell type (1,010) | descending / ascending | condition |

Connectome features are synapse counts of each descending or ascending neuron with partner groups (super class
and hemilineage, else cell class; inputs and outputs separately) on each side of the neck: brain-side partners
have their soma in the brain, nerve-cord-side partners in the nerve cord; partners that cross the neck and
non-neuronal partners are excluded; neurons need at least 50 synapses on each side. For Patch-seq and the
connectomes each unit contributes few cells, so cells are centred within condition across units; for droplet data
within sample and condition.

Secondary analysis (`banc_crossanimal`): as `banc`, but the unpaired pool is not BANC. Brain-side-only neurons come
from the FAFB brain connectome (FlyWire) and nerve-cord-side-only neurons from the MANC nerve-cord connectome, as
they would in practice; the paired cells are the same BANC reservoir neurons, and the held-out targets are the same.

## Roles

* Folds: units in SHA-256 order of `generality-fold-v1|<data set>|<unit>`; the i-th unit is held out in fold
  i mod 3. Every unit is held out exactly once; in each fold the cells of the other units are training cells.
* Training cells: 25% form the paired calibration reservoir by the development study's barcode hash
  (`hybrid-reservoir-v1`); the others form the unpaired pool, contributing X if `generality-part-v1|<cell>` hashes
  below one half and Y otherwise, never both.
* Held-out cells: split into two sub-halves by `generality-half-v1|<cell>` for the noise-unbiased endpoint.

## Features (per fold, from its training cells only)

* X: the 200 most variable features by the external test's binned-dispersion rule among candidate features detected
  in at least 5% of training cells, as log(1 + 10^4 count / library). Gouwens: no counts are available, so the 200
  highest-variance genes are used and the latent correlation equals the measured one (count splitting is skipped).
* Y: surface proteins detected in at least 1% of training cells (at most 200, by variance of log counts), centred
  log-ratios; chromatin peaks and nerve-cord wiring: 200 by the binned-dispersion rule, log(1 + 10^4 count /
  library); electrophysiology: every feature, raw values (the estimators scale by pooled standard deviations).
* Training populations need at least 10 cells in each assay half.

## Arms (the code of extension/semipaired, unchanged)

* **Proposed** (`bjs2/mapped`): two-sided block James-Stein in the unpaired eigenbases with the condition map.
* **Self-tuning** (`selftune/mapped`), the prespecified second arm: variant V2 of `semipaired/selftune.py`
  (preconditioned shrinkage within the same blocks, tau^2 by the moment rule), with the same map, chosen before this
  plan by the rule of `semipaired/DEV_SELFTUNE.md`.
* Comparators, each pooled, refitted per condition and with the proposed map: James-Stein in marker coordinates,
  SCOSE, FCOSE and cross-validated low rank (paired cells only), SemiCCA, and reference ridge regression with a
  paired or unpaired Gram matrix. Champollion is not included (about 9 s per fit).
* Ablation: the proposed shrinkage in bases estimated from the paired cells themselves (pooled).
* Reference: every training cell used as a paired cell.
* Budgets 25, 50, 100, 200, 400 and 800 paired cells, up to the fold's reservoir size; 5 draws per budget and
  fold (15 per budget); every arm receives the same paired cells and the same unpaired pool (seeds in `grun.py`).

## Endpoint and savings

* Recovered fraction: sum over folds and conditions of 2<C, T> - E||C||^2, divided by the sum of <T_A, T_B>, where
  T is the within-population cross-correlation of held-out cells pooled within condition over the fold's held-out
  units, T_A and T_B those of the two sub-halves, and C a prediction (the expectation over draws). As in the
  external test.
* Paired cells needed to reach a target: running maximum of the budget curve, log-linear interpolation between
  budgets (budgets common to all folds). Targets are fractions of the reference's recovered fraction (every
  training cell paired): **0.5 (primary)**, 0.25 and 0.75, so that the target is reachable in every data set.
* Saving: cells the comparator envelope needs divided by cells the proposed arm needs. A curve that never reaches
  the target counts as needing the largest budget (conservative for the proposed arm when it is the one that fails).
* Intervals: 2,000 bootstrap resamples of units (over all folds), recomputing targets, the reference and the
  endpoint. Draws of paired cells are averaged, not resampled; units are the replication units.

## Hypotheses

* **G1 (primary)**: over the nine primary data sets, the geometric mean saving of the proposed estimator against the
  paired-only envelope (James-Stein, SCOSE, FCOSE, low rank; pooled or per condition) at the primary target
  exceeds one: lower bound of the 95% interval above one, the interval combining the r-th bootstrap resample of every
  data set. G1 is met only if all nine data sets are scored.
* **G2**: the same against SemiCCA (pooled or per condition).
* **G3 (self-tuning, secondary, two-sided)**: geometric mean over data sets of (cells the fixed estimator needs) /
  (cells the self-tuning arm needs) at the primary target, with its 95% interval; reported without a threshold.
* **G4 (structural law)**: across the nine primary data sets and the three earlier ones (Frangieh, Papalexi,
  OverCITE-seq), the Spearman correlation between the predicted saving and the observed saving against the
  paired-only envelope at the primary target is positive: one-sided permutation p < 0.05 (10,000 permutations).
  * Predicted saving (`glaw.py`, training cells only, never held-out cells): in fold 0, with the reservoir as the
    population, the mean per-cell product in the unpaired two-sided eigenbasis gives for each block b
    a_b^2 = max(||theta_b||^2 - tau_b/N, 0), with tau_b the trace of the covariance of the per-cell products and N
    the reservoir size. The oracle linear shrinkage risk of the blocks, sum_b a_b^2 tau_b / (B a_b^2 + tau_b), and of
    a single block of all coefficients, A tau / (B A + tau) with A = sum_b a_b^2 and tau = sum_b tau_b, reach half of
    A at budgets B_blocks and B_single = tau / A; the predicted saving is B_single / B_blocks. For the three earlier
    data sets the reservoirs of `semipaired/bound_check.py` are used.
  * Observed saving of the earlier data sets: at half of their reference's recovered fraction, from their stored
    curves (development results, external test).
* Reported without hypotheses: per-data-set savings and curves, savings against every comparator in its original
  form and against every comparator given the map, the ablation and the secondary cross-animal analysis.
  Uncorrected losses are not computed in this benchmark.

## Integrity

`grun.py predict` writes every prediction of a data set, hashes it in `results/<name>_manifest.json`, and only then
does `grun.py evaluate` read the held-out statistics, after checking those hashes and the freeze record. The harness
was tested on simulated data (`gdata.py synthetic`) before the freeze; its results are not reported.
