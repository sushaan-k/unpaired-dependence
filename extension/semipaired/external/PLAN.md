# External evaluation of the semi-paired estimator (frozen before scoring)

Question: in an independent screen, does the covariance-guided estimator, fixed after development on the
Frangieh and Papalexi screens, reach a prespecified accuracy for within-condition RNA-protein dependence with
fewer paired cells than estimation from paired cells alone and than SemiCCA, and with no more than Champollion
needs, given the same paired cells and the same unpaired observations?

## Data and cell roles (sealed before this plan; `extract.py`, `results/seal.json`)

OverCITE-seq (Legut et al. 2022; GEO GSE193736): primary human CD8+ T cells, each transduced with one ORF, cultured
24 h with IL-2 alone ("rest") or with CD3/CD28 activator ("stim"), 14 surface proteins. ORF calls: largest ORF
count >= 3 and >= 3x the second; hashtag calls: largest >= 2x the second; the four hashtags with the largest
median RNA library size are the stimulated wells (medians 12,959-13,254 against 5,999-6,295). 3,067 of 4,312
cells were called. Every third ORF in SHA-256 order is held out (11 ORFs); their cells were split by barcode hash
into two files (453 and 455 cells) and, within each, into two sub-halves; the training file holds the other 21
ORFs (2,159 cells), with assay halves fixed by barcode hash.

No arm of this plan adapts to held-out cells (the condition maps, Champollion's transport and every comparator use
training cells only), so both held-out files are scoring cells (908 cells). Their sub-half labels, fixed at
extraction, give the two halves of the noise-unbiased endpoint.

## Computed before this plan was frozen

* The seal: counts, technical calls, library sizes and the number of cells of each held-out ORF x condition
  group (no other statistic of a held-out cell).
* The gene panel (`run_external.py panel`, training cells only; `results/panel.json`), regenerated once before
  the freeze to exclude ORF transgenes (the 32 ORFs of the seal). The first version,
  `results/panel_v1_with_orf_genes.json`, contained DUPD1 and NGFR, and 5 of its 200 genes changed; the pilots of
  estimator variants (`results/pilot_explore6.json`, `results/pilot_explore8.json`) used it, and every other pilot
  used the final panel.
* A pilot on the training file only (`pilot.py`, `pilot_champ.py`; `results/pilot*.json`): the 21 training ORFs
  in three folds, each held out in turn with every arm fitted on the other ORFs, at 25-200 paired cells. It set
  the accuracy target and Champollion's lasso weights and compared variants of the proposed estimator.
* Development on the Frangieh and Papalexi screens (`../RESULTS.md`), which fixed the estimator, including the
  two-sided eigenbasis (`../explore8.py`), and the comparators, with Champollion run on the same paired draws
  (`../champ_dev.py`; lasso weight chosen per budget on the evaluation groups). Against paired-only estimation the
  proposed estimator needed 4.4-5.9 (Frangieh) and 1.9-2.5 (Papalexi) times fewer paired cells to reach recovered
  fractions of 0.3-0.6. Against Champollion it needed 1.4-1.7 times fewer in Frangieh (all 141 groups; lower
  bounds above 1.3), whereas in 7 held-out Papalexi targets Champollion needed fewer paired cells to reach 0.3
  (ratio 0.55, 0.35-0.79) and the two were similar at 0.5-0.6. Propagating each protein's covariance with its
  encoding genes (`../explore9.py`) helped in Papalexi but not in Frangieh or in the pilot, and was not adopted.
* A code test of the prediction and scoring functions on training ORFs only (no held-out file was read).

## Frozen rules

Everything not stated here is as in `../run_study.py` (development), unchanged.

* Features: `results/panel.json`: the genes encoding the 14 target antibodies that are detected in at least 1% of
  training cells, then the most variable genes (dispersion index in 20 mean bins, detection >= 5%), 200 genes, no
  ORF transgene; the 14 antibodies, log1p counts centred over the 14 per cell.
* Populations: ORF x condition. Training populations need at least 10 cells in each assay half, and pool
  populations at least 10 pool cells in each half.
* Pool and reservoir: training cells of eligible populations split by barcode hash (salt `hybrid-reservoir-v1`)
  into a 25% paired calibration reservoir and a 75% unpaired pool; pool cells contribute RNA from assay half 0
  and protein from half 1, never both.
* Budgets: 25, 50, 100, 200 and 400 paired cells drawn from the reservoir; 20 draws each; every arm uses the same
  paired cells in a draw and the same unpaired pool.
* Proposed estimator (`bjs2/mapped`): positive-part block James-Stein of the mean paired cross-products in two
  unpaired eigenbases, the latent (count-split) RNA correlation of the pool and the pool's protein correlation;
  blocks are RNA eigen-blocks 1-5, 6-20, 21-60 and the rest times protein eigen-blocks 1-3 and the rest; the
  estimate is mapped to each condition through the condition's unpaired latent RNA covariance on the leading 120
  eigendirections, shrunk towards the identity by the split-half reliability of the condition-minus-pooled RNA
  covariance, and rescaled by the condition's standard deviations.
* Comparators, each pooled and per condition (refitted on the condition's paired cells and unpaired summaries):
  paired only: James-Stein in marker coordinates, SCOSE, FCOSE and cross-validated low rank; semi-paired: SemiCCA
  (projection of the paired cross-covariance on SemiCCA subspaces, beta and rank by cross-validation) and
  Champollion (fitted on the paired cells, epsilon 1 (its default), 2,000 iterations, lasso weight c_B / sqrt(B)
  with c_B = 3.4, 2.25, 2.25, 1.5 and 1.5 at 25, 50, 100, 200 and 400 paired cells, the best values of the pilot
  grid {0.67, 1, 1.5, 2.25, 3.4} on training ORFs (`../logs/champ_choice.json`; 400 cells take the value at 200);
  transport between the unpaired RNA and protein pool cells of each condition; the plan's cross-correlation is
  the estimate); reference ridge regression with a paired or an unpaired Gram.
  Every denoiser is also run with the proposed map (reported, not part of any hypothesis). References: all
  training cells of the pool populations used as paired cells, and independence.
* Endpoint: recovered fraction of within-condition cross-correlation among held-out ORFs (`targets.py`): each
  held-out cell is centred on the mean of its ORF x condition group (within the subset: all cells or one
  sub-half), the centred cells of a condition are pooled into one cross-correlation T_c, and
  RF = sum_c (2 <C_c, T_c> - ||C_c||^2) / sum_c <T_c^A, T_c^B>, with the numerator averaged over draws;
  independence scores 0 and a perfect estimate 1. Uncorrected relative loss: sum_c ||C_c - T_c||^2 / sum_c ||T_c||^2,
  averaged over draws.
* Paired cells needed to reach a target: running maximum of the budget curve, log-linear interpolation between
  budgets; a curve at or above the target at 25 cells needs 25 ("at most"), one below it at 400 cells needs 400
  ("more than"; conservative for a comparator).
* Intervals: 2,000 bootstrap resamples of held-out ORFs (seed 20261045), re-pooling the targets in each resample.

## Hypotheses (prespecified)

Accuracy target: recovered fraction 0.3. In the pilot (smaller pools, held-out training ORFs; 25, 50, 100 and
200 paired cells), the proposed estimator recovered -6.9, 15.1, 30.5 and 42.8%, paired-only estimation at most
1.1, 6.6, 12.1 and 19.1%, SemiCCA 4.0, 14.4, 21.3 and 33.5%, reference regression 8.2, 16.2, 26.3 and 35.1%, and
Champollion with its best lasso weight 1.5, 14.6, 26.5 and 37.7%. The proposed estimator thus reached 0.3 at about
100 paired cells, and paired-only estimation did not reach it within 200. Secondary targets 0.2 and 0.4 are
reported. A comparison is decided only if the proposed curve reaches the target within 400 cells. For each
comparison, the ratio is (paired cells the comparator needs) / (paired cells the proposed estimator needs), with
its 95% bootstrap interval.

* H1 (primary): fewer paired cells than paired-only estimation (upper envelope of James-Stein, SCOSE, FCOSE and
  low rank, pooled or per condition): lower bound of the ratio above 1.
* H2: fewer paired cells than SemiCCA (the better of pooled and per condition at each budget): lower bound above 1.
* H3 (non-inferiority): no more than 1.25 times the paired cells Champollion needs: lower bound of the ratio
  above 0.8. With its lasso weight tuned to the evaluation or training groups, Champollion was less efficient
  than the proposed estimator in Frangieh and in the pilot but more efficient at small budgets in Papalexi,
  whereas the proposed estimator is closed-form, untuned and more than a hundred times faster (tens of
  milliseconds per fit against seconds); superiority (lower bound above 1) is reported without a claim.
* Reported without a claim: the ratio against the upper envelope of every comparator in its original form and of
  every comparator including those with the proposed map; the difference from the best comparator at each
  budget; the uncorrected loss of every arm and budget; recovered fractions with intervals. No budget or target
  other than those above is used for a claim.

## Deviations

None yet.
