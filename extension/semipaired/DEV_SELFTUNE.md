# Self-tuning variant: development rule (written before any variant was run)

Written 27 September 2026, after the external test and before `selftune.py` or `dev_selftune.py` had produced
any result. The fixed estimator (`bjs2/mapped`) stays the proposed estimator of the external test and the primary
arm of the generality benchmark; this note decides which self-tuning variant is carried into that benchmark as a
prespecified second arm.

## Why

The fixed estimator uses one block partition for every data set (RNA eigen-coordinates 1-5, 6-20, 21-60 and the
rest; protein eigen-coordinates 1-3 and the rest) and one James-Stein factor per block. Both were set on
RNA-protein development data. A self-tuning variant chooses them for each data set from its own paired and
unpaired training cells, with Stein's unbiased risk estimate (SURE), and never uses held-out data. It is meant to
adapt the estimator to new modalities (electrophysiology, connectivity, chromatin accessibility) and to each data
set's spectrum.

## Variants (all in the unpaired two-sided eigenbasis, all followed by the unchanged condition map)

Notation: block coefficients m (means of per-cell products), their noise covariance S (from the per-cell products,
as in the fixed estimator), t = tr S, s_j the diagonal of S.

* **V0** (reference): the fixed estimator, positive-part James-Stein (c = t) in the fixed partition.
* **V1, self-tuned partition**: the partition is chosen among 42 candidates, RNA edges in {(5,20,60),
  (2,5,10,20,40,80), (3,10,30,100), (10,40), (5,20), (1,3,10,30,100), ()} times protein edges in {(3,), (),
  (1,3), (2,5,10), (3,10,30), (5,)}, by the smallest total SURE of positive-part James-Stein; James-Stein in the
  chosen partition. SURE of a block with ||m||^2 > t: t - t^2/||m||^2 + 4 t m'Sm/||m||^4; otherwise ||m||^2 - t.
* **V2, preconditioned shrinkage**: the fixed partition; within each block each coefficient is shrunk by
  tau^2 / (tau^2 + s_j), with the moment estimate tau^2 = max(0, (||m||^2 - t)/d) for a block of d coefficients
  (heteroscedastic empirical Bayes shrinkage; refs. Efron & Morris 1973, Xie, Kou & Brown 2012). With equal s_j
  this is exactly the fixed estimator's positive-part James-Stein factor 1 - t/||m||^2, so the rule differs from
  V0 only by preconditioning each coefficient by its own noise. Its SURE, exact for Gaussian block means with
  covariance S including the dependence of tau^2 on m, is
  sum_j b_j^2 m_j^2 + 2 sum_j s_j (1 - b_j) + (4/d) sum_j m_j s_j (Sm)_j / (tau^2 + s_j)^2 - t, with
  b_j = s_j/(tau^2 + s_j) (||m||^2 - t when tau^2 = 0).
* **V3, full self-tuning**: every block uses the rule (James-Stein or preconditioned) with the smaller SURE, and
  the partition is chosen among the same 42 candidates by the smallest total of these per-block SUREs.

## Development data and draws

The two development screens only (Frangieh: all held-out target groups, budgets 50-3,200, 10 draws; Papalexi:
each target held out in turn, budgets 50-1,600, 4 draws per fold), with the paired cells of the development study
(same seeds as `run_study.py`), so V0 reproduces the stored `bjs2/mapped` curve.

## Rule

Criterion: the recovered fraction of within-group cross-correlation (all genes and proteins, pooled over groups)
averaged over the budgets of a screen, then over the two screens. The variant among V1-V3 with the highest
criterion is the self-tuning arm of the generality benchmark. It is carried whether or not it beats V0 in
development, because its purpose is adaptation to data sets unlike the development screens; its development
result is reported either way. No further variants are tried after this rule is applied.

## Amendment (before any development data were used)

The first version of V2 chose tau^2 by minimizing SURE over a grid. A unit test on synthetic Gaussian data
(`selftune.py`, no development data) showed that the minimized SURE is optimistic when the noise of a block is
correlated (mean SURE 1.13 against a mean loss of 1.81 in a block of 30 coefficients), which would bias V3's
per-block choice towards V2. V2 was therefore changed, before `dev_selftune.py` was run on either screen, to the
moment estimate of tau^2 with its exact SURE (synthetic check: mean SURE 2.512 against mean loss 2.525, 0.493
against 0.489 and 10.64 against 10.64 in blocks of 30, 8 and 100 coefficients). The positive-part James-Stein
SURE was unbiased in the same test (1.832 against 1.844).

## Result (dev_selftune.py, 16:57 EDT, 27 September 2026)

V0 reproduced the stored `bjs2/mapped` numbers (largest difference 5e-15 in Frangieh, 0 in Papalexi).
Recovered fraction (%) by budget, and the criterion (mean over budgets):

| | Frangieh 50 / 100 / 200 / 400 / 800 / 1,600 / 3,200 | mean | Papalexi 50 / 100 / 200 / 400 / 800 / 1,600 | mean | criterion |
|---|---|---|---|---|---|
| V0 fixed | 37.4 / 48.1 / 62.9 / 72.4 / 78.9 / 83.9 / 86.9 | 67.22 | 16.0 / 37.1 / 55.1 / 67.3 / 73.0 / 77.6 | 54.34 | 60.78 |
| V1 self-tuned partition | 26.9 / 46.2 / 60.0 / 71.9 / 78.6 / 83.6 / 86.8 | 64.87 | 4.3 / 31.4 / 50.0 / 65.0 / 71.4 / 77.0 | 49.84 | 57.35 |
| V2 preconditioned | 41.4 / 50.5 / 64.2 / 72.1 / 78.4 / 83.5 / 86.6 | 68.09 | 19.1 / 40.3 / 56.0 / 67.7 / 73.1 / 77.8 | 55.66 | 61.87 |
| V3 full | 29.0 / 45.4 / 60.2 / 71.7 / 78.8 / 83.6 / 86.8 | 65.06 | 2.3 / 31.9 / 49.8 / 64.8 / 71.3 / 77.0 | 49.52 | 57.29 |

By the rule, **V2 (preconditioned, self-tuning shrinkage) is the self-tuning arm** of the generality benchmark. It
gained most where paired cells are fewest (50 cells: +4.0 and +3.1 points) and lost at most 0.5 points at large
budgets. Choosing the partition by SURE (V1, V3) lost recovery at small budgets, where SURE is noisy and the
selection among 42 partitions overfits. Time per fit: V2 8 ms (Frangieh) and 4 ms (Papalexi).
