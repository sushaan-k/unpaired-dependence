# Program-level recovery and perturbation choice (perturbation test, part 2)

Written on 27 September 2026 after the sealed perturbation test
(`PLAN.md`, `results/evaluation.json`) and in response to review, which asked
(i) whether the 6.8% aggregate recovery hides accurate recovery of one biological
program, with programs defined independently of held-out outcomes and compared
with paired transfer and measurement repeatability, and (ii) whether a rule
computed from unpaired data predicts which perturbations to measure.

Status: the held-out and non-targeting scoring files were opened by
`evaluate.py` for the aggregate endpoint. These analyses are therefore not
sealed. Their definitions, code and hypotheses are hashed by
`freeze_posthoc.py` before any of their statistics is computed; no program-level
or subset-level quantity of held-out cells had been computed before. The fitted
channels, predictions and groups of the sealed test are used unchanged.

## A. Program-level recovery (`programs.py`)

Programs are MSigDB Hallmark gene sets (downloaded from gsea-msigdb.org on
27 September 2026; SHA-256 in `programs.py`) intersected with the frozen panel.
The interferon program is the union of the Hallmark interferon-gamma and
interferon-alpha response sets; its proteins are the antibodies whose encoding
genes belong to that union (by the encoding table of `pdata.py`). Other
programs with at least five panel genes are paired with all 20 antibodies.

Blocks of the within-group cross-correlation, and their roles:

* `ifn`: interferon genes x interferon antibodies (primary block).
* `along_S`: the RNA score along the primary channel's single supported
  direction (from the training fit) x all antibodies; this is where the method
  predicts support.
* `complement`: all other genes x all other antibodies.
* `ifn_genes` and one block per other program: program genes x all antibodies.

Arms: the primary pairing-free channel, the paired closed form, the transferred
within-condition correlation, reference regression, the group's own adaptation
pairs (ridge), and the group's adaptation half used directly as an independent
replicate measurement of its cross-correlation (measurement repeatability).
Endpoint: recovered fraction of each block (as in `PLAN.md`); split-half
reliability of each block's scoring target. Support of a program: the median
over held-out groups of the mean squared correlation between its genes and the
RNA score along S, from adaptation RNA only.

Hypotheses (95% intervals from 2,000 resamples of held-out targets, seed 20261016):

* P1: in the `ifn` block, pairing-free recovery is at least half of the paired
  closed form's (lower bound of the share above 0.5).
* P2: the same along S.
* P3: pairing-free recovery is higher in the `ifn` block than in the
  complement (lower bound of the difference above 0).
* Descriptive: rank correlation between support and pairing-free recovery
  across programs; all arms and blocks; non-targeting cells.

## B. Which perturbations make a program learnable (`design.py`)

Exposure of a training target is computed from RNA only (training RNA halves):
summed over its condition populations, d' R_c d - tr(R_c^2)/n, with d the
population's RNA mean shift from its condition mean and R_c the pooled
within-population RNA covariance of the condition (pooled within-population SD
units), n its RNA cells. It measures how far a perturbation moves RNA along
directions that vary within conditions. Effect strength is |d|^2 - tr(R_c)/n.

Matched-count ablation: the primary channel is refitted by the frozen rules on
all 186 training targets except the k with the highest exposure, except the k
with the highest effect strength, and except k random targets (50 draws, seed
20261017), for k = 5, 10 and 20, and evaluated on the held-out groups and
non-targeting cells with the frozen predictions' inputs, for the full
cross-correlation and for the interferon block of analysis A. The channel is
also refitted without each training target in turn.

Development on training targets only (47 of them in the held-out role, as in
`dev.py`; `dev_design2.py` and `dev_design3.py` in `logs/`, exploratory) showed
that channels fitted on small random or exposure-selected subsets (10 to 40
targets) were unstable (penalties chosen by cross-validation on few targets left
many spurious directions), whereas removing a few targets from the full set left
the estimator stable. Forward selection was therefore replaced by this ablation
before any held-out statistic of analysis B was computed.

Hypotheses (95% intervals from 2,000 resamples of held-out targets, all sets
rescored in each resample, seed 20261018 + 10k + block):

* D1 (primary; k = 5 and k = 10 separately; full cross-correlation and
  interferon block): recovery after removing k random targets (mean over draws)
  minus recovery after removing the k targets of highest exposure has a lower
  bound above 0, and the latter lies below the 5th percentile of random removals.
* D2 (secondary): the same comparison for effect strength, and k = 20.
* D3 (secondary): rank correlation between exposure and the loss of held-out
  recovery when a single target is left out (permutation P, 10,000 permutations);
  redundancy among targets that move the same direction limits single-target
  effects.

## C. Replication in an independent screen

Specified separately in `extension/perturbation_replication/PLAN.md`
(Papalexi et al. 2021 ECCITE-seq), sealed.

## Deviations

None.
