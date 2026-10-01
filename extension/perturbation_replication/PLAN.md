# Replication in an independent perturbation screen (sealed)

Question: does the mechanism found in the Frangieh screen (perturbation-induced
population variation exposes directions along which pairing-free transfer
recovers within-condition RNA-protein dependence, strongly within the
interferon program, and a few perturbations predicted from RNA alone determine
whether it is learnable) hold in an independent screen, with the estimator and
selection rules frozen?

Written on 27 September 2026 after the split and seal (`extract.py`,
`results/seal.json`) and before any statistic of test or non-targeting cells
other than library sizes was computed. Only training-half cells were read for
the gene panel and for code checks (`logs/dev_dryrun.log`: the pipeline run on
training halves, with two training targets standing in for test targets).
`freeze.py` records the SHA-256 of this file and the code.

## Data

Papalexi et al. 2021 ECCITE-seq screen (scPerturb release, Zenodo 7041849;
SHA-256 in `rdata.py`): 20,729 THP-1 cells stimulated with interferon-gamma,
knockouts of 25 genes of the interferon-gamma and PD-L1 pathways and
non-targeting guides, three biological replicates (rep1, rep3, rep4; the 77
cells of rep2 are not used), RNA and four surface proteins (CD86, PD-L1, PD-L2,
CD366). One guide per cell; targets are read from guide names.

## Frozen rules reused from `extension/perturbation`

Gene panel (encoding genes detected in at least 1% of training cells, then the
most variable genes up to 200, from training halves only), preprocessing, the
condition-centred pairing-free channel with the replicate in the role of the
condition (populations are target x replicate, centred within replicate),
cross-validation over targets, the matched controls (pseudo-populations and
derangements within replicate, ten draws each), the paired comparators, the
own-pairs benchmark, eligibility (training populations with at least 20 cells in
each half; test groups with at least 40 adaptation cells), the recovered
fraction, the program blocks of `PLAN_programs.md` (interferon genes: Hallmark
interferon-gamma and interferon-alpha response sets on the panel; interferon
antibodies: those whose encoding genes are in these sets, CD86 and PD-L1), and
the exposure score.

## Design

The screen has 25 targets, so every target with an eligible test group is held
out in turn (leave-one-target-out). Each target x replicate group was split by
cell hash into a training half and a test half; a target's training half trains
the channels of every other fold, and its test half is used only in its own
fold (adaptation cells as inputs, scoring cells for the endpoint). Channels for
the non-targeting cells (three groups, one per replicate) use all targets.

Arms: the primary pairing-free channel (measured RNA correlation); with corrected
RNA correlation; controls; paired closed form; reference regression; transferred
correlation of the replicate; own adaptation pairs (ridge); the adaptation half
used directly as a replicate measurement; and, per fold, the channel refitted
without the 5 training targets of highest exposure, without the 5 of highest
effect strength, and without 5 random training targets (50 draws, seed
20261019).

## Endpoint and uncertainty

Recovered fraction pooled over all eligible test groups, and for the
interferon block, its complement and the supported subspace of each fold.
Intervals: 2,000 bootstrap resamples of test targets (seed 20261020); for
non-targeting cells, of guides within replicates. Recovery is also reported
separately for each biological replicate.

## Hypotheses (95% lower bounds)

* R1: pairing-free recovery above 0 over all test groups.
* R2: above the pseudo-population and the derangement controls.
* R3: in the interferon block, pairing-free recovery at least half of the paired
  closed form's.
* R4: above 0 in non-targeting cells.
* R5: recovery after removing 5 random training targets minus recovery after
  removing the 5 of highest exposure above 0.

## Deviations

None.
