# Replication in an independent screen (sealed)

Papalexi et al. (2021) ECCITE-seq (scPerturb release, Zenodo
7041849; SHA-256 in `rdata.py`): THP-1 cells stimulated with interferon-gamma,
knockouts of 25 genes of the interferon-gamma and PD-L1 pathways, three
biological replicates, four surface proteins. The estimator and selection
rules of `../perturbation` were reused unchanged, with the replicate in the
role of the condition. Manuscript: Limitations, Methods "Learning without pairs"
and Supplementary Note 10.

Order: `extract.py` (train/test halves per target and replicate, seal), a dry
run on training halves (`logs/dev_dryrun.*`), `freeze.py` at 08:18:29 EDT on 27
September 2026, `predict.py` (leave-one-target-out fits and predictions, then
`results/manifest.json`), `evaluate.py`. `posthoc_penalty.py` was added after
unsealing.

Results: the frozen estimator retained 18-61 directions from the 64 training
populations of each fold and was worse than independence over all genes and proteins
(-212%) and in the interferon block (-3.0%); all five prespecified criteria
failed. With the penalty fixed after unsealing at the first screen's value
(one direction), interferon-block recovery was 32% but recovery over all genes
and proteins -76%; a tenfold larger penalty gave 45% and 4.6%.

`make_assets.py --out ../../revision/source` writes the macros and table.
Checks: `python ../verify_perturbation.py`.
