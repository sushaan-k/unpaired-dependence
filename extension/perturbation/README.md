# Learning without pairs in the melanoma screen (sealed)

Frangieh et al. (2021) Perturb-CITE-seq (scPerturb release,
Zenodo 7041849; SHA-256 in `pdata.py`): patient-derived melanoma cells with
knockouts of 248 genes, cultured alone, with interferon-gamma or with
autologous T cells, RNA and 20 surface proteins. Manuscript: Results "Without
paired cells, population differences reveal dependence only where populations
differ", Fig. 6b,c, Methods "Learning without pairs", Supplementary Note 9.

## Order of events

1. `extract.py`: held-out targets (every fourth in SHA-256 order; 62 targets),
   hash splits into adaptation and scoring files, `results/seal.json`.
2. `dev.py`, `dev_dryrun.py` (training targets only): choice of the
   condition-centred channel as primary (`results/dev.json`, `dev_cf.json`).
3. `fit.py`: channels and comparators on training targets (`results/channels.*`).
4. `freeze.py`: SHA-256 of `PLAN.md`, code and fits at 07:20:18 EDT, 27 September
   2026, before any statistic of held-out or non-targeting cells other than
   library sizes (`results/freeze.json`).
5. `predict.py`: predictions and their digests, then `results/manifest.json`,
   before the scoring files were opened.
6. `evaluate.py`: endpoint, intervals and verdicts (`results/evaluation.json`).
7. `PLAN_programs.md`, `freeze_posthoc.py` (08:18:28 EDT), then `programs.py` and
   `design.py`: program-level recovery and perturbation choice, specified and
   hashed before computation but after the scoring cells had been opened.

## Results in brief

* Held-out targets (141 groups from 51 targets): pairing-free recovery 6.8%
  (4.1-9.4%) of within-condition dependence, above pseudo-population and
  derangement controls; 19% in non-targeting cells; paired closed form 37%, so
  the share criterion (H4) failed. H1-H3 met.
* Interferon program (Hallmark interferon-gamma and -alpha sets): 43% (32-51%)
  pairing-free against 64% (paired closed form) and 94% (paired correlations);
  5.7% outside the program. P1 and P3 met, P2 not met.
* Removing the five targets of strongest RNA effect or highest exposure left no
  supported direction, unlike any of 50 random removals; the prespecified
  exposure criteria (D1) were not met.

`make_assets.py --out ../../revision/source` writes the manuscript macros,
Fig. 6 and the tables, and `results/RESULTS.md`. Checks:
`python ../verify_perturbation.py`. Raw counts are not redistributed.
