# Measured identified intervals (post hoc extension, 2026-09-26)

## Why this analysis

The paper's central theory (released main text Eq. 2; Supplementary Information, Proposition S1)
says that, with a fixed source interaction, a recipient described only by its
individual marker frequencies identifies each binary RNA-protein query only up
to an interval. The released paper never computed that interval in a real
recipient; its Limitations section says "The examples establish ambiguity, not
its biological magnitude." The empirical results compare reconstructions under
different recipient information, but do not show how much the frequency-only
information leaves undetermined.

This extension measures the interval in every recipient of the adaptation-only
analysis (24 Cambridge, 56 Newcastle, 15 T-ALL) and in all 11 colon donors,
and asks whether held-out cells agree with the direction chosen by the
recipient's own assay patterns when that direction differs from the
frequency-only prediction.

## What was done

- `PLAN.md` fixed the analysis before any summary was computed
  (SHA-256 `f2f31b50...c23c2`, recorded in `results/summary.json`). All
  analyses are retrospective. The frozen colon protocol and its endpoint are
  untouched.
- Inputs are released arrays only: the unchanged source interaction
  (`source_fit.npz`), the recipients' smoothed adaptation histograms, the
  released frequency-only and full-pattern predictions, and the released
  held-out 2x2 tables. No raw data, refitting, or new recipients.
- **Certified inner bounds.** Every value used as an interval endpoint is
  attained by an explicit pair of strictly positive assay laws with the
  recipient's 18 marker frequencies, reconstructed with the fixed interaction.
  So the reported interval lies inside the sharp one. The search runs over
  star laws, in which the other markers are conditionally independent given
  the queried marker. A short first-order argument (Supplementary Information,
  Proposition S2, a corollary of the response expansion) shows these laws
  attain the extremes of the weak-interaction identified range. The search uses L-BFGS-B with exact adjoint gradients from
  two deterministic starts per direction. No global optimum is claimed.
- **Minimax bound.** Proposition S1 applied to the inner interval gives a
  per-query lower bound on the worst-case deviance of any frequency-only
  prediction.
- **Reversals.** A query is a reversal when the frequency-only and
  full-pattern predictions lie on opposite sides of independence. Both are
  attained members of the same identified set, so each reversal certifies
  direction ambiguity without any extreme construction. The analysis records
  whether the held-out direction agrees with full patterns. The bootstrap is
  paired by recipient, with Bonferroni correction across four cohorts.

## Results

`results/RESULTS.md` is generated from the result files and gives every number.
In brief:

- Individual frequencies left the association direction unidentified for every
  one of the 8,586 recipient-queries (S1), including all same-marker pairs.
- When recipient patterns reversed the frequency-only direction (19-28% of
  queries; at least one reversal in every person), held-out cells agreed with
  the patterns in 59.6%, 75.3%, 84.5% and 68.0% of reversals in Cambridge,
  Newcastle, T-ALL and colon. All four Bonferroni lower bounds exceed 50%.
- Unfavourable, prespecified descriptive result: interval size did not predict
  which queries gained from patterns (S4).

## Reproduce

Use the same environment as the handoff (`../../requirements.txt`). Keep this
folder at `extension/identified_set` next to `release/`.

```bash
python run_identified.py compute --shard 0 --of 2   # 4.3 CPU-hours in total (~100 s per recipient)
python run_identified.py compute --shard 1 --of 2   # shards skip finished recipients, so reruns resume
python run_identified.py merge                      # re-certifies all 17,172 endpoints
python sensitivity.py                               # unrestricted search, first colon donor
python analyze_identified.py                        # S1-S4 -> results/summary.json
python s3_sensitivity.py                            # added S3 checks
python make_manuscript_assets.py --out ../../revision/source
python -m unittest -v test_identified.py
python verify_extension.py                          # add --full to re-certify all endpoints
```

## Files

| File | Content |
|---|---|
| `PLAN.md` | Post hoc analysis plan, fixed before any summary was computed |
| `identified.py` | Star laws, matrix scaling, adjoint gradients, first-order range, minimax, certification |
| `run_identified.py` | Per-recipient computation and merge with re-certification |
| `analyze_identified.py` | Summaries S1-S4 |
| `s3_sensitivity.py` | Two added S3 checks (labelled as not prespecified) |
| `sensitivity.py` | Planned unrestricted local search for the first colon donor |
| `make_manuscript_assets.py` | Generates LaTeX macros, figure, table and `results/RESULTS.md` |
| `test_identified.py` | Unit tests |
| `verify_extension.py` | End-to-end verification |
| `results/star_endpoints.npz` | Certified endpoints and their star parameters (106 x 81) |
| `results/recipient_queries.csv` | All 8,586 recipient-query rows |
| `results/summary.json`, `results/s3_sensitivity.json`, `results/sensitivity_free_search.json`, `results/certification.json` | Summaries |

## Scope and caveats

- Intervals are inner bounds from local searches over a restricted family.
  The unrestricted search in one donor widened every one of its 81 intervals,
  so the widths and minimax bounds are conservative. The direction result (S1)
  can only strengthen.
- The smoothed adaptation frequencies are treated as exact. Extreme compatible
  populations need not be biologically plausible. The reversal analysis does
  not rely on them.
- Per-query minimax bounds do not bound the worst case of the averaged loss.
- Held-out directions come from 256 cells, and their noise pulls agreement
  toward one half.
- Colon results here are post hoc analyses of the prespecified test's data.
  They are not part of its frozen endpoint.
