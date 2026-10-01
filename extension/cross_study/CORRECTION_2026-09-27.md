# Correction, 2026-09-27: recorded times of post hoc analyses P1 and P2

`PLAN.md` and `results/plan_sha256.txt` are kept exactly as frozen and hashed on
2026-09-26. This separate, dated note corrects two clock times that they state.
Neither file was edited, and no hash was recomputed or re-recorded.

## Erroneous times

| Record | Stated | Correct statement |
|---|---|---|
| `PLAN.md`, heading of post hoc analysis P1, and the P1 line of `results/plan_sha256.txt` | written 2026-09-26 15:40 EDT | written after scoring (13:58 EDT) and no later than 15:31:56 EDT |
| `PLAN.md`, heading of post hoc analysis P2, and the P2 line of `results/plan_sha256.txt` | written 2026-09-26 16:05 EDT | written after P1's results (15:30:53 EDT) and no later than 15:31:56 EDT, before P2 was computed (15:41:33 EDT) |

Both stated times are later than the last modification of the files that
already contain the two sections, so they cannot have been read from the
system clock when the sections were written.

## Evidence

File-system modification times on the machine that ran the analyses (clock set
to America/New_York). Reproduce with `stat -c '%y %n' <file>` in this folder.
The handoff archives keep these times.

| File | Last modified (EDT, 2026-09-26) | Meaning |
|---|---|---|
| `estimators.py` | 12:50:41 | rank cap of amendment 1 implemented |
| `results/predictions.npz`, `results/predictions_manifest.json` | 13:58:12 | frozen predictions written |
| `results/scores.json`, `logs/evaluate.log` | 13:58:54, 13:58:52 | sealed scoring done |
| `posthoc_within.py` | 15:30:24 | P1 code |
| `results/posthoc_within.json`, `logs/posthoc_within.log` | 15:30:53 | P1 results |
| `PLAN.md` | 15:31:56 | last change; the file contains P1 and P2 |
| `results/plan_sha256.txt` | 15:31:56 | last change; contains the P1 and P2 hash lines |
| `posthoc_mixture.py` | 15:38:23 | P2 code |
| `results/posthoc_mixture.json`, `logs/posthoc_mixture.log` | 15:41:33 | P2 results |
| `results/mixture_diagnostics.json` | 15:43:19 | P2 diagnostics |

## What the records do and do not establish

* P2: the stated order holds. Its text reports P1's result (86 identified
  directions), so it was written after P1 was computed, and it was in the plan by
  15:31:56, about ten minutes before P2's code was last changed and its results
  were written. Only the clock time was wrong.
* P1: written after scoring, as stated. A last-modification time cannot show
  when the P1 text was first added, so these records neither confirm nor
  contradict the statement in `results/plan_sha256.txt` that P1 was added before
  it was computed (P1's results were written at 15:30:53, 63 seconds before the
  plan's last change, which added P2). That statement should be treated as
  unverified.
* Amendment 1 (stated 12:50 EDT, before any recipient prediction): consistent
  with the records above.

## Consequences

None for any result or claim. P1 and P2 are labelled post hoc and exploratory
in the plan, in `README.md` and in the manuscript (Methods, "Post hoc
analyses"; the legend of the between/within-type figure), and no confirmatory
test depends on when they were written. The frozen protocol, amendment 1 and
the sealed scoring are unaffected.
