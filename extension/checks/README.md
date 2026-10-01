# Checks added in revision

## Draw scoring (`draw_scoring.py`)

Every recovery curve in the paper is the mean, over the draws of paired cells at a budget, of the score of each
draw's prediction on its own. For a prediction C and held-out truth T the score is G(C) = 2<C, T> - ||C||^2, and

    G(mean_d C_d) = mean_d G(C_d) + mean_d ||C_d - mean_d C_d||^2,

so scoring the averaged prediction would credit a study with the paired cells of all its draws. The tests store each
arm's mean prediction and msq, the mean over draws of ||C_d||^2, and compute 2<mean C, T> - msq, which is
mean_d G(C_d) exactly; ||mean_d C_d||^2 is never used. The check reruns the frozen prediction code of each test
unchanged, watches each draw's prediction at the line where the code accumulates it (a trace function reads the local
variables; the code itself is not modified), scores it on its own, and compares:

1. the regenerated mean predictions and msq with the stored ones (the rerun is the published run);
2. the curve averaged over per-draw scores with the published curve (or, for a partial rerun, with the stored terms
   of each held-out unit);
3. the published curve with the curve that scoring the averaged prediction would have given.

| Command | What it reruns | Time (two cores) |
|---|---|---|
| `python draw_scoring.py identity` | nothing: the identity above on random matrices | seconds |
| `python draw_scoring.py external` | nothing: the external test stored every draw's prediction | 1 min |
| `python draw_scoring.py deployment` | the bone-marrow test, all 12 held-out batches | 15 min |
| `python draw_scoring.py validation --from-review` | nothing: reads the per-draw scores of the validation test's regeneration in `../second_review` | 1 min |
| `python draw_scoring.py validation` | the validation test itself, all 34 held-out samples | 6-10 h |
| `python draw_scoring.py law NAME` | one data set of the prospective test of the law | 1-10 min |
| `python draw_scoring.py benchmark NAME --folds 0` | the first of the three folds of one benchmark data set (`stephenson` through `../generality/with_object_arrays.py`, as the benchmark reads that file) | 5-40 min |
| `python draw_scoring.py report` | nothing: `results/draw_scoring.json` and a table | seconds |
| `python draw_scoring.py verify` | nothing: recomputes every curve from the per-draw scores in `results/` | seconds |

`verify` needs no data: `results/<test>.json` holds, for every rerun held-out unit, arm and budget, the score of every
draw, with the units' denominators and the published curves beside them. Each benchmark data set was rerun for the
first of its three folds, its three held-out units; there `verify` compares the first fold's mean per-draw scores with
the fold's stored terms, 2<mean C, T> - msq, and the published curve with the curve pooled from the stored terms of all
three folds. The other commands need the public data
(`../DATA.md`) and write regenerated prediction files only under `results/scratch/`, which they delete after the
comparison. `run_all.sh` runs everything in sequence and logs any step that fails without stopping the others.

The development comparison needs no rerun: `semipaired/run_study.run_fold` scores each draw's prediction inside the
draw loop (`add` calls `common.scores`) and averages the scores; the validation test's readout does the same.
