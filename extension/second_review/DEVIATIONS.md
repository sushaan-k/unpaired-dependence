# Deviations from PLAN.md in running it

Recorded while the runs were in progress (30 September 2026, EDT). None changes a script, a setting or a value; the
plan and every script remain as hashed in `results/freeze.json`.

1. **stephenson.npz read with the benchmark's reader (07:15).** `benchmark_review.py run stephenson` stopped at
   07:06 because `stephenson.npz` stores its gene and protein names as object arrays, which `numpy.load` reads only
   when allowed to. The benchmark met the same file and recorded the fix as its Deviation 1
   (`../generality/DEVIATIONS.md`): `../generality/with_object_arrays.py` runs a frozen script unchanged with
   `numpy.load` allowed to read that one file's object arrays. `benchmark_review.py run stephenson` and the
   draw-scoring check's `draw_scoring.py benchmark stephenson` were run through it; `run_all.sh` and
   `../checks/run_all.sh` now do the same. The wrapper runs a script with `runpy`, which does not put the script's
   folder on the module path, so the first such run of `benchmark_review.py` (07:49) stopped at its import of
   `validation_review`; it was run again with this folder on `PYTHONPATH` (`results/logs/lane_d.sh`).
2. **The atlas bootstrap rerun.** Its first start, at 07:06 beside both parts of the validation review, was stopped
   by the memory limit of the 8 GB machine before it had written or logged anything. Part 1 of the validation review
   was then stopped between two folds (07:49), and the atlas bootstrap was started again unchanged, with the same
   replicates and seeds (08:10); part 1 was restarted when part 0 had finished (08:42) and resumed at its first
   unfinished fold, as the script does. `results/logs/lane_c.sh` and `lane_d.sh` record the order and `lane_c.out`
   the times.

The lane scripts in `results/logs/` (`lane_a4.sh`, `lane_b3.sh`, `lane_c.sh`, `lane_d.sh`) are the records of how
the runs were scheduled on the two cores of the machine; they refer to process numbers of that session and are not
meant to be run again. `run_all.sh` lists the same runs in order.
