# Post hoc diagnostics of the generality benchmark (not part of the frozen analysis)

* `diag_reference.py <dataset> [fold]`: per condition of one fold, the held-out target's size (||T||^2 and the
  noise-unbiased <T_A, T_B>) against the fully paired reference prediction (<C, T>, ||C||^2), from the hashed
  predictions. It showed why the plan's reference-relative targets failed in stephenson and colon: the unshrunk
  reference is dominated by noise in conditions with few cells (DEVIATIONS.md, amendment 1).

Run as `python posthoc/diag_reference.py colon 0` (stephenson through `with_object_arrays.py`).
