# Deviations from the frozen plan (PLAN.md, frozen 2026-09-27T21:39:36-0400)

## Deviation 1: no cell cap (recorded 2026-09-27, before any extraction)

The plan caps droplet data at 3,000 cells per population, a rule carried over from the generality benchmark, where
it limited memory for data sets of more than 100,000 cells. Applied here it would discard most cells of the three
data sets with a single population (pbmc10k 6,855 cells, malt10k 6,838, bmcite 30,672), shrink their reservoirs
below the budgets at which one-block James-Stein reaches the primary target in development (500-1,600 cells), and
so censor their savings. The largest data set here has about 31,000 cells, which fits in memory. No cap is applied.
This was decided before any file was extracted and before any statistic of any of these data sets was computed;
it changes which cells enter, not any estimator, prediction, endpoint or hypothesis.

## Deviation 2: extraction script renamed (recorded 2026-09-27, before any prediction)

The plan names the extraction code `pdata.py`. A file of that name in this directory shadows
`extension/perturbation/pdata.py`, which the frozen estimator code imports (`from pdata import clr, lognorm`), so
the first attempt to write the data record and to run the predictions stopped at import, before computing anything
(`logs_prospective.txt`). The script was renamed `pextract.py`, unchanged otherwise; the extracted files, which it had
already written, were not regenerated. Because the frozen `freeze.py` hashes a file named `pdata.py` in its data
mode, the data record is written by `freeze_data.py`, which records the same items with `pextract.py`.
