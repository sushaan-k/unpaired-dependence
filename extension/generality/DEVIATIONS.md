# Deviations from the frozen plan

`PLAN.md` and `results/freeze.json` are kept exactly as frozen (17:26:25 EDT, 27 September 2026). Deviations are
recorded here with times from the system clock.

## Deviation 1 (recorded 17:27 EDT, 27 September 2026, before any held-out statistic of Stephenson was computed)

**What.** `stephenson.npz` was written before `gdata.py` converted string arrays to fixed-width Unicode, so its gene,
antibody and label arrays are object arrays, which the frozen loader (`np.load(..., allow_pickle=False)`) refuses.
The first attempt to run Stephenson stopped at loading, before any computation (`logs/stephenson.log`).

**What was done.** Nothing frozen was changed: the data file keeps the hash in the freeze record and `grun.py` is
unchanged. `with_object_arrays.py` runs the frozen scripts with `numpy.load` allowed to read object arrays for this
one file; the values read are the same strings. Stephenson is run as
`python with_object_arrays.py grun.py run stephenson`, and the structural law as
`python with_object_arrays.py glaw.py predict`.

## Amendment 1 (recorded 18:02:38 EDT, 27 September 2026, after stephenson, colon and hao had been scored and before any other data set had been scored)

**What was seen.** In stephenson and colon the fully paired reference (every training cell used as a paired cell,
standardized within each population and not shrunk) had a negative recovered fraction (stephenson -10.4%, colon
-2.4%). A post hoc diagnostic (`posthoc/diag_reference.py`) shows why. In the large conditions the reference
recovers 42-87% (stephenson) and up to 67% (colon). In conditions with few cells (for example 4 to 33 reservoir
cells) the unshrunk reference is dominated by noise, and its squared norm (up to 1,300) swamps the noise-unbiased
denominator. The plan's targets are fractions of this reference, so they are negative, every curve starts above
them, and every saving is 1 by construction. The rule does not measure what it was meant to measure. The shrinkage
estimators do not share the problem: in colon the proposed estimator reached 11.1% at 100 paired cells and 25.3% at
800, and the paired-only envelope 7.7% at 800.

**What is added.** Nothing frozen is changed, and the plan's analysis is reported as specified (G1-G4 with targets
relative to the fully paired reference). An amended target rule is added and reported alongside it. Targets are
fractions 0.5 (primary), 0.25 and 0.75 of the largest recovered fraction that any method reached at any budget. The
methods are the proposed and self-tuning arms and every comparator, pooled, per condition or mapped; the fully
paired reference and the ablation are excluded. Bootstrap resamples recompute this level. Savings, censoring,
intervals, the hypotheses G1'-G3' (geometric means over the nine data sets) and the structural law G4' (observed
savings at the amended primary target, including the three earlier data sets from their stored curves) are
otherwise as in the plan. `grun_amend.py` computes them from the hashed predictions. It first reproduces every
saving of the frozen rule exactly, as a check, and writes `results/<name>_amend1.json` and
`results/summary_amend1.json`. For stephenson, colon and hao the amendment is post hoc. Hao finished scoring while this record was being
written, and its printed summary line was seen; its details were not examined. For the other six data sets the
amendment was recorded before they were scored.

## Amendment 2 (recorded 18:06:44 EDT, 27 September 2026, after stephenson, colon, hao and scala_m1 had been scored and before any other data set had been scored)

**What was seen.** In scala_m1 no method recovered positive dependence at any common budget (25-100 paired
cells). Every recovered fraction was negative, and cross-validated low rank predicted zero (recovered fraction 0
exactly). The fully paired reference was -2.7%. Amendment 1's level (the largest recovered fraction any method
reached) is then 0, so its target is 0. That target is met by predicting zero, and a saving computed at it does not
measure paired-cell efficiency.

**What is added.** A data set is informative for savings only if the amended level has a bootstrap 95% lower bound
above zero, that is, if some method recovers dependence in held-out units. Otherwise its savings are reported as
not estimable and it is left out of the geometric means G1'-G3', which are reported both with this rule and without
it (with the censored savings of every data set). The rule applies to every data set. It is post hoc for the four
data sets already scored and was recorded before the other five and the secondary analysis were scored.
`grun_amend.py` already stores the level and its interval (`level`, `level_ci`); the summary applies the rule.

## Erratum to amendments 1 and 2 (recorded 19:06 EDT, 27 September 2026)

The recovered fractions of the fully paired reference quoted above were read as percentages from values stored as
fractions, so they are 100 times too small in magnitude. The correct values, from `results/<name>.json`
(`reference_rf_paired_all`), are stephenson -1,042%, colon -237% and scala_m1 -274%. The later-scored
bmmc_multiome was -148%. The per-condition figures (42-87% in stephenson's large conditions, squared norms up to
1,300) and the colon recoveries (11.1% at 100 paired cells, 25.3% at 800; paired-only 7.7%) were quoted correctly.
The references are negative either way, so the reasoning and the amendments are unchanged. In hao, which had been
scored when amendment 1 was recorded, the reference was 70.0%. Its prespecified target (35%) was reached neither by
the proposed estimator (30.3% at 800 paired cells) nor by the paired-only envelope, so its prespecified saving is 1
because both curves were censored, not because the target was negative.

## Deviation 2 (recorded 19:19 EDT, 27 September 2026; computational, no change of analysis)

The frozen `glaw.py predict` builds every per-cell product vector of a block. With a reservoir of about 12,000
cells (blood) and a 140 x 197 block this takes more than 2.6 GB, twice over, and the process was stopped by the
memory limit three times, before writing any output. `law_lowmem.py` runs the frozen `glaw.predict()` with a
`block_moments` that computes the same two quantities from the identity
sum_i ||z_i - m||^2 = sum_i ||x_i||^2 ||y_i||^2 - N ||m||^2. On three data sets its values equal the frozen
function's to relative differences below 1e-13, and the predicted savings agree to 15 digits
(`python law_lowmem.py check`). The predictions are made with
`python with_object_arrays.py law_lowmem.py predict`.

**Note on deviation 2** (recorded 20:01 EDT). Run alone, with nothing else in memory, the frozen
`glaw.py predict` later completed (`finalize.sh`, 19:41 EDT). Its output replaced that of `law_lowmem.py`
(19:19 EDT) in `results/law_predicted.json`, and the two agree to 15 digits in every predicted saving. The reported
predictions therefore come from the frozen code. `law_lowmem.py` remains as the check.

## Clerical note (recorded after scoring)

The data-set table of `PLAN.md` gives 264 mice for scala_m1. The extraction that was frozen (hash in
`results/freeze.json`) excludes cells whose RNA family is "low quality", and so has 263 mice and 1,203 cells. That
extraction is the one analysed. The count in `PLAN.md` was taken from an earlier extraction before the exclusion,
and the frozen file is not edited.
