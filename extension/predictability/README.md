# Predicting the saving of paired cells (theory, development and a prospective test)

How many paired cells does the unpaired block structure save, and can that number be predicted? Theory:
`THEORY.md` (manuscript Supplementary Note 7, Propositions S9-S11 in the 29 September 2026 revision; `THEORY.md` keeps its development labels S13-S15). Development on the nine benchmark data sets
(post hoc): `DEV.md`. Prospective test on ten further data sets: plan `PLAN.md`, frozen 2026-09-27 21:39:36 EDT
before extraction (`results/freeze.json`); data record 21:56:40 EDT (`results/data_freeze.json`); deviations
`DEVIATIONS.md`; results `RESULTS.md`.

## Theory in one paragraph

For the implemented rule (block means of non-Gaussian per-cell products, a noise trace estimated from the same
cells, positive part), each block's root-risk is within (36 l_b/B + 22 kappa_b tau_b/B^2)^(1/2) of the oracle-linear
root-risk, which depends only on the block's signal r_b^2 and per-cell noise trace tau_b (S13). The saving of block
shrinkage over the same shrinkage without unpaired structure is then a function of the signal shares w and noise
shares pi alone: at least 1, equal to 1 if and only if w = pi, at most 1/min pi_b over blocks carrying signal, tending
to 1 + chi^2(w || pi) at low accuracy and to the inverse noise share of the signal blocks at high accuracy (S14).
Unpaired cells reveal pi (exactly under independence; up to r_b^2 for Gaussian data) but not w: every saving in
[1, 1/min pi] and every budget above a minimum is compatible with the same unpaired data (S15). A pilot of m paired
cells estimates r_b^2 without bias, with relative error about 2 (d_eff m r_b^2/tau_b)^(-1/2).

## Files

| File | Role |
|---|---|
| `THEORY.md` | Propositions S13-S15 with proofs |
| `check_theory.py` | Numerical checks of S13-S15 (`results/check_theory.json`; not proofs) |
| `ptools.py` | Roles, unpaired summaries (the proposed estimator's, unchanged), fast block and one-block James-Stein (equal to `sp_estimators` to 1e-15), recovered fraction against truth cells, block moments, the law's curves and closed form, bootstrap law |
| `dev.py`, `dev2.py`, `dev_summary.py`, `dev2_summary.py` | Development (post hoc) on the nine benchmark data sets (`results/dev/`) |
| `results/transfer_profile.json` | Development signal-share profile used for the zero-pair prediction |
| `PLAN.md`, `DEV.md` | Frozen plan and development record |
| `freeze.py` | Plan freeze (`results/freeze.json`) |
| `pextract.py` | Extraction of the ten prospective data sets (written after the freeze; Deviation 2 renamed it from `pdata.py`) |
| `freeze_data.py` | Data record (`results/data_freeze.json`; Deviation 2) |
| `prun.py` | Prospective runner: `predict` (hashed in `results/<name>_manifest.json` before any truth-cell statistic), `evaluate`, `summary` (H1-H6) |
| `psecondary.py` | Prespecified secondary analyses (bootstrap law, curves) and exploratory ones (censoring, bias) |
| `make_assets.py` | Manuscript macros, figure and table (`revision/source/pred_*.tex`) and `RESULTS.md` |
| `run_extract.sh`, `run_prospective.sh`, `run_dev*.sh` | Run queues used |
| `test/` | Harness test of `prun.py` on development data sets (not a result) |

## Results (details in RESULTS.md)

* Development (post hoc): the law was within 22% of the observed saving in all 14 estimable problems (median 7%,
  savings 1.4-14.7); random bases gave no saving.
* Prospective, prespecified: 29 problems from ten data sets; 10 problems from four data sets had an estimable saving
  at the primary target (half of the population's dependence recovered). H1-H6 were all met: law median error 21%
  (95% interval over data sets 11-29%), Spearman 0.98 (P < 0.001), random-basis control 1.00, pilot median error
  12% (4-147%), zero-pair transfer 40%, no observed saving outside the unpaired bounds.
* The law overpredicts: observed/predicted 0.83 at the primary target, 0.57 at 0.3, 0.96 at 0.7. The bias sits in the
  proposed estimator's small blocks (it needed 1.23x the predicted cells; paired-only 1.04x); a bootstrap law removed
  it (median error 6%). For sparse SNARE-seq chromatin the proposed estimator needed 783-1,357 cells against
  396-408 predicted.
* Not estimable: three CITE-seq data sets without cell-type conditions (target reached at 25 cells), the two
  chromatin data sets and the 613-cell Patch-seq data set (targets not reached within the budgets).

## Reproduce

```bash
python check_theory.py
python dev.py run <name> [<panel>]; python dev2.py <name>; python dev_summary.py; python dev2_summary.py
python pextract.py <name>                  # raw files: see PLAN.md
./run_prospective.sh                       # all predictions, then all evaluations, then the summary
python psecondary.py && python make_assets.py
python ../verify_predictability.py [--data]
```
