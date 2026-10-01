# Synthesis: manuscript assets that combine studies

Nothing here fits a model. The scripts read stored results of the other folders and write manuscript files.

| Script | Reads | Writes |
|---|---|---|
| `savings_summary.py` | the external test (`../semipaired/external`) and the nine benchmark data sets (`../generality/results/*_amend1.json`, with their bootstrap resamples) | `results/savings_summary.json`: each test data set's saving over paired-only estimation at its primary target, and the geometric mean over the informative ones with an interval that combines the r-th bootstrap resample of every data set (post hoc: computed after all tests had been scored) |
| `make_assets.py` | `results/savings_summary.json`, `../deployment/results/` and `../deployment/results_posthoc/`, `../validation/results/`, `../validation/readout/results/`, `../validation/results_posthoc/`, `../validation/posthoc_review/results/`, `../predictability/results/rows.json`, `../semipaired`, `../cross_study`, `../perturbation` | `synth_macros.tex`; Fig. 2 (`atlas_figure.tex`: the validation test with the other methods given the same atlas, its readout, and the atlas's size and mix of cell types), Fig. 3 (`transfer_figure.tex`: the bone-marrow test and the fly connectome), Fig. 4 (`savings_figure.tex`) and Fig. 6 (`pairfree_figure.tex`); SI tables of the savings in every test data set (`savings_table.tex`), the bone-marrow test (`deploy_table.tex`, `deploy_site_table.tex`, `deploy_contrast_table.tex`, `fly_table.tex`), the validation test (`val_table.tex`, `audit_table.tex`), its donor allocation (`allocation_table.tex`), its readout (`readout_table.tex`) and the readout's calibration (`calibration_table.tex`, `multiplicity_table.tex`), the other methods (`alternatives_table.tex`) and its robustness (`robust_table.tex`, `pool_table.tex`, `donor_table.tex`). After scoring, it also recomputes the validation's saving with the budget axis counted in contrast rows and compares the bone-marrow paired-only control under both centrings (`sensitivity_macros`) |
| `registry.py` | the result records of every test, `../generality/DEVIATIONS.md`, `pert_macros.tex` and `rep_macros.tex` | `registry_table.tex` (Supplementary Table 1), `../HYPOTHESES.md` and `../hypotheses.json`: every prespecified hypothesis with its criterion, estimate and result, and every analysis done after scoring |
| `si_refs.py` | `../../revision/source/supplement.tex` and the files it inputs | `si_refs.tex`: the numbers of Supplementary Notes, Tables, Figures and Propositions, read in source order, so that the main text cites them by label (`\Sref{tab:pred}`) and compiles on its own; with `--aux`, the numbers are also checked against a compiled `supplement.aux` |

```bash
python savings_summary.py
python make_assets.py --out ../../revision/source
python registry.py --out ../../revision/source
python si_refs.py --out ../../revision/source
cd .. && python verify_manuscript_assets.py      # regenerates every generated .tex file and compares
```
