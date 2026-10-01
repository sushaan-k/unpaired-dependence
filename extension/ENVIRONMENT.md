# Computing environment

Every analysis, check and manuscript asset in this folder was produced with the environment below, on Linux x86-64
with two CPU cores and 7 GB of memory. Runners limit BLAS threads to two (`threadpoolctl`), so that results do not
depend on the core count; the verifiers compare regenerated files byte for byte.

## Python

| Package | Version | Needed for |
|---|---|---|
| Python | 3.11.15 | everything |
| numpy | 2.0.2 | everything |
| scipy | 1.13.1 | everything |
| threadpoolctl | 3.6.0 | runners (thread limits) |
| scikit-learn | 1.8.0 | cross-validated low rank and ridge comparators, cross-study analyses |
| pandas | 2.3.3 | extraction scripts (metadata tables) |
| pyarrow | 25.0.1 | fly connectome extraction (feather tables) |
| h5py | 3.16.0 | extraction of 10x HDF5 and h5ad files |
| anndata | 0.12.19 | extraction of h5ad files |
| pyreadr | 0.5.6 | extraction of the validation study's metadata (RDS) |

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install numpy==2.0.2 scipy==1.13.1 threadpoolctl==3.6.0 scikit-learn==1.8.0 pandas==2.3.3 pyarrow==25.0.1 \
    h5py==3.16.0 anndata==0.12.19 pyreadr==0.5.6
```

`../requirements.txt` pins the three packages that the results-only checks need (numpy, scipy, threadpoolctl).

## Champollion (comparator only)

Champollion (github.com/cantinilab/champollion, commit 39b3a83f9718f9aaa1a87dc430967ea0e433057b) needs Python >= 3.12
and PyTorch. It was installed in a separate virtual environment (Python 3.12.3, torch 2.5.1+cpu, the package
`champollion-omics` 0.1.0 from that commit), whose interpreter `semipaired/champ_tune.py` calls through its `VENV`
setting. The development comparison, the external test and the validation's alternatives
(`validation/posthoc_review`) use it.

## Manuscript

pdfTeX 3.141592653-2.6-1.40.25 (TeX Live 2023) with latexmk 4.83, BibTeX, TikZ and PGFPlots 1.18. The journal class
`sn-jnl.cls` is included in `../revision/source`. Build from `../revision/source`:

```bash
python ../../extension/synthesis/si_refs.py --out .       # supplementary numbers cited by the main text
latexmk -pdf supplement.tex reader.tex main.tex
```

## Runtimes on two cores

| Step | Time |
|---|---|
| Results-only verifiers (all) | one to three minutes |
| `verify_manuscript_assets.py` (regenerates the 71 generated files) | under a minute |
| Validation test, `vrun.py predict` / `evaluate` | 21 minutes / under a minute |
| Readout of the validation, `readout/brun.py run` | about 38 minutes |
| Robustness of the validation, `posthoc_robustness.py` | about 50 minutes |
| Alternatives and calibration, `posthoc_review/rrun.py run --part 0` and `--part 1` | about 75 minutes each, run side by side |
| Champollion in the atlas setting, `posthoc_review/rrun.py champ` | about 150 seconds per fold run alone (85 minutes for 34 folds); two runs side by side slow each several-fold |
| Bone-marrow test, `drun.py predict` | 11 minutes |
| Benchmark (nine data sets, `grun.py`) | about 3.3 hours |
| Draw-scoring check, `checks/run_all.sh` (every test rerun with its frozen code) | about 3 hours; `draw_scoring.py verify` seconds |
| Analyses added in revision: noise calibration, `second_review/noise_calibration.py run` | 70 minutes (validation), 30 minutes (bone marrow) |
| Analyses added in revision: `validation_review.py run --part 0` and `--part 1` | 6 to 22 minutes per held-out sample; about 5 hours for both parts side by side |
| Analyses added in revision: `atlas_bootstrap.py run -1 0 ... 9` | 8 to 12 minutes per replicate, about 1.5 hours; about 4 GB of memory |
| Analyses added in revision: `law_review.py pilot` (ten data sets), `benchmark_review.py run` (ten) | 7 minutes; about an hour |
| `second_review/verify.py` (hashes, order and every scored record recomputed) | one to four minutes |
