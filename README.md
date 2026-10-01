# Unpaired Dependence

**Using unpaired measurements to reduce the number of paired cells needed to estimate dependence between modalities**

Sushaan Kandukoori, Pranava Kumar, Shrikrishna Ramesh

[Project page](https://sushaan-k.github.io/unpaired-dependence/) ·
[Paper](docs/assets/papers/reading-copy.pdf) ·
[Supplementary Information](docs/assets/papers/supplement.pdf) ·
[Manuscript source](revision/source/) ·
[Data sources](extension/DATA.md)

Unpaired measurements provide each modality's principal axes and covariance within cell types.
The estimator uses this structure to estimate cross-modal dependence from a small paired sample:
contrasts remove population averages, block shrinkage estimates the pooled dependence along the
unpaired axes, and regularized covariance maps produce cell-type estimates.

In a prespecified test on an independent blood study, 147 paired cells with another study's atlas
estimated RNA-protein correlations within broad cell types better than paired-only methods with
800. Reference regression given the same atlas came close, reaching the target with 169 cells.
The paper separates this empirical saving from its oracle-linear theory and pilot budget predictions.

## Repository

| Path | Contents |
| --- | --- |
| `extension/` | Estimator code, frozen protocols, result records, prediction manifests and verification scripts |
| `revision/source/` | Current manuscript, supplement, proofs and generated figures/tables |
| `docs/` | Project page, current PDFs and web renders of the paper's figures |
| `release/` | Immutable historical release required by the reproducibility checks |
| `scripts/` | Project-page and figure export scripts |

The current estimator and development analyses are in [`extension/semipaired/`](extension/semipaired/).
The external-atlas test is in [`extension/validation/`](extension/validation/), with subsequent checks
in [`extension/second_review/`](extension/second_review/). The saving-law tests are in
[`extension/predictability/`](extension/predictability/).

## Verification

Python 3.11 is the recorded environment. Install the small dependency set for the package-only checks:

```bash
python -m pip install -r requirements.txt
cd extension
python verify_semipaired.py
python verify_generality.py
python verify_predictability.py
python verify_deployment.py
python verify_validation.py
python verify_perturbation.py
(cd cross_study && python verify_cross_study.py)
python verify_manuscript_assets.py
python checks/draw_scoring.py identity
python checks/draw_scoring.py verify
python second_review/verify.py
python verify_release_portable.py
python -m unittest discover -s checks -p test_theory_corrections.py
```

These checks verify the packaged results, hashes, identities and available manuscript assets. They do
not rerun the experiments from raw counts. Raw matrices and large prediction arrays are not included;
checks requiring omitted inputs report skips or partial coverage. [`extension/DATA.md`](extension/DATA.md)
provides acquisition and reconstruction instructions, and [`extension/ENVIRONMENT.md`](extension/ENVIRONMENT.md)
records the analysis environments. Plans, amendments and outcomes are indexed in
[`extension/HYPOTHESES.md`](extension/HYPOTHESES.md).

## Manuscript and Website

Build the PDFs using the instructions in [`revision/source/README.md`](revision/source/README.md).
The submitted-layout manuscript is also available [here](docs/assets/papers/manuscript.pdf).
The PDFs and LaTeX files preserve the reviewed submission version, including its blinded metadata;
the author names and contact details are given on this page and the project website.

The project page takes its scientific prose and numerical values directly from the manuscript source.
Its figures are rendered from the same TikZ files, not redrawn from screenshots.

```bash
python -m pip install pylatexenc==2.10 PyMuPDF==1.26.4
python scripts/build_site.py
python scripts/export_figures.py  # requires a TeX installation with TikZ/PGFPlots
```

GitHub Pages serves `docs/` on `main`.

## History

This repository was previously named `coupling-fields-benchmark`. The default branch now contains the
current semi-paired estimation paper. Earlier material is preserved in Git history and at the
[`archive/assay-resolution-2026-10-01`](https://github.com/sushaan-k/unpaired-dependence/tree/archive/assay-resolution-2026-10-01)
tag. Files inside `release/` retain their original names and bytes so their recorded checksums remain valid.

## Contact

- Sushaan Kandukoori: [sushaankandukoori@gmail.com](mailto:sushaankandukoori@gmail.com)
- Pranava Kumar: [pranavak@mit.edu](mailto:pranavak@mit.edu)
- Shrikrishna Ramesh: [shrikrish.ramesh@gmail.com](mailto:shrikrish.ramesh@gmail.com)
