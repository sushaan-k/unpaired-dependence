# Generality benchmark: tissues, modality pairs and nervous systems

Does the semi-paired estimator's saving of paired cells hold when independent replication units are held out,
across tissues, species, modality pairs and nervous systems, and does its size follow from how the dependence is
concentrated in the unpaired eigen-blocks? Frozen plan: `PLAN.md` (hashed with code and extracted data at
17:26 EDT, 27 September 2026, `results/freeze.json`); deviations: `DEVIATIONS.md`; results: `RESULTS.md`.

## Data sets (extracted by `gdata.py`; raw data are not redistributed)

| Name | Pair (X, Y) | Units held out | Source |
|---|---|---|---|
| hao | RNA, 228 surface proteins | 8 donors | Hao et al. 2021 PBMC CITE-seq (GEO GSE164378) |
| stephenson | RNA, 192 surface proteins | 118 patients | Stephenson et al. 2021 COVID-19 PBMC (ArrayExpress E-MTAB-10026) |
| bmmc_cite | RNA, 134 surface proteins | 9 donors | NeurIPS 2021 bone marrow CITE-seq (GEO GSE194122) |
| colon | RNA, 177 surface proteins | 12 patients | Mennillo et al. 2024 (Figshare 21919356 v3) |
| bmmc_multiome | RNA, chromatin peaks | 10 donors | NeurIPS 2021 bone marrow multiome (GEO GSE194122) |
| scala_m1 | RNA, 29 electrophysiological features | 263 mice | Scala et al. 2021 Patch-seq (github.com/berenslab/mini-atlas) |
| gouwens_visp | RNA (log CPM), 68 electrophysiological features | 362 recording days | Gouwens et al. 2020 Patch-seq, processed by Gala et al. 2021 (github.com/AllenInstitute/coupledAE-patchseq) |
| banc | brain-side wiring, nerve-cord-side wiring of descending/ascending neurons | 913 cell types | BANC connectome v888 (Bates et al. 2026) |
| malecns | the same | 1,010 cell types | male CNS connectome v0.9 (Berg et al. 2025) |
| (secondary) banc_crossanimal | as banc; unpaired pool from FAFB v783 (brain only) and MANC v1.2.1 (nerve cord only) | 913 cell types | Dorkenwald et al. 2024; Schlegel et al. 2024; Takemura et al. 2024 |

Fly connectome files: compiled data of the fly connectome tutorial,
`https://storage.googleapis.com/lee-lab_brain-and-nerve-cord-fly-connectome/compiled_data/` (`*_meta.feather`,
`*_simple_edgelist.feather`). SHA-256 of the extracted files are in `results/freeze.json` (`data`); the large raw
files (Stephenson h5ad, multiome h5ad, male CNS edge list) were deleted after extraction to free disk
(`/home/claude/cbio/rawdata/generality_sources_sha256.txt` lists the hashes of the multiome and male CNS sources).

## Files

| File | Role |
|---|---|
| `PLAN.md` | Frozen plan: data sets, roles, features, arms, endpoint, hypotheses G1-G4 |
| `gdata.py` | Extraction of every data set into a common format (`/home/claude/cbio/rawdata/generality/<name>.npz`) |
| `grun.py` | Cross-fitted benchmark: `predict` (training units only; predictions hashed in `results/<name>_manifest.json`), `evaluate` (held-out units; bootstrap over units), `summary` (G1-G3) |
| `glaw.py` | Structural law (G4): predicted saving from training reservoirs, rank correlation with observed savings |
| `freeze.py` | Hashes plan, code and data (`results/freeze.json`) |
| `with_object_arrays.py` | Deviation 1: runs frozen scripts so that `stephenson.npz` (object string arrays) loads |
| `run_queue.sh` | The two run queues used |
| `make_assets.py` | Manuscript macros, table and figure (`revision/source/gen_*.tex`) and `RESULTS.md` |
| `test/` | Harness test on simulated data before the freeze (not a result) |
| `results/` | Per-data-set results (`<name>.json`), predictions, manifests, `summary.json`, `law_predicted.json`, `law.json` |

## Reproduce

```bash
python gdata.py <name>                       # each data set (raw files needed)
python freeze.py                             # once, before any held-out statistic
./run_queue.sh                               # predict + evaluate every data set
python with_object_arrays.py grun.py run stephenson   # deviation 1
python grun.py summary && python with_object_arrays.py glaw.py predict && python glaw.py test
python make_assets.py
python ../verify_generality.py               # results only; --data re-scores three data sets
```

## Results (details in RESULTS.md)

* Prespecified (targets relative to the unshrunk fully paired reference): G1 1.50 (1.40-1.55) and G2 1.34
  (1.05-1.39), both met as specified. Five savings were exactly 1 because the targets were degenerate
  (DEVIATIONS.md). A 1 here encodes an unresolved comparison; it is neither a measured saving nor a lower bound,
  so these means say little beyond meeting the criterion.
* Amendments 1-2 (targets at half the best recovery any method reached; data sets where no method recovers
  dependence are not informative; the complete rule preceded the scoring of five of the nine data sets, so these
  are supporting evidence, not a prespecified test): G1' 2.90 (2.26-3.03) and G2' 2.29 (1.82-2.38) over eight informative data sets,
  and 2.21 and 2.09 over all nine with censored savings.
* G3 (self-tuning against fixed): 0.58 (0.55-0.69), so the fixed estimator needs fewer paired cells.
* G4 (structural law): not met (prespecified -0.09; amended 0.42, P = 0.10). Post hoc and exploratory
  (`posthoc/law_concordance.py`): Harrell's C 0.92, P = 0.008.
* Secondary cross-animal analysis: the proposed estimator failed (below zero at every budget), as did paired-only
  shrinkage centred on the other animals' means; reference regression (31% at 100 paired neurons) and SemiCCA up
  to 100 paired neurons kept positive recovery. A failure under this distribution shift, not a proof that unpaired
  data must come from the same population.

## Run notes

* Queue B (`run_queue.sh`) was stopped after bmmc_multiome started; gouwens_visp and malecns ran in parallel with it
  (`run_rest.sh`). The amendment reruns and the law's predictions then ran one at a time (`finalize.sh`), because
  parallel jobs were stopped by the memory limit. Stopped jobs left no output and were rerun whole.
* Deviation 2: `law_lowmem.py` runs the frozen `glaw.predict()` with an algebraically identical, memory-light
  `block_moments`. Run alone, the frozen script later completed, and its output (equal to 15 digits) is the one
  reported.
