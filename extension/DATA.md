# Data: sources, extraction and checksums

Raw counts are not redistributed. Each extraction script downloads or reads the public files named below and writes
one compact file per data set (under `/home/claude/cbio/rawdata/`; change the `RAW`/`DATA` constants at the top of
each script for another location). Extraction uses cell and feature metadata and statistics of one modality at a
time, never a statistic of dependence between modalities. The SHA-256 of every extracted file was recorded in the
test's freeze before any estimate was computed, and the loaders check it; the freeze files are listed per test.

## Data sets

| Data set | Used in | Public source | Extraction | Checksums recorded in |
|---|---|---|---|---|
| Melanoma Perturb-CITE-seq (Frangieh et al. 2021) | development; pairing-free test | scPerturb, Zenodo record 7041849: `FrangiehIzar2021_RNA.h5ad`, `FrangiehIzar2021_protein.h5ad` | `perturbation/extract.py` (split and seal; loaders in `pdata.py`) | `perturbation/results/seal.json`, `freeze.json` |
| THP-1 ECCITE-seq (Papalexi et al. 2021) | development; replication | Zenodo record 7041849: `PapalexiSatija2021_eccite_RNA.h5ad`, `..._protein.h5ad` | `perturbation_replication/extract.py` (split and seal; loaders in `rdata.py`) | `perturbation_replication/results/seal.json` |
| T-cell OverCITE-seq (Legut et al. 2022) | external test | GEO GSE193736 (GSM5819657-GSM5819660) | `semipaired/external/extract.py` | `semipaired/external/results/seal.json` |
| Blood CITE-seq, 8 donors (Hao et al. 2021) | benchmark; pairing-free test | GEO GSE164378 (3' RNA and ADT: GSM5008737, GSM5008738) | `generality/gdata.py hao`; `cross_study/data.py` | `generality/results/freeze.json` |
| Blood CITE-seq, COVID-19 (Stephenson et al. 2021) | benchmark; atlas of the validation; pairing-free test | ArrayExpress E-MTAB-10026 (h5ad, 7.2 GB) | `generality/gdata.py stephenson`; `cross_study/data.py` | `generality/results/freeze.json`; `validation/results/freeze.json` |
| Bone-marrow CITE-seq and multiome (NeurIPS 2021) | benchmark; deployment test | GEO GSE194122 | `generality/gdata.py bmmc_cite`, `bmmc_multiome` | `generality/results/freeze.json`; `deployment/results/freeze.json` |
| Colon CITE-seq (Mennillo et al. 2024) | benchmark; pairing-free test | Figshare article 21919356, version 3 | `generality/gdata.py colon`; `spectral_transfer/README.md` | `generality/results/freeze.json` |
| Mouse motor-cortex Patch-seq (Scala et al. 2021) | benchmark | github.com/berenslab/mini-atlas | `generality/gdata.py scala_m1` | `generality/results/freeze.json` |
| Mouse visual-cortex Patch-seq (Gouwens et al. 2020; Gala et al. 2021) | benchmark | github.com/AllenInstitute/coupledAE-patchseq | `generality/gdata.py gouwens_visp` | `generality/results/freeze.json` |
| Fly connectomes: BANC v888, male CNS v0.9, FAFB v783, MANC v1.2.1 | benchmark; cross-animal analysis; law | compiled data of the fly connectome tutorial, `gs://lee-lab_brain-and-nerve-cord-fly-connectome/compiled_data` | `generality/gdata.py banc`, `malecns`, `fafb_brain`, `manc_vnc`; `predictability/pextract.py fafb_vpn`, `banc_vpn` | `generality/results/freeze.json`; `predictability/results/data_freeze.json` |
| Mouse lymphoid organs, 111 and 206 antibodies (Gayoso et al. 2021) | law | github.com/YosefLab/totalVI_reproducibility (`spleen_lymph_111.h5ad`, `spleen_lymph_206.h5ad`; GEO GSE150599) | `predictability/pextract.py sln111`, `sln206` | `predictability/results/data_freeze.json` |
| 10x PBMC 10k and MALT 10k v3 | law | github.com/YosefLab/totalVI_reproducibility (`pbmc_10k_protein_v3.h5ad`, `malt_10k_protein_v3.h5ad`) | `predictability/pextract.py pbmc10k`, `malt10k` | `predictability/results/data_freeze.json` |
| Bone-marrow CITE-seq (Stuart et al. 2019) | law | GEO GSE128639 (GSM3681518, GSM3681519) | `predictability/pextract.py bmcite` | `predictability/results/data_freeze.json` |
| Human fetal cortex multiome (Trevino et al. 2021) | law | GEO GSE162170 | `predictability/pextract.py fetal_cortex` | `predictability/results/data_freeze.json` |
| Adult mouse cortex SNARE-seq (Chen et al. 2019) | law | GEO GSE126074 | `predictability/pextract.py snare_cortex` | `predictability/results/data_freeze.json` |
| Human cortex GABAergic Patch-seq (Lee, Dalley et al. 2023) | law | github.com/AllenInstitute/human_patchseq_gaba | `predictability/pextract.py human_gaba` | `predictability/results/data_freeze.json` |
| Healthy-adult blood CITE-seq, ImmunoMicrobiome study | validation and its readout | GEO GSE314416: pools DB8-DB14 (GEX and ADT HDF5 files), `GSE314416_cell_metadata_followup.rds`, `GSE314416_TotalSeqC_feature_ref.csv.gz` | `validation/vdata.py download`, `extract` | `validation/results/freeze.json` (raw files and extract) |
| MSigDB Hallmark interferon-gamma and interferon-alpha response sets | melanoma program analysis | gsea-msigdb.org | `perturbation/programs.py` | `perturbation/results/programs.json` (`gene_sets_sha256`) |

## Order of steps in every test

1. Extraction (above), recording checksums.
2. A smoke run of the whole pipeline on synthetic values with the real design.
3. `freeze.py`: SHA-256 of the plan, the runner, every module it imports and the data files, with a timestamp.
4. `predict`: checks the freeze, writes predictions and hashes them into `manifest.json` before any held-out cell is
   read.
5. `evaluate`: checks the hashes, reads held-out cells and scores.

`HYPOTHESES.md` lists every hypothesis with its freeze time and result; the `verify_*.py` scripts check
the freezes, the order of the steps and the verdicts.
