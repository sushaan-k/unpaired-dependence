"""OverCITE-seq (Legut et al. 2022, GEO GSE193736): constants, cell roles and sealed loaders.

Primary human CD8+ T cells, each transduced with one of 41 open reading frames (ORFs; tNGFR is the control),
pooled, cultured 24 h with IL-2 alone or with CD3/CD28 activator, hashed into four wells per condition (eight
hashtags), and profiled for RNA, 14 surface proteins (TotalSeq-C) and ORF transcripts (GEO samples GSM5819657-60;
4,312 cells after the authors' quality control). The condition of each hashtag is not given in the deposited
metadata; the four hashtags with the largest median RNA library size are taken as the stimulated wells (a
library-size rule fixed before any other statistic; stimulated T cells enlarge within 24 h).

Cell roles, fixed before any statistic other than library sizes: every third ORF in SHA-256 order is held out with
all its cells, which are split by barcode hash into adaptation and scoring cells, and the scoring cells into two
sub-halves; the cells of the other ORFs form the training file, with assay halves (RNA part 0, protein part 1)
fixed by barcode hash. Preprocessing as elsewhere: RNA log1p(1e4 count / library) with the library summed over all genes; protein
log1p(count) minus its mean over the 14 antibodies.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
RAW = Path("/home/claude/cbio/rawdata/overcite")
SOURCE_SHA256 = {"GSM5819657_HTO_counts.csv.gz": "d0d8362857b126deb44d2903c9dbf404adb6263f47f0b4265e7df23b15c68e9e",
                 "GSM5819658_ADT_counts.csv.gz": "971861d3ea270bbdbf47a3e9df77ff342e759869161ae9f5c71b682f7d117969",
                 "GSM5819659_ORF_counts.csv.gz": "e89caa62e52f4c0bda2fad78776da7401b867a319bb33d7efc4a30b766f853cc",
                 "GSM5819660_GEX_counts.csv.gz": "9fa2dce1d2184b37f7d7cf0ae3e4fa7d8183d776173d282bb1e415328da21e5c"}
SALT = "overcite-roles-v1"
ENCODING = {"CD11c": ["ITGAX"], "CD14": ["CD14"], "CD16": ["FCGR3A", "FCGR3B"], "CD19": ["CD19"], "CD25": ["IL2RA"],
            "CD3": ["CD3E", "CD3D", "CD3G"], "CD4": ["CD4"], "CD45": ["PTPRC"], "CD45RA": ["PTPRC"],
            "CD45RO": ["PTPRC"], "CD56": ["NCAM1"], "CD69": ["CD69"], "CD8": ["CD8A", "CD8B"], "NGFR": ["NGFR"]}
ORF_MIN, ORF_RATIO = 3, 3.0      # ORF call: largest ORF count >= 3 and >= 3x the second largest
HTO_RATIO = 2.0                  # hashtag call: largest >= 2x the second largest
PANEL_SIZE = 200
ENCODING_MIN_DETECTION = 0.01
HVG_MIN_DETECTION = 0.05
HVG_BINS = 20
MIN_TRAIN_HALF = 10              # training population (ORF x condition): >= 10 cells in each assay half
MIN_ADAPT = 20                   # held-out ORF x condition group: >= 20 adaptation cells


def heldout_orfs(orfs):
    order = sorted(set(orfs), key=lambda t: hashlib.sha256(f"overcite-heldout-v1|{t}".encode()).hexdigest())
    return sorted(order[::3])


def h(key):
    return int(hashlib.sha256(f"{SALT}|{key}".encode()).hexdigest()[:12], 16) / 16 ** 12


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def seal():
    return json.loads((HERE / "results/seal.json").read_text())


def clr(adt):
    logged = np.log1p(np.asarray(adt, float))
    return logged - logged.mean(axis=1, keepdims=True)


def lognorm(counts, library):
    return np.log1p(1e4 * counts / np.maximum(library, 1)[:, None])


def load(part, genes=None):
    """part: train, test_adaptation, test_scoring (seal-checked). With genes, returns dense counts of those genes."""
    record = seal()
    path = RAW / f"overcite_{part}.npz"
    assert sha(path) == record["files"][part]["sha256"], f"{path} does not match the seal"
    d = np.load(path, allow_pickle=False)
    X = sp.csr_matrix((d["data"], d["indices"], d["indptr"]), shape=(len(d["cell"]), len(d["all_genes"])))
    out = {"library": d["library"].astype(float), "adt": d["adt"].astype(float), "y": clr(d["adt"]),
           "cell": d["cell"].astype(str), "target": d["orf"].astype(str), "condition": d["condition"].astype(str),
           "guide": d["orf"].astype(str), "part": d["part"].astype(int), "all_genes": d["all_genes"].astype(str),
           "proteins": list(d["proteins"].astype(str)), "sparse": X}
    if genes is not None:
        gi = {g: i for i, g in enumerate(out["all_genes"])}
        out["counts"] = X[:, [gi[g] for g in genes]].toarray().astype(float)
        out["genes"] = list(genes)
    return out
