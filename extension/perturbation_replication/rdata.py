"""Papalexi et al. (2021) ECCITE-seq screen: constants, cell classes, split and sealed loaders.

Source: scPerturb release (Zenodo record 7041849): PapalexiSatija2021_eccite_RNA.h5ad
and PapalexiSatija2021_eccite_protein.h5ad (same cells in the same order).
THP-1 cells stimulated with interferon-gamma, CRISPR knockouts of 25 genes of
the interferon-gamma and PD-L1 pathways and non-targeting guides, three
biological replicates (hashtags rep1-tx, rep3-tx, rep4-tx; the 77 cells of rep2
are not used), RNA and four surface proteins (CD86, PD-L1, PD-L2, CD366).

The frozen estimator and selection rules of extension/perturbation are reused
unchanged, with the replicate in the role of the condition: populations are
target x replicate, and population means are centred within replicate.
Preprocessing: RNA log1p(1e4 * count / library), library summed over all genes;
protein log1p(count) minus the cell's mean over the four antibodies.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
RAW = Path("/home/claude/cbio/rawdata/papalexi")
SOURCE_SHA256 = {"PapalexiSatija2021_eccite_RNA.h5ad": "03e9602d124c261281936717d5795445805e6e704a52b0310bcde94225aa0929",
                 "PapalexiSatija2021_eccite_protein.h5ad": "e1e293c75dfebe09301bf32093703cfd5f50c17d6fb53144d8213e52e0e4e150"}
CELL_SALT = "papalexi-cells-v1"
REPLICATES = ("rep1-tx", "rep3-tx", "rep4-tx")
ENCODING = {"CD86": ["CD86"], "PDL1": ["CD274"], "PDL2": ["PDCD1LG2"], "CD366": ["HAVCR2"]}
PANEL_SIZE = 200
ENCODING_MIN_DETECTION = 0.01
HVG_MIN_DETECTION = 0.05
HVG_BINS = 20
MIN_TRAIN_HALF = 20
MIN_ADAPT = 40


def h(salt, key):
    return hashlib.sha256(f"{salt}|{key}".encode()).hexdigest()


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def decode(a):
    return np.array([x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in a])


def target_of(guide):
    return re.sub(r"g\d+$", "", guide)


def clr(counts):
    logged = np.log1p(np.asarray(counts, float))
    return logged - logged.mean(axis=1, keepdims=True)


def lognorm(counts, library):
    return np.log1p(1e4 * counts / np.maximum(library, 1)[:, None])


def seal():
    return json.loads((HERE / "results/seal.json").read_text())


def load(part):
    """part: train, test_adaptation, test_scoring, nt_adaptation, nt_scoring (seal-checked)."""
    record = seal()
    path = RAW / f"papalexi_{part}.npz"
    assert sha(path) == record["files"][part]["sha256"], f"{path} does not match the seal"
    d = np.load(path, allow_pickle=False)
    assert list(d["genes"]) == record["genes"] and list(d["proteins"]) == record["proteins"]
    counts = sp.csr_matrix((d["rna_data"], d["rna_indices"], d["rna_indptr"]),
                           shape=(len(d["cell"]), len(d["genes"]))).toarray().astype(float)
    return {"counts": counts, "library": d["library"].astype(float), "y": clr(d["adt"]), "cell": d["cell"].astype(str),
            "target": d["target"].astype(str), "condition": d["replicate"].astype(str), "guide": d["guide"].astype(str),
            "part": d["part"].astype(int), "genes": list(d["genes"]), "proteins": list(d["proteins"])}
