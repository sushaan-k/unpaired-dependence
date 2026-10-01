"""Frangieh et al. (2021) Perturb-CITE-seq: constants, cell classes and sealed loaders.

Source: scPerturb release of the Frangieh-Izar screen (Zenodo record 7041849,
FrangiehIzar2021_RNA.h5ad and FrangiehIzar2021_protein.h5ad; same cells in the
same order). Patient-derived melanoma cells carrying CRISPR knockouts of 248
genes, cultured alone, with interferon-gamma, or with autologous tumour-
infiltrating lymphocytes; RNA and 24 antibody-derived tags per cell.

Targets are read from the detected guides (guide_id), not from the release's
"perturbation" column, which names only the first of several targeted genes and
labels cells with a mixture of targeting and non-targeting guides as controls.
A cell is "single" if all its guides target the same gene and "NT" if all its
guides are non-targeting (NO_SITE_*, ONE_NON-GENE_SITE_*); other cells are not used.

Preprocessing is the same as in the other cohorts: RNA log1p(1e4 * count /
library), with the library summed over all 23,712 genes; protein log1p(count)
minus its mean over the 20 target antibodies of the cell (isotype controls excluded).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
RAW = Path("/home/claude/cbio/rawdata/perturb")
RNA_FILE = RAW / "FrangiehIzar2021_RNA.h5ad"
PROTEIN_FILE = RAW / "FrangiehIzar2021_protein.h5ad"
SOURCE_SHA256 = {"FrangiehIzar2021_RNA.h5ad": "cc42ef38bcf703a00e0c77c7945dd53159b12d814c438ae8afeaed9bc71f48d1",
                 "FrangiehIzar2021_protein.h5ad": "1f85827b5afad11a30d8ac99399772231110a1c3723bed6ecf7981a12cc3dbcc"}

HELDOUT_SALT = "perturb-heldout-v1"   # held-out targets: every fourth target in SHA-256 order
CELL_SALT = "perturb-cells-v1"        # order of cells within a population or group
ISOTYPES = ("Rat_IgG2a", "Mouse_IgG1", "Mouse_IgG2a", "Mouse_IgG2b")
ENCODING = {"CD117": ["KIT"], "CD119": ["IFNGR1"], "CD140a": ["PDGFRA"], "CD140b": ["PDGFRB"], "CD172a": ["SIRPA"],
            "CD184": ["CXCR4"], "CD202b": ["TEK"], "CD274": ["CD274"], "CD29": ["ITGB1"], "CD309": ["KDR"],
            "CD44": ["CD44"], "CD47": ["CD47"], "CD49f": ["ITGA6"], "CD58": ["CD58"], "CD59": ["CD59"],
            "CD61": ["ITGB3"], "HLA_A": ["HLA-A", "HLA-B", "HLA-C", "B2M"], "HLA_E": ["HLA-E"], "CD9": ["CD9"],
            "CD279": ["PDCD1"]}
PANEL_SIZE = 200
ENCODING_MIN_DETECTION = 0.01   # encoding genes kept if detected in >= 1% of training cells
HVG_MIN_DETECTION = 0.05
HVG_BINS = 20
MIN_TRAIN_HALF = 20             # training population: >= 20 cells in each half
MIN_ADAPT = 40                  # held-out group: >= 40 adaptation cells
MIN_SCORE_HALF = 20             # and >= 20 cells in each scoring sub-half


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


def target_set(guides):
    if guides == "nan":
        return None
    out = set()
    for g in guides.split(";"):
        out.add("NT" if g.startswith(("NO_SITE", "ONE_NON-GENE_SITE")) else g.rsplit("_", 1)[0])
    return out


def cell_class(guides):
    t = target_set(guides)
    if t is None:
        return "none", ""
    if t == {"NT"}:
        return "NT", "NT"
    if len(t) == 1:
        return "single", next(iter(t))
    return ("mixedNT" if "NT" in t else "multi"), ""


def heldout_targets(targets):
    order = sorted(set(targets), key=lambda t: h(HELDOUT_SALT, t))
    return sorted(order[::4])


def clr(counts):
    logged = np.log1p(np.asarray(counts, float))
    return logged - logged.mean(axis=1, keepdims=True)


def lognorm(counts, library):
    return np.log1p(1e4 * counts / np.maximum(library, 1)[:, None])


def seal():
    return json.loads((HERE / "results/seal.json").read_text())


def load(part):
    """part: training, heldout_adaptation, heldout_scoring, nt_adaptation, nt_scoring (seal-checked)."""
    record = seal()
    path = RAW / f"perturb_{part}.npz"
    assert sha(path) == record["files"][part]["sha256"], f"{path} does not match the seal"
    d = np.load(path, allow_pickle=False)
    assert list(d["genes"]) == record["genes"] and list(d["proteins"]) == record["proteins"]
    counts = sp.csr_matrix((d["rna_data"], d["rna_indices"], d["rna_indptr"]),
                           shape=(len(d["cell"]), len(d["genes"]))).toarray().astype(float)
    return {"counts": counts, "library": d["library"].astype(float), "adt": d["adt"].astype(float),
            "y": clr(d["adt"]), "cell": d["cell"].astype(str), "target": d["target"].astype(str),
            "condition": d["condition"].astype(str), "guide": d["guide"].astype(str), "part": d["part"].astype(int),
            "genes": list(d["genes"]), "proteins": list(d["proteins"])}
