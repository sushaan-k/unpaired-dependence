"""Loading and preprocessing shared by every step (PLAN.md, "Features" and "Scales").

RNA: log1p(1e4 * count / total RNA count of the cell). Protein: log1p(count)
minus its mean over the matched targets of the cell. Identical in every study.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from adt_matching import COLON, HAO, STEPHENSON, names_for

HERE = Path(__file__).resolve().parent
DATA = Path("/home/claude/cbio/rawdata/cross_study")
COLON_DIR = Path("/home/claude/cbio/rawdata/blood_colon")
SHRINK = 0.1
RIDGE_FLOOR = 1e-3


def features():
    return json.loads((HERE / "results/features.json").read_text())


def clr(counts):
    logged = np.log1p(np.asarray(counts, float))
    return logged - logged.mean(axis=1, keepdims=True)


def load_training():
    d = np.load(DATA / "stephenson_training.npz", allow_pickle=False)
    feats = features()
    assert list(d["genes"]) == feats["genes"] and list(d["proteins"]) == feats["proteins"]
    return {"x": d["x"].astype(float), "y": clr(d["adt"]), "patient": d["patient"], "site": d["site"],
            "half": d["half"], "initial": d["initial"], "full": d["full"], "barcode": d["barcode"]}


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_recipient(cohort, part):
    """cohort in {'hao', 'colon'}; part in {'adaptation', 'scoring'}.

    Scoring files are checked against their seals before use.
    """
    feats = features()
    genes, proteins = feats["genes"], feats["proteins"]
    if cohort == "colon":
        path = COLON_DIR / f"colon_{part}.npz"
        seal = json.loads((HERE / "seal.json").read_text())
        d = np.load(path, allow_pickle=False)
        rna = sp.csr_matrix((d["rna_data"], d["rna_indices"], d["rna_indptr"]))
        library = np.asarray(rna.sum(axis=1)).ravel()
        names = list(d["rna_names"])
        cols = [names.index(g) for g in genes]
        adt_names = list(d["adt_names"])
        adt = d["adt"][:, [adt_names.index(n) for n in names_for(COLON, proteins)]]
        donor, labels = d["CoLabs_patient"].astype(str), {"coarse": d["coarse_annotations_MK"].astype(str)}
        extra = {"sample": d["CoLabs_sample"].astype(str), "condition": d["condition"].astype(str)}
    else:
        path = DATA / f"hao_{part}.npz"
        seal = json.loads((HERE / "hao_seal.json").read_text())
        d = np.load(path, allow_pickle=False)
        rna = sp.csr_matrix((d["rna_data"], d["rna_indices"], d["rna_indptr"]))
        library = d["library"]
        names = list(d["rna_names"])
        cols = [names.index(g) for g in genes]
        adt_names = list(d["adt_names"])
        adt = d["adt"][:, [adt_names.index(n) for n in names_for(HAO, proteins)]]
        donor, labels = d["donor"].astype(str), {"l1": d["l1"].astype(str), "l2": d["l2"].astype(str)}
        extra = {"time": d["time"].astype(str)}
    assert _sha256(path) == seal["files"][part]["sha256"], f"{path} does not match its seal"
    counts = rna[:, cols].toarray().astype(float)
    x = np.log1p(1e4 * counts / np.maximum(library, 1)[:, None])
    return {"x": x, "y": clr(adt), "donor": donor, "labels": labels, "barcode": d["barcodes"].astype(str),
            **extra}


def shrink(S):
    S = (1 - SHRINK) * S + SHRINK * np.diag(np.diag(S))
    return S + RIDGE_FLOOR * np.mean(np.diag(S)) * np.eye(len(S))


def standardize(v):
    """Centre and scale columns; constant columns become zero."""
    c = v - v.mean(0)
    sd = c.std(0)
    return c / np.where(sd > 0, sd, 1.0), sd


def correlation_moments(x, y):
    """Shrunk within-assay correlation matrices and raw cross-correlation."""
    xs, _ = standardize(x)
    ys, _ = standardize(y)
    n = len(x)
    return shrink(xs.T @ xs / n), shrink(ys.T @ ys / n), xs.T @ ys / n


def cross_correlation(x, y):
    xs, _ = standardize(x)
    ys, _ = standardize(y)
    return xs.T @ ys / len(x)


def type_centre(v, labels, keep):
    out = np.array(v, float, copy=True)
    for t in keep:
        m = labels == t
        out[m] -= out[m].mean(0)
    return out
