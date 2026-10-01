"""Loading the bone-marrow CITE-seq files written by extract_bmmc.py.

Both files are checked against results/bmmc_seal.json. The scoring file may
be opened only by evaluate_bmmc.py, which first checks the prediction manifest.
Preprocessing is identical to the other cohorts: RNA log1p(1e4 * count /
library); protein log1p(count) minus its mean over the kept targets of the cell.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
RAW = Path("/home/claude/cbio/rawdata/bmmc")


def seal():
    return json.loads((HERE / "results/bmmc_seal.json").read_text())


def clr(counts):
    logged = np.log1p(np.asarray(counts, float))
    return logged - logged.mean(axis=1, keepdims=True)


def load(part):
    record = seal()
    path = RAW / f"bmmc_{part}.npz"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record["files"][part]["sha256"], f"{path} seal"
    d = np.load(path, allow_pickle=False)
    assert list(d["rna_names"]) == record["genes"] and list(d["adt_names"]) == record["proteins"]
    counts = sp.csr_matrix((d["rna_data"], d["rna_indices"], d["rna_indptr"])).toarray().astype(float)
    return {"counts": counts, "library": d["library"].astype(float), "y": clr(d["adt"]),
            "donor": d["donor"].astype(str), "batch": d["batch"].astype(str), "site": d["site"].astype(str),
            "labels": {"fine": d["fine"].astype(str), "coarse": d["coarse"].astype(str)},
            "barcode": d["barcodes"].astype(str)}
