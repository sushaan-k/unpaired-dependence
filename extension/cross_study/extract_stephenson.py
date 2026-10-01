#!/usr/bin/env python3
"""Training populations from Stephenson et al. 2021 (PLAN.md, "Data").

    python extract_stephenson.py stats      # pass 1: gene dispersion over selected cells
    python extract_stephenson.py extract    # pass 2: final genes and matched proteins

Selection: no LPS stimulation; initial_clustering not Platelets/RBC; per patient
the first 2,000 cells in ascending SHA-256 of stephenson-reference-v1|patient|barcode.
Raw counts come from layers/raw (RNA and antibody counts).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import scipy.sparse as sp

H5AD = Path("/home/claude/cbio/rawdata/stephenson.h5ad")
MD5 = "add2501947c585ea6b6ef7429e4bd9f2"
OUT = Path("/home/claude/cbio/rawdata/cross_study")
HERE = Path(__file__).resolve().parent
CAP = 2000
BLOCK = 50000


def column(f, group, name):
    data = f[group][name][:]
    cats = f[group].get("__categories")
    if cats is not None and name in cats:
        c = cats[name]
        c = c.asstr()[:] if c.dtype.kind in "OSU" else c[:]
        return c[data]
    return data.astype(str) if data.dtype.kind in "OS" else data


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def selection(f):
    obs = {k: column(f, "obs", k) for k in ("patient_id", "sample_id", "Site", "time_after_LPS",
                                            "initial_clustering", "full_clustering")}
    barcodes = column(f, "obs", f["obs"].attrs["_index"])
    keep = (obs["time_after_LPS"] == "nan") & ~np.isin(obs["initial_clustering"], ["Platelets", "RBC"])
    chosen = []
    for patient in sorted(set(obs["patient_id"][keep])):
        idx = np.flatnonzero(keep & (obs["patient_id"] == patient))
        keys = [sha(f"stephenson-reference-v1|{patient}|{barcodes[i]}") for i in idx]
        chosen.extend(idx[np.argsort(keys)][:CAP].tolist())
    chosen = np.array(sorted(chosen))
    return chosen, obs, barcodes


def features(f):
    names = f["var"]["_index"].asstr()[:]
    types = column(f, "var", "feature_types")
    return names, types != "Gene Expression"


def blocks(f, rows):
    """Yield (row indices, CSR block restricted to rows) in file order."""
    indptr = f["layers/raw/indptr"][:]
    ncol = len(f["var"]["_index"])
    for start in range(0, len(indptr) - 1, BLOCK):
        stop = min(start + BLOCK, len(indptr) - 1)
        want = rows[(rows >= start) & (rows < stop)]
        if len(want) == 0:
            continue
        a, b = indptr[start], indptr[stop]
        block = sp.csr_matrix((f["layers/raw/data"][a:b], f["layers/raw/indices"][a:b],
                               indptr[start:stop + 1] - a), shape=(stop - start, ncol))
        yield want, block[want - start]


def normalise(rna):
    library = np.asarray(rna.sum(axis=1)).ravel()
    norm = sp.diags(1e4 / np.maximum(library, 1)) @ rna
    norm.data = np.log1p(norm.data)
    return norm.tocsr(), library


def stats():
    with h5py.File(H5AD, "r") as f:
        rows, obs, _ = selection(f)
        names, is_adt = features(f)
        genes = np.flatnonzero(~is_adt)
        total, square, count = np.zeros(len(genes)), np.zeros(len(genes)), 0
        for want, block in blocks(f, rows):
            norm, _ = normalise(block[:, genes])
            total += np.asarray(norm.sum(axis=0)).ravel()
            square += np.asarray(norm.multiply(norm).sum(axis=0)).ravel()
            count += len(want)
            print(count, flush=True)
    mean = total / count
    var = square / count - mean ** 2
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "stephenson_gene_stats.npz", genes=names[genes].astype(str), mean=mean, var=var,
                        cells=count)
    summary = {"cells": int(count), "patients": int(len(set(obs["patient_id"][rows]))),
               "sites": {s: int(len(set(obs["patient_id"][rows][obs["Site"][rows] == s])))
                         for s in sorted(set(obs["Site"][rows]))}}
    print(json.dumps(summary))


def extract():
    from adt_matching import STEPHENSON, names_for, matched_targets, COLON, HAO
    chosen = json.loads((HERE / "results/features.json").read_text())
    targets = matched_targets(STEPHENSON, COLON, HAO)
    assert targets == chosen["proteins"]
    with h5py.File(H5AD, "r") as f:
        rows, obs, barcodes = selection(f)
        names, is_adt = features(f)
        genes = np.flatnonzero(~is_adt)
        position = {n: i for i, n in enumerate(names)}
        gene_cols = np.array([np.flatnonzero(names[genes] == g)[0] for g in chosen["genes"]])
        adt_cols = np.array([position[n] for n in names_for(STEPHENSON, targets)])
        xs, counts = [], []
        for want, block in blocks(f, rows):
            norm, _ = normalise(block[:, genes])
            xs.append(norm[:, gene_cols].toarray().astype(np.float32))
            counts.append(block[:, adt_cols].toarray().astype(np.float32))
            print(len(xs) * BLOCK, flush=True)
    x, adt = np.vstack(xs), np.vstack(counts)
    patient = obs["patient_id"][rows]
    half = np.zeros(len(rows), np.int8)
    for p in sorted(set(patient)):
        idx = np.flatnonzero(patient == p)
        keys = [sha(f"pf-halves-v1|{p}|{barcodes[rows[i]]}") for i in idx]
        order = idx[np.argsort(keys)]
        half[order[len(order) // 2:]] = 1           # 0: RNA moments only; 1: protein moments only
    path = OUT / "stephenson_training.npz"
    text = lambda v: np.asarray(v).astype(str)
    np.savez_compressed(path, x=x, adt=adt, genes=text(chosen["genes"]), proteins=text(targets),
                        patient=text(patient), sample=text(obs["sample_id"][rows]), site=text(obs["Site"][rows]),
                        initial=text(obs["initial_clustering"][rows]), full=text(obs["full_clustering"][rows]),
                        barcode=text(barcodes[rows]), half=half)
    record = {"cells": int(len(rows)), "patients": int(len(set(patient))), "genes": len(chosen["genes"]),
              "proteins": len(targets), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "source_md5": MD5}
    (HERE / "results/stephenson_extract.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("stats", "extract"))
    mode = parser.parse_args().mode
    stats() if mode == "stats" else extract()
