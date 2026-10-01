#!/usr/bin/env python3
"""Split the Hao et al. 2021 3' CITE-seq cells into adaptation and scoring files.

Per donor, cells are sorted by SHA-256 of blood-hao-v1|donor|barcode; the first
half (floor) is adaptation, the rest scoring. The two halves are written to
separate files and the scoring file's SHA-256 is recorded in hao_seal.json.
Only candidate genes (results/gene_candidates.json, fixed from Stephenson and
gene symbols alone) are kept, with every cell's total RNA count. Gene detection
rates are computed on adaptation cells only, for the eligibility rule.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

RAW = Path("/home/claude/cbio/rawdata/hao")
OUT = Path("/home/claude/cbio/rawdata/cross_study")
HERE = Path(__file__).resolve().parent
SALT = "blood-hao-v1"


def lines(name):
    return [l.decode().rstrip("\n").split("\t") for l in gzip.open(RAW / name)]


def read_mtx(name, chunk=20_000_000):
    """Stream a gzipped MatrixMarket coordinate file as (row, col, value) chunks (0-based)."""
    with gzip.open(RAW / name, "rt") as handle:
        header = 0
        for line in handle:
            header += 1
            if not line.startswith("%"):
                shape = tuple(int(v) for v in line.split())
                break
    reader = pd.read_csv(RAW / name, sep=" ", skiprows=header, header=None, dtype=np.int64,
                         chunksize=chunk, compression="gzip", engine="c")
    return shape, reader


def main():
    genes = [g[0] for g in lines("GSM5008737_RNA_3P-features.tsv.gz")]
    cells = [b[0] for b in lines("GSM5008737_RNA_3P-barcodes.tsv.gz")]
    assert cells == [b[0] for b in lines("GSM5008738_ADT_3P-barcodes.tsv.gz")]
    adt_names = [a[0] for a in lines("GSM5008738_ADT_3P-features.tsv.gz")]
    meta = pd.read_csv(RAW / "meta3p.csv.gz", index_col=0).loc[cells]
    donor = meta["donor"].to_numpy().astype(str)
    half = np.zeros(len(cells), np.int8)
    for d in sorted(set(donor)):
        idx = np.flatnonzero(donor == d)
        keys = [hashlib.sha256(f"{SALT}|{d}|{cells[i]}".encode()).hexdigest() for i in idx]
        half[idx[np.argsort(keys)][len(idx) // 2:]] = 1
    adapt = half == 0

    candidates = json.loads((HERE / "results/gene_candidates.json").read_text())["genes"]
    first = {}
    for i, g in enumerate(genes):
        first.setdefault(g, i)
    cand_rows = np.array([first[g] for g in candidates])
    lookup = np.full(len(genes), -1)
    lookup[cand_rows] = np.arange(len(candidates))

    (ng, nc, nnz), reader = read_mtx("GSM5008737_RNA_3P-matrix.mtx.gz")
    assert (ng, nc) == (len(genes), len(cells))
    library = np.zeros(nc)
    detected = np.zeros(ng)
    keep_r, keep_c, keep_v, seen = [], [], [], 0
    for block in reader:
        g, c, v = block[0].to_numpy() - 1, block[1].to_numpy() - 1, block[2].to_numpy()
        library += np.bincount(c, weights=v, minlength=nc)
        a = adapt[c] & (v > 0)
        detected += np.bincount(g[a], minlength=ng)
        m = lookup[g] >= 0
        keep_r.append(lookup[g[m]].astype(np.int32)); keep_c.append(c[m].astype(np.int32))
        keep_v.append(v[m].astype(np.float32))
        seen += len(g)
        print(f"RNA {seen}/{nnz}", flush=True)
    assert seen == nnz
    counts = sp.csr_matrix((np.concatenate(keep_v), (np.concatenate(keep_c), np.concatenate(keep_r))),
                           shape=(nc, len(candidates)))

    (na, nc2, nnz2), reader = read_mtx("GSM5008738_ADT_3P-matrix.mtx.gz")
    assert (na, nc2) == (len(adt_names), nc)
    adt = np.zeros((nc, na), np.float32)
    for block in reader:
        adt[block[1].to_numpy() - 1, block[0].to_numpy() - 1] = block[2].to_numpy()

    OUT.mkdir(parents=True, exist_ok=True)
    record = {"salt": SALT, "files": {}}
    for name, flag in (("adaptation", 0), ("scoring", 1)):
        m = half == flag
        sub = counts[m].tocsr()
        path = OUT / f"hao_{name}.npz"
        np.savez_compressed(path, rna_data=sub.data, rna_indices=sub.indices, rna_indptr=sub.indptr,
                            rna_names=np.array(candidates), library=library[m], adt=adt[m],
                            adt_names=np.array(adt_names), barcodes=np.array(cells)[m], donor=donor[m],
                            time=meta["time"].to_numpy()[m].astype(str),
                            l1=meta["celltype.l1"].to_numpy()[m].astype(str),
                            l2=meta["celltype.l2"].to_numpy()[m].astype(str),
                            l3=meta["celltype.l3"].to_numpy()[m].astype(str))
        record["files"][name] = {"cells": int(m.sum()),
                                 "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    record["cells_per_donor"] = {d: [int(np.sum((donor == d) & (half == 0))),
                                     int(np.sum((donor == d) & (half == 1)))] for d in sorted(set(donor))}
    (HERE / "hao_seal.json").write_text(json.dumps(record, indent=2) + "\n")
    rate = detected / adapt.sum()
    np.savez_compressed(OUT / "hao_adaptation_detection.npz", genes=np.array(genes), rate=rate)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
