#!/usr/bin/env python3
"""Replication, step 0: split the ECCITE-seq screen, choose the gene panel from training halves, seal.

Every target x replicate group (targets other than non-targeting) is split by
the SHA-256 order of cell names (salt papalexi-cells-v1): the first half is the
training half (its first half part 0, RNA moments; the rest part 1, protein
moments), the second half the test half (its first half adaptation cells,
part -1; the rest scoring cells alternating between sub-halves A = 0 and B = 1).
A target's training half trains the channels of every leave-one-target-out fold
except its own; its test half is used only in its own fold. Non-targeting cells
never train: per replicate, the first half is adaptation and the rest scoring.
The gene panel follows the frozen rule of extension/perturbation/extract.py,
applied to training-half cells only. For test and non-targeting cells only
library sizes are computed here. Writes five npz files to the raw-data folder
(not redistributed) and results/seal.json.
"""

from __future__ import annotations

import datetime
import json

import h5py
import numpy as np
import scipy.sparse as sp

from rdata import (CELL_SALT, ENCODING, ENCODING_MIN_DETECTION, HERE, HVG_BINS, HVG_MIN_DETECTION, PANEL_SIZE, RAW,
                   REPLICATES, SOURCE_SHA256, decode, h, sha, target_of)


def obs_column(f, key):
    g = f["obs"][key]
    if isinstance(g, h5py.Group):
        return decode(g["categories"][:])[g["codes"][:]]
    return g[:]


def main():
    for name, digest in SOURCE_SHA256.items():
        assert sha(RAW / name) == digest, name
    rna, prot = h5py.File(RAW / "PapalexiSatija2021_eccite_RNA.h5ad", "r"), h5py.File(
        RAW / "PapalexiSatija2021_eccite_protein.h5ad", "r")
    cells = decode(rna["obs"][rna["obs"].attrs["_index"]][:])
    assert (cells == decode(prot["obs"][prot["obs"].attrs["_index"]][:])).all()
    guide = obs_column(rna, "guide_id")
    replicate = obs_column(rna, "hto")
    target = np.array([target_of(g) for g in guide])
    keep = np.isin(replicate, REPLICATES)
    is_nt = keep & (target == "NT")
    is_t = keep & (target != "NT")
    part = np.full(len(cells), -9, int)
    role = np.full(len(cells), "", dtype=object)
    for key in sorted(set(zip(target[is_t], replicate[is_t]))):
        idx = np.flatnonzero(is_t & (target == key[0]) & (replicate == key[1]))
        idx = idx[np.argsort([h(CELL_SALT, cells[i]) for i in idx])]
        tr, te = idx[: len(idx) // 2], idx[len(idx) // 2:]
        part[tr[: len(tr) // 2]], part[tr[len(tr) // 2:]] = 0, 1
        role[tr] = "train"
        ad, sc = te[: len(te) // 2], te[len(te) // 2:]
        part[ad] = -1
        role[ad] = "test_adaptation"
        part[sc[0::2]], part[sc[1::2]] = 0, 1
        role[sc] = "test_scoring"
    for r in REPLICATES:
        idx = np.flatnonzero(is_nt & (replicate == r))
        idx = idx[np.argsort([h(CELL_SALT, cells[i]) for i in idx])]
        ad, sc = idx[: len(idx) // 2], idx[len(idx) // 2:]
        part[ad] = -1
        role[ad] = "nt_adaptation"
        part[sc[0::2]], part[sc[1::2]] = 0, 1
        role[sc] = "nt_scoring"
    train = role == "train"

    X = rna["X"]
    indptr = X["indptr"][:]
    data, indices = X["data"][:].astype(np.float64), X["indices"][:]
    gene_of = np.repeat(np.arange(len(indptr) - 1), np.diff(indptr))
    library = np.bincount(indices, weights=data, minlength=len(cells))
    genes = list(decode(rna["var"][rna["var"].attrs["_index"]][:]))
    m = train[indices]
    v = np.log1p(1e4 * data[m] / library[indices[m]])
    n_genes, n_train = len(genes), int(train.sum())
    det = np.bincount(gene_of[m], minlength=n_genes)
    s1 = np.bincount(gene_of[m], weights=v, minlength=n_genes)
    s2 = np.bincount(gene_of[m], weights=v * v, minlength=n_genes)
    frac, mean = det / n_train, s1 / n_train
    var = s2 / n_train - mean ** 2
    index = {g: i for i, g in enumerate(genes)}
    encoding = []
    for antibody, names in ENCODING.items():
        for g in names:
            if g in index and frac[index[g]] >= ENCODING_MIN_DETECTION and g not in encoding:
                encoding.append(g)
    ok = np.array([frac[i] >= HVG_MIN_DETECTION and not g.startswith(("MT-", "RPL", "RPS")) and g not in encoding
                   for i, g in enumerate(genes)])
    cand = np.flatnonzero(ok)
    disp = np.log(var[cand] / mean[cand])
    edges = np.quantile(mean[cand], np.linspace(0, 1, HVG_BINS + 1))
    b = np.clip(np.searchsorted(edges, mean[cand], side="right") - 1, 0, HVG_BINS - 1)
    z = np.zeros(len(cand))
    for k in range(HVG_BINS):
        sel = b == k
        if sel.sum() > 1:
            z[sel] = (disp[sel] - disp[sel].mean()) / disp[sel].std()
    order = sorted(range(len(cand)), key=lambda j: (-z[j], genes[cand[j]]))
    panel = encoding + [genes[cand[j]] for j in order[:PANEL_SIZE - len(encoding)]]
    csc = sp.csc_matrix((data, indices, indptr), shape=(len(cells), n_genes))
    panel_csr = csc[:, [index[g] for g in panel]].tocsr()

    pnames = list(decode(prot["var"][prot["var"].attrs["_index"]][:]))
    P = prot["X"]
    adt = sp.csc_matrix((P["data"][:], P["indices"][:], P["indptr"][:]), shape=(len(cells), len(pnames))).toarray()
    proteins = list(ENCODING)
    adt = np.rint(adt[:, [pnames.index(p) for p in proteins]]).astype(np.int64)
    files = {}
    for name in ("train", "test_adaptation", "test_scoring", "nt_adaptation", "nt_scoring"):
        rows = np.flatnonzero(role == name)
        sub = panel_csr[rows]
        path = RAW / f"papalexi_{name}.npz"
        np.savez_compressed(path, rna_data=sub.data.astype(np.int32), rna_indices=sub.indices.astype(np.int32),
                            rna_indptr=sub.indptr.astype(np.int64), library=library[rows], adt=adt[rows].astype(np.int32),
                            genes=np.array(panel), proteins=np.array(proteins), cell=cells[rows], target=target[rows],
                            replicate=replicate[rows], guide=guide[rows], part=part[rows])
        files[name] = {"sha256": sha(path), "cells": int(len(rows))}
    targets = sorted(set(target[is_t]))
    record = {"created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"), "source": SOURCE_SHA256,
              "cells": int(len(cells)), "used_cells": int(keep.sum()), "targets": targets, "replicates": list(REPLICATES),
              "genes": panel, "encoding_genes": encoding, "proteins": proteins, "files": files,
              "note": "test and non-targeting cells: only library sizes were computed before this seal"}
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results/seal.json").write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k != "genes"}, indent=1))


if __name__ == "__main__":
    main()
