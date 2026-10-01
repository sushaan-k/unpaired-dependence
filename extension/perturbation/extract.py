#!/usr/bin/env python3
"""Perturbation test, step 0: split the screen by target, choose the gene panel from training cells, seal.

Held-out targets are every fourth of the 248 targets in SHA-256 order (salt
perturb-heldout-v1); every cell whose guides target a held-out gene is kept out
of training and tuning. Within each target x condition population (and each
condition of the non-targeting cells), cells are ordered by SHA-256 of their
name (salt perturb-cells-v1):
  training populations  first half -> part 0 (RNA moments), second half -> part 1 (protein moments);
  held-out and NT groups first half -> adaptation (part -1), second half -> scoring, whose cells
                        alternate between sub-halves A (part 0) and B (part 1).

The gene panel is fixed from training cells only: genes encoding the 20 target
antibodies that are detected in at least 1% of training cells, then the most
variable genes (detected in at least 5%; not mitochondrial or ribosomal; log
dispersion z-scored within 20 bins of mean log expression) up to 200 genes.
For held-out and non-targeting cells only library sizes are computed here. No
statistic combining RNA and protein is computed. Writes the five npz files to the
raw-data folder (not redistributed) and results/seal.json.
"""

from __future__ import annotations

import datetime
import json
import time

import h5py
import numpy as np
import scipy.sparse as sp

from pdata import (CELL_SALT, ENCODING, ENCODING_MIN_DETECTION, HVG_BINS, HVG_MIN_DETECTION, ISOTYPES, PANEL_SIZE,
                   PROTEIN_FILE, RAW, RNA_FILE, SOURCE_SHA256, HERE, cell_class, decode, h, heldout_targets, sha)

CONDITION = {"Control": "control", "IFNγ": "ifng", "Co-culture": "coculture"}
BLOCK = 25_000_000


def obs_column(f, key):
    g = f["obs"][key]
    if isinstance(g, h5py.Group):
        return decode(g["categories"][:])[g["codes"][:]]
    return g[:]


def stream(X, train, library):
    """Library per cell and, over training cells, per-gene detection, sum and sum of squares of log expression."""
    indptr = X["indptr"][:]
    n_genes = len(indptr) - 1
    lib = np.zeros(len(train))
    det, s1, s2 = np.zeros(n_genes), np.zeros(n_genes), np.zeros(n_genes)
    total = indptr[-1]
    for start in range(0, total, BLOCK):
        end = min(start + BLOCK, total)
        data = X["data"][start:end].astype(np.float64)
        idx = X["indices"][start:end]
        gene = np.searchsorted(indptr, np.arange(start, end), side="right") - 1
        lib += np.bincount(idx, weights=data, minlength=len(train))
        m = train[idx]
        v = np.log1p(1e4 * data[m] / library[idx[m]])
        det += np.bincount(gene[m], minlength=n_genes)
        s1 += np.bincount(gene[m], weights=v, minlength=n_genes)
        s2 += np.bincount(gene[m], weights=v * v, minlength=n_genes)
        print(f"  streamed {end:,} of {total:,}", flush=True)
    return lib, det, s1, s2


def choose_panel(genes, det, s1, s2, n_train):
    frac = det / n_train
    mean = s1 / n_train
    var = s2 / n_train - mean ** 2
    index = {g: i for i, g in enumerate(genes)}
    encoding = []
    for antibody, names in ENCODING.items():
        for g in names:
            if g in index and frac[index[g]] >= ENCODING_MIN_DETECTION and g not in encoding:
                encoding.append(g)
    excluded = set(encoding)
    ok = np.array([frac[i] >= HVG_MIN_DETECTION and not g.startswith(("MT-", "RPL", "RPS")) and g not in excluded
                   for i, g in enumerate(genes)])
    cand = np.flatnonzero(ok)
    disp = np.log(var[cand] / mean[cand])
    edges = np.quantile(mean[cand], np.linspace(0, 1, HVG_BINS + 1))
    b = np.clip(np.searchsorted(edges, mean[cand], side="right") - 1, 0, HVG_BINS - 1)
    z = np.zeros(len(cand))
    for k in range(HVG_BINS):
        m = b == k
        if m.sum() > 1:
            z[m] = (disp[m] - disp[m].mean()) / disp[m].std()
    order = sorted(range(len(cand)), key=lambda j: (-z[j], genes[cand[j]]))
    hvg = [genes[cand[j]] for j in order[:PANEL_SIZE - len(encoding)]]
    stats = {g: {"detection": float(frac[index[g]]), "mean": float(mean[index[g]]), "var": float(var[index[g]])}
             for g in encoding + hvg}
    return encoding, hvg, stats


def parts(cells, keys, kind):
    """Assign parts within each population or group by the SHA-256 order of cell names."""
    part = np.full(len(cells), -9, int)
    for key in sorted(set(keys)):
        idx = np.flatnonzero(keys == key)
        idx = idx[np.argsort([h(CELL_SALT, cells[i]) for i in idx])]
        half = len(idx) // 2
        if kind == "training":
            part[idx[:half]], part[idx[half:]] = 0, 1
        else:
            part[idx[:half]] = -1
            score = idx[half:]
            part[score[0::2]], part[score[1::2]] = 0, 1
    return part


def write(name, rows, panel_csc, library, adt, meta, genes, proteins):
    csr = panel_csc[rows].tocsr()
    path = RAW / f"perturb_{name}.npz"
    np.savez_compressed(path, rna_data=csr.data.astype(np.int32), rna_indices=csr.indices.astype(np.int32),
                        rna_indptr=csr.indptr.astype(np.int64), library=library[rows], adt=adt[rows].astype(np.int32),
                        genes=np.array(genes), proteins=np.array(proteins),
                        **{k: v[rows] for k, v in meta.items()})
    return {"sha256": sha(path), "cells": int(len(rows))}


def main():
    t0 = time.time()
    for name, digest in SOURCE_SHA256.items():
        assert sha(RAW / name) == digest, name
    rna, prot = h5py.File(RNA_FILE, "r"), h5py.File(PROTEIN_FILE, "r")
    cells = decode(rna["obs/cell_name"][:])
    assert (cells == decode(prot["obs"][prot["obs"].attrs["_index"]][:])).all()
    guide = obs_column(rna, "guide_id")
    condition = np.array([CONDITION[c] for c in obs_column(rna, "perturbation_2")])
    moi = obs_column(rna, "MOI")
    klass, target = map(np.array, zip(*[cell_class(g) for g in guide]))
    single, nt = klass == "single", klass == "NT"
    targets = sorted(set(target[single]))
    held = heldout_targets(targets)
    training_targets = [t for t in targets if t not in held]
    train = single & ~np.isin(target, held)
    heldout = single & np.isin(target, held)
    print(f"cells {len(cells):,}; single {single.sum():,}; NT {nt.sum():,}; targets {len(targets)}; held out {len(held)}",
          flush=True)

    genes = list(decode(rna["var"][rna["var"].attrs["_index"]][:]))
    ncounts = rna["obs/ncounts"][:]
    lib, det, s1, s2 = stream(rna["X"], train, ncounts)
    assert np.allclose(lib, ncounts), "obs ncounts differs from the summed counts"
    encoding, hvg, stats = choose_panel(genes, det, s1, s2, int(train.sum()))
    panel = encoding + hvg
    print(f"panel: {len(encoding)} encoding genes, {len(hvg)} variable genes ({time.time() - t0:.0f}s)", flush=True)

    X = rna["X"]
    indptr = X["indptr"][:]
    cols_data, cols_idx, cols_ptr = [], [], [0]
    for g in panel:
        j = genes.index(g)
        cols_data.append(X["data"][indptr[j]:indptr[j + 1]].astype(np.int64))
        cols_idx.append(X["indices"][indptr[j]:indptr[j + 1]])
        cols_ptr.append(cols_ptr[-1] + len(cols_idx[-1]))
    panel_csc = sp.csc_matrix((np.concatenate(cols_data), np.concatenate(cols_idx), np.array(cols_ptr)),
                              shape=(len(cells), len(panel))).tocsr()

    pnames = list(decode(prot["var"][prot["var"].attrs["_index"]][:]))
    P = prot["X"]
    adt_all = sp.csc_matrix((P["data"][:], P["indices"][:], P["indptr"][:]), shape=(len(cells), len(pnames))).toarray()
    proteins = [p for p in pnames if p not in ISOTYPES]
    assert proteins == list(ENCODING), proteins
    adt = np.rint(adt_all[:, [pnames.index(p) for p in proteins]]).astype(np.int64)

    part = np.full(len(cells), -9, int)
    pop = np.array([f"{t}|{c}" for t, c in zip(target, condition)])
    part[train] = parts(cells[train], pop[train], "training")
    part[heldout] = parts(cells[heldout], pop[heldout], "test")
    part[nt] = parts(cells[nt], pop[nt], "test")
    meta = {"cell": cells, "target": target, "condition": condition, "guide": guide, "moi": moi, "part": part}
    files = {}
    files["training"] = write("training", np.flatnonzero(train), panel_csc, lib, adt, meta, panel, proteins)
    files["heldout_adaptation"] = write("heldout_adaptation", np.flatnonzero(heldout & (part == -1)), panel_csc, lib,
                                        adt, meta, panel, proteins)
    files["heldout_scoring"] = write("heldout_scoring", np.flatnonzero(heldout & (part >= 0)), panel_csc, lib, adt,
                                     meta, panel, proteins)
    files["nt_adaptation"] = write("nt_adaptation", np.flatnonzero(nt & (part == -1)), panel_csc, lib, adt, meta,
                                   panel, proteins)
    files["nt_scoring"] = write("nt_scoring", np.flatnonzero(nt & (part >= 0)), panel_csc, lib, adt, meta, panel,
                                proteins)
    classes = {str(k): int(v) for k, v in zip(*np.unique(klass, return_counts=True))}
    record = {"created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
              "source": SOURCE_SHA256, "cells": int(len(cells)), "classes": classes,
              "targets": len(targets), "heldout_targets": held, "training_targets": training_targets,
              "genes": panel, "encoding_genes": encoding, "proteins": proteins, "panel_stats": stats,
              "library_equals_ncounts": True, "files": files,
              "note": "held-out and non-targeting cells: only library sizes were computed before this seal"}
    (HERE / "results/seal.json").write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k not in ("panel_stats", "training_targets")}, indent=1))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
