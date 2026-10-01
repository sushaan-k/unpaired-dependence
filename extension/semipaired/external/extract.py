#!/usr/bin/env python3
"""External evaluation, step 0: parse OverCITE-seq, assign ORFs, hashtags and conditions, fix cell roles, seal.

    python extract.py

Reads only technical counts (ORF and hashtag UMIs) and RNA library sizes before sealing: ORF and hashtag calls,
the condition of each hashtag by median library size, and the cell roles. Writes
rawdata/overcite/overcite_{train,test_adaptation,test_scoring}.npz and results/seal.json with their SHA-256.
"""

from __future__ import annotations

import csv
import datetime
import gzip
import json

import numpy as np
import scipy.sparse as sp

import odata as od


def read_matrix(name, dense=True):
    path = od.RAW / name
    assert od.sha(path) == od.SOURCE_SHA256[name], name
    with gzip.open(path, "rt") as f:
        r = csv.reader(f)
        cells = next(r)[1:]
        rows, feats = [], []
        if dense:
            for line in r:
                feats.append(line[0])
                rows.append(np.array(line[1:], dtype=np.int64))
            return feats, cells, np.array(rows)
        data, indices, indptr = [], [], [0]
        for line in r:
            feats.append(line[0])
            v = np.array(line[1:], dtype=np.int64)
            nz = np.flatnonzero(v)
            data.append(v[nz])
            indices.append(nz)
            indptr.append(indptr[-1] + len(nz))
        M = sp.csr_matrix((np.concatenate(data), np.concatenate(indices), np.array(indptr)),
                          shape=(len(feats), len(cells)))
        return feats, cells, M


def main():
    hto_f, cells, hto = read_matrix("GSM5819657_HTO_counts.csv.gz")
    adt_f, c2, adt = read_matrix("GSM5819658_ADT_counts.csv.gz")
    orf_f, c3, orf = read_matrix("GSM5819659_ORF_counts.csv.gz")
    genes, c4, gex = read_matrix("GSM5819660_GEX_counts.csv.gz", dense=False)
    assert cells == c2 == c3 == c4
    gex = gex.T.tocsr()                                  # cells x genes
    library = np.asarray(gex.sum(1)).ravel().astype(float)
    # ORF calls
    o = np.sort(orf, axis=0)
    top, second = o[-1], o[-2]
    orf_call = np.array([orf_f[i] for i in np.argmax(orf, axis=0)])
    orf_ok = (top >= od.ORF_MIN) & (top >= od.ORF_RATIO * np.maximum(second, 0.5))
    # hashtag calls
    hs = np.sort(hto, axis=0)
    hto_call = np.array([hto_f[i] for i in np.argmax(hto, axis=0)])
    hto_ok = hs[-1] >= od.HTO_RATIO * np.maximum(hs[-2], 0.5)
    keep = orf_ok & hto_ok
    # condition of each hashtag: the four with the largest median RNA library size are the stimulated wells
    med = {t: float(np.median(library[keep & (hto_call == t)])) for t in hto_f}
    stim = set(sorted(hto_f, key=lambda t: -med[t])[:4])
    condition = np.array(["stim" if t in stim else "rest" for t in hto_call])
    cells = np.array(cells)
    held = set(od.heldout_orfs(orf_call[keep]))
    test = keep & np.isin(orf_call, list(held))
    train = keep & ~test
    hv = np.array([od.h(c) for c in cells])
    role = np.full(len(cells), "", dtype=object)
    role[train] = "train"
    role[test & (hv < 0.5)] = "test_adaptation"
    role[test & (hv >= 0.5)] = "test_scoring"
    part = (np.array([od.h(c + "|part") for c in cells]) >= 0.5).astype(int)   # assay half or scoring sub-half
    record = {"sealed_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
              "cells_total": int(len(cells)), "cells_kept": int(keep.sum()),
              "orf_calls_failed": int((~orf_ok).sum()), "hashtag_calls_failed": int((~hto_ok).sum()),
              "hashtag_median_library": med, "stimulated_hashtags": sorted(stim),
              "heldout_orfs": sorted(held), "training_orfs": sorted(set(orf_call[train])),
              "proteins": adt_f, "files": {}}
    for name in ("train", "test_adaptation", "test_scoring"):
        m = role == name
        Xs = gex[m]
        path = od.RAW / f"overcite_{name}.npz"
        np.savez_compressed(path, data=Xs.data.astype(np.int32), indices=Xs.indices.astype(np.int32),
                            indptr=Xs.indptr.astype(np.int64), library=library[m], adt=adt[:, m].T.astype(np.int32),
                            cell=cells[m], orf=orf_call[m], condition=condition[m], hto=hto_call[m], part=part[m],
                            all_genes=np.array(genes), proteins=np.array(adt_f))
        record["files"][name] = {"sha256": od.sha(path), "cells": int(m.sum())}
    (od.HERE / "results").mkdir(exist_ok=True)
    (od.HERE / "results/seal.json").write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k != "training_orfs"}, indent=1))


if __name__ == "__main__":
    main()
