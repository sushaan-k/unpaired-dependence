#!/usr/bin/env python3
"""Split the NeurIPS 2021 bone-marrow CITE-seq cells (GSE194122) into adaptation and scoring files.

Per sample (site x donor batch), cells are sorted by SHA-256 of
bmmc-recoverability-v1|batch|barcode; the first half (floor) is adaptation and
the rest scoring. Kept features: the frozen genes of cross_study/results/features.json
present among the deposited GEX features, and the frozen protein targets whose
antibody name in this panel equals the target name (exact match only). Raw
counts come from layers/counts; library size is the sum over all deposited GEX
features. Writes the two files to rawdata/bmmc and their SHA-256 to
results/bmmc_seal.json. No statistic of either file is computed here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
CS = HERE.parent / "cross_study"
RAW = Path("/home/claude/cbio/rawdata/bmmc/GSE194122_cite_BMMC_processed.h5ad")
OUT = Path("/home/claude/cbio/rawdata/bmmc")
SALT = "bmmc-recoverability-v1"

COARSE = {
    "B": ["B1 B IGKC+", "B1 B IGKC-", "Naive CD20+ B IGKC+", "Naive CD20+ B IGKC-", "Transitional B",
          "Plasma cell IGKC+", "Plasma cell IGKC-", "Plasmablast IGKC+", "Plasmablast IGKC-"],
    "CD4 T": ["CD4+ T CD314+ CD45RA+", "CD4+ T activated", "CD4+ T activated integrinB7+", "CD4+ T naive",
              "T reg"],
    "CD8 T": ["CD8+ T CD49f+", "CD8+ T CD57+ CD45RA+", "CD8+ T CD57+ CD45RO+", "CD8+ T CD69+ CD45RA+",
              "CD8+ T CD69+ CD45RO+", "CD8+ T TIGIT+ CD45RA+", "CD8+ T TIGIT+ CD45RO+", "CD8+ T naive",
              "CD8+ T naive CD127+ CD26- CD101-"],
    "other T": ["MAIT", "gdT CD158b+", "gdT TCRVD2+", "dnT", "T prog cycling"],
    "NK": ["NK", "NK CD158e1+", "ILC", "ILC1"],
    "Mono": ["CD14+ Mono", "CD16+ Mono"],
    "DC": ["cDC1", "cDC2", "pDC"],
    "progenitor": ["HSC", "Lymph prog", "G/M prog", "MK/E prog"],
    "erythroid": ["Proerythroblast", "Erythroblast", "Normoblast", "Reticulocyte"],
}


def column(group, key):
    values = group[key][:]
    if "__categories" in group and key in group["__categories"]:
        return group["__categories"][key][:].astype(str)[values]
    return values.astype(str) if values.dtype.kind in "OS" else values


def main():
    feats = json.loads((CS / "results/features.json").read_text())
    with h5py.File(RAW, "r") as f:
        obs, var = f["obs"], f["var"]
        names = column(var, "_index")
        kinds = column(var, "feature_types")
        barcodes = column(obs, "_index")
        batch = column(obs, "batch")
        donor = column(obs, "DonorID").astype(str)
        site = column(obs, "Site")
        fine = column(obs, "cell_type")
        g = f["layers/counts"]
        counts = sp.csr_matrix((g["data"][:], g["indices"][:], g["indptr"][:]), shape=(len(barcodes), len(names)))
    assert len(set(barcodes)) == len(barcodes)
    gex = np.flatnonzero(kinds == "GEX")
    adt = np.flatnonzero(kinds == "ADT")
    gex_index = {n: i for i, n in zip(gex, names[gex])}
    adt_index = {n: i for i, n in zip(adt, names[adt])}
    assert len(gex_index) == len(gex) and len(adt_index) == len(adt)
    genes = [g_ for g_ in feats["genes"] if g_ in gex_index]
    proteins = [p for p in feats["proteins"] if p in adt_index]
    coarse_of = {t: c for c, ts in COARSE.items() for t in ts}
    assert set(coarse_of) == set(fine), sorted(set(fine) ^ set(coarse_of))
    coarse = np.array([coarse_of[t] for t in fine])

    library = np.asarray(counts[:, gex].sum(axis=1)).ravel()
    rna = counts[:, [gex_index[g_] for g_ in genes]].tocsr()
    prot = counts[:, [adt_index[p] for p in proteins]].toarray().astype(np.float32)
    assert np.allclose(rna.data, np.rint(rna.data)) and np.allclose(prot, np.rint(prot))

    half = np.zeros(len(barcodes), np.int8)
    for b in sorted(set(batch)):
        idx = np.flatnonzero(batch == b)
        keys = [hashlib.sha256(f"{SALT}|{b}|{barcodes[i]}".encode()).hexdigest() for i in idx]
        half[idx[np.argsort(keys)][len(idx) // 2:]] = 1

    record = {"salt": SALT, "source": RAW.name, "source_gz_sha256": None, "genes": genes, "proteins": proteins,
              "genes_missing": [g_ for g_ in feats["genes"] if g_ not in gex_index],
              "proteins_missing": [p for p in feats["proteins"] if p not in adt_index],
              "coarse": COARSE, "files": {}}
    for name, flag in (("adaptation", 0), ("scoring", 1)):
        m = half == flag
        sub = rna[m].tocsr()
        path = OUT / f"bmmc_{name}.npz"
        np.savez_compressed(path, rna_data=sub.data.astype(np.float32), rna_indices=sub.indices,
                            rna_indptr=sub.indptr, rna_names=np.array(genes), library=library[m],
                            adt=prot[m], adt_names=np.array(proteins), barcodes=barcodes[m], batch=batch[m],
                            donor=donor[m], site=site[m], fine=fine[m], coarse=coarse[m])
        record["files"][name] = {"cells": int(m.sum()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    record["cells_per_batch"] = {b: [int(np.sum((batch == b) & (half == 0))), int(np.sum((batch == b) & (half == 1)))]
                                 for b in sorted(set(batch))}
    record["donor_of_batch"] = {b: sorted(set(donor[batch == b]))[0] for b in sorted(set(batch))}
    gz = RAW.with_suffix(".h5ad.gz")
    record["source_gz_sha256"] = hashlib.sha256(gz.read_bytes()).hexdigest() if gz.exists() else None
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results/bmmc_seal.json").write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k not in ("genes", "coarse")}, indent=1))


if __name__ == "__main__":
    main()
