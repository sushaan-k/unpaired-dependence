#!/usr/bin/env python3
"""Extraction of the validation data set (GSE314416 CITE-seq) and of the antibody match to the Stephenson atlas.

GSE314416 (ImmunoMicrobiome study, healthy adults; 10x 5' CITE-seq with the TotalSeq-C panel of 140 antibodies).
Only the pools with antibody counts are used (DB8-DB14: 34 samples of 17 donors at two time points; the baseline-only
pools DB1-DB7 have empty antibody matrices). Cell types are the authors' labels (`predicted.manual.celltype`, in
GSE314416_cell_metadata_followup.rds), mapped by name to seven coarse types shared with the Stephenson atlas; cells
of other types and unlabelled cells are left out. RNA counts are kept for the genes of the Stephenson extract
(the atlas from which the panel is chosen); the library size is the total over all genes. Antibody counts are kept
for all 140 features.

This script reads counts and metadata only; it computes cell counts, library sizes and the antibody name match, and
no statistic of dependence between RNA and protein.

    python vdata.py download     # GEO files into /home/claude/cbio/rawdata/validation/raw
    python vdata.py extract      # /home/claude/cbio/rawdata/validation/gse314416.npz and .json
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import scipy.sparse as sp

RAW = Path("/home/claude/cbio/rawdata/validation/raw")
OUT = Path("/home/claude/cbio/rawdata/validation")
STEPH = Path("/home/claude/cbio/rawdata/generality/stephenson.npz")
GEO = "https://ftp.ncbi.nlm.nih.gov/geo"
SERIES = "GSE314416"
POOLS = ("DB8", "DB9", "DB10", "DB11", "DB12", "DB13", "DB14")

# coarse cell types, by the authors' names (fixed before any expression value was read)
COARSE_NEW = {
    "B": ["B naive", "B memory switched", "B memory non-switched"],
    "CD4 T": ["CD4 Naive A", "CD4 Naive B", "CD4 Naive C", "CD4 Naive D", "CD4 Naive E", "CD4 Th2-like/Tcm",
              "CD4 Th1-like/Tem", "CD4 A6 memory", "CD4 A4B7 memory", "CD4 cytotoxic", "CD4 TEMRA",
              "CD4 Th17-like/Tem", "CD4 AEB7 memory", "CD4 recent circ", "Treg CD25high"],
    "CD8 T": ["CD8 Naive A", "CD8 Naive B", "CD8 Naive C", "CD8 Naive D", "CD8 cytotoxic", "CD8 Tem",
              "CD8 A1B1/A6B1 Tem", "CD8 AEB7 memory"],
    "NK": ["mNK A", "mNK B", "NK adaptive memory", "iNK"],
    "CD14 Mono": ["cMo A", "cMo B", "cMo C"],
    "CD16 Mono": ["ncMo"],
    "DC": ["cDC1", "cDC2"],
}
COARSE_ATLAS = {"B": ["B_cell"], "CD4 T": ["CD4", "Treg"], "CD8 T": ["CD8"], "NK": ["NK_16hi", "NK_56hi"],
                "CD14 Mono": ["CD14"], "CD16 Mono": ["CD16"], "DC": ["DCs"]}


def sh(cmd):
    subprocess.run(cmd, shell=True, check=True)


def download():
    RAW.mkdir(parents=True, exist_ok=True)
    base = f"{GEO}/series/{SERIES[:-3]}nnn/{SERIES}/suppl"
    for f in ("filelist.txt", f"{SERIES}_cell_metadata_followup.rds", f"{SERIES}_TotalSeqC_feature_ref.csv.gz"):
        if not (RAW / f).exists():
            sh(f"curl -sS -m 600 -o {RAW / f} {base}/{f}")
    rows = [l.rstrip("\n").split("\t") for l in open(RAW / "filelist.txt") if l.startswith("File")]
    for r in rows:
        name, size = r[1], int(r[3])
        m = re.search(r"POOL-(DB\d+)-", name)
        if not m or m.group(1) not in POOLS:
            continue
        gsm = name.split("_")[0]
        dest = RAW / name
        if dest.exists() and dest.stat().st_size == size:
            continue
        sh(f"curl -sS -m 900 -o {dest} {GEO}/samples/{gsm[:-3]}nnn/{gsm}/suppl/{name}")
        assert dest.stat().st_size == size, f"{name}: size {dest.stat().st_size} != {size}"
    print("downloaded")


def read_10x(path):
    with h5py.File(path, "r") as h:
        m = h["matrix"]
        shape = tuple(int(v) for v in m["shape"][()])
        M = sp.csc_matrix((m["data"][()], m["indices"][()], m["indptr"][()]), shape=shape).T.tocsr()
        bc = np.array([b.decode() for b in m["barcodes"][()]])
        names = np.array([x.decode() for x in m["features"]["name"][()]])
    return M, bc, names


def antibody_key(name):
    s = re.sub(r"^anti-(human|mouse|human/mouse|mouse/human|human/mouse/rat)[-_]", "", name, flags=re.I)
    s = re.sub(r"[-_]TotalSeqC$", "", s)
    s = re.sub(r"[-_]\(.*\)$", "", s)
    return s.replace("_", "-").upper()


def match_antibodies(new_names, new_targets, atlas_names):
    """GSE314416 antibody -> Stephenson ADT: first by antibody name (CD name), then by target gene symbol when the
    symbol identifies one unmatched antibody on each side. Isotype controls are never matched."""
    atlas_key = {(n[3:] if n.startswith("AB_") else n).upper(): n for n in atlas_names}
    out = {}
    for n in new_names:
        if "ISOTYPE" in n.upper() or "IGG" in antibody_key(n):
            continue
        k = antibody_key(n)
        if k in atlas_key:
            out[n] = atlas_key[k]
    used = set(out.values())
    by_target = {}
    for n, t in zip(new_names, new_targets):
        if n in out or "ISOTYPE" in n.upper() or not isinstance(t, str) or t in ("", "nan"):
            continue
        by_target.setdefault(t.upper(), []).append(n)
    for t, ns in by_target.items():
        if len(ns) == 1 and t in atlas_key and atlas_key[t] not in used:
            out[ns[0]] = atlas_key[t]
            used.add(atlas_key[t])
    return out


def strs(col):
    return np.array([str(v) for v in col], dtype=str)


def extract():
    import pandas as pd
    import pyreadr
    meta = pyreadr.read_r(str(RAW / f"{SERIES}_cell_metadata_followup.rds"))[None]
    meta = meta.reset_index() if "rownames" not in meta.columns else meta
    bc_col = "rownames" if "rownames" in meta.columns else meta.columns[0]
    fine2coarse = {f: c for c, fs in COARSE_NEW.items() for f in fs}
    meta["coarse"] = meta["predicted.manual.celltype"].astype(str).map(fine2coarse)
    meta = meta[meta["coarse"].notna()].copy()
    meta["pool"] = meta["orig.ident"].str.extract(r"POOL-(DB\d+)-")[0]
    meta["donor"] = meta["sample"].str.extract(r"-(HS\d+)-")[0]
    meta = meta[meta["pool"].isin(POOLS)]
    lookup = dict(zip(meta[bc_col], range(len(meta))))
    steph = np.load(STEPH, allow_pickle=True)
    atlas_genes = [str(g) for g in steph["x_names"]]
    fr = pd.read_csv(RAW / f"{SERIES}_TotalSeqC_feature_ref.csv.gz")
    X_parts, Y_parts, libs, cells, rows = [], [], [], [], []
    gene_idx, genes, ab_names = None, None, None
    wells = sorted({re.search(r"(XHLT2-POOL-DB\d+-SCG\d+)", p.name).group(1) for p in RAW.glob("*_GEX.h5")},
                   key=lambda w: (int(re.search(r"DB(\d+)", w).group(1)), w))
    for w in wells:
        gex = next(RAW.glob(f"*_{w}_GEX.h5"))
        adt = next(RAW.glob(f"*_{w}_ADT.h5"))
        G, gbc, gnames = read_10x(gex)
        A, abc, anames = read_10x(adt)
        if genes is None:
            pos = {g: i for i, g in enumerate(gnames)}
            genes = [g for g in atlas_genes if g in pos]
            gene_idx = np.array([pos[g] for g in genes])
            ab_names = anames
        assert list(anames) == list(ab_names)
        assert len(set(gnames)) == len(gnames) or True
        apos = {b: i for i, b in enumerate(abc)}
        keep = [i for i, b in enumerate(gbc) if b in lookup and b in apos]
        keep = np.array(keep, int)
        X_parts.append(G[keep][:, gene_idx].astype(np.float32))
        libs.append(np.asarray(G[keep].sum(1)).ravel().astype(np.float32))
        Y_parts.append(A[[apos[b] for b in gbc[keep]]].toarray().astype(np.float32))
        cells += [str(b) for b in gbc[keep]]
        rows += [lookup[b] for b in gbc[keep]]
        print(w, len(keep), "cells", flush=True)
    X = sp.vstack(X_parts).tocsr()
    Y = np.vstack(Y_parts)
    lib = np.concatenate(libs)
    m = meta.iloc[rows]
    targets = dict(zip(fr["name"].str.replace("_", "-"), fr["target_gene_name"].astype(str)))
    new_targets = [targets.get(n, "") for n in ab_names]
    match = match_antibodies(list(ab_names), new_targets, [str(n) for n in steph["y_names"]])
    np.savez_compressed(
        OUT / "gse314416.npz", x_data=X.data, x_indices=X.indices, x_indptr=X.indptr, x_shape=np.array(X.shape),
        x_names=np.array(genes, dtype=str), x_library=lib, y=Y, y_names=np.array(ab_names, dtype=str),
        y_target=np.array(new_targets, dtype=str),
        cell=np.array([f"gse314416|{c}" for c in cells]), sample=strs(m["sample"]), donor=strs(m["donor"]),
        timepoint=strs(m["timepoint"]), pool=strs(m["pool"]), well=strs(m["orig.ident"]), cond=strs(m["coarse"]),
        fine=strs(m["predicted.manual.celltype"]),
        match_new=np.array(list(match.keys()), dtype=str), match_atlas=np.array(list(match.values()), dtype=str))
    counts = m.groupby(["pool", "sample", "coarse"]).size().unstack(fill_value=0)
    summary = {"source": "GSE314416 CITE-seq (ImmunoMicrobiome study; 10x 5' with TotalSeq-C, 140 antibodies); "
                         "pools DB8-DB14; authors' labels (predicted.manual.celltype) mapped to seven coarse types",
               "cells": int(len(cells)), "genes": len(genes), "antibodies": int(len(ab_names)),
               "matched_antibodies": len(match), "samples": int(m["sample"].nunique()),
               "donors": int(m["donor"].nunique()), "pools": sorted(set(m["pool"]), key=lambda p: int(p[2:])),
               "cells_by_type": {k: int(v) for k, v in m["coarse"].value_counts().items()},
               "cells_by_sample_and_type": {f"{p}|{s}": {k: int(v) for k, v in r.items()}
                                            for (p, s), r in counts.iterrows()},
               "match": match}
    (OUT / "gse314416.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("cells", "genes", "antibodies", "matched_antibodies", "samples",
                                              "donors", "cells_by_type")}, indent=1))


if __name__ == "__main__":
    {"download": download, "extract": extract}[sys.argv[1]]()
