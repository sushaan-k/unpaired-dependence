#!/usr/bin/env python3
"""Extraction of the prospective data sets (PLAN.md) into the common format of extension/generality.

    python pextract.py <name>   # /home/claude/cbio/rawdata/predictability/<name>.npz and <name>.json

Written after the plan was frozen (results/freeze.json). It uses cell and feature metadata and marginal statistics
of one modality at a time (detection rates, library sizes, missing values) and no statistic of dependence between
modalities. Deviation 1 (DEVIATIONS.md): no cell cap.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "generality"))
import gdata as gd  # noqa: E402  (frozen; its helpers are used read-only)

RAW = Path("/home/claude/cbio/rawdata/predictability/raw")
FLY = Path("/home/claude/cbio/rawdata/fly")
OUT = Path("/home/claude/cbio/rawdata/predictability")
MAX_CAND, MIN_DET, PEAK_DET = 3000, 0.01, 0.05


def save(name, meta, **arrays):
    OUT.mkdir(parents=True, exist_ok=True)
    arrays = {k: gd.unicode(v) for k, v in arrays.items()}
    np.savez_compressed(OUT / f"{name}.npz", **arrays)
    meta = dict(meta, cells=int(len(arrays["cell"])), units=int(len(set(arrays["unit"]))),
                x_candidates=int(len(arrays["x_names"])), y_features=int(len(arrays["y_names"])),
                conditions={c: int(np.sum(arrays["cond"] == c)) for c in sorted(set(arrays["cond"]))},
                populations=int(len(set(arrays["pop"]))))
    (OUT / f"{name}.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps({k: v for k, v in meta.items() if k != "conditions"}), flush=True)


def x_parts(X, names, library):
    """Candidate X features (detected in >= 1% of cells, at most 3,000 by detection) as CSR parts."""
    X = sp.csr_matrix(X, dtype=np.float32)
    c = gd.candidates(X, np.asarray(names), max_cand=MAX_CAND, min_det=MIN_DET)
    return dict(gd.csr_parts(X[:, c]), x_names=np.asarray(names)[c].astype(str), x_library=np.asarray(library, float))


def peaks(Y, names):
    """Peaks detected in >= 5% of cells, at most 3,000 by detection; library over all peaks."""
    Y = sp.csc_matrix(Y, dtype=np.float32)
    det = np.diff((Y > 0).astype(np.int8).tocsc().indptr) / Y.shape[0]
    ok = [j for j in range(Y.shape[1]) if det[j] >= PEAK_DET]
    ok = sorted(sorted(ok, key=lambda j: (-det[j], str(names[j])))[:MAX_CAND])
    return {"y": Y[:, ok].toarray().astype(np.float32), "y_names": np.asarray(names)[ok].astype(str),
            "y_library": np.asarray(Y.sum(1)).ravel(), "y_kind": np.array("lognorm")}


def dec(a):
    return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in a])


def h5ad(path, block=2000):
    """Counts (cells x genes, CSR), protein counts and obs/uns fields of the totalVI files."""
    with h5py.File(path, "r") as z:
        n = z["X"].shape[0]
        parts = [sp.csr_matrix(z["X"][i:i + block]) for i in range(0, n, block)]
        X = sp.vstack(parts).tocsr()
        P = z["obsm/protein_expression"][:].astype(np.float32)
        obs = z["obs"][:]
        uns = {k: dec(z["uns"][k][:]) for k in z["uns"].keys() if k != "version"}
        var = z["var"][:]
    return X, P, obs, uns, var


def isotype(names):
    return np.array([("isotype" in s.lower()) or ("igg" in s.lower() and "ctrl" in s.lower()) for s in names])


def sln(name, file):
    X, P, obs, uns, var = h5ad(RAW / file)
    genes = dec(var["index"])
    types = uns["cell_types_categories"][obs["cell_types"]]
    tissue = uns["hash_id_categories"][obs["hash_id"]]
    mouse = obs["batch_indices"].astype(int)
    bad = np.array([("doublet" in t.lower()) or ("low quality" in t.lower()) for t in types])
    keep = ~bad & (tissue != "Negative")
    pn = uns["protein_names"]
    iso = isotype(pn)
    barcodes = dec(obs["index"])
    Xk = X[np.flatnonzero(keep)]
    save(name, {"source": f"{file}, github.com/YosefLab/totalVI_reproducibility (Gayoso et al. 2021; GEO GSE150599)",
                "x": "RNA counts", "y": f"protein counts ({int((~iso).sum())} antibodies, isotype controls excluded: "
                                        f"{', '.join(pn[iso]) or 'none'})",
                "unit": "mouse (batch_indices)", "cond": "cell type (doublets and low-quality cells excluded)",
                "pop": "mouse x tissue", "excluded_cells": int((~keep).sum())},
         **x_parts(Xk, genes, np.asarray(Xk.sum(1)).ravel()), y=P[keep][:, ~iso], y_names=pn[~iso],
         y_kind=np.array("clr"), cell=np.array([f"{m}|{b}" for m, b in zip(mouse[keep], barcodes[keep])]),
         unit=np.array([f"mouse{m}" for m in mouse[keep]]), cond=types[keep],
         pop=np.array([f"mouse{m}|{t}" for m, t in zip(mouse[keep], tissue[keep])]))


def tenx(name, file, label):
    X, P, obs, uns, var = h5ad(RAW / file)
    genes = dec(var["index"])
    pn = uns["protein_names"]
    iso = isotype(pn)
    n = X.shape[0]
    save(name, {"source": f"{file}, github.com/YosefLab/totalVI_reproducibility (10x Genomics {label})",
                "x": "RNA counts", "y": f"protein counts ({int((~iso).sum())} antibodies)", "unit": "one donor",
                "cond": "all", "pop": "all"},
         **x_parts(X, genes, np.asarray(X.sum(1)).ravel()), y=P[:, ~iso], y_names=pn[~iso], y_kind=np.array("clr"),
         cell=dec(obs["index"]), unit=np.array(["donor"] * n), cond=np.array(["all"] * n), pop=np.array(["all"] * n))


def dense_tsv_rows(path, header=True):
    """Stream a features x cells dense TSV (first column a feature name when header=True) into a sparse
    cells x features matrix; returns (matrix, feature names, cell names)."""
    rows, cols, vals, names = [], [], [], []
    with gzip.open(path, "rt") as f:
        cells = f.readline().rstrip("\n").split("\t") if header else None
        if not header:
            f.seek(0)
        for j, line in enumerate(f):
            if header:
                name, rest = line.split("\t", 1)
                names.append(name)
            else:
                rest = line
                names.append(str(j))
            v = np.fromstring(rest, sep="\t")
            nz = np.flatnonzero(v)
            rows.append(nz)
            cols.append(np.full(len(nz), j, np.int64))
            vals.append(v[nz].astype(np.float32))
    ncell = len(cells) if header else len(v)
    M = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(ncell, len(names)))
    return M, np.array(names), (np.array(cells) if header else None)


def atac_stream(path, keep, pnames):
    """Peaks x cells dense TSV without header, streamed: per-cell library over all peaks, and the columns of peaks
    detected in >= 5% of the kept cells (at most 3,000 by detection); returns the peaks() dictionary."""
    lib, kept, det, names = None, [], [], []
    with gzip.open(path, "rt") as f:
        for j, line in enumerate(f):
            v = np.fromstring(line, sep="\t")[keep]
            lib = v.copy() if lib is None else lib + v
            d = float(np.mean(v > 0))
            if d >= PEAK_DET:
                kept.append(sp.csr_matrix(v.astype(np.float32)))
                det.append(d)
                names.append(pnames[j])
    assert j + 1 == len(pnames), "peak rows and consensus peaks differ"
    order = sorted(sorted(range(len(det)), key=lambda i: (-det[i], names[i]))[:MAX_CAND])
    Y = sp.vstack([kept[i] for i in order]).T.toarray().astype(np.float32)
    return {"y": Y, "y_names": np.array([names[i] for i in order]), "y_library": lib, "y_kind": np.array("lognorm")}


def bmcite():
    R, genes, cells = dense_tsv_rows(RAW / "GSM3681518_MNC_RNA_counts.tsv.gz")
    A, prots, cells_a = dense_tsv_rows(RAW / "GSM3681519_MNC_ADT_counts.tsv.gz")
    idx_a = {c: i for i, c in enumerate(cells_a)}
    both = [i for i, c in enumerate(cells) if c in idx_a]
    R = R[both]
    A = A[[idx_a[cells[i]] for i in both]].toarray()
    n = len(both)
    save("bmcite", {"source": "GEO GSE128639 (Stuart et al. 2019 bone-marrow CITE-seq): GSM3681518 RNA, GSM3681519 ADT",
                    "x": "RNA counts", "y": f"ADT counts ({len(prots)} antibodies)", "unit": "all (samples not "
                    "demultiplexed)", "cond": "all", "pop": "all"},
         **x_parts(R, genes, np.asarray(R.sum(1)).ravel()), y=A.astype(np.float32), y_names=prots,
         y_kind=np.array("clr"), cell=cells[both], unit=np.array(["all"] * n), cond=np.array(["all"] * n),
         pop=np.array(["all"] * n))


def fetal_cortex():
    meta = pd.read_csv(RAW / "GSE162170_multiome_cell_metadata.txt.gz", sep="\t")
    names = pd.read_csv(RAW / "GSE162170_multiome_cluster_names.txt.gz", sep="\t")
    rna_names = dict(zip(names[names["Assay"] == "Multiome RNA"]["Cluster.ID"],
                         names[names["Assay"] == "Multiome RNA"]["Cluster.Name"]))
    R, genes, cells = dense_tsv_rows(RAW / "GSE162170_multiome_rna_counts.tsv.gz")
    assert list(cells) == list(meta["Cell.ID"]), "RNA columns and metadata rows differ in order"
    pk = pd.read_csv(RAW / "GSE162170_multiome_atac_consensus_peaks.txt.gz", sep="\t")
    pnames = (pk["seqnames"].astype(str) + ":" + pk["start"].astype(str) + "-" + pk["end"].astype(str)).values
    keep = (meta["DF_classification"] == "Singlet").values
    A = atac_stream(RAW / "GSE162170_multiome_atac_counts.tsv.gz", keep, pnames)
    R = R[np.flatnonzero(keep)]
    m = meta[keep]
    save("fetal_cortex", {"source": "GEO GSE162170 (Trevino et al. 2021 human fetal cortex multiome): RNA and ATAC "
                                    "counts, cell metadata; ATAC columns in the order of the RNA columns (the file has "
                                    "no header; its column count equals the metadata's rows)",
                          "x": "RNA counts", "y": "ATAC peak counts, peaks detected in >= 5% of cells (<= 3,000)",
                          "unit": "sample", "cond": "RNA cluster name", "pop": "sample"},
         **x_parts(R, genes, np.asarray(R.sum(1)).ravel()), **A, cell=m["Cell.ID"].values,
         unit=m["Sample.ID"].values, cond=np.array([rna_names[c] for c in m["seurat_clusters"].values]),
         pop=m["Sample.ID"].values)


def snare_cortex():
    p = "GSE126074_AdBrainCortex_SNAREseq_"
    R = sio.mmread(gzip.open(RAW / f"{p}cDNA.counts.mtx.gz")).T.tocsr()
    A = sio.mmread(gzip.open(RAW / f"{p}chromatin.counts.mtx.gz")).T.tocsr()
    bc_r = pd.read_csv(RAW / f"{p}cDNA.barcodes.tsv.gz", header=None)[0].astype(str).values
    bc_a = pd.read_csv(RAW / f"{p}chromatin.barcodes.tsv.gz", header=None)[0].astype(str).values
    genes = pd.read_csv(RAW / f"{p}cDNA.genes.tsv.gz", header=None)[0].astype(str).values
    pn = pd.read_csv(RAW / f"{p}chromatin.peaks.tsv.gz", header=None)[0].astype(str).values
    ia = {b: i for i, b in enumerate(bc_a)}
    both = [i for i, b in enumerate(bc_r) if b in ia]
    R = R[both]
    A = A[[ia[bc_r[i]] for i in both]]
    lib = np.array([b.split("_")[0] for b in bc_r[both]])
    n = len(both)
    save("snare_cortex", {"source": "GEO GSE126074 (Chen et al. 2019 SNARE-seq, adult mouse cortex): cDNA and chromatin "
                                    "counts matched by barcode",
                          "x": "RNA counts", "y": "chromatin peak counts, peaks detected in >= 5% of cells (<= 3,000)",
                          "unit": "library (barcode prefix)", "cond": "all", "pop": "library"},
         **x_parts(R, genes, np.asarray(R.sum(1)).ravel()), **peaks(A, pn), cell=bc_r[both], unit=lib,
         cond=np.array(["all"] * n), pop=lib)


EXCLUDED_PARTNERS = {"visual_projection", "visual_centrifugal", "descending", "ascending", "sensory_ascending",
                     "sensory_descending", "ascending_visceral_circulatory", "glia", "not_a_neuron", "trachea",
                     "unknown"}


def fly_vpn(name, meta_file, edge_file, idcol):
    meta = pd.read_feather(FLY / meta_file)
    meta[idcol] = meta[idcol].astype(str)
    nm = meta[meta["super_class"].isin(["visual_projection", "visual_centrifugal"]) & meta["cell_type"].notna()
              & (meta["cell_type"].astype(str) != "nan")]
    neurons = nm[idcol].tolist()
    edges = gd.incident_edges(FLY / edge_file, neurons)
    m = meta.set_index(idcol)
    ok_partner = m["super_class"].notna() & ~m["super_class"].isin(EXCLUDED_PARTNERS)

    def counts(region, group):
        part = m[ok_partner & (m["region"] == region)]
        g = group(part)
        rows = []
        for direction, me, other in (("in", "post", "pre"), ("out", "pre", "post")):
            e = edges[edges[me].isin(neurons) & edges[other].isin(g.index)]
            rows.append(pd.DataFrame({"neuron": e[me].values, "group": direction + "|" + g.loc[e[other].values].values,
                                      "count": e["count"].values}))
        df = pd.concat(rows).groupby(["neuron", "group"])["count"].sum().unstack(fill_value=0)
        return df.reindex(neurons).fillna(0)

    def optic_group(part):
        ct = part["cell_type"].where(part["cell_type"].notna() & (part["cell_type"].astype(str) != "nan"))
        cc = part["cell_class"].where(part["cell_class"].notna() & (part["cell_class"].astype(str) != "nan"))
        return part["super_class"].astype(str) + ":" + ct.fillna(cc).fillna("NA").astype(str)

    Xo = counts("optic_lobe", optic_group)
    Yc = counts("central_brain", gd.partner_group)
    keep = (Xo.sum(axis=1) >= gd.MIN_SIDE) & (Yc.sum(axis=1) >= gd.MIN_SIDE)
    ids = [n for n, k in zip(neurons, keep) if k]
    Xo, Yc = Xo.loc[ids], Yc.loc[ids]
    Ydet = (Yc.values > 0).mean(0)
    yc = [j for j in range(Yc.shape[1]) if Ydet[j] >= MIN_DET]
    info = nm.set_index(idcol).loc[ids]
    save(name, {"source": f"{name}: compiled data of the fly connectome tutorial (lee-lab_brain-and-nerve-cord-fly-"
                          f"connectome/compiled_data): {meta_file}, {edge_file}",
                "x": "optic-lobe synapse counts with partner groups (super class : cell type, else cell class), inputs "
                     "and outputs", "y": "central-brain synapse counts with partner groups (super class : hemilineage, "
                                         "else cell class), inputs and outputs",
                "unit": "cell type", "cond": "super class (visual projection, visual centrifugal)", "pop": "all",
                "neurons": f"visual projection and visual centrifugal neurons with a cell type and >= {gd.MIN_SIDE} "
                           f"synapses on each side"},
         **x_parts(sp.csr_matrix(Xo.values.astype(np.float32)), Xo.columns.values.astype(str), Xo.values.sum(1)),
         y=Yc.values[:, yc].astype(np.float32), y_names=Yc.columns.values[yc].astype(str), y_library=Yc.values.sum(1),
         y_kind=np.array("lognorm"), cell=np.array(ids), unit=info["cell_type"].values.astype(str),
         cond=info["super_class"].values.astype(str), pop=np.array(["all"] * len(ids)))


def human_gaba():
    import pyreadr
    e1 = pyreadr.read_r(RAW / "complete_patchseq_data_sets1.RData")["datPatch1"]
    e2 = pyreadr.read_r(RAW / "complete_patchseq_data_sets2.RData")["datPatch2"]
    expr = pd.concat([e1, e2], axis=1)
    meta = pd.read_csv(RAW / "LeeDalley_manuscript_metadata_v2.csv")
    fx = pd.read_csv(RAW / "LeeDalley_ephys_fx.csv")
    meta = meta[meta["patched_cell_container"].isin(expr.columns)]
    meta = meta[meta["specimen_id_x"].isin(fx["specimen_id"])]
    fx = fx.set_index("specimen_id").loc[meta["specimen_id_x"].values]
    feats = [c for c in fx.columns if fx[c].isna().mean() <= 0.10]
    ok = fx[feats].notna().all(axis=1).values
    meta, fx = meta[ok], fx[ok]
    E = expr[meta["patched_cell_container"].values].T.values.astype(np.float32)   # cells x genes, log values
    genes = expr.index.values.astype(str)
    det = (E > 0).mean(0)
    cand = [j for j in range(E.shape[1]) if det[j] >= MIN_DET]
    cand = sorted(sorted(cand, key=lambda j: (-det[j], genes[j]))[:MAX_CAND])
    n = len(meta)
    save("human_gaba", {"source": "github.com/AllenInstitute/human_patchseq_gaba (Lee, Dalley et al. 2023 human cortex "
                                  "GABAergic Patch-seq): complete_patchseq_data_sets1-2.RData, LeeDalley_ephys_fx.csv, "
                                  "LeeDalley_manuscript_metadata_v2.csv; expression as provided (log values, no counts)",
                        "x": "expression (log values) of genes detected in >= 1% of cells (<= 3,000)",
                        "y": f"{len(feats)} electrophysiological features with <= 10% missing values",
                        "unit": "donor", "cond": "subclass (subclass_label)", "pop": "all"},
         x_dense=E[:, cand], x_names=genes[cand], y=fx[feats].values.astype(np.float32), y_names=np.array(feats),
         y_kind=np.array("raw"), cell=meta["patched_cell_container"].values.astype(str),
         unit=meta["Donor"].values.astype(str), cond=meta["subclass_label"].values.astype(str),
         pop=np.array(["all"] * n))


if __name__ == "__main__":
    name = sys.argv[1]
    {"sln111": lambda: sln("sln111", "spleen_lymph_111.h5ad"), "sln206": lambda: sln("sln206", "spleen_lymph_206.h5ad"),
     "pbmc10k": lambda: tenx("pbmc10k", "pbmc_10k_protein_v3.h5ad", "PBMC 10k v3"),
     "malt10k": lambda: tenx("malt10k", "malt_10k_protein_v3.h5ad", "MALT 10k v3"),
     "bmcite": bmcite, "fetal_cortex": fetal_cortex, "snare_cortex": snare_cortex,
     "fafb_vpn": lambda: fly_vpn("fafb_vpn", "fafb_783_meta.feather", "fafb_783_simple_edgelist.feather", "fafb_783_id"),
     "banc_vpn": lambda: fly_vpn("banc_vpn", "banc_888_meta.feather", "banc_888_edgelist_simple_v3.feather",
                                 "banc_888_id"),
     "human_gaba": human_gaba}[name]()
