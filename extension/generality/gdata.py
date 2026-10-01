#!/usr/bin/env python3
"""Data sets of the generality benchmark (PLAN.md): one compact file per data set in a common format.

    python gdata.py <name>        # writes /home/claude/cbio/rawdata/generality/<name>.npz and <name>.json

Common format. Cells (neurons for connectomes) with
  x_*        X-modality counts (CSR over candidate features) and x_library (total over all features), or x_dense
             (continuous values, no counts)
  y          Y-modality raw values (dense) and y_library (total over all Y features, for count modalities)
  cell       identifier, unit (replication unit, held out whole), cond (condition), pop (centring population
             within which cells are centred; populations nest in units or equal the condition)
Candidate features: X features detected in at least 1% of cells (at most 3,000, by detection); Y features as
described per data set. Cell caps: the first N cells of each sample in SHA-256 order of
"generality-cap-v1|<data set>|<cell>". Everything here uses marginal statistics of one modality at a time and no
statistic of dependence between modalities.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

RAW = Path("/home/claude/cbio/rawdata")
OUT = RAW / "generality"
MAX_CAND, MIN_DET = 3000, 0.01


def h(text):
    return hashlib.sha256(text.encode()).hexdigest()


def cap(name, cells, groups, n):
    """Indices of the first n cells of each group in SHA-256 order."""
    keep = []
    for g in sorted(set(groups)):
        idx = np.flatnonzero(groups == g)
        order = sorted(idx, key=lambda i: h(f"generality-cap-v1|{name}|{cells[i]}"))
        keep.extend(order[:n])
    return np.array(sorted(keep))


def candidates(X, names, max_cand=MAX_CAND, min_det=MIN_DET, exclude=()):
    """Columns detected in >= min_det of cells, at most max_cand by detection (ties by name)."""
    X = sp.csr_matrix(X)
    det = np.bincount(X.indices, minlength=X.shape[1]) / X.shape[0]
    ok = [j for j in range(X.shape[1]) if det[j] >= min_det and not str(names[j]).startswith(exclude)]
    ok = sorted(ok, key=lambda j: (-det[j], str(names[j])))[:max_cand]
    ok = sorted(ok)
    return ok


def unicode(a):
    a = np.asarray(a)
    return np.array([str(v) for v in a.ravel()]).reshape(a.shape) if a.dtype == object else a


def save(name, meta, **arrays):
    OUT.mkdir(parents=True, exist_ok=True)
    arrays = {k: unicode(v) for k, v in arrays.items()}
    np.savez_compressed(OUT / f"{name}.npz", **arrays)
    meta = dict(meta, cells=int(len(arrays["cell"])), units=int(len(set(arrays["unit"]))),
                conditions={c: int(np.sum(arrays["cond"] == c)) for c in sorted(set(arrays["cond"]))})
    (OUT / f"{name}.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps({k: v for k, v in meta.items() if k != "conditions"}), flush=True)


def csr_parts(X):
    X = sp.csr_matrix(X)
    return {"x_data": X.data.astype(np.float32), "x_indices": X.indices.astype(np.int32),
            "x_indptr": X.indptr.astype(np.int64), "x_shape": np.array(X.shape)}


def h5col(group, key):
    v = group[key][:]
    cats = group.get("__categories")
    if cats is not None and key in cats:
        c = cats[key]
        c = c.asstr()[:] if c.dtype.kind in "OSU" else c[:]
        return np.asarray(c)[v].astype(str)
    if "categories" in getattr(group[key], "attrs", {}):
        pass
    return v.astype(str) if v.dtype.kind in "OS" else v


def h5_stream(f, layer, rows, block=4000):
    """CSR blocks of the selected rows (sorted) of an h5ad sparse layer."""
    g = f[layer]
    indptr = g["indptr"][:]
    ncol = int(g.attrs["shape"][1])
    for s in range(0, len(rows), block):
        rr = rows[s:s + block]
        lo, hi = indptr[rr[0]], indptr[rr[-1] + 1]
        blk = sp.csr_matrix((g["data"][lo:hi], g["indices"][lo:hi], indptr[rr[0]:rr[-1] + 2] - lo),
                            shape=(rr[-1] - rr[0] + 1, ncol))[rr - rr[0]]
        print(f"  rows {s + len(rr)}/{len(rows)}", flush=True)
        yield blk.tocsr()


def h5_select(f, layer, rows, groups):
    """Two passes: per-column detection and per-group row totals, then the columns chosen by `groups`, a list of
    (column mask, chooser) where chooser(detection, names_index) returns the kept column indices."""
    ncol = int(f[layer].attrs["shape"][1])
    det, n = np.zeros(ncol), 0
    totals = [[] for _ in groups]
    for blk in h5_stream(f, layer, rows):
        det += np.bincount(blk.indices, minlength=ncol)
        n += blk.shape[0]
        for i, (mask, _) in enumerate(groups):
            totals[i].append(np.asarray(blk[:, np.flatnonzero(mask)].sum(1)).ravel())
    det /= n
    cols = [chooser(det) for _, chooser in groups]
    allc = np.concatenate(cols)
    parts = [blk[:, allc] for blk in h5_stream(f, layer, rows)]
    M = sp.vstack(parts).tocsr()
    out, start = [], 0
    for c in cols:
        out.append(M[:, start:start + len(c)])
        start += len(c)
    return out, cols, [np.concatenate(t) for t in totals]


def top_detected(mask, names, min_det, max_cand, exclude=()):
    def chooser(det):
        ok = [j for j in np.flatnonzero(mask) if det[j] >= min_det and not str(names[j]).startswith(exclude)]
        return np.array(sorted(sorted(ok, key=lambda j: (-det[j], str(names[j])))[:max_cand]), dtype=np.int64)
    return chooser


# ------------------------------------------------------------------ droplet CITE-seq and multiome

COARSE_BMMC = {  # extension/recoverability/extract_bmmc.py (CITE), extended by keyword for multiome labels
    "B": ["B1 B", "Naive CD20+ B", "Transitional B", "Plasma cell", "Plasmablast"],
    "CD4 T": ["CD4+ T", "T reg"], "CD8 T": ["CD8+ T"], "other T": ["MAIT", "gdT", "dnT", "T prog cycling"],
    "NK": ["NK", "ILC"], "Mono": ["CD14+ Mono", "CD16+ Mono"], "DC": ["cDC1", "cDC2", "pDC"],
    "progenitor": ["HSC", "Lymph prog", "G/M prog", "MK/E prog", "ID2-hi myeloid prog"],
    "erythroid": ["Proerythroblast", "Erythroblast", "Normoblast", "Reticulocyte"],
}


def coarse_bmmc(label):
    for c, prefixes in COARSE_BMMC.items():
        if any(label.startswith(p) for p in prefixes):
            return c
    return "other"


def hao():
    parts = [np.load(RAW / "cross_study" / f"hao_{p}.npz") for p in ("adaptation", "scoring")]
    X = sp.vstack([sp.csr_matrix((d["rna_data"], d["rna_indices"], d["rna_indptr"]),
                                 shape=(len(d["barcodes"]), len(d["rna_names"]))) for d in parts]).tocsr()
    cat = lambda k: np.concatenate([d[k] for d in parts])
    cell, donor, time, l1 = cat("barcodes"), cat("donor"), cat("time"), cat("l1")
    sample = np.array([f"{a}_{b}" for a, b in zip(donor, time)])
    keep = cap("hao", cell, sample, 3000)
    genes = parts[0]["rna_names"]
    X = X[keep]
    c = candidates(X, genes, exclude=("MT-", "RPL", "RPS"))
    save("hao", {"source": "Hao et al. 2021 PBMC CITE-seq (GSE164378), 3' data; 606-gene cross-study panel as candidates",
                 "x": "RNA counts", "y": "ADT counts (228)", "unit": "donor", "cond": "celltype.l1",
                 "pop": "donor x time point (sample)"},
         **csr_parts(X[:, c]), x_names=genes[c], x_library=cat("library")[keep], y=cat("adt")[keep],
         y_names=parts[0]["adt_names"], y_kind=np.array("clr"), cell=cell[keep], unit=donor[keep], cond=l1[keep],
         pop=sample[keep])


def stephenson():
    path = RAW / "stephenson.h5ad"
    with h5py.File(path, "r") as f:
        obs = {k: h5col(f["obs"], k) for k in ("patient_id", "sample_id", "time_after_LPS", "initial_clustering")}
        cells = h5col(f["obs"], f["obs"].attrs["_index"])
        names = f["var"]["_index"].asstr()[:]
        kind = h5col(f["var"], "feature_types")
        ok = (obs["time_after_LPS"] == "nan") & ~np.isin(obs["initial_clustering"], ["Platelets", "RBC", "Doublets"])
        idx = np.flatnonzero(ok)
        keep = idx[cap("stephenson", cells[idx], obs["patient_id"][idx], 600)]
        gene = kind == "Gene Expression"
        (G, A), (gc, ac), (lib, _) = h5_select(f, "layers/raw", keep, [
            (gene, top_detected(gene, names, MIN_DET, MAX_CAND, ("MT-", "RPL", "RPS"))),
            (~gene, lambda det: np.flatnonzero(~gene))])
    save("stephenson", {"source": "Stephenson et al. 2021 COVID-19 PBMC CITE-seq (E-MTAB-10026); no LPS; not "
                                  "platelets, red cells or doublets; 600 cells per patient",
                        "x": "RNA counts", "y": f"ADT counts ({len(ac)})", "unit": "patient",
                        "cond": "initial_clustering", "pop": "sample"},
         **csr_parts(G), x_names=names[gc], x_library=lib, y=A.toarray().astype(np.float32),
         y_names=names[ac], y_kind=np.array("clr"), cell=cells[keep], unit=obs["patient_id"][keep],
         cond=obs["initial_clustering"][keep], pop=obs["sample_id"][keep])


def bmmc(name, file, ykind):
    with h5py.File(RAW / file, "r") as f:
        obs = {k: h5col(f["obs"], k) for k in ("DonorID", "batch", "cell_type")}
        cells = h5col(f["obs"], "_index")
        names = h5col(f["var"], "_index")
        kind = h5col(f["var"], "feature_types")
        keep = cap(name, cells, obs["batch"], 3000)
        gene = kind == "GEX"
        ychoose = (top_detected(~gene, names, 0.05, MAX_CAND) if ykind == "atac"
                   else (lambda det: np.flatnonzero(~gene)))
        (G, Ym), (gc, yc), (lib, ylib) = h5_select(f, "layers/counts", keep, [
            (gene, top_detected(gene, names, MIN_DET, MAX_CAND, ("MT-", "RPL", "RPS"))), (~gene, ychoose)])
    save(name, {"source": f"NeurIPS 2021 bone-marrow {'CITE-seq' if ykind == 'clr' else 'multiome'} (GSE194122); "
                          "3,000 cells per batch", "x": "RNA counts",
                "y": "ADT counts" if ykind == "clr" else "ATAC peak counts, peaks detected in >= 5% of cells (<= 3,000)",
                "unit": "donor", "cond": "coarse cell type", "pop": "batch (site x donor)"},
         **csr_parts(G), x_names=names[gc], x_library=lib, y=Ym.toarray().astype(np.float32),
         y_names=names[yc], y_library=ylib, y_kind=np.array("clr" if ykind == "clr" else "lognorm"),
         cell=cells[keep], unit=obs["DonorID"][keep].astype(str),
         cond=np.array([coarse_bmmc(t) for t in obs["cell_type"][keep]]), pop=obs["batch"][keep])


def colon():
    d = np.load(RAW / "colon_paired.npz")
    X = sp.csr_matrix((d["rna_data"], d["rna_indices"], d["rna_indptr"]), shape=(len(d["barcodes"]), len(d["rna_names"])))
    library = np.asarray(X.sum(1)).ravel()
    c = candidates(X, d["rna_names"], exclude=("MT-", "RPL", "RPS"))
    save("colon", {"source": "Mennillo et al. 2024 colon CITE-seq (Figshare 21919356 v3)", "x": "RNA counts",
                   "y": "ADT counts (177)", "unit": "patient", "cond": "coarse annotation", "pop": "sample"},
         **csr_parts(X[:, c]), x_names=d["rna_names"][c], x_library=library, y=d["adt"].astype(np.float32),
         y_names=d["adt_names"], y_kind=np.array("clr"), cell=d["barcodes"], unit=d["CoLabs_patient"],
         cond=d["coarse_annotations_MK"], pop=d["CoLabs_sample"])


# ------------------------------------------------------------------ Patch-seq

def scala():
    base = RAW / "patchseq"
    meta = pd.read_csv(base / "m1_patchseq_meta_data.csv", sep="\t")
    eph = pd.read_csv(base / "m1_patchseq_ephys_features.csv", sep="\t" if "\t" in open(base / "m1_patchseq_ephys_features.csv").readline() else ",")
    with gzip.open(base / "m1_patchseq_exon_counts.csv.gz", "rt") as fh:
        ex = pd.read_csv(fh, index_col=0)
    eph = eph.set_index("cell id")
    meta = meta.set_index("Cell")
    feats = [c for c in eph.columns if eph[c].isna().mean() <= 0.10]
    ok = [c for c in meta.index if c in eph.index and c in ex.columns and isinstance(meta.loc[c, "RNA family"], str)
          and meta.loc[c, "RNA family"] != "low quality"
          and not isinstance(meta.loc[c, "Exclusion reasons"], str) and not eph.loc[c, feats].isna().any()]
    X = sp.csr_matrix(ex[ok].values.T.astype(np.float32))
    genes = ex.index.values.astype(str)
    library = np.asarray(X.sum(1)).ravel()
    c = candidates(X, genes, exclude=("mt-", "Rpl", "Rps"))
    save("scala_m1", {"source": "Scala et al. 2021 mouse motor cortex Patch-seq (github.com/berenslab/mini-atlas); "
                                "cells without exclusion reasons, with an RNA family other than 'low quality' and all "
                                "kept features",
                      "x": "exon read counts", "y": f"electrophysiological features ({len(feats)}, <= 10% missing)",
                      "unit": "mouse", "cond": "RNA family", "pop": "condition"},
         **csr_parts(X[:, c]), x_names=genes[c], x_library=library, y=eph.loc[ok, feats].values.astype(np.float32),
         y_names=np.array(feats), y_kind=np.array("raw"), cell=np.array(ok), unit=meta.loc[ok, "Mouse"].values.astype(str),
         cond=meta.loc[ok, "RNA family"].values.astype(str), pop=np.array(["all"] * len(ok)))


def gouwens():
    import scipy.io as sio
    d = sio.loadmat(RAW / "patchseq" / "gouwens_PS_v5.mat")
    flat = lambda k: np.array([str(v[0]) if hasattr(v, "__len__") else str(v) for v in d[k].ravel()])
    sample = flat("sample_id")
    cluster = flat("cluster")
    assert np.all(d["T_spec_id_label"].ravel() == d["E_spec_id_label"].ravel())
    day = np.array([s.split("_")[1] for s in sample])
    sub = np.array([c.split(" ")[0] for c in cluster])
    y = np.hstack([d["E_feature"], d["E_pc_scaled"]]).astype(np.float32)
    ynames = np.concatenate([flat("feature_name"), flat("pc_name")])
    ok = np.all(np.isfinite(y), 1) & np.all(np.isfinite(d["T_dat"]), 1)
    save("gouwens_visp", {"source": "Gouwens et al. 2020 mouse visual-cortex GABAergic Patch-seq, processed by Gala et al. "
                                    "2021 (github.com/AllenInstitute/coupledAE-patchseq); expression log CPM (no counts)",
                          "x": "log CPM of 1,252 genes (no counts: latent correlation = measured)",
                          "y": "24 electrophysiological features and 44 sparse principal components of responses",
                          "unit": "recording day", "cond": "subclass (first word of the t-type)", "pop": "condition"},
         x_dense=d["T_dat"][ok].astype(np.float32), x_names=flat("gene_id"), y=y[ok], y_names=ynames,
         y_kind=np.array("raw"), cell=np.array([str(v) for v in d["T_spec_id_label"].ravel()])[ok], unit=day[ok],
         cond=sub[ok], pop=np.array(["all"] * int(ok.sum())))


# ------------------------------------------------------------------ connectomes

CROSSING = {"descending", "ascending", "sensory_ascending", "sensory_descending", "ascending_visceral_circulatory"}
NON_NEURONAL = {"glia", "not_a_neuron", "trachea", "unknown"}
BRAIN, VNC = ("central_brain", "optic_lobe"), ("ventral_nerve_cord",)
MIN_SIDE = 50


def partner_group(meta):
    lin = meta["hemilineage"].where(meta["hemilineage"].notna() & (meta["hemilineage"].astype(str) != "nan"))
    cls = meta["cell_class"].where(meta["cell_class"].notna() & (meta["cell_class"].astype(str) != "nan"))
    return meta["super_class"].astype(str) + ":" + lin.fillna(cls).fillna("NA").astype(str)


def side_counts(edges, meta, neurons, idcol, regions):
    """Synapse counts of each neuron with partners of each group on one side (partners whose soma region is on
    that side and which do not cross the neck), inputs and outputs separately."""
    m = meta.set_index(idcol)
    m = m[m["region"].isin(regions) & ~m["super_class"].isin(CROSSING | NON_NEURONAL) & m["super_class"].notna()]
    grp = partner_group(m)
    rows = []
    for direction, me, other in (("in", "post", "pre"), ("out", "pre", "post")):
        e = edges[edges[me].isin(neurons) & edges[other].isin(grp.index)]
        rows.append(pd.DataFrame({"neuron": e[me].values, "group": direction + "|" + grp.loc[e[other].values].values,
                                  "count": e["count"].values}))
    df = pd.concat(rows).groupby(["neuron", "group"])["count"].sum().unstack(fill_value=0)
    return df.reindex(neurons).fillna(0)


def incident_edges(path, ids):
    """Edges with a pre- or postsynaptic neuron in ids, read in record batches (the male CNS edge list does not fit
    in memory)."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.ipc as ipc
    want = pa.array(sorted(ids), type=pa.string())
    out = []
    with pa.memory_map(str(path)) as src:
        reader = ipc.open_file(src)
        for i in range(reader.num_record_batches):
            b = reader.get_batch(i)
            pre = pc.cast(b.column("pre"), pa.string())
            post = pc.cast(b.column("post"), pa.string())
            keep = pc.or_(pc.is_in(pre, value_set=want), pc.is_in(post, value_set=want))
            t = pa.table({"pre": pre, "post": post, "count": b.column("count")}).filter(keep)
            out.append(t)
    return pa.concat_tables(out).to_pandas()


def connectome(name, meta_file, edge_file, idcol):
    fly = RAW / "fly"
    meta = pd.read_feather(fly / meta_file)
    meta[idcol] = meta[idcol].astype(str)
    nm = meta[meta["super_class"].isin(["descending", "ascending"]) & meta["cell_type"].notna()
              & (meta["cell_type"].astype(str) != "nan")]
    neurons = nm[idcol].tolist()
    edges = incident_edges(fly / edge_file, neurons)
    Xb = side_counts(edges, meta, neurons, idcol, BRAIN)
    Yv = side_counts(edges, meta, neurons, idcol, VNC)
    ok = (Xb.sum(axis=1) >= MIN_SIDE) & (Yv.sum(axis=1) >= MIN_SIDE)
    ids = [n for n, k in zip(neurons, ok) if k]
    Xb, Yv = Xb.loc[ids], Yv.loc[ids]
    X = sp.csr_matrix(Xb.values.astype(np.float32))
    c = candidates(X, Xb.columns.values)
    Ydet = (Yv.values > 0).mean(0)
    yc = [j for j in range(Yv.shape[1]) if Ydet[j] >= MIN_DET]
    info = nm.set_index(idcol).loc[ids]
    save(name, {"source": f"{name} connectome, compiled data of the fly connectome tutorial (gs://lee-lab_brain-and-nerve-"
                          f"cord-fly-connectome/compiled_data): {meta_file}, {edge_file}",
                "x": "brain-side synapse counts with partner groups (super class : hemilineage, else cell class), "
                     "inputs and outputs", "y": "nerve-cord-side synapse counts, same groups",
                "unit": "cell type", "cond": "super class (descending, ascending)", "pop": "condition",
                "neurons": f"descending and ascending neurons with a cell type and >= {MIN_SIDE} synapses on each side"},
         **csr_parts(X[:, c]), x_names=Xb.columns.values[c].astype(str), x_library=Xb.values.sum(1),
         y=Yv.values[:, yc].astype(np.float32), y_names=Yv.columns.values[yc].astype(str), y_library=Yv.values.sum(1),
         y_kind=np.array("lognorm"), cell=np.array(ids), unit=info["cell_type"].values.astype(str),
         cond=info["super_class"].values.astype(str), pop=np.array(["all"] * len(ids)))


def fly_unpaired(name, meta_file, edge_file, idcol, side):
    """Descending and ascending neurons of a brain-only (side 'brain') or nerve-cord-only (side 'vnc') connectome:
    counts of the one side it contains, for the secondary cross-animal analysis."""
    fly = RAW / "fly"
    meta = pd.read_feather(fly / meta_file)
    meta[idcol] = meta[idcol].astype(str)
    nm = meta[meta["super_class"].isin(["descending", "ascending"])]
    neurons = nm[idcol].tolist()
    edges = incident_edges(fly / edge_file, neurons)
    D = side_counts(edges, meta, neurons, idcol, BRAIN if side == "brain" else VNC)
    ok = D.sum(axis=1) >= MIN_SIDE
    D = D[ok.values]
    info = nm.set_index(idcol).loc[D.index]
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{name}.npz", counts=D.values.astype(np.float32), names=unicode(D.columns.values),
                        cell=unicode(D.index.values), cond=unicode(info["super_class"].values),
                        cell_type=unicode(info["cell_type"].astype(str).values), library=D.values.sum(axis=1))
    print(name, D.shape, flush=True)


def synthetic():
    """Simulated data for testing the harness (never scored as a result): 12 donors x 2 samples x 3 cell types;
    latent programmes drive Poisson RNA counts and continuous Y."""
    rng = np.random.default_rng(7)
    p, q, k = 300, 40, 8
    Wx = rng.standard_normal((k, p)) * 0.4
    Wy = rng.standard_normal((k, q)) * 0.6
    base = rng.normal(0.0, 1.0, p)
    rows, cells, unit, cond, pop, ys, libs = [], [], [], [], [], [], []
    for d in range(12):
        for smp in range(2):
            for t in range(3):
                n = 180
                z = rng.standard_normal((n, k)) * (1 + 0.3 * t)
                lam = np.exp(base + 0.5 * t + z @ Wx * 0.5 + rng.normal(0, 0.1, p))
                lib_scale = rng.lognormal(0, 0.3, n)[:, None]
                rows.append(rng.poisson(lam * lib_scale))
                ys.append(z @ Wy + rng.standard_normal((n, q)))
                cells += [f"d{d}s{smp}t{t}c{i}" for i in range(n)]
                unit += [f"donor{d}"] * n
                cond += [f"type{t}"] * n
                pop += [f"donor{d}_s{smp}"] * n
    X = sp.csr_matrix(np.vstack(rows).astype(np.float32))
    lib = np.asarray(X.sum(1)).ravel() + 500
    names = np.array([f"g{j}" for j in range(p)])
    c = candidates(X, names)
    save("synthetic", {"source": "simulation for testing the harness"}, **csr_parts(X[:, c]), x_names=names[c],
         x_library=lib, y=np.vstack(ys).astype(np.float32), y_names=np.array([f"y{j}" for j in range(q)]),
         y_kind=np.array("raw"), cell=np.array(cells), unit=np.array(unit), cond=np.array(cond), pop=np.array(pop))


if __name__ == "__main__":
    which = sys.argv[1]
    {"hao": hao, "stephenson": stephenson, "colon": colon, "scala_m1": scala, "gouwens_visp": gouwens,
     "bmmc_cite": lambda: bmmc("bmmc_cite", "bmmc/GSE194122_cite_BMMC_processed.h5ad", "clr"),
     "bmmc_multiome": lambda: bmmc("bmmc_multiome", "multiome/GSE194122_multiome_BMMC_processed.h5ad", "atac"),
     "banc": lambda: connectome("banc", "banc_888_meta.feather", "banc_888_edgelist_simple_v3.feather", "banc_888_id"),
     "malecns": lambda: connectome("malecns", "malecns_09_meta.feather", "malecns_09_simple_edgelist.feather",
                                   "malecns_09_id"),
     "fafb_brain": lambda: fly_unpaired("fafb_brain", "fafb_783_meta.feather", "fafb_783_simple_edgelist.feather",
                                        "fafb_783_id", "brain"),
     "synthetic": synthetic,
     "manc_vnc": lambda: fly_unpaired("manc_vnc", "manc_121_meta.feather", "manc_121_simple_edgelist.feather",
                                      "manc_121_id", "vnc")}[which]()
