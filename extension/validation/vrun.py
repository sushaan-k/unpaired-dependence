#!/usr/bin/env python3
"""Validation test (PLAN.md): the contrast-centred semi-paired estimator with an external reference, in an untouched
CITE-seq data set (GSE314416), at small budgets of paired cells.

    python vrun.py smoke      # the whole pipeline on synthetic values with the real design (results_smoke/)
    python vrun.py pools      # checks before the freeze: unpaired summaries of the references only (results_check/)
    python vrun.py predict    # checks the freeze; predictions of every fold, hashed to results/manifest.json
    python vrun.py evaluate   # checks the hashes; held-out statistics, scores, bootstrap over donors (summary.json)

The estimator, its blocks, map and comparators are those of the benchmark (extension/semipaired, via the deployment
runner drun.py), imported unchanged; the within-population contrasts are posthoc_contrasts.contrasts, the
centring defined after the deployment test was scored and tested here.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
EXT = HERE.parent
sys.path.insert(0, str(EXT / "deployment"))

import drun  # noqa: E402  (context, own_std, pool_std_same, mapped; frozen in the deployment test)
import posthoc_contrasts as pc  # noqa: E402  (contrasts)

cm, rs, es, grun, pm = drun.cm, drun.rs, drun.es, drun.grun, drun.pm
DATA = Path("/home/claude/cbio/rawdata/validation/gse314416.npz")
ATLAS = Path("/home/claude/cbio/rawdata/generality/stephenson.npz")
RES = HERE / "results"
CONDS = ("B", "CD4 T", "CD8 T", "NK", "CD14 Mono")   # the types with enough atlas cells per patient (PLAN.md)
ATLAS_COARSE = {"B_cell": "B", "CD4": "CD4 T", "Treg": "CD4 T", "CD8": "CD8 T", "NK_16hi": "NK", "NK_56hi": "NK",
                "CD14": "CD14 Mono"}
BUDGETS = (25, 50, 100, 200, 400, 800)
DRAWS = 5
SEED = 20261202
PANEL_X, PANEL_Y = 200, 200
MIN_PERCOND = 6
PAIRED_ONLY = ("js", "scose", "fcose", "lowrank")
PO_ARMS = tuple(f"{a}/{m}" for a in PAIRED_ONLY for m in ("pooled", "percond"))
PROPOSED = ("atlas", "other", "own", "own_poolstd", "atlas_dfcentre")
ARMS = PO_ARMS + PROPOSED
TARGETS = (0.5, 0.25, 0.75)
BOOT, BOOT_SEED = 2000, 20261203
CALIB_BUDGETS = (25, 50, 100, 200)


def h(text):
    return int(hashlib.sha256(text.encode()).hexdigest()[:12], 16) / 16 ** 12


sha = drun.sha


# ------------------------------------------------------------------ data

def load_new(synthetic=False):
    z = np.load(DATA, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    d["x_counts"] = sp.csr_matrix((d["x_data"], d["x_indices"], d["x_indptr"]), shape=tuple(d["x_shape"]))
    for k in ("cell", "sample", "donor", "timepoint", "pool", "well", "cond", "x_names", "y_names", "match_new",
              "match_atlas"):
        d[k] = d[k].astype(str)
    d["unit"], d["pop"] = d["donor"], d["sample"]
    keep = np.flatnonzero(np.isin(d["cond"], CONDS))
    for k in list(d):
        if isinstance(d[k], np.ndarray) and d[k].ndim >= 1 and len(d[k]) == len(d["cell"]) and k != "cell":
            d[k] = d[k][keep]
    d["x_counts"], d["cell"] = d["x_counts"][keep], d["cell"][keep]
    if synthetic:
        d = synthesize(d, per_pop=300, seed=7)
    return d


def load_atlas(synthetic=False):
    z = np.load(ATLAS, allow_pickle=True)
    cond = np.array([ATLAS_COARSE.get(str(c), "") for c in z["cond"]])
    keep = np.flatnonzero(cond != "")
    X = sp.csr_matrix((z["x_data"], z["x_indices"], z["x_indptr"]), shape=tuple(z["x_shape"]))
    d = {"x_counts": X[keep], "x_library": z["x_library"][keep], "y": z["y"][keep],
         "x_names": np.array([str(g) for g in z["x_names"]]), "y_names": np.array([str(n) for n in z["y_names"]]),
         "cell": np.array([f"stephenson|{c}" for c in z["cell"][keep]]),
         "unit": np.array([str(u) for u in z["unit"][keep]]), "pop": np.array([str(p) for p in z["pop"][keep]]),
         "cond": cond[keep]}
    if synthetic:
        d = synthesize(d, per_pop=150, seed=8)
    return d


def synthesize(d, per_pop, seed):
    """Smoke-test values: the real design (cells, units, samples, cell types; per_pop cells per sample) with
    random counts from a cell-type-specific low-rank model; no measured value is read."""
    rng = np.random.default_rng(seed)
    sub = np.concatenate([np.flatnonzero(d["pop"] == u)[:per_pop] for u in sorted(set(d["pop"]))])
    for k in list(d):
        if isinstance(d[k], np.ndarray) and d[k].ndim >= 1 and len(d[k]) == len(d["cell"]) and k != "cell":
            d[k] = d[k][sub]
    d["x_counts"] = d["x_counts"][sub]
    d["cell"] = d["cell"][sub]
    n, p, q = len(d["cell"]), d["x_counts"].shape[1], d["y"].shape[1]
    types = sorted(set(d["cond"]))
    base = {t: rng.gamma(0.6, 1.0, p) for t in types}
    load_, wy = rng.standard_normal((p, 5)) * 0.4, rng.standard_normal((5, q)) * 0.5
    fac = rng.standard_normal((n, 5))
    lam = np.stack([base[t] for t in d["cond"]]) * np.exp(np.clip(fac @ load_.T, -3, 3))
    C = rng.poisson(lam).astype(np.float32)
    d["x_counts"] = sp.csr_matrix(C)
    d["x_library"] = C.sum(1).astype(np.float32) + 1
    base_y = {t: rng.gamma(2.0, 5.0, q) for t in types}
    d["y"] = rng.poisson(np.stack([base_y[t] for t in d["cond"]]) * np.exp(np.clip(fac @ wy, -3, 3))).astype(np.float32)
    return d


def roles(d):
    part = np.array([int(h(f"validation-part-v1|{c}") >= 0.5) for c in d["cell"]])
    half = np.array([int(h(f"validation-half-v1|{c}") >= 0.5) for c in d["cell"]])
    res = cm.hm.reservoir(d["cell"])
    return part, half, res


def folds(d):
    """One fold per sample: held out; paired samples, the next two of its pool in hash order; other-pool reference,
    the samples of other pools whose donor does not appear in the held-out sample's pool."""
    out = []
    pools = sorted(set(d["pool"]), key=lambda p: int(p[2:]))
    donor_of = {s: d["donor"][d["sample"] == s][0] for s in sorted(set(d["sample"]))}
    pool_of = {s: d["pool"][d["sample"] == s][0] for s in donor_of}
    for P in pools:
        order = sorted([s for s in donor_of if pool_of[s] == P], key=lambda s: h(f"validation-order-v1|{s}"))
        donors_here = {donor_of[s] for s in order}
        other = sorted(s for s in donor_of if pool_of[s] != P and donor_of[s] not in donors_here)
        n = len(order)
        for i, s in enumerate(order):
            out.append({"heldout": s, "pool": P, "donor": donor_of[s],
                        "paired": [order[(i + 1) % n], order[(i + 2) % n]], "other": other})
    for f, fd in enumerate(out):
        fd["fold"] = f
    return out


def panels(atlas, new, part_a):
    """200 genes by the benchmark's dispersion rule and the matched antibodies detected in >= 1% of cells, both from
    the atlas pool (RNA from part-0 cells, protein from part-1 cells); fixed for every fold."""
    ga = {g: i for i, g in enumerate(atlas["x_names"])}
    gn = {g: i for i, g in enumerate(new["x_names"])}
    common = [g for g in atlas["x_names"] if g in gn]
    ci = np.array([ga[g] for g in common])
    rows = np.flatnonzero(part_a == 0)
    L = grun.lognorm_sparse(atlas["x_counts"][rows][:, ci], atlas["x_library"][rows])
    det = np.bincount(L.indices, minlength=L.shape[1]) / L.shape[0]
    mean = np.asarray(L.mean(0)).ravel()
    var = np.asarray(L.multiply(L).mean(0)).ravel() - mean ** 2
    sel = grun.hvg((mean, var), det, PANEL_X)
    genes = [common[j] for j in sel]
    ya = {n: i for i, n in enumerate(atlas["y_names"])}
    yn = {n: i for i, n in enumerate(new["y_names"])}
    pairs = [(yn[a], ya[b]) for a, b in zip(new["match_new"], new["match_atlas"]) if a in yn and b in ya]
    pairs.sort(key=lambda t: t[1])
    Y = atlas["y"][np.flatnonzero(part_a == 1)][:, [b for _, b in pairs]]
    ok = np.flatnonzero((Y > 0).mean(0) >= 0.01)
    if len(ok) > PANEL_Y:
        ok = np.sort(ok[np.argsort(-np.log1p(Y[:, ok]).var(0), kind="stable")[:PANEL_Y]])
    pairs = [pairs[j] for j in ok]
    return genes, np.array([gn[g] for g in genes]), np.array([ga[g] for g in genes]), \
        np.array([a for a, _ in pairs]), np.array([b for _, b in pairs])


def rna(d, rows, gidx):
    C = d["x_counts"][rows][:, gidx].toarray().astype(np.float64)
    lib = d["x_library"][rows].astype(np.float64)
    return np.log1p(1e4 * C / np.maximum(lib, 1)[:, None]), C, lib


def prot(d, rows, yidx):
    y = np.log1p(d["y"][rows][:, yidx].astype(np.float64))
    return y - y.mean(1, keepdims=True)


def popkey(d, rows):
    return np.array([f"{p}|{c}" for p, c in zip(d["pop"][rows], d["cond"][rows])])


# ------------------------------------------------------------------ references (unpaired summaries)

def atlas_context(atlas, part_a, gidx_a, yidx_a):
    rows = np.arange(len(atlas["cell"]))
    x, C, lib = rna(atlas, rows, gidx_a)
    y = prot(atlas, rows, yidx_a)
    cells = grun.outside_reservoir("validation-atlas", atlas["cell"])
    return drun.context(x, y, C, lib, popkey(atlas, rows), part_a, cells, atlas["unit"]) + (len(rows),)


def other_context(new, fd, part, gidx, yidx):
    rows = np.flatnonzero(np.isin(new["sample"], fd["other"]))
    x, C, lib = rna(new, rows, gidx)
    y = prot(new, rows, yidx)
    cells = grun.outside_reservoir("validation-other", new["cell"][rows])
    return drun.context(x, y, C, lib, popkey(new, rows), part[rows], cells, new["unit"][rows]) + (len(rows),)


def own_context(new, fd, part, res, gidx, yidx):
    rows = np.flatnonzero(np.isin(new["sample"], fd["paired"]) & ~res)
    x, C, lib = rna(new, rows, gidx)
    y = prot(new, rows, yidx)
    return drun.context(x, y, C, lib, popkey(new, rows), part[rows], new["cell"][rows], new["unit"][rows]) \
        + (len(rows),)


# ------------------------------------------------------------------ audit of the paired draws

def accounting(pop):
    cnt = {}
    for k in pop:
        cnt[k] = cnt.get(k, 0) + 1
    return {"populations": len(cnt), "singletons": sum(1 for v in cnt.values() if v == 1),
            "rows_contrasts": sum(v - 1 for v in cnt.values() if v >= 2),
            "rows_dfcentre": sum(v for v in cnt.values() if v >= 2)}


def block_moments(X, Y, U, V, rblocks, pblocks):
    """Per block: the mean product vector and its estimated noise trace (sp_estimators.noise_terms)."""
    XU, YV = X @ U, Y @ V
    out = []
    for rows in rblocks:
        for cols in pblocks:
            Z = (XU[:, rows][:, :, None] * YV[:, cols][:, None, :]).reshape(len(X), -1)
            m, tr, _ = es.noise_terms(Z, iters=1)
            out.append((m, tr))
    return out


# ------------------------------------------------------------------ predictions

def predict(synthetic=False, out_dir=RES, fold_ids=None, budgets=BUDGETS, draws=DRAWS, log=print):
    new, atlas = load_new(synthetic), load_atlas(synthetic)
    part, half, res = roles(new)
    part_a = np.array([int(h(f"validation-part-v1|{c}") >= 0.5) for c in atlas["cell"]])
    genes, gidx, gidx_a, yidx, yidx_a = panels(atlas, new, part_a)
    out_dir.mkdir(exist_ok=True)
    A_pool, A, n_atlas = atlas_context(atlas, part_a, gidx_a, yidx_a)
    V_a, _ = es.eigenbasis(A.Ry)
    pblocks = es.eigen_blocks(V_a.shape[1], es.PROTEIN_EDGES)
    log(f"atlas reference: {n_atlas} cells, {len(A_pool.keys)} populations, reliability {A.reliability}")
    store, info, calib, acct = {}, {}, {}, {}
    other_cache = {}
    t0 = time.time()
    for fd in folds(new):
        f = fd["fold"]
        if fold_ids is not None and f not in fold_ids:
            continue
        if fd["pool"] not in other_cache:
            other_cache[fd["pool"]] = other_context(new, fd, part, gidx, yidx)
        O_pool, O, n_other = other_cache[fd["pool"]]
        W_pool, W, n_own = own_context(new, fd, part, res, gidx, yidx)
        rres = np.flatnonzero(np.isin(new["sample"], fd["paired"]) & res)
        xr, _, _ = rna(new, rres, gidx)
        yr = prot(new, rres, yidx)
        popr, condr = popkey(new, rres), new["cond"][rres]
        fb = [B for B in budgets if B <= len(rres)]
        acc = {}
        for B in fb:
            cal = {}
            for dr in range(draws):
                rng = np.random.default_rng([SEED, f, B, dr])
                pick = rng.choice(len(rres), size=B, replace=False)
                x, y, pp, cc = xr[pick], yr[pick], popr[pick], condr[pick]
                acct.setdefault(str(B), []).append(accounting(pp))
                preds = {}
                hc = pc.contrasts(x, y, pp)
                if hc is not None:
                    Xh, Yh, ih = hc
                    ch = cc[ih]
                    for a in PAIRED_ONLY:
                        crng = np.random.default_rng([SEED + 1, f, B, dr])
                        Pp = rs.denoise(a, Xh, Yh, A.Rx, A.Ry, A.n_x, A.n_y, A.Uz, A.blocks, crng)
                        for c in CONDS:
                            preds[(f"{a}/pooled", c)] = Pp
                            m = ch == c
                            preds[(f"{a}/percond", c)] = (rs.denoise(a, Xh[m], Yh[m], A.Rx, A.Ry, A.n_x, A.n_y,
                                                                     A.Uz, A.blocks, crng)
                                                          if m.sum() >= MIN_PERCOND else np.zeros_like(Pp))
                    for arm, ctx in (("atlas", A), ("other", O), ("own", W)):
                        Pe = es.two_sided_js(Xh, Yh, ctx.Uz, ctx.Ry, ctx.blocks)[0]
                        for c in CONDS:
                            preds[(arm, c)] = drun.mapped(ctx, Pe, c)
                    if B in CALIB_BUDGETS:
                        cal.setdefault("contrasts", []).append(block_moments(Xh, Yh, A.Uz, V_a, A.blocks, pblocks))
                o = drun.own_std(x, y, pp)
                if o is not None:
                    Pe = es.two_sided_js(o[0], o[1], A.Uz, A.Ry, A.blocks)[0]
                    for c in CONDS:
                        preds[("atlas_dfcentre", c)] = drun.mapped(A, Pe, c)
                    if B in CALIB_BUDGETS:
                        cal.setdefault("dfcentre", []).append(block_moments(o[0], o[1], A.Uz, V_a, A.blocks, pblocks))
                Xs, Ys, _ = drun.pool_std_same(W_pool, x, y, pp)
                Pe = es.two_sided_js(Xs, Ys, W.Uz, W.Ry, W.blocks)[0]
                for c in CONDS:
                    preds[("own_poolstd", c)] = drun.mapped(W, Pe, c)
                for (arm, c), Q in preds.items():
                    k = f"{f}/{arm}/{B}/{c}"
                    s, q = acc.get(k, (0.0, 0.0))
                    acc[k] = (s + Q / draws, q + float(np.sum(Q * Q)) / draws)
            for meth, per_draw in cal.items():
                nb = len(per_draw[0])
                n = len(per_draw)
                t_mean = [float(np.mean([pdw[b][1] for pdw in per_draw])) for b in range(nb)]
                v_hat = []
                for b in range(nb):
                    M = np.stack([pdw[b][0] for pdw in per_draw])
                    v_hat.append(float((np.sum(M * M) - np.sum(M.sum(0) ** 2) / n) / (n - 1)) if n > 1 else float("nan"))
                calib[f"{f}/{meth}/{B}"] = {"draws": n, "reservoir": int(len(rres)), "t_mean": t_mean,
                                            "var_across_draws": v_hat}
            log(f"fold {f} ({fd['heldout']}) B={B} done [{time.time() - t0:.0f}s]")
        for k, (s, q) in acc.items():
            store[f"P/{k}"] = np.asarray(s, np.float32)
            store[f"msq/{k}"] = np.array(q)
        info[str(f)] = {"heldout": fd["heldout"], "pool": fd["pool"], "donor": fd["donor"], "paired": fd["paired"],
                        "other": fd["other"], "reservoir": int(len(rres)), "budgets": fb,
                        "pool_cells": {"atlas": int(n_atlas), "other": int(n_other), "own": int(n_own)},
                        "pool_keys": {"atlas": len(A_pool.keys), "other": len(O_pool.keys), "own": len(W_pool.keys)},
                        "map_reliability": {"atlas": A.reliability, "other": O.reliability, "own": W.reliability}}
    meta = {"genes": genes, "proteins_new": [str(new["y_names"][j]) for j in yidx],
            "proteins_atlas": [str(atlas["y_names"][j]) for j in yidx_a]}
    path = out_dir / "predictions.npz"
    np.savez_compressed(path, **store)
    (out_dir / "predictions_info.json").write_text(json.dumps({"folds": info, "panels": meta}, indent=1))
    (out_dir / "audit.json").write_text(json.dumps({"calibration": calib, "accounting": acct}, indent=1))
    man = {"predictions": sha(path), "info": sha(out_dir / "predictions_info.json"),
           "audit": sha(out_dir / "audit.json"), "written": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
           "synthetic": synthetic}
    (out_dir / "manifest.json").write_text(json.dumps(man, indent=1))
    log(f"predictions hashed: {man['predictions'][:16]}")


# ------------------------------------------------------------------ evaluation

def heldout(new, fd, half, gidx, yidx):
    r = np.flatnonzero(new["sample"] == fd["heldout"])
    x, _, _ = rna(new, r, gidx)
    y = prot(new, r, yidx)
    cond, hv = new["cond"][r], half[r]
    out = {}
    for c in CONDS:
        rec = {}
        for s, sel in (("T", np.ones(len(r), bool)), ("TA", hv == 0), ("TB", hv == 1)):
            m = (cond == c) & sel
            if m.sum() < 2:
                rec[s] = np.zeros((len(gidx), len(yidx)))
                continue
            xc, yc = x[m] - x[m].mean(0), y[m] - y[m].mean(0)
            den = np.sqrt(np.maximum((xc * xc).sum(0), 1e-12)[:, None] * np.maximum((yc * yc).sum(0), 1e-12)[None, :])
            rec[s] = (xc.T @ yc) / den
        out[c] = rec
    return out


def fold_terms(store, info, stats):
    terms = {}
    for f, fi in info.items():
        num, den = {}, 0.0
        for c in CONDS:
            T = stats[f][c]
            den += float(np.sum(T["TA"] * T["TB"]))
            for arm in ARMS:
                for B in fi["budgets"]:
                    k = f"{f}/{arm}/{B}/{c}"
                    if f"P/{k}" in store:
                        num[(arm, B)] = num.get((arm, B), 0.0) + 2 * float(np.sum(store[f"P/{k}"] * T["T"])) \
                            - float(store[f"msq/{k}"])
        terms[f] = (num, den)
    return terms


def curves(terms, weights, budgets):
    den = sum(w * terms[f][1] for f, w in weights.items())
    out = {a: np.array([sum(w * terms[f][0].get((a, B), np.nan) for f, w in weights.items()) / den
                        for B in budgets]) for a in ARMS}
    out["paired_only"] = np.nanmax(np.stack([out[a] for a in PO_ARMS]), axis=0)
    return out


def comparisons(cv, budgets, frac):
    level = max(float(np.nanmax(cv[a])) for a in ARMS)
    target = frac * level
    n = {a: grun.needed(cv[a], target, budgets) for a in ("paired_only",) + PROPOSED}
    return level, target, n


RATIOS = {"V1_atlas_vs_paired_only": ("paired_only", "atlas"),
          "V3_other_vs_paired_only": ("paired_only", "other"),
          "V4_atlas_over_own": ("atlas", "own"),
          "own_vs_paired_only": ("paired_only", "own"),
          "own_poolstd_vs_paired_only": ("paired_only", "own_poolstd"),
          "atlas_dfcentre_vs_paired_only": ("paired_only", "atlas_dfcentre"),
          "atlas_over_other": ("atlas", "other")}


def audit_summary(audit):
    """Degrees of freedom of the paired draws, and the calibration of the noise term: over folds and blocks, the sum
    of the mean estimated noise traces against the sum of the across-draw variances of the block means, each
    divided by the finite-reservoir factor 1 - B/N."""
    acct = {B: {k: float(np.mean([r[k] for r in rows])) for k in rows[0]} for B, rows in audit["accounting"].items()}
    calib = {}
    for key, r in audit["calibration"].items():
        f, meth, B = key.split("/")
        fpc = 1 - int(B) / r["reservoir"]
        c = calib.setdefault(meth, {}).setdefault(B, {"t": 0.0, "v": 0.0, "t_blocks": None, "v_blocks": None})
        t, v = np.array(r["t_mean"]), np.array(r["var_across_draws"]) / fpc
        c["t"] += float(t.sum())
        c["v"] += float(v.sum())
        c["t_blocks"] = t if c["t_blocks"] is None else c["t_blocks"] + t
        c["v_blocks"] = v if c["v_blocks"] is None else c["v_blocks"] + v
    out = {}
    for meth, byB in calib.items():
        out[meth] = {B: {"ratio_estimated_to_actual": c["t"] / c["v"],
                         "ratio_by_block": [float(a / b) for a, b in zip(c["t_blocks"], c["v_blocks"])]}
                     for B, c in sorted(byB.items(), key=lambda kv: int(kv[0]))}
    return {"accounting": acct, "calibration": out}


def evaluate(synthetic=False, out_dir=RES, log=print):
    man = json.loads((out_dir / "manifest.json").read_text())
    for key, f in (("predictions", "predictions.npz"), ("info", "predictions_info.json"), ("audit", "audit.json")):
        assert sha(out_dir / f) == man[key], f"{f} changed since it was hashed"
    store = dict(np.load(out_dir / "predictions.npz"))
    pinfo = json.loads((out_dir / "predictions_info.json").read_text())
    info = pinfo["folds"]
    new = load_new(synthetic)
    part, half, res = roles(new)
    yn = {n: i for i, n in enumerate(new["y_names"])}
    gn = {g: i for i, g in enumerate(new["x_names"])}
    gidx = np.array([gn[g] for g in pinfo["panels"]["genes"]])
    yidx = np.array([yn[p] for p in pinfo["panels"]["proteins_new"]])
    fds = {str(fd["fold"]): fd for fd in folds(new)}
    stats = {f: heldout(new, fds[f], half, gidx, yidx) for f in info}
    terms = fold_terms(store, info, stats)
    budgets = sorted(set.intersection(*[set(fi["budgets"]) for fi in info.values()]))
    cv = curves(terms, {f: 1.0 for f in info}, budgets)
    donors = sorted({fi["donor"] for fi in info.values()})
    folds_of = {d: [f for f, fi in info.items() if fi["donor"] == d] for d in donors}
    rng = np.random.default_rng(BOOT_SEED)
    boots = []
    for _ in range(BOOT):
        w = {}
        for dnr in rng.choice(donors, size=len(donors), replace=True):
            for f in folds_of[dnr]:
                w[f] = w.get(f, 0.0) + 1.0
        boots.append(curves(terms, w, budgets))
    out = {"budgets": budgets, "folds": {f: {k: fi[k] for k in ("heldout", "pool", "donor", "reservoir", "pool_cells")}
                                         for f, fi in info.items()},
           "rf": {a: [float(v) for v in cv[a]] for a in cv}, "targets": {}}
    out["rf_ci"] = {a: [[float(np.percentile([b[a][i] for b in boots], 2.5)),
                         float(np.percentile([b[a][i] for b in boots], 97.5))] for i in range(len(budgets))]
                    for a in ("paired_only",) + PROPOSED}
    for frac in TARGETS:
        level, target, n = comparisons(cv, budgets, frac)
        rec = {"level": level, "target": target,
               "cells": {a: {"cells": v[0], "relation": v[1]} for a, v in n.items()}}
        bn = [comparisons(bc, budgets, frac)[2] for bc in boots]
        for name, (a, b) in RATIOS.items():
            bs = [nb[a][0] / nb[b][0] for nb in bn]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            rec[name] = {"ratio": n[a][0] / n[b][0], "ci": [float(lo), float(hi)], "numerator_relation": n[a][1],
                         "denominator_relation": n[b][1]}
        out["targets"][str(frac)] = rec
    prim = out["targets"]["0.5"]
    i50, i100 = budgets.index(50), budgets.index(100)
    out["hypotheses"] = {
        "V1": bool(prim["V1_atlas_vs_paired_only"]["ci"][0] > 1),
        "V2": bool(out["rf_ci"]["atlas"][i50][0] > 0 and out["rf_ci"]["atlas"][i100][0] > 0),
        "V3": bool(prim["V3_other_vs_paired_only"]["ci"][0] > 1),
        "V4": bool(prim["V4_atlas_over_own"]["ci"][1] < 1.25)}
    out["pools"] = {}
    for P in sorted({fi["pool"] for fi in info.values()}, key=lambda p: int(p[2:])):
        cs = curves(terms, {f: 1.0 for f, fi in info.items() if fi["pool"] == P}, budgets)
        level, target, n = comparisons(cs, budgets, 0.5)
        out["pools"][P] = {"rf": {a: [float(v) for v in cs[a]] for a in ("paired_only",) + PROPOSED},
                           "level": level, "atlas_vs_paired_only": n["paired_only"][0] / n["atlas"][0],
                           "relations": {a: n[a][1] for a in n}}
    out["audit"] = audit_summary(json.loads((out_dir / "audit.json").read_text()))
    out["evaluated"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    (out_dir / "summary.json").write_text(json.dumps(out, indent=1))
    log(json.dumps(out["hypotheses"]))
    for k in ("V1_atlas_vs_paired_only", "V3_other_vs_paired_only", "V4_atlas_over_own"):
        log(f"{k}: {prim[k]['ratio']:.2f} ({prim[k]['ci'][0]:.2f}-{prim[k]['ci'][1]:.2f}) "
            f"[{prim[k]['numerator_relation']}/{prim[k]['denominator_relation']}]")
    for a in ("paired_only",) + PROPOSED:
        log(f"{a:16s} " + " ".join(f"{100 * v:7.1f}" for v in cv[a]))
    for meth, byB in out["audit"]["calibration"].items():
        log(f"calibration {meth}: " + ", ".join(f"B={B} {v['ratio_estimated_to_actual']:.2f}" for B, v in byB.items()))


# ------------------------------------------------------------------ checks before the freeze

def pools_check(log=print):
    """Unpaired summaries of the references only (sizes, reliabilities, map norms, latent spectrum); no paired-cell
    estimate and no statistic of held-out cells."""
    new, atlas = load_new(), load_atlas()
    part, half, res = roles(new)
    part_a = np.array([int(h(f"validation-part-v1|{c}") >= 0.5) for c in atlas["cell"]])
    genes, gidx, gidx_a, yidx, yidx_a = panels(atlas, new, part_a)
    out = {"genes": len(genes), "proteins": int(len(yidx))}
    A_pool, A, n_atlas = atlas_context(atlas, part_a, gidx_a, yidx_a)
    out["atlas"] = {"cells": int(n_atlas), "populations": len(A_pool.keys), "reliability": A.reliability,
                    "latent_eigenvalues_top10": [float(v) for v in A.lz[:10]],
                    "map_norms": {c: float(np.linalg.norm(A.maps[c][0] - np.eye(len(A.lz)))) for c in A.maps}}
    fds = folds(new)
    for fd in (fds[0], fds[len(fds) // 2]):
        O_pool, O, n_other = other_context(new, fd, part, gidx, yidx)
        out[f"other_{fd['pool']}"] = {"cells": int(n_other), "populations": len(O_pool.keys),
                                      "reliability": O.reliability}
    out["folds"] = [{k: fd[k] for k in ("fold", "heldout", "pool", "paired")} | {"other": len(fd["other"])}
                    for fd in fds]
    out["reservoir_cells"] = {fd["heldout"]: int(np.sum(np.isin(new["sample"], fd["paired"]) & res)) for fd in fds}
    d = HERE / "results_check"
    d.mkdir(exist_ok=True)
    (d / "pools_check.json").write_text(json.dumps(out, indent=1, default=float))
    log(json.dumps({k: v for k, v in out.items() if k != "folds"}, indent=1, default=float)[:3000])


# ------------------------------------------------------------------ freeze check

def check_freeze():
    rec = json.loads((RES / "freeze.json").read_text())
    for rel, digest in rec["files"].items():
        assert sha(EXT / rel) == digest, f"{rel} changed since the freeze"
    for name, (path, digest) in rec["data"].items():
        assert sha(path) == digest, f"{name} changed since the freeze"


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    logd = {"smoke": HERE / "results_smoke", "pools": HERE / "results_check"}.get(cmd, RES)
    logd.mkdir(exist_ok=True)
    lp = logd / f"{cmd}.log"

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(lp, "a") as fh:
            fh.write(line + "\n")

    with threadpool_limits(limits=2):
        if cmd == "smoke":
            out = HERE / "results_smoke"
            predict(synthetic=True, out_dir=out, fold_ids={0, 17}, budgets=(25, 50, 100), draws=2, log=log)
            evaluate(synthetic=True, out_dir=out, log=log)
        elif cmd == "pools":
            pools_check(log=log)
        elif cmd == "predict":
            check_freeze()
            predict(log=log)
        elif cmd == "evaluate":
            check_freeze()
            evaluate(log=log)


if __name__ == "__main__":
    main()
