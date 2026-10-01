#!/usr/bin/env python3
"""Deployment test (PLAN.md): unpaired cells from other sites and from another assay.

    python drun.py smoke      # the whole pipeline on synthetic values with the real design (results_smoke/)
    python drun.py predict    # checks the freeze, predictions of every fold, hashed to results/manifest.json
    python drun.py evaluate   # checks the hashes, then held-out statistics, scores and bootstrap (results/summary.json)

The estimator, blocks, condition map and comparators are those of extension/semipaired, imported unchanged; the
panel rule, interpolation and censoring are those of the generality benchmark (extension/generality/grun.py).
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
sys.path.insert(0, str(EXT / "semipaired"))
sys.path.insert(0, str(EXT / "generality"))

import common as cm  # noqa: E402
import run_study as rs  # noqa: E402
import sp_estimators as es  # noqa: E402
import grun  # noqa: E402  (panel rule, interpolation; frozen, imported unchanged)

pm = cm.pm
DATA = Path("/home/claude/cbio/rawdata/generality")
RES = HERE / "results"
CONDS = ("B", "CD4 T", "CD8 T", "DC", "Mono", "NK", "erythroid", "progenitor")
BUDGETS = (25, 50, 100, 200, 400, 800)
DRAWS = 5
SEED = 20261101
PANEL_X, PANEL_Y = 200, 200
MIN_TRAIN_HALF = 10
MIN_PERCOND = 6
PAIRED_ONLY = ("js", "scose", "fcose", "lowrank")
PO_ARMS = tuple(f"{a}/{m}" for a in PAIRED_ONLY for m in ("pooled", "percond"))
PROPOSED = ("proposed_same", "proposed_other", "proposed_assay", "proposed_other_poolstd", "proposed_same_poolstd")
ARMS = PO_ARMS + PROPOSED
TARGETS = (0.5, 0.25, 0.75)
BOOT, BOOT_SEED = 2000, 20261102


def h(text):
    return int(hashlib.sha256(text.encode()).hexdigest()[:12], 16) / 16 ** 12


def sha(path):
    d = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            d.update(b)
    return d.hexdigest()


# ------------------------------------------------------------------ data

def load(name, synthetic=False):
    z = np.load(DATA / f"{name}.npz", allow_pickle=False)
    d = {k: z[k] for k in z.files}
    d["x_counts"] = sp.csr_matrix((d["x_data"], d["x_indices"], d["x_indptr"]), shape=tuple(d["x_shape"]))
    for k in ("cell", "unit", "cond", "pop"):
        d[k] = d[k].astype(str)
    d["x_names"] = d["x_names"].astype(str)
    d["cell"] = np.array([f"{name}|{c}" for c in d["cell"]])
    d["site"] = np.array([p[:2] for p in d["pop"]])
    if synthetic:
        d = synthesize(name, d)
    keep = np.isin(d["cond"], CONDS)
    for k in ("cell", "unit", "cond", "pop", "site", "x_library"):
        d[k] = d[k][keep]
    d["x_counts"] = d["x_counts"][np.flatnonzero(keep)]
    if name == "bmmc_cite":
        d["y"] = d["y"][keep]
        d["y_names"] = d["y_names"].astype(str)
    return d


def synthesize(name, d):
    """Smoke-test values: the real design (cells, batches, donors, cell types; 400 cells per batch) with random
    counts from a cell-type-specific low-rank model; no measured value is read."""
    rng = np.random.default_rng(7)
    sub = np.concatenate([np.flatnonzero(d["pop"] == b)[:400] for b in sorted(set(d["pop"]))])
    for k in ("cell", "unit", "cond", "pop", "site", "x_library"):
        d[k] = d[k][sub]
    d["x_counts"] = d["x_counts"][sub]
    if name == "bmmc_cite":
        d["y"] = d["y"][sub]
    else:
        d.pop("y", None)
    n, p = len(d["cell"]), d["x_counts"].shape[1]
    types = sorted(set(d["cond"]))
    base = {t: rng.gamma(0.6, 1.0, p) for t in types}
    load_ = rng.standard_normal((p, 5)) * 0.4
    fac = rng.standard_normal((n, 5))
    lam = np.stack([base[t] for t in d["cond"]]) * np.exp(np.clip(fac @ load_.T, -3, 3))
    C = rng.poisson(lam).astype(np.float32)
    d["x_counts"] = sp.csr_matrix(C)
    d["x_library"] = C.sum(1).astype(np.float32) + 1
    if "y" in d:
        q = d["y"].shape[1]
        wy = rng.standard_normal((5, q)) * 0.5
        base_y = {t: rng.gamma(2.0, 5.0, q) for t in types}
        mu = np.stack([base_y[t] for t in d["cond"]]) * np.exp(np.clip(fac @ wy, -3, 3))
        d["y"] = rng.poisson(mu).astype(np.float32)
    return d


def folds(cite, mult):
    """One fold per CITE-seq batch (sorted): held-out batch, paired batches, other-site pools."""
    batches = sorted(set(cite["pop"]))
    donor_of = {b: cite["unit"][cite["pop"] == b][0] for b in batches}
    mdonor_of = {b: mult["unit"][mult["pop"] == b][0] for b in sorted(set(mult["pop"]))}
    out = []
    for f, b in enumerate(batches):
        S = b[:2]
        site_batches = [c for c in batches if c[:2] == S]
        site_donors = {donor_of[c] for c in site_batches}
        out.append({"fold": f, "heldout": b, "site": S, "paired": [c for c in site_batches if c != b],
                    "other": [c for c in batches if c[:2] != S and donor_of[c] not in site_donors],
                    "assay": [c for c in sorted(mdonor_of) if c[:2] != S and mdonor_of[c] not in site_donors]})
    return out


def roles(cite):
    part = np.array([int(h(f"deployment-part-v1|{c}") >= 0.5) for c in cite["cell"]])
    half = np.array([int(h(f"deployment-half-v1|{c}") >= 0.5) for c in cite["cell"]])
    res = cm.hm.reservoir(cite["cell"])
    return part, half, res


def panels(cite, mult, rows_rna, rows_prot):
    """200 genes by the benchmark's binned-dispersion rule among genes of both assays, from other-site RNA pool
    cells; proteins detected in >= 1% of other-site protein pool cells (at most 200)."""
    common = sorted(set(cite["x_names"]) & set(mult["x_names"]))
    ci = np.array([list(cite["x_names"]).index(g) for g in common])
    L = grun.lognorm_sparse(cite["x_counts"][rows_rna][:, ci], cite["x_library"][rows_rna])
    det = np.bincount(L.indices, minlength=L.shape[1]) / L.shape[0]
    mean = np.asarray(L.mean(0)).ravel()
    var = np.asarray(L.multiply(L).mean(0)).ravel() - mean ** 2
    sel = grun.hvg((mean, var), det, PANEL_X)
    genes = [common[j] for j in sel]
    Y = cite["y"][rows_prot]
    ok = np.flatnonzero((Y > 0).mean(0) >= 0.01)
    if len(ok) > PANEL_Y:
        ok = np.sort(ok[np.argsort(-np.log1p(Y[:, ok]).var(0), kind="stable")[:PANEL_Y]])
    return genes, ok


def rna(d, rows, genes):
    idx = np.array([list(d["x_names"]).index(g) for g in genes])
    C = d["x_counts"][rows][:, idx].toarray().astype(np.float64)
    lib = d["x_library"][rows].astype(np.float64)
    return np.log1p(1e4 * C / np.maximum(lib, 1)[:, None]), C, lib


def prot(cite, rows, yp):
    y = np.log1p(cite["y"][rows][:, yp].astype(np.float64))
    return y - y.mean(1, keepdims=True)


def outside(prefix, ids):
    return grun.outside_reservoir(prefix, ids)


# ------------------------------------------------------------------ unpaired summaries

def context(x, y, C, lib, pop, part, cells, guide):
    keys = pm.eligible_training(pop, part, MIN_TRAIN_HALF)
    pool = cm.hm.Pool(x, y, pop, part, guide, keys)
    un = cm.Unpaired(pool, x, y, C, lib, pop, part, pool.keys)
    ctx = rs.build_context({"cell": cells, "part": part}, x, y, pop, pool.keys, pool, un)
    return pool, ctx


def pools(cite, mult, fd, part, res, genes, yp):
    out = {}
    # same site: non-reservoir cells of the paired batches
    m = np.isin(cite["pop"], fd["paired"]) & ~res
    r = np.flatnonzero(m)
    x, C, lib = rna(cite, r, genes)
    y = prot(cite, r, yp)
    pop = np.array([f"{p}|{c}" for p, c in zip(cite["pop"][r], cite["cond"][r])])
    out["same"] = context(x, y, C, lib, pop, part[r], cite["cell"][r], cite["unit"][r]) + (len(r),)
    # other sites: every cell of the other-site batches
    r = np.flatnonzero(np.isin(cite["pop"], fd["other"]))
    x, C, lib = rna(cite, r, genes)
    y = prot(cite, r, yp)
    pop = np.array([f"{p}|{c}" for p, c in zip(cite["pop"][r], cite["cond"][r])])
    out["other"] = context(x, y, C, lib, pop, part[r], outside("deploy-other", cite["cell"][r]),
                           cite["unit"][r]) + (len(r),)
    # other assay: multiome RNA of the other-site batches, protein of the other-site CITE-seq pool (part 1)
    rm = np.flatnonzero(np.isin(mult["pop"], fd["assay"]))
    rp = r[part[r] == 1]
    xm, Cm, libm = rna(mult, rm, genes)
    xa = np.vstack([xm, np.zeros((len(rp), len(genes)))])
    Ca = np.vstack([Cm, np.zeros((len(rp), len(genes)))])
    la = np.concatenate([libm, np.ones(len(rp))])
    ya = np.vstack([np.zeros((len(rm), len(yp))), prot(cite, rp, yp)])
    popa = np.array([f"{p}|{c}" for p, c in zip(np.concatenate([mult["pop"][rm], cite["pop"][rp]]),
                                                 np.concatenate([mult["cond"][rm], cite["cond"][rp]]))])
    parta = np.concatenate([np.zeros(len(rm), int), np.ones(len(rp), int)])
    cells = np.concatenate([outside("deploy-assay", mult["cell"][rm]), outside("deploy-assayp", cite["cell"][rp])])
    guide = np.concatenate([mult["unit"][rm], cite["unit"][rp]])
    out["assay"] = context(xa, ya, Ca, la, popa, parta, cells, guide) + (len(rm) + len(rp),)
    return out


# ------------------------------------------------------------------ standardization of paired cells

def own_std(x, y, pop):
    """Centre on the drawn cells' own population means (df-corrected), scale by their own pooled within-population
    SDs; populations with one drawn cell are dropped."""
    keep, xs, ys = [], [], []
    for k in sorted(set(pop)):
        i = np.flatnonzero(pop == k)
        if len(i) < 2:
            continue
        f = np.sqrt(len(i) / (len(i) - 1))
        xs.append((x[i] - x[i].mean(0)) * f)
        ys.append((y[i] - y[i].mean(0)) * f)
        keep.append(i)
    if not keep:
        return None
    keep = np.concatenate(keep)
    X, Y = np.vstack(xs), np.vstack(ys)
    sx, sy = np.sqrt((X * X).mean(0)), np.sqrt((Y * Y).mean(0))
    sx[sx == 0], sy[sy == 0] = 1.0, 1.0
    return X / sx, Y / sy, keep


def pool_std_same(pool, x, y, pop):
    ok = np.flatnonzero(np.isin(pop, pool.keys))
    X, Y = pool.centre_cells(x[ok], y[ok], pop[ok])
    return X, Y, ok


def pool_std_other(pool, x, y, cond):
    ok = np.flatnonzero(np.isin(cond, pool.conds))
    X = x[ok] / pool.sd_x - np.stack([pool.cx[c] for c in cond[ok]])
    Y = y[ok] / pool.sd_y - np.stack([pool.cy[c] for c in cond[ok]])
    return X, Y, ok


def mapped(ctx, P, c):
    return ctx.mapped(P, c) if c in getattr(ctx, "maps", {}) else P


# ------------------------------------------------------------------ predictions

def predict(synthetic=False, out_dir=RES, fold_ids=None, budgets=BUDGETS, draws=DRAWS, log=print):
    cite, mult = load("bmmc_cite", synthetic), load("bmmc_multiome", synthetic)
    part, half, res = roles(cite)
    out_dir.mkdir(exist_ok=True)
    store, info = {}, {}
    t_start = time.time()
    for fd in folds(cite, mult):
        f = fd["fold"]
        if fold_ids is not None and f not in fold_ids:
            continue
        rr = np.flatnonzero(np.isin(cite["pop"], fd["other"]) & (part == 0))
        rp = np.flatnonzero(np.isin(cite["pop"], fd["other"]) & (part == 1))
        genes, yp = panels(cite, mult, rr, rp)
        P = pools(cite, mult, fd, part, res, genes, yp)
        rres = np.flatnonzero(np.isin(cite["pop"], fd["paired"]) & res)
        xr, _, _ = rna(cite, rres, genes)
        yr = prot(cite, rres, yp)
        popr = np.array([f"{p}|{c}" for p, c in zip(cite["pop"][rres], cite["cond"][rres])])
        condr = cite["cond"][rres]
        ctx_any = P["other"][1]
        fb = [B for B in budgets if B <= len(rres)]
        acc = {}
        for B in fb:
            for dr in range(draws):
                rng = np.random.default_rng([SEED, f, B, dr])
                pick = rng.choice(len(rres), size=B, replace=False)
                x, y, pp, cc = xr[pick], yr[pick], popr[pick], condr[pick]
                o = own_std(x, y, pp)
                preds = {}
                if o is not None:
                    Xo, Yo, keep = o
                    co = cc[keep]
                    for a in PAIRED_ONLY:
                        crng = np.random.default_rng([SEED + 1, f, B, dr])
                        Pp = rs.denoise(a, Xo, Yo, ctx_any.Rx, ctx_any.Ry, ctx_any.n_x, ctx_any.n_y, ctx_any.Uz,
                                        ctx_any.blocks, crng)
                        for c in CONDS:
                            preds[(f"{a}/pooled", c)] = Pp
                            m = co == c
                            preds[(f"{a}/percond", c)] = (rs.denoise(a, Xo[m], Yo[m], ctx_any.Rx, ctx_any.Ry,
                                                                     ctx_any.n_x, ctx_any.n_y, ctx_any.Uz,
                                                                     ctx_any.blocks, crng)
                                                          if m.sum() >= MIN_PERCOND else np.zeros_like(Pp))
                    for arm, key in (("proposed_same", "same"), ("proposed_other", "other"),
                                     ("proposed_assay", "assay")):
                        ctx = P[key][1]
                        Pe = es.two_sided_js(Xo, Yo, ctx.Uz, ctx.Ry, ctx.blocks)[0]
                        for c in CONDS:
                            preds[(arm, c)] = mapped(ctx, Pe, c)
                pool_s, ctx_s = P["same"][0], P["same"][1]
                Xs, Ys, _ = pool_std_same(pool_s, x, y, pp)
                Pe = es.two_sided_js(Xs, Ys, ctx_s.Uz, ctx_s.Ry, ctx_s.blocks)[0]
                for c in CONDS:
                    preds[("proposed_same_poolstd", c)] = mapped(ctx_s, Pe, c)
                pool_o, ctx_o = P["other"][0], P["other"][1]
                Xt, Yt, _ = pool_std_other(pool_o, x, y, cc)
                Pe = es.two_sided_js(Xt, Yt, ctx_o.Uz, ctx_o.Ry, ctx_o.blocks)[0]
                for c in CONDS:
                    preds[("proposed_other_poolstd", c)] = mapped(ctx_o, Pe, c)
                for (arm, c), Q in preds.items():
                    k = f"{f}/{arm}/{B}/{c}"
                    s, q = acc.get(k, (0.0, 0.0))
                    acc[k] = (s + Q / draws, q + float(np.sum(Q * Q)) / draws)
            log(f"fold {f} ({fd['heldout']}) B={B} done [{time.time() - t_start:.0f}s]")
        for k, (s, q) in acc.items():
            store[f"P/{k}"] = np.asarray(s, np.float32)
            store[f"msq/{k}"] = np.array(q)
        info[str(f)] = {"heldout": fd["heldout"], "site": fd["site"], "paired_batches": fd["paired"],
                        "other_batches": fd["other"], "assay_batches": fd["assay"], "genes": genes,
                        "proteins": [str(cite["y_names"][j]) for j in yp], "reservoir": int(len(rres)),
                        "pool_cells": {k: int(v[2]) for k, v in P.items()},
                        "pool_keys": {k: len(v[0].keys) for k, v in P.items()},
                        "map_reliability": {k: v[1].reliability for k, v in P.items()},
                        "budgets": fb}
    path = out_dir / "predictions.npz"
    np.savez_compressed(path, **store)
    (out_dir / "predictions_info.json").write_text(json.dumps(info, indent=1))
    man = {"predictions": sha(path), "info": sha(out_dir / "predictions_info.json"),
           "written": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "synthetic": synthetic}
    (out_dir / "manifest.json").write_text(json.dumps(man, indent=1))
    log(f"predictions hashed: {man['predictions'][:16]}")


# ------------------------------------------------------------------ held-out statistics and scores

def heldout(cite, fd, half, genes, yp):
    r = np.flatnonzero(cite["pop"] == fd["heldout"])
    x, _, _ = rna(cite, r, genes)
    y = prot(cite, r, yp)
    cond, hv = cite["cond"][r], half[r]
    out = {}
    for c in CONDS:
        rec = {}
        for s, sel in (("T", np.ones(len(r), bool)), ("TA", hv == 0), ("TB", hv == 1)):
            m = (cond == c) & sel
            if m.sum() < 2:
                rec[s] = np.zeros((len(genes), len(yp)))
                continue
            xc, yc = x[m] - x[m].mean(0), y[m] - y[m].mean(0)
            den = np.sqrt(np.maximum((xc * xc).sum(0), 1e-12)[:, None] * np.maximum((yc * yc).sum(0), 1e-12)[None, :])
            rec[s] = (xc.T @ yc) / den
        out[c] = rec
    return out


def fold_terms(store, info, stats):
    """Per fold: numerators (arm x budget) and the denominator of the recovered fraction."""
    terms = {}
    for f, fi in info.items():
        num = {}
        den = 0.0
        for c in CONDS:
            T = stats[f][c]
            den += float(np.sum(T["TA"] * T["TB"]))
            for arm in ARMS:
                for B in fi["budgets"]:
                    k = f"{f}/{arm}/{B}/{c}"
                    if f"P/{k}" not in store:
                        continue
                    v = 2 * float(np.sum(store[f"P/{k}"] * T["T"])) - float(store[f"msq/{k}"])
                    num[(arm, B)] = num.get((arm, B), 0.0) + v
        terms[f] = (num, den)
    return terms


def curves(terms, weights, budgets):
    """Recovered-fraction curves (arm -> array over budgets) for fold weights (dict fold -> weight)."""
    den = sum(w * terms[f][1] for f, w in weights.items())
    out = {}
    for arm in ARMS:
        out[arm] = np.array([sum(w * terms[f][0].get((arm, B), np.nan) for f, w in weights.items()) / den
                             for B in budgets])
    out["paired_only"] = np.nanmax(np.stack([out[a] for a in PO_ARMS]), axis=0)
    return out


def comparisons(cv, budgets, frac):
    level = max(float(np.nanmax(cv[a])) for a in ARMS)
    target = frac * level
    n = {a: grun.needed(cv[a], target, budgets) for a in ("paired_only",) + PROPOSED}
    return level, target, n


def evaluate(synthetic=False, out_dir=RES, log=print):
    man = json.loads((out_dir / "manifest.json").read_text())
    assert sha(out_dir / "predictions.npz") == man["predictions"], "predictions changed since they were hashed"
    assert sha(out_dir / "predictions_info.json") == man["info"], "prediction info changed since it was hashed"
    store = dict(np.load(out_dir / "predictions.npz"))
    info = json.loads((out_dir / "predictions_info.json").read_text())
    cite, mult = load("bmmc_cite", synthetic), load("bmmc_multiome", synthetic)
    part, half, res = roles(cite)
    fds = {str(fd["fold"]): fd for fd in folds(cite, mult)}
    stats = {}
    for f, fi in info.items():
        yp = np.array([list(cite["y_names"]).index(p) for p in fi["proteins"]])
        stats[f] = heldout(cite, fds[f], half, fi["genes"], yp)
    terms = fold_terms(store, info, stats)
    budgets = sorted(set.intersection(*[set(fi["budgets"]) for fi in info.values()]))
    all_w = {f: 1.0 for f in info}
    cv = curves(terms, all_w, budgets)
    out = {"budgets": budgets, "folds": {f: {k: fi[k] for k in ("heldout", "site", "reservoir", "pool_cells")}
                                         for f, fi in info.items()},
           "rf": {a: [float(v) for v in cv[a]] for a in cv}, "targets": {}}
    sites = sorted({fi["site"] for fi in info.values()})
    by_site = {s: [f for f, fi in info.items() if fi["site"] == s] for s in sites}
    rng = np.random.default_rng(BOOT_SEED)
    boots = []
    for _ in range(BOOT):
        w = {}
        for s in sites:
            fs = by_site[s]
            for f in rng.choice(fs, size=len(fs), replace=True):
                w[f] = w.get(f, 0.0) + 1.0
        boots.append(curves(terms, w, budgets))
    for frac in TARGETS:
        level, target, n = comparisons(cv, budgets, frac)
        rec = {"level": level, "target": target,
               "cells": {a: {"cells": v[0], "relation": v[1]} for a, v in n.items()}}
        ratios = {"D1_other_vs_paired_only": ("paired_only", "proposed_other"),
                  "D2_assay_vs_paired_only": ("paired_only", "proposed_assay"),
                  "D3_other_over_same": ("proposed_other", "proposed_same"),
                  "same_vs_paired_only": ("paired_only", "proposed_same"),
                  "other_poolstd_vs_paired_only": ("paired_only", "proposed_other_poolstd"),
                  "same_poolstd_vs_paired_only": ("paired_only", "proposed_same_poolstd")}
        for name, (a, b) in ratios.items():
            est = n[a][0] / n[b][0]
            bs = []
            for bc in boots:
                _, _, nb = comparisons(bc, budgets, frac)
                bs.append(nb[a][0] / nb[b][0])
            lo, hi = np.percentile(bs, [2.5, 97.5])
            rec[name] = {"ratio": est, "ci": [float(lo), float(hi)], "numerator_relation": n[a][1],
                         "denominator_relation": n[b][1]}
        out["targets"][str(frac)] = rec
    prim = out["targets"]["0.5"]
    out["hypotheses"] = {"D1": bool(prim["D1_other_vs_paired_only"]["ci"][0] > 1),
                         "D2": bool(prim["D2_assay_vs_paired_only"]["ci"][0] > 1),
                         "D3": bool(prim["D3_other_over_same"]["ci"][1] < 1.25)}
    # per site, descriptive
    out["sites"] = {}
    for s in sites:
        cs = curves(terms, {f: 1.0 for f in by_site[s]}, budgets)
        level, target, n = comparisons(cs, budgets, 0.5)
        out["sites"][s] = {"rf": {a: [float(v) for v in cs[a]] for a in ("paired_only",) + PROPOSED},
                           "level": level,
                           "other_vs_paired_only": n["paired_only"][0] / n["proposed_other"][0],
                           "assay_vs_paired_only": n["paired_only"][0] / n["proposed_assay"][0],
                           "relations": {a: n[a][1] for a in n}}
    out["evaluated"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    (out_dir / "summary.json").write_text(json.dumps(out, indent=1))
    log(json.dumps(out["hypotheses"]))
    for k in ("D1_other_vs_paired_only", "D2_assay_vs_paired_only", "D3_other_over_same"):
        log(f"{k}: {prim[k]['ratio']:.2f} ({prim[k]['ci'][0]:.2f}-{prim[k]['ci'][1]:.2f})")


# ------------------------------------------------------------------ freeze check

def check_freeze():
    rec = json.loads((RES / "freeze.json").read_text())
    for rel, digest in rec["files"].items():
        assert sha(EXT / rel) == digest, f"{rel} changed since the freeze"
    for name, digest in rec["data"].items():
        assert sha(DATA / f"{name}.npz") == digest, f"{name}.npz changed since the freeze"


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    logf = (RES if cmd != "smoke" else HERE / "results_smoke")
    logf.mkdir(exist_ok=True)
    lp = logf / f"{cmd}.log"

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(lp, "a") as fh:
            fh.write(line + "\n")

    with threadpool_limits(limits=2):
        if cmd == "smoke":
            out = HERE / "results_smoke"
            predict(synthetic=True, out_dir=out, fold_ids={0, 4}, budgets=(25, 100), draws=1, log=log)
            evaluate(synthetic=True, out_dir=out, log=log)
        elif cmd == "predict":
            check_freeze()
            predict(log=log)
        elif cmd == "evaluate":
            check_freeze()
            evaluate(log=log)


if __name__ == "__main__":
    main()
