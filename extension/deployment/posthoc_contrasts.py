#!/usr/bin/env python3
"""Post hoc (after the deployment test was scored): own centring by within-population contrasts.

In the deployment test the paired cells were centred on the means of their own populations (cell type x donor)
with centred values multiplied by sqrt(n/(n-1)) (drun.own_std). Centring makes the cells of a population
dependent: for a population of n drawn cells, the per-cell cross-products after centring have the correct mean but
their spread understates the variance of their sum by the factor n/(n-1) (for n = 2 the two products are equal).
Every shrinkage estimator that takes its noise term from the spread of per-cell products (James-Stein, SCOSE,
FCOSE) then shrinks too little, and cross-validation over cells (low rank) sees each pair on both sides of a split.

Here each population's n drawn cells are replaced by their n - 1 orthonormal Helmert contrasts,
h_j = (z_1 + ... + z_j - j z_{j+1}) / sqrt(j (j + 1)), j = 1..n-1, which have mean zero, the within-population
covariance, and are independent for Gaussian cells; their cross-products sum to the within-population
cross-products, so the estimand is unchanged. Contrasts are scaled by their pooled standard deviations. Everything
else (folds, reservoir, pools, panels, budgets, draws, seeds, estimators, map, scoring, bootstrap) is the frozen
deployment design, imported unchanged from drun.py. The frozen own-centred arm is recomputed for fold 0 and must
match the hashed predictions.

    python posthoc_contrasts.py predict    # results_posthoc/contrasts_predictions.npz (per fold, resumable)
    python posthoc_contrasts.py evaluate   # results_posthoc/contrasts_summary.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import drun  # noqa: E402

grun, cm, rs, es = drun.grun, drun.cm, drun.rs, drun.es
OUT = HERE / "results_posthoc"
PRED_DIR = OUT / "contrasts_folds"
C_PROPOSED = ("c_proposed_same", "c_proposed_other", "c_proposed_assay")
C_PO_ARMS = tuple(f"c_{a}" for a in drun.PO_ARMS)
C_ARMS = C_PO_ARMS + C_PROPOSED


def helmert(Z):
    """The n - 1 orthonormal Helmert contrasts of the rows of Z (n x d)."""
    n = len(Z)
    j = np.arange(1, n, dtype=float)[:, None]
    return (np.cumsum(Z, 0)[:-1] - j * Z[1:]) / np.sqrt(j * (j + 1))


def contrasts(x, y, pop):
    """Within-population Helmert contrasts of the drawn cells, scaled by their pooled standard deviations; returns
    X, Y and, per contrast, the index of a drawn cell of its population (for its condition)."""
    xs, ys, idx = [], [], []
    for k in sorted(set(pop)):
        i = np.flatnonzero(pop == k)
        if len(i) < 2:
            continue
        xs.append(helmert(x[i]))
        ys.append(helmert(y[i]))
        idx.append(i[1:])
    if not xs:
        return None
    X, Y = np.vstack(xs), np.vstack(ys)
    sx, sy = np.sqrt((X * X).mean(0)), np.sqrt((Y * Y).mean(0))
    sx[sx == 0], sy[sy == 0] = 1.0, 1.0
    return X / sx, Y / sy, np.concatenate(idx)


def self_check():
    rng = np.random.default_rng(0)
    for n in (2, 3, 7):
        Z = rng.standard_normal((n, 4))
        H = helmert(Z)
        Zc = Z - Z.mean(0)
        assert H.shape == (n - 1, 4)
        assert np.allclose(H.T @ H, Zc.T @ Zc)


def predict(log=print):
    self_check()
    cite, mult = drun.load("bmmc_cite"), drun.load("bmmc_multiome")
    part, half, res = drun.roles(cite)
    frozen = np.load(drun.RES / "predictions.npz")
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for fd in drun.folds(cite, mult):
        f = fd["fold"]
        path = PRED_DIR / f"fold{f}.npz"
        if path.exists():
            continue
        rr = np.flatnonzero(np.isin(cite["pop"], fd["other"]) & (part == 0))
        rp = np.flatnonzero(np.isin(cite["pop"], fd["other"]) & (part == 1))
        genes, yp = drun.panels(cite, mult, rr, rp)
        P = drun.pools(cite, mult, fd, part, res, genes, yp)
        rres = np.flatnonzero(np.isin(cite["pop"], fd["paired"]) & res)
        xr, _, _ = drun.rna(cite, rres, genes)
        yr = drun.prot(cite, rres, yp)
        popr = np.array([f"{p}|{c}" for p, c in zip(cite["pop"][rres], cite["cond"][rres])])
        condr = cite["cond"][rres]
        ctx_any = P["other"][1]
        fb = [B for B in drun.BUDGETS if B <= len(rres)]
        acc, check = {}, {}
        for B in fb:
            for dr in range(drun.DRAWS):
                rng = np.random.default_rng([drun.SEED, f, B, dr])
                pick = rng.choice(len(rres), size=B, replace=False)
                x, y, pp, cc = xr[pick], yr[pick], popr[pick], condr[pick]
                preds = {}
                if f == 0:  # the frozen own-centred arm, to check that the design is reproduced exactly
                    o = drun.own_std(x, y, pp)
                    if o is not None:
                        Pe = es.two_sided_js(o[0], o[1], P["other"][1].Uz, P["other"][1].Ry, P["other"][1].blocks)[0]
                        for c in drun.CONDS:
                            k = f"{c}"
                            check[(B, k)] = check.get((B, k), 0.0) + drun.mapped(P["other"][1], Pe, c) / drun.DRAWS
                h = contrasts(x, y, pp)
                if h is not None:
                    Xh, Yh, ih = h
                    ch = cc[ih]
                    for a in drun.PAIRED_ONLY:
                        crng = np.random.default_rng([drun.SEED + 1, f, B, dr])
                        Pp = rs.denoise(a, Xh, Yh, ctx_any.Rx, ctx_any.Ry, ctx_any.n_x, ctx_any.n_y, ctx_any.Uz,
                                        ctx_any.blocks, crng)
                        for c in drun.CONDS:
                            preds[(f"c_{a}/pooled", c)] = Pp
                            m = ch == c
                            preds[(f"c_{a}/percond", c)] = (rs.denoise(a, Xh[m], Yh[m], ctx_any.Rx, ctx_any.Ry,
                                                                       ctx_any.n_x, ctx_any.n_y, ctx_any.Uz,
                                                                       ctx_any.blocks, crng)
                                                            if m.sum() >= drun.MIN_PERCOND else np.zeros_like(Pp))
                    for arm, key in (("c_proposed_same", "same"), ("c_proposed_other", "other"),
                                     ("c_proposed_assay", "assay")):
                        ctx = P[key][1]
                        Pe = es.two_sided_js(Xh, Yh, ctx.Uz, ctx.Ry, ctx.blocks)[0]
                        for c in drun.CONDS:
                            preds[(arm, c)] = drun.mapped(ctx, Pe, c)
                for (arm, c), Q in preds.items():
                    k = f"{f}/{arm}/{B}/{c}"
                    s, q = acc.get(k, (0.0, 0.0))
                    acc[k] = (s + Q / drun.DRAWS, q + float(np.sum(Q * Q)) / drun.DRAWS)
            log(f"fold {f} B={B} [{time.time() - t0:.0f}s]")
        if f == 0:
            diffs = [float(np.max(np.abs(np.asarray(v, np.float32) - frozen[f"P/0/proposed_other/{B}/{c}"])))
                     for (B, c), v in check.items()]
            log(f"fold 0 reproduction of the frozen own-centred arm: max abs difference {max(diffs):.2e}")
            assert max(diffs) < 1e-5, "the frozen design is not reproduced"
        store = {}
        for k, (s, q) in acc.items():
            store[f"P/{k}"] = np.asarray(s, np.float32)
            store[f"msq/{k}"] = np.array(q)
        np.savez_compressed(path, **store)


def evaluate(log=print, out_path=None):
    info = json.loads((drun.RES / "predictions_info.json").read_text())
    frozen = dict(np.load(drun.RES / "predictions.npz"))
    store = {}
    for f in info:
        store.update(dict(np.load(PRED_DIR / f"fold{f}.npz")))
    store.update(frozen)
    cite, mult = drun.load("bmmc_cite"), drun.load("bmmc_multiome")
    part, half, res = drun.roles(cite)
    fds = {str(fd["fold"]): fd for fd in drun.folds(cite, mult)}
    stats = {}
    for f, fi in info.items():
        yp = np.array([list(cite["y_names"]).index(p) for p in fi["proteins"]])
        stats[f] = drun.heldout(cite, fds[f], half, fi["genes"], yp)
    arms = drun.ARMS + C_ARMS
    terms = {}
    for f, fi in info.items():
        num, den = {}, 0.0
        for c in drun.CONDS:
            T = stats[f][c]
            den += float(np.sum(T["TA"] * T["TB"]))
            for arm in arms:
                for B in fi["budgets"]:
                    k = f"{f}/{arm}/{B}/{c}"
                    if f"P/{k}" not in store:
                        continue
                    num[(arm, B)] = num.get((arm, B), 0.0) + 2 * float(np.sum(store[f"P/{k}"] * T["T"])) \
                        - float(store[f"msq/{k}"])
        terms[f] = (num, den)
    budgets = sorted(set.intersection(*[set(fi["budgets"]) for fi in info.values()]))

    def curves(weights):
        den = sum(w * terms[f][1] for f, w in weights.items())
        out = {a: np.array([sum(w * terms[f][0].get((a, B), np.nan) for f, w in weights.items()) / den
                            for B in budgets]) for a in arms}
        out["paired_only"] = np.nanmax(np.stack([out[a] for a in drun.PO_ARMS]), axis=0)
        out["c_paired_only"] = np.nanmax(np.stack([out[a] for a in C_PO_ARMS]), axis=0)
        return out

    cv = curves({f: 1.0 for f in info})
    level = max(float(np.nanmax(cv[a])) for a in arms)
    frozen_level = max(float(np.nanmax(cv[a])) for a in drun.ARMS)
    sites = sorted({fi["site"] for fi in info.values()})
    by_site = {s: [f for f, fi in info.items() if fi["site"] == s] for s in sites}
    rng = np.random.default_rng(drun.BOOT_SEED)
    boots = []
    for _ in range(drun.BOOT):
        w = {}
        for s in sites:
            for f in rng.choice(by_site[s], size=len(by_site[s]), replace=True):
                w[f] = w.get(f, 0.0) + 1.0
        boots.append(curves(w))
    ratios = {"other_vs_paired_only": ("c_paired_only", "c_proposed_other"),
              "same_vs_paired_only": ("c_paired_only", "c_proposed_same"),
              "assay_vs_paired_only": ("c_paired_only", "c_proposed_assay"),
              "other_over_same": ("c_proposed_other", "c_proposed_same"),
              "contrasts_vs_frozen_other": ("proposed_other", "c_proposed_other")}
    out = {"budgets": budgets, "rf": {a: [float(v) for v in cv[a]] for a in cv},
           "level_all_arms": level, "level_frozen_arms": frozen_level, "targets": {}}
    for frac in drun.TARGETS:
        # the target is a fraction of the frozen test's best recovery, so that frozen and contrast arms share it
        target = frac * frozen_level
        rec = {"target": target, "cells": {}}
        for a in ("paired_only", "c_paired_only", "proposed_other", "proposed_same", "c_proposed_other",
                  "c_proposed_same", "c_proposed_assay", "proposed_same_poolstd"):
            v = grun.needed(cv[a], target, budgets)
            rec["cells"][a] = {"cells": v[0], "relation": v[1]}
        for name, (a, b) in ratios.items():
            na, nb = grun.needed(cv[a], target, budgets), grun.needed(cv[b], target, budgets)
            bs = []
            for bc in boots:
                bs.append(grun.needed(bc[a], target, budgets)[0] / grun.needed(bc[b], target, budgets)[0])
            lo, hi = np.percentile(bs, [2.5, 97.5])
            rec[name] = {"ratio": na[0] / nb[0], "ci": [float(lo), float(hi)], "numerator_relation": na[1],
                         "denominator_relation": nb[1]}
        out["targets"][str(frac)] = rec
    out["evaluated"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    (out_path or OUT / "contrasts_summary.json").write_text(json.dumps(out, indent=1))
    for a in ("paired_only", "c_paired_only", "proposed_other", "c_proposed_other", "c_proposed_same",
              "c_proposed_assay", "proposed_same_poolstd"):
        log(f"{a:24s} " + " ".join(f"{100 * v:7.1f}" for v in cv[a]))
    p = out["targets"]["0.5"]
    for name in ratios:
        log(f"{name}: {p[name]['ratio']:.2f} ({p[name]['ci'][0]:.2f}-{p[name]['ci'][1]:.2f}) "
            f"[{p[name]['numerator_relation']}/{p[name]['denominator_relation']}]")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "predict"
    OUT.mkdir(exist_ok=True)
    lp = OUT / f"contrasts_{cmd}.log"

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(lp, "a") as fh:
            fh.write(line + "\n")

    with threadpool_limits(limits=2):
        drun.check_freeze()
        if cmd == "predict":
            predict(log)
        elif cmd == "evaluate":
            evaluate(log)


if __name__ == "__main__":
    main()
