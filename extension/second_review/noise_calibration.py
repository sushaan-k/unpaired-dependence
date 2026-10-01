#!/usr/bin/env python3
"""A. Calibration of the noise estimate after contrasts (PLAN.md, section A).

    python noise_calibration.py smoke                          # synthetic validation values: 2 folds, budgets 25
                                                               # and 100, 50 resamples; checks the pipeline
    python noise_calibration.py run validation|bone_marrow     # checks results/freeze.json; results/noise/
    python noise_calibration.py score                          # results/noise_calibration.json

Each resample draws B cells with replacement from a fold's reservoir of paired cells and runs the complete pipeline:
centring by Helmert contrasts within populations, scaling by the contrasts' pooled standard deviations, rotation into
the reference's eigenbases and the eight blocks. The block mean m_b does not depend on the order of cells or on the
contrast basis; the noise estimates do. Resamples are processed in 20 batches of 25; the variance of m_b across
resamples is the mean of the within-batch variances, so that batches can be resampled for intervals.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
EXT = HERE.parent
sys.path.insert(0, str(EXT / "validation"))
import vrun  # noqa: E402  (frozen: validation/results/freeze.json)

drun, pc, es = vrun.drun, vrun.pc, vrun.es
OUT = HERE / "results" / "noise"
BUDGETS = (25, 50, 100, 200, 400, 800)
RESAMPLES, BATCH = 500, 25
ORDER_RESAMPLES, ORDERINGS = 100, 20
BOOT = 2000
SEED, BOOT_SEED, ORDER_SEED = 20261301, 20261302, 20261303
STUDIES = {"validation": 1, "bone_marrow": 2}
ESTIMATES = ("helmert", "helmert_shuffled", "orthonormal", "jackknife", "own_centring")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------ studies

def validation(synthetic=False, fold_ids=None):
    new, atlas = vrun.load_new(synthetic), vrun.load_atlas(synthetic)
    part, half, res = vrun.roles(new)
    part_a = np.array([int(vrun.h(f"validation-part-v1|{c}") >= 0.5) for c in atlas["cell"]])
    genes, gidx, gidx_a, yidx, yidx_a = vrun.panels(atlas, new, part_a)
    _, ctx, _ = vrun.atlas_context(atlas, part_a, gidx_a, yidx_a)
    del atlas
    folds = []
    for fd in vrun.folds(new):
        if fold_ids is not None and fd["fold"] not in fold_ids:
            continue
        rres = np.flatnonzero(np.isin(new["sample"], fd["paired"]) & res)
        st = vrun.heldout(new, fd, half, gidx, yidx)
        folds.append(SimpleNamespace(
            fold=fd["fold"], unit=str(fd["donor"]), x=vrun.rna(new, rres, gidx)[0], y=vrun.prot(new, rres, yidx),
            pop=vrun.popkey(new, rres), T={c: st[c]["T"] for c in vrun.CONDS},
            den=sum(float(np.sum(st[c]["TA"] * st[c]["TB"])) for c in vrun.CONDS), ctx=ctx))
    return folds, vrun.CONDS


def bone_marrow(fold_ids=None):
    cite, mult = drun.load("bmmc_cite"), drun.load("bmmc_multiome")
    part, half, res = drun.roles(cite)
    folds = []
    for fd in drun.folds(cite, mult):
        if fold_ids is not None and fd["fold"] not in fold_ids:
            continue
        rr = np.flatnonzero(np.isin(cite["pop"], fd["other"]) & (part == 0))
        rp = np.flatnonzero(np.isin(cite["pop"], fd["other"]) & (part == 1))
        genes, yp = drun.panels(cite, mult, rr, rp)
        r = np.flatnonzero(np.isin(cite["pop"], fd["other"]))          # the other sites (drun.pools, "other")
        x, C, lib = drun.rna(cite, r, genes)
        y = drun.prot(cite, r, yp)
        pop = np.array([f"{p}|{c}" for p, c in zip(cite["pop"][r], cite["cond"][r])])
        _, ctx = drun.context(x, y, C, lib, pop, part[r], drun.outside("deploy-other", cite["cell"][r]),
                              cite["unit"][r])
        rres = np.flatnonzero(np.isin(cite["pop"], fd["paired"]) & res)
        st = drun.heldout(cite, fd, half, genes, yp)
        folds.append(SimpleNamespace(
            fold=fd["fold"], unit=str(fd["heldout"]), x=drun.rna(cite, rres, genes)[0], y=drun.prot(cite, rres, yp),
            pop=np.array([f"{p}|{c}" for p, c in zip(cite["pop"][rres], cite["cond"][rres])]),
            T={c: st[c]["T"] for c in drun.CONDS},
            den=sum(float(np.sum(st[c]["TA"] * st[c]["TB"])) for c in drun.CONDS), ctx=ctx))
        log(f"bone marrow fold {fd['fold']} ({fd['heldout']}): reservoir {len(rres)}")
    return folds, drun.CONDS


# ------------------------------------------------------------------ one resample

def orthonormal_contrasts(n, rng):
    """A random orthonormal basis of the vectors of length n summing to zero (rows), Haar distributed."""
    Q, R = np.linalg.qr(np.column_stack([np.ones(n), rng.standard_normal((n, n - 1))]))
    Q = Q * np.sign(np.diag(R))[None, :]
    return Q[:, 1:].T


def row_trace(hx, hy, rblocks, pblocks, M, N):
    """Unbiased-under-independence trace estimate of Cov(m_b) from the rows (hx, hy) of every block."""
    nx, ny = hx * hx, hy * hy
    out = np.empty((len(rblocks), len(pblocks)))
    for i, rows in enumerate(rblocks):
        sx = nx[:, rows].sum(1)
        for j, cols in enumerate(pblocks):
            n2 = float(np.sum(M[np.ix_(rows, cols)] ** 2))
            out[i, j] = (float(sx @ ny[:, cols].sum(1)) - N * n2) / (N * (N - 1))
    return out.ravel()


def resample(f, B, rng, U, V, rblocks, pblocks, orderings, rng_order):
    x, y, pop = f.x, f.y, f.pop
    idx = rng.integers(0, len(x), B)
    x, y, pop = x[idx], y[idx], pop[idx]
    groups = [np.flatnonzero(pop == k) for k in sorted(set(pop))]
    groups = [g for g in groups if len(g) >= 2]
    if not groups:
        return None
    xc = np.vstack([x[g] - x[g].mean(0) for g in groups])
    yc = np.vstack([y[g] - y[g].mean(0) for g in groups])
    sizes = np.array([len(g) for g in groups])
    n, N = int(sizes.sum()), int(sizes.sum() - len(groups))
    sx, sy = np.sqrt((xc * xc).sum(0) / N), np.sqrt((yc * yc).sum(0) / N)
    sx[sx == 0], sy[sy == 0] = 1.0, 1.0
    xr, yr = (xc / sx) @ U, (yc / sy) @ V
    starts = np.concatenate([[0], np.cumsum(sizes)])
    local = [np.arange(a, b) for a, b in zip(starts[:-1], starts[1:])]
    M = xr.T @ yr / N
    blocks = [(r_, c_) for r_ in rblocks for c_ in pblocks]
    m = [M[np.ix_(r_, c_)].ravel() for r_, c_ in blocks]
    out = {"m": m, "n2": np.array([float(v @ v) for v in m])}

    def helmert_rows(order=None):
        ids = local if order is None else order
        return (np.vstack([pc.helmert(xr[i]) for i in ids]), np.vstack([pc.helmert(yr[i]) for i in ids]))

    hx, hy = helmert_rows()
    out["helmert"] = row_trace(hx, hy, rblocks, pblocks, M, N)
    shuffled = [rng_order.permutation(i) for i in local]
    out["helmert_shuffled"] = row_trace(*helmert_rows(shuffled), rblocks, pblocks, M, N)
    Q = [orthonormal_contrasts(len(i), rng_order) for i in local]
    out["orthonormal"] = row_trace(np.vstack([q @ xr[i] for q, i in zip(Q, local)]),
                                   np.vstack([q @ yr[i] for q, i in zip(Q, local)]), rblocks, pblocks, M, N)
    # delete-one-cell jackknife with the scaling held fixed: removing cell i of population p takes
    # n_p/(n_p - 1) x_i y_i' from the centred cross-product sum and one degree of freedom
    c = np.concatenate([np.full(s, s / (s - 1)) for s in sizes])
    jk = np.empty(len(blocks))
    for b, (r_, c_) in enumerate(blocks):
        a_bar = (xr[:, r_] * c[:, None]).T @ yr[:, c_] / n
        s2 = float(np.sum(c * c * (xr[:, r_] ** 2).sum(1) * (yr[:, c_] ** 2).sum(1)))
        jk[b] = (n - 1) / n / (N - 1) ** 2 * (s2 - n * float(np.sum(a_bar * a_bar)))
    out["jackknife"] = jk
    # own-population centring with the sqrt(n/(n-1)) correction (drun.own_std): its own mean and trace
    ox, oy = xc * np.sqrt(c)[:, None], yc * np.sqrt(c)[:, None]
    tx, ty = np.sqrt((ox * ox).mean(0)), np.sqrt((oy * oy).mean(0))
    tx[tx == 0], ty[ty == 0] = 1.0, 1.0
    oxr, oyr = (ox / tx) @ U, (oy / ty) @ V
    Mo = oxr.T @ oyr / n
    out["m_own"] = [Mo[np.ix_(r_, c_)].ravel() for r_, c_ in blocks]
    out["own_centring"] = row_trace(oxr, oyr, rblocks, pblocks, Mo, n)
    out["M"], out["shape"] = M, (len(rblocks), len(pblocks))
    if orderings:
        tt = np.array([row_trace(*helmert_rows([rng_order.permutation(i) for i in local]), rblocks, pblocks, M, N)
                       for _ in range(orderings)])
        fac = np.maximum(1 - tt / np.maximum(out["n2"], 1e-300)[None, :], 0)
        out["order_cv"] = tt.std(0, ddof=1) / np.maximum(np.abs(tt.mean(0)), 1e-300)
        out["order_factor_range"] = fac.max(0) - fac.min(0)
    return out


def shrunk(M, t, n2, rblocks, pblocks, U, V):
    A = np.zeros_like(M)
    k = 0
    for rows in rblocks:
        for cols in pblocks:
            A[np.ix_(rows, cols)] = (max(1.0 - t[k] / n2[k], 0.0) if n2[k] > 0 else 0.0) * M[np.ix_(rows, cols)]
            k += 1
    return U @ A @ V.T


def gain(ctx, P, T, conds):
    """Recovered-fraction numerator of a pooled estimate passed through the reference's condition maps."""
    total = 0.0
    for c in conds:
        Q = drun.mapped(ctx, P, c)
        total += 2 * float(np.sum(Q * T[c])) - float(np.sum(Q * Q))
    return total


def run_fold(study, f, conds, budgets, resamples, log):
    ctx = f.ctx
    U, (V, _) = ctx.Uz, es.eigenbasis(ctx.Ry)
    rblocks, pblocks = ctx.blocks, es.eigen_blocks(V.shape[1], es.PROTEIN_EDGES)
    nb = len(rblocks) * len(pblocks)
    batches = resamples // BATCH
    rec = {}
    for B in budgets:
        acc = {e: np.zeros((batches, nb)) for e in ESTIMATES}
        var, var_own = np.zeros((batches, nb)), np.zeros((batches, nb))
        recov = {"helmert": np.zeros(batches), "jackknife": np.zeros(batches)}
        order_cv, order_range = [], []
        for j in range(batches):
            ms, mo = [], []
            for r in range(j * BATCH, (j + 1) * BATCH):
                rng = np.random.default_rng([SEED, STUDIES[study], f.fold, B, r])
                rng_o = np.random.default_rng([ORDER_SEED, STUDIES[study], f.fold, B, r])
                o = resample(f, B, rng, U, V, rblocks, pblocks, ORDERINGS if r < ORDER_RESAMPLES else 0, rng_o)
                if o is None:
                    continue
                ms.append(o["m"])
                mo.append(o["m_own"])
                for e in ESTIMATES:
                    acc[e][j] += o[e] / BATCH
                for e in recov:
                    P = shrunk(o["M"], o[e], o["n2"], rblocks, pblocks, U, V)
                    recov[e][j] += gain(ctx, P, f.T, conds) / BATCH
                if "order_cv" in o:
                    order_cv.append(o["order_cv"])
                    order_range.append(o["order_factor_range"])
            for b in range(nb):
                for target, vecs in ((var, ms), (var_own, mo)):
                    Mb = np.array([v[b] for v in vecs])
                    target[j, b] = float(np.sum((Mb - Mb.mean(0)) ** 2)) / (len(Mb) - 1)
        rec[B] = {"var": var, "var_own": var_own, **{f"t_{e}": acc[e] for e in ESTIMATES},
                  "recovery_helmert": recov["helmert"], "recovery_jackknife": recov["jackknife"],
                  "order_cv": np.array(order_cv), "order_factor_range": np.array(order_range)}
        log(f"{study} fold {f.fold} B={B}: global ratio "
            f"{acc['helmert'].sum() / var.sum():.4f} (jackknife {acc['jackknife'].sum() / var.sum():.4f})")
    return rec


def save(study, f, rec, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = {"unit": np.array(f.unit), "den": np.array(f.den), "budgets": np.array(sorted(rec))}
    for B, d in rec.items():
        for k, v in d.items():
            arrays[f"{B}/{k}"] = v
    np.savez_compressed(out_dir / f"{study}_fold{f.fold:02d}.npz", **arrays)


# ------------------------------------------------------------------ scoring

def score(out_dir=OUT):
    out = {}
    for study in STUDIES:
        files = sorted(out_dir.glob(f"{study}_fold*.npz"))
        if not files:
            continue
        Z = [dict(np.load(p)) for p in files]
        units = [str(z["unit"]) for z in Z]
        uniq = sorted(set(units))
        of_unit = {u: [i for i, v in enumerate(units) if v == u] for u in uniq}
        budgets = [int(b) for b in Z[0]["budgets"]]
        rng = np.random.default_rng([BOOT_SEED, STUDIES[study]])
        nbat = Z[0][f"{budgets[0]}/var"].shape[0]
        boots = []
        for _ in range(BOOT):
            mult = np.zeros(len(Z))
            for u in rng.choice(uniq, len(uniq)):
                mult[of_unit[u]] += 1
            boots.append((mult, rng.integers(0, nbat, (len(Z), nbat))))
        res = {"folds": len(Z), "units": len(uniq), "budgets": {}}
        for B in budgets:
            V = np.stack([z[f"{B}/var"] for z in Z])            # folds x batches x blocks
            Vo = np.stack([z[f"{B}/var_own"] for z in Z])
            rb = {}
            for e in ESTIMATES:
                T = np.stack([z[f"{B}/t_{e}"] for z in Z])
                den = Vo if e == "own_centring" else V
                point = T.mean(1).sum(0) / den.mean(1).sum(0)
                glob = T.mean(1).sum() / den.mean(1).sum()
                bb, bg = [], []
                for mult, pick in boots:
                    Tb = np.take_along_axis(T, pick[:, :, None], 1).mean(1)
                    Db = np.take_along_axis(den, pick[:, :, None], 1).mean(1)
                    num, dd = (mult[:, None] * Tb).sum(0), (mult[:, None] * Db).sum(0)
                    bb.append(num / dd)
                    bg.append(num.sum() / dd.sum())
                bb = np.array(bb)
                rb[e] = {"blocks": point.tolist(), "blocks_ci": np.percentile(bb, [2.5, 97.5], 0).T.tolist(),
                         "global": float(glob), "global_ci": np.percentile(bg, [2.5, 97.5]).tolist()}
            cv = np.concatenate([z[f"{B}/order_cv"] for z in Z])
            fr = np.concatenate([z[f"{B}/order_factor_range"] for z in Z])
            den_all = sum(float(z["den"]) for z in Z)
            rh = sum(float(z[f"{B}/recovery_helmert"].mean()) for z in Z) / den_all
            rj = sum(float(z[f"{B}/recovery_jackknife"].mean()) for z in Z) / den_all
            res["budgets"][str(B)] = {"ratio": rb, "ordering": {"median_cv": np.median(cv, 0).tolist(),
                                                                "median_factor_range": np.median(fr, 0).tolist(),
                                                                "resamples": int(len(cv))},
                                      "recovered_fraction": {"helmert": rh, "jackknife": rj}}
        out[study] = res
    path = HERE / ("results" if out_dir == OUT else "results_smoke") / "noise_calibration.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    for study, res in out.items():
        for B, d in res["budgets"].items():
            print(study, B, {e: round(d["ratio"][e]["global"], 4) for e in ESTIMATES},
                  {k: round(v, 4) for k, v in d["recovered_fraction"].items()})
    return out


# ------------------------------------------------------------------ smoke and run

def smoke():
    folds, conds = validation(synthetic=True, fold_ids=[0, 17])
    f = folds[0]
    ctx = f.ctx
    U, (V, _) = ctx.Uz, es.eigenbasis(ctx.Ry)
    rblocks, pblocks = ctx.blocks, es.eigen_blocks(V.shape[1], es.PROTEIN_EDGES)
    # the resample pipeline reproduces the implemented estimator on the same cells
    rng = np.random.default_rng(1)
    idx = rng.integers(0, len(f.x), 100)
    o = resample(SimpleNamespace(x=f.x[idx], y=f.y[idx], pop=f.pop[idx]), 100, np.random.default_rng(0), U, V,
                 rblocks, pblocks, 0, np.random.default_rng(2))
    rng0 = np.random.default_rng(0)
    pick = rng0.integers(0, 100, 100)
    Xh, Yh, _ = pc.contrasts(f.x[idx][pick], f.y[idx][pick], f.pop[idx][pick])
    ref = es.two_sided_js(Xh, Yh, ctx.Uz, ctx.Ry, ctx.blocks)[0]
    mine = shrunk(o["M"], o["helmert"], o["n2"], rblocks, pblocks, U, V)
    d = float(np.max(np.abs(ref - mine)))
    log(f"reproduces es.two_sided_js: max abs difference {d:.2e}")
    assert d < 1e-10
    out_dir = HERE / "results_smoke" / "noise"
    for fo in folds:
        save("validation", fo, run_fold("validation", fo, conds, (25, 100), 2 * BATCH, log), out_dir)
    score(out_dir)


def check_freeze():
    rec = json.loads((HERE / "results" / "freeze.json").read_text())
    import hashlib
    bad = [f for f, dg in rec["files"].items() if hashlib.sha256((EXT / f).read_bytes()).hexdigest() != dg]
    assert not bad, f"changed since the freeze: {bad}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("smoke", "run", "score"))
    ap.add_argument("study", nargs="?", choices=tuple(STUDIES))
    ap.add_argument("--folds", type=int, nargs="*")
    a = ap.parse_args()
    with threadpool_limits(limits=1):
        if a.what == "smoke":
            smoke()
        elif a.what == "run":
            check_freeze()
            folds, conds = validation(fold_ids=a.folds) if a.study == "validation" else bone_marrow(a.folds)
            for f in folds:
                if (OUT / f"{a.study}_fold{f.fold:02d}.npz").exists():
                    continue
                save(a.study, f, run_fold(a.study, f, conds, BUDGETS, RESAMPLES, log), OUT)
        else:
            score()


if __name__ == "__main__":
    main()
