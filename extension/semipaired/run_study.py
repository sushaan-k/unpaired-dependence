#!/usr/bin/env python3
"""Semi-paired finishing study (development): the covariance-guided estimator against the closest methods at
matched budgets, with mechanism ablations and the unpaired dose-response.

    python run_study.py frangieh|papalexi [--draws N] [--smoke]

Every arm in a draw uses the same B paired calibration cells and the same unpaired pool. Denoisers, each run
pooled, per condition (refitted on the condition's paired cells and unpaired summaries) and mapped (the pooled
estimate mapped to each condition through its unpaired co-expression, the step of the proposed estimator):

  bjs2       block James-Stein in the unpaired eigenbases of both assays: latent RNA eigen-blocks (1-5, 6-20,
             21-60, rest) x protein eigen-blocks (1-3, rest) (proposed denoiser)
  bjs        the same with all protein coordinates in every block (one-sided; ablation)
  js         James-Stein towards zero in marker coordinates
  scose      scalar shrinkage with leave-one-out intensity (Muandet-style SCOSE)
  fcose      spectral shrinkage with cross-validated penalty (FCOSE)
  lowrank    cross-validated truncated SVD
  semicca    paired cross-covariance projected on SemiCCA subspaces (unpaired PCA part), beta and rank by CV
  ridge_p    reference ridge regression, Gram from the paired cells
  ridge_u    reference ridge regression, Gram from unpaired cells

The proposed estimator is bjs2/mapped. Ablations (pooled unless stated): bjs2 with the paired cells' own
eigenbases (bjs_pairbasis) and in marker coordinates with random blocks of the same sizes (bjs_marker); bjs2/mapped
with the maps estimated from the condition's paired cells instead of unpaired cells (map_paired) or from measured
instead of latent unpaired RNA covariances (map_measured); and bjs2/mapped with unpaired summaries from 500, 2,000
or 8,000 unpaired cells of each assay (dose_<n>). Writes results/<dataset>.json.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
from threadpoolctl import threadpool_limits

import common as cm
import competitors as cp
import sp_estimators as es
from sp_estimators import eigenbasis as eigenbasis_

pm = cm.pm
SETTINGS = {"frangieh": {"budgets": (50, 100, 200, 400, 800, 1600, 3200), "draws": 10},
            "papalexi": {"budgets": (50, 100, 200, 400, 800, 1600), "draws": 4}}
DENOISERS = ("bjs2", "bjs", "js", "scose", "fcose", "lowrank", "semicca", "ridge_p", "ridge_u")
MODES = ("pooled", "percond", "mapped")
DOSES = (500, 2000, 8000)
K_MAP = 120
DRAW_SEED = 20261036


class Context:
    """Unpaired quantities of one fold (or of a subsample of its unpaired cells)."""

    def __init__(self, un, pool, conds_sd):
        self.un, self.pool = un, pool
        R = un.R["__all__"]
        self.Uz, self.lz = es.eigenbasis(R["Rz"])
        self.blocks = es.eigen_blocks(len(self.lz))
        self.Rx, self.Ry = R["Rx"], R["Ry"]
        self.n_x, self.n_y = R["n_rna"], R["n_prot"]
        self.cond = {}
        for c in un.conds:
            rx, ry = conds_sd[c]
            Rc = un.R[c]
            Uc, _ = es.eigenbasis(Rc["Rz"])
            self.cond[c] = {"rx": rx, "ry": ry, "Sz": rx[:, None] * Rc["Rz"] * rx[None, :], "Uz": Uc,
                            "Rx": Rc["Rx"], "Ry": Rc["Ry"], "n_x": Rc["n_rna"], "n_y": Rc["n_prot"]}

    def set_maps(self, reliability, k=K_MAP):
        Uk = self.Uz[:, :k]
        Pk = Uk @ np.diag(1 / self.lz[:k]) @ Uk.T
        Qk = np.eye(len(self.lz)) - Uk @ Uk.T
        I = np.eye(len(self.lz))
        self.maps = {}
        for c, d in self.cond.items():
            s = reliability[c]
            G = d["Sz"] @ Pk + Qk
            self.maps[c] = (I + s * (G - I), 1 + s * (d["rx"] - 1), 1 + s * (d["ry"] - 1))

    def mapped(self, C, c):
        G, sx, sy = self.maps[c]
        return (1 / sx)[:, None] * (G @ C) * (1 / sy)[None, :]

    def mapped_measured(self, C, c, k=K_MAP):
        """Ablation: the same map built from measured (not noise-corrected) RNA covariances, without shrinkage."""
        if not hasattr(self, "_mm"):
            Ux, lx = eigenbasis_(self.Rx)
            Uk = Ux[:, :k]
            Pk = Uk @ np.diag(1 / lx[:k]) @ Uk.T
            Qk = np.eye(len(lx)) - Uk @ Uk.T
            self._mm = {cc: (self.cond[cc]["rx"][:, None] * self.cond[cc]["Rx"] * self.cond[cc]["rx"][None, :]) @ Pk + Qk
                        for cc in self.cond}
        d = self.cond[c]
        return (1 / d["rx"])[:, None] * (self._mm[c] @ C) * (1 / d["ry"])[None, :]


def condition_sds(pool, conds):
    out = {}
    for c in conds:
        ks = [k for k in pool.keys if cm.cond_of(k) == c]
        r = [pool.raw[pool.keys.index(k)] for k in ks]
        nx = np.array([q["nx"] for q in r], float)
        ny = np.array([q["ny"] for q in r], float)
        out[c] = (np.sqrt(sum(n * np.diag(q["Sx"]) for n, q in zip(nx, r)) / nx.sum()) / pool.sd_x,
                  np.sqrt(sum(n * np.diag(q["Sy"]) for n, q in zip(ny, r)) / ny.sum()) / pool.sd_y)
    return out


def half_reliability(xs, pop, keys, conds, Uk, seed=20261033):
    """Split-half reliability of each condition's RNA covariance minus the pooled one, on the top-k coordinates."""
    half = np.random.default_rng(seed).integers(0, 2, len(xs))
    cov = []
    for h in (0, 1):
        m = half == h
        res = {}
        for c in ["__all__"] + conds:
            ks = keys if c == "__all__" else [k for k in keys if cm.cond_of(k) == c]
            acc, n = 0.0, 0
            for k in ks:
                mm = m & (pop == k)
                if mm.sum() < 2:
                    continue
                d = xs[mm] - xs[mm].mean(0)
                acc = acc + d.T @ d
                n += int(mm.sum())
            res[c] = acc / n
        cov.append(res)
    rel = {}
    for c in conds:
        d1 = Uk.T @ (cov[0][c] - cov[0]["__all__"]) @ Uk
        d2 = Uk.T @ (cov[1][c] - cov[1]["__all__"]) @ Uk
        rel[c] = float(np.clip(np.sum(d1 * d2) / (0.5 * (np.sum(d1 * d1) + np.sum(d2 * d2))), 0, 1))
    return rel


def build_context(train, x, y, pop, keys, pool, un, rna_idx=None, prot_idx=None):
    ctx = Context(un, pool, condition_sds(pool, un.conds))
    use = np.isin(pop, pool.keys) & (train["part"] == 0) & ~cm.hm.reservoir(train["cell"])
    if rna_idx is not None:
        sel = np.zeros(len(pop), bool)
        sel[rna_idx] = True
        use &= sel
    rel = half_reliability(x[use] / pool.sd_x, pop[use], pool.keys, un.conds, ctx.Uz[:, :K_MAP])
    ctx.set_maps(rel)
    ctx.reliability = rel
    return ctx


def subsample_unpaired(train, x, y, pop, keys, pool, n_u, seed):
    """Unpaired summaries from about n_u RNA and n_u protein pool cells: whole training populations taken in random
    order (within every condition in proportion) until the cell count is reached, so within-population centring
    keeps working."""
    rng = np.random.default_rng(seed)
    base = np.isin(pop, pool.keys) & ~cm.hm.reservoir(train["cell"])
    keep = np.zeros(len(pop), bool)
    conds = sorted({cm.cond_of(k) for k in pool.keys})
    for c in conds:
        ks = [k for k in pool.keys if cm.cond_of(k) == c]
        share = sum(int(np.sum(base & (pop == k) & (train["part"] == 0))) for k in ks) / max(
            int(np.sum(base & (train["part"] == 0))), 1)
        target, got = n_u * share, 0
        for k in [ks[i] for i in rng.permutation(len(ks))]:
            if got >= target:
                break
            m = base & (pop == k)
            keep |= m
            got += int(np.sum(m & (train["part"] == 0)))
    un = cm.Unpaired(pool, x[keep], y[keep], train["counts"][keep], train["library"][keep], pop[keep],
                     train["part"][keep], pool.keys)
    ctx = Context(un, pool, condition_sds(pool, un.conds))
    rna = np.flatnonzero(keep & (train["part"] == 0))
    xs = x[rna] / pool.sd_x
    ctx.set_maps(half_reliability(xs, pop[rna], pool.keys, un.conds, ctx.Uz[:, :K_MAP]))
    return ctx


def denoise(name, X, Y, Rx, Ry, n_x, n_y, U, blocks, rng):
    """Pooled-type estimate of the cross-correlation from paired cells X, Y with unpaired summaries (Rx, Ry, U)."""
    if name == "bjs2":
        return es.two_sided_js(X, Y, U, Ry, blocks)[0]
    if name == "bjs":
        return es.block_js(X, Y, U, blocks)[0]
    if name == "js":
        return es.js_matrix(X, Y)[0]
    if name == "scose":
        return cp.scose(X, Y)
    if name == "fcose":
        return cp.fcose(X, Y, rng=rng)[0]
    if name == "lowrank":
        return cp.low_rank(X, Y, rng=rng)[0]
    if name == "semicca":
        return cp.semicca(X, Y, Rx, Ry, rng=rng)[0]
    if name == "factor":
        return cp.factor_model(X, Y, Rx, n_x, Ry, n_y, ks=(2, 5, 10, 20), rng=rng)[0]
    if name in ("ridge_p", "ridge_u"):
        Wt, _ = cp.ridge(X, Y, Rx, name == "ridge_u", rng=rng)
        return Rx @ Wt
    raise ValueError(name)


def run_fold(dataset, fi, gs, train, x, y, pop, keys, masks, budgets, draws, log, smoke):
    pool, un, X_res, Y_res, cond_res = cm.setup_fold(train, x, y, pop, keys)
    ctx = build_context(train, x, y, pop, keys, pool, un)
    doses = {n: subsample_unpaired(train, x, y, pop, keys, pool, n, 20261037 + n) for n in DOSES}
    T = np.array([g["T"] for g in gs])
    den = np.einsum("gpq,bpq->gb", np.array([g["TA"] * g["TB"] for g in gs]), masks)
    tn = np.einsum("gpq,bpq->gb", T * T, masks)
    gcond = [g["cond"] for g in gs]
    conds = un.conds
    rng_marker = np.random.default_rng(20261038)
    marker_order = rng_marker.permutation(len(ctx.lz))
    Umarker = np.eye(len(ctx.lz))[:, marker_order]
    Vmarker = np.eye(ctx.Ry.shape[0])[:, rng_marker.permutation(ctx.Ry.shape[0])]
    pblocks = es.eigen_blocks(ctx.Ry.shape[0], es.PROTEIN_EDGES)
    paired_all = {}
    for c in conds:
        acc_, n_ = 0.0, 0
        for kk in [kk for kk in pool.keys if cm.cond_of(kk) == c]:
            m = pop == kk
            xs, _ = pm.standardize(x[m])
            ys, _ = pm.standardize(y[m])
            acc_ = acc_ + xs.T @ ys
            n_ += int(m.sum())
        paired_all[c] = acc_ / n_
    out = {"den": den, "tn": tn, "num": {}, "loss": {}, "info": {"reliability": ctx.reliability,
                                                                 "n_reservoir": int(len(X_res))}}

    def add(arm, B, C):
        num, loss = cm.scores(C, T, None, None, masks)
        out["num"].setdefault(arm, {}).setdefault(B, np.zeros_like(den))
        out["loss"].setdefault(arm, {}).setdefault(B, np.zeros_like(den))
        out["num"][arm][B] += num
        out["loss"][arm][B] += loss

    add("paired_all", 0, np.array([paired_all[c] for c in gcond]))
    add("independence", 0, np.zeros_like(T))
    for B in budgets:
        if B > len(X_res):
            continue
        t_b = time.time()
        for dr in range(draws):
            rng = np.random.default_rng([DRAW_SEED, {"frangieh": 1, "papalexi": 2}[dataset], fi, B, dr])
            pick = rng.choice(len(X_res), size=B, replace=False)
            X, Y, cc = X_res[pick], Y_res[pick], cond_res[pick]
            for name in DENOISERS:
                if smoke and name in ("fcose", "factor", "semicca") and B > 200:
                    continue
                crng = np.random.default_rng([DRAW_SEED + 1, fi, B, dr])
                P = denoise(name, X, Y, ctx.Rx, ctx.Ry, ctx.n_x, ctx.n_y, ctx.Uz, ctx.blocks, crng)
                add(f"{name}/pooled", B, P)
                add(f"{name}/mapped", B, np.array([ctx.mapped(P, c) for c in gcond]))
                per = {}
                for c in conds:
                    m = cc == c
                    d = ctx.cond[c]
                    if m.sum() >= 6:
                        per[c] = denoise(name, X[m], Y[m], d["Rx"], d["Ry"], d["n_x"], d["n_y"], d["Uz"], ctx.blocks,
                                         crng)
                    else:
                        per[c] = np.zeros_like(P)
                add(f"{name}/percond", B, np.array([per[c] for c in gcond]))
            # ablations of the proposed estimator
            Pb = es.block_js2(X, Y, np.linalg.eigh(X.T @ X / B)[1][:, ::-1], np.linalg.eigh(Y.T @ Y / B)[1][:, ::-1],
                              ctx.blocks, pblocks)[0]
            add("abl/bjs_pairbasis", B, Pb)
            add("abl/bjs_marker", B, es.block_js2(X, Y, Umarker, Vmarker, ctx.blocks, pblocks)[0])
            Pz = es.two_sided_js(X, Y, ctx.Uz, ctx.Ry, ctx.blocks)[0]
            # maps from the condition's paired cells (measured covariance, no unpaired information)
            mp = []
            for g_c in gcond:
                m = cc == g_c
                Sp = X[m].T @ X[m] / max(m.sum(), 1)
                Uk = ctx.Uz[:, :K_MAP]
                G = Sp @ Uk @ np.diag(1 / ctx.lz[:K_MAP]) @ Uk.T + (np.eye(len(ctx.lz)) - Uk @ Uk.T)
                mp.append(G @ Pz)
            add("abl/map_paired", B, np.array(mp))
            add("abl/map_measured", B, np.array([ctx.mapped_measured(Pz, c) for c in gcond]))
            for n_u, dctx in doses.items():
                Pd = es.two_sided_js(X, Y, dctx.Uz, dctx.Ry, dctx.blocks)[0]
                add(f"dose/{n_u}", B, np.array([dctx.mapped(Pd, c) for c in gcond]))
        for arm in list(out["num"]):
            if B in out["num"][arm]:
                out["num"][arm][B] /= draws if arm not in ("paired_all", "independence") else 1
                out["loss"][arm][B] /= draws if arm not in ("paired_all", "independence") else 1
        log(f"  B={B} ({time.time() - t_b:.0f}s): " + ", ".join(
            f"{a} {100 * out['num'][a][B][:, 0].sum() / den[:, 0].sum():.1f}" for a in
            ("bjs2/mapped", "bjs/mapped", "bjs2/pooled", "js/pooled", "fcose/mapped", "semicca/mapped",
             "ridge_u/mapped", "abl/bjs_pairbasis") if B in out["num"].get(a, {})))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=sorted(SETTINGS))
    ap.add_argument("--draws", type=int)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    st = SETTINGS[args.dataset]
    draws = args.draws or st["draws"]
    budgets = st["budgets"][:2] if args.smoke else st["budgets"]
    t0 = time.time()

    def log(s):
        print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)
    mod, train, ev = cm.hr.load_dataset(args.dataset)
    x, y = pm.features(train)
    pop = pm.keys_of(train)
    keys = pm.eligible_training(pop, train["part"], mod.MIN_TRAIN_HALF)
    names, masks, defs = cm.hr.block_masks(train["genes"], train["proteins"], mod.ENCODING)
    groups_all = ev["test"]
    folds = [None] if args.dataset == "frangieh" else sorted({g["target"] for g in groups_all})
    if args.smoke:
        folds = folds[:1]
    rows, infos = [], []
    for fi, held in enumerate(folds):
        gs = [g for g in groups_all if held is None or g["target"] == held]
        ks = [k for k in keys if held is None or k.split("|")[0] != held]
        log(f"fold {held}: {len(gs)} groups")
        out = run_fold(args.dataset, fi, gs, train, x, y, pop, ks, masks, budgets, draws, log, args.smoke)
        infos.append(dict(out["info"], fold=held))
        for i, g in enumerate(gs):
            rows.append({"key": g["key"], "target": g["target"], "cond": g["cond"], "den": out["den"][i].tolist(),
                         "tn": out["tn"][i].tolist(),
                         "num": {a: {str(b): v[i].tolist() for b, v in d.items()} for a, d in out["num"].items()},
                         "loss": {a: {str(b): v[i].tolist() for b, v in d.items()} for a, d in out["loss"].items()}})
    res = {"dataset": args.dataset, "blocks": names, "block_defs": defs, "budgets": list(budgets), "draws": draws,
           "rows": rows, "fold_info": infos, "smoke": args.smoke}
    (cm.HERE / "results" / f"{args.dataset}{'_smoke' if args.smoke else ''}.json").write_text(json.dumps(res) + "\n")
    log("done")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
