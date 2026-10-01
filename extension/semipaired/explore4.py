#!/usr/bin/env python3
"""Exploration 4 (development data): shrinking the condition map by its unpaired split-half reliability.

    python explore4.py frangieh|papalexi [--draws 6] [--budgets ...] [--k 120] [--folds 21]

Arms: bjs_z (pooled), map (condition map, rank k), map_s (the map shrunk towards the identity by the
split-half reliability of the condition-minus-pooled RNA covariance, estimated from unpaired RNA cells only),
map_s_hier (map_s plus the condition's paired deviation from its mapped pooled mean, James-Stein shrunk).
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
from threadpoolctl import threadpool_limits

import common as cm
import sp_estimators as es

pm = cm.pm


def half_covariances(xs, pop, keys, conds, seed=20261033):
    """Within-population RNA covariance (pooled-SD units) per condition and pooled, in two random halves of cells."""
    rng = np.random.default_rng(seed)
    half = rng.integers(0, 2, len(xs))
    out = []
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
        out.append(res)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--draws", type=int, default=6)
    ap.add_argument("--budgets", default="50,100,200,400,800,1600")
    ap.add_argument("--k", type=int, default=120)
    ap.add_argument("--folds", type=int, default=21)
    args = ap.parse_args()
    budgets = [int(b) for b in args.budgets.split(",")]
    t0 = time.time()
    mod, train, ev = cm.hr.load_dataset(args.dataset)
    x, y = pm.features(train)
    pop = pm.keys_of(train)
    keys = pm.eligible_training(pop, train["part"], mod.MIN_TRAIN_HALF)
    names, masks, _ = cm.hr.block_masks(train["genes"], train["proteins"], mod.ENCODING)
    groups_all = ev["test"]
    folds = [None] if args.dataset == "frangieh" else sorted({g["target"] for g in groups_all})[: args.folds]
    arms = ["bjs_z", "map", "map_s", "map_s_hier"]
    tot = {a: {b: np.zeros(6) for b in budgets} for a in arms}
    rel_log = []
    for fi, held in enumerate(folds):
        gs = [g for g in groups_all if held is None or g["target"] == held]
        ks = [k for k in keys if held is None or k.split("|")[0] != held]
        pool, un, X_res, Y_res, cond_res = cm.setup_fold(train, x, y, pop, ks)
        T = np.array([g["T"] for g in gs])
        den = np.einsum("gpq,bpq->gb", np.array([g["TA"] * g["TB"] for g in gs]), masks)
        tn = np.einsum("gpq,bpq->gb", T * T, masks)
        gcond = [g["cond"] for g in gs]
        Sz, rx, ry = {}, {}, {}
        for c in un.conds:
            ksc = [k for k in pool.keys if cm.cond_of(k) == c]
            r = [pool.raw[pool.keys.index(k)] for k in ksc]
            nx = np.array([q["nx"] for q in r], float)
            ny = np.array([q["ny"] for q in r], float)
            rx[c] = np.sqrt(sum(n * np.diag(q["Sx"]) for n, q in zip(nx, r)) / nx.sum()) / pool.sd_x
            ry[c] = np.sqrt(sum(n * np.diag(q["Sy"]) for n, q in zip(ny, r)) / ny.sum()) / pool.sd_y
            Sz[c] = rx[c][:, None] * un.R[c]["Rz"] * rx[c][None, :]
        Uz, lz = es.eigenbasis(un.R["__all__"]["Rz"])
        blocks = es.eigen_blocks(Uz.shape[0])
        k = args.k
        Uk = Uz[:, :k]
        Pk = Uk @ np.diag(1 / lz[:k]) @ Uk.T
        Qk = np.eye(len(lz)) - Uk @ Uk.T
        # split-half reliability of condition-minus-pooled RNA covariance on the top-k coordinates (unpaired only)
        use = np.isin(pop, pool.keys) & (train["part"] == 0) & ~cm.hm.reservoir(train["cell"])
        hc = half_covariances(x[use] / pool.sd_x, pop[use], pool.keys, un.conds)
        rel = {}
        for c in un.conds:
            d1 = Uk.T @ (hc[0][c] - hc[0]["__all__"]) @ Uk
            d2 = Uk.T @ (hc[1][c] - hc[1]["__all__"]) @ Uk
            rel[c] = float(np.clip(np.sum(d1 * d2) / (0.5 * (np.sum(d1 * d1) + np.sum(d2 * d2))), 0, 1))
        rel_log.append({"fold": held, "reliability": rel})
        G = {c: Sz[c] @ Pk + Qk for c in un.conds}
        Gs = {c: np.eye(len(lz)) + rel[c] * (G[c] - np.eye(len(lz))) for c in un.conds}
        sx_s = {c: 1 + rel[c] * (rx[c] - 1) for c in un.conds}
        sy_s = {c: 1 + rel[c] * (ry[c] - 1) for c in un.conds}
        print(f"[{time.time()-t0:.0f}s] fold {held}: {len(gs)} groups; map reliability {rel}", flush=True)
        for B in budgets:
            if B > len(X_res):
                continue
            for dr in range(args.draws):
                rng = np.random.default_rng([20261032, fi, B, dr])
                pick = rng.choice(len(X_res), size=B, replace=False)
                X, Y, cc = X_res[pick], Y_res[pick], cond_res[pick]
                Cz, _ = es.block_js(X, Y, Uz, blocks)
                pm_ = X.T @ Y / B
                per = {c: (1 / rx[c])[:, None] * (G[c] @ Cz) * (1 / ry[c])[None, :] for c in un.conds}
                pers = {c: (1 / sx_s[c])[:, None] * (Gs[c] @ Cz) * (1 / sy_s[c])[None, :] for c in un.conds}
                hier = {}
                for c in un.conds:
                    m = cc == c
                    if m.sum() >= 3:
                        Zc = (X[m][:, :, None] * Y[m][:, None, :]).reshape(int(m.sum()), -1)
                        mc, trc, _ = es.noise_terms(Zc)
                        dvec = mc - (Gs[c] @ pm_).ravel()
                        f = es.js_factor(float(dvec @ dvec), trc)
                        hier[c] = pers[c] + (1 / sx_s[c])[:, None] * (f * dvec.reshape(pm_.shape)) * (1 / sy_s[c])[None, :]
                    else:
                        hier[c] = pers[c]
                preds = {"bjs_z": Cz, "map": np.array([per[c] for c in gcond]),
                         "map_s": np.array([pers[c] for c in gcond]), "map_s_hier": np.array([hier[c] for c in gcond])}
                for arm in arms:
                    num, loss = cm.scores(preds[arm], T, None, None, masks)
                    tot[arm][B] += [num[:, 0].sum(), den[:, 0].sum(), loss[:, 0].sum(), tn[:, 0].sum(),
                                    num[:, 1].sum(), den[:, 1].sum()]
            if held is None or fi == len(folds) - 1:
                print(f"[{time.time()-t0:.0f}s]  B={B}: " + "  ".join(
                    f"{a} {100*tot[a][B][0]/tot[a][B][1]:.1f}/{tot[a][B][2]/tot[a][B][3]:.3f}" for a in arms), flush=True)
    out = {a: {b: {"rf_all": float(v[0] / v[1]), "loss_rel": float(v[2] / v[3]), "rf_ifn": float(v[4] / v[5])}
               for b, v in d.items() if v[1] > 0} for a, d in tot.items()}
    out["reliability"] = rel_log
    (cm.HERE / "logs" / f"explore4_{args.dataset}_k{args.k}.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
