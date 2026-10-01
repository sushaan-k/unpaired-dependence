#!/usr/bin/env python3
"""Exploration 3 (development data): mapping the pooled estimate to each recipient's own unpaired co-expression.

    python explore3.py frangieh|papalexi [--draws 6] [--budgets ...] [--k 60] [--folds 21]

Arms: bjs_z (pooled), map (condition co-expression), map_g<a> (recipient co-expression from its own adaptation
cells, S_g = a S_group + (1 - a) S_condition, a in {0.25, 0.5, 1}).
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
from threadpoolctl import threadpool_limits

import common as cm
import sp_estimators as es
from noise import noise_fraction, signal_correlation, split_halves

pm = cm.pm
ALPHAS = (0.25, 0.5, 1.0)


def group_latent(mod, part, pool):
    """Latent RNA covariance (pooled-SD units) and SDs of every eligible recipient group, from adaptation cells."""
    d = mod.load(f"{part}_adaptation")
    keys = pm.keys_of(d)
    out = {}
    for key in sorted(set(keys)):
        idx = np.flatnonzero(keys == key)
        if len(idx) < mod.MIN_ADAPT:
            continue
        xg = pm.lognorm(d["counts"][idx], d["library"][idx])
        yg = d["y"][idx]
        sx, sy = xg.std(0), yg.std(0)
        xs = (xg - xg.mean(0)) / np.where(sx > 0, sx, 1)
        Rx = pm.shrink(xs.T @ xs / len(idx))
        h1, h2 = split_halves(d["counts"][idx], d["library"][idx])
        nu = noise_fraction(h1, h2)
        Rz = signal_correlation(Rx, nu)
        dsc = np.where(sx > 0, sx, pool.sd_x) / pool.sd_x
        out[key] = {"Sz": dsc[:, None] * Rz * dsc[None, :], "rx": dsc, "ry": np.where(sy > 0, sy, pool.sd_y) / pool.sd_y}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--draws", type=int, default=6)
    ap.add_argument("--budgets", default="50,100,200,400,800,1600")
    ap.add_argument("--k", type=int, default=60)
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
    part = "heldout" if args.dataset == "frangieh" else "test"
    folds = [None] if args.dataset == "frangieh" else sorted({g["target"] for g in groups_all})[: args.folds]
    arms = ["bjs_z", "map"] + [f"map_g{a}" for a in ALPHAS]
    tot = {a: {b: np.zeros(4) for b in budgets} for a in arms}
    glat = None
    for fi, held in enumerate(folds):
        gs = [g for g in groups_all if held is None or g["target"] == held]
        ks = [k for k in keys if held is None or k.split("|")[0] != held]
        pool, un, X_res, Y_res, cond_res = cm.setup_fold(train, x, y, pop, ks)
        if glat is None or held is not None:
            glat = group_latent(mod, part, pool)
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
        Pk = Uz[:, :k] @ np.diag(1 / lz[:k]) @ Uz[:, :k].T
        Qk = np.eye(len(lz)) - Uz[:, :k] @ Uz[:, :k].T

        def mapped(S, sx, sy, C):
            return (1 / sx)[:, None] * ((S @ Pk + Qk) @ C) * (1 / sy)[None, :]
        print(f"[{time.time()-t0:.0f}s] fold {held}: {len(gs)} groups", flush=True)
        for B in budgets:
            if B > len(X_res):
                continue
            for dr in range(args.draws):
                rng = np.random.default_rng([20261032, fi, B, dr])
                pick = rng.choice(len(X_res), size=B, replace=False)
                X, Y = X_res[pick], Y_res[pick]
                Cz, _ = es.block_js(X, Y, Uz, blocks)
                preds = {"bjs_z": Cz, "map": np.array([mapped(Sz[c], rx[c], ry[c], Cz) for c in gcond])}
                for a in ALPHAS:
                    preds[f"map_g{a}"] = np.array([
                        mapped(a * glat[g["key"]]["Sz"] + (1 - a) * Sz[g["cond"]],
                               a * glat[g["key"]]["rx"] + (1 - a) * rx[g["cond"]],
                               a * glat[g["key"]]["ry"] + (1 - a) * ry[g["cond"]], Cz) for g in gs])
                for arm in arms:
                    num, loss = cm.scores(preds[arm], T, None, None, masks)
                    tot[arm][B] += [num[:, 0].sum(), den[:, 0].sum(), loss[:, 0].sum(), tn[:, 0].sum()]
            if held is None or fi == len(folds) - 1:
                print(f"[{time.time()-t0:.0f}s]  B={B}: " + "  ".join(
                    f"{a} {100*tot[a][B][0]/tot[a][B][1]:.1f}/{tot[a][B][2]/tot[a][B][3]:.3f}" for a in arms), flush=True)
    out = {a: {b: {"rf_all": float(v[0] / v[1]), "loss_rel": float(v[2] / v[3])} for b, v in d.items() if v[1] > 0}
           for a, d in tot.items()}
    (cm.HERE / "logs" / f"explore3_{args.dataset}_k{args.k}.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
