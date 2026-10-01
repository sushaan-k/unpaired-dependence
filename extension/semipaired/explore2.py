#!/usr/bin/env python3
"""Exploration 2 (development data): condition-specific estimation at matched total budgets.

    python explore2.py frangieh|papalexi [--draws 6] [--budgets ...] [--folds 21] [--k 60]

Arms (all from the same B paired cells and the same unpaired pool):
  bjs          pooled block James-Stein in the measured unpaired RNA eigenbasis
  bjs_z        the same in the latent (noise-corrected) eigenbasis
  map          bjs_z mapped to each condition through unpaired co-expression: C_c = S_c S^-1 C on the top-k
               latent eigendirections (shared channel, condition-specific latent RNA covariance), then scaled to
               the condition's correlation units
  cond_direct  block James-Stein on the condition's own paired cells in its own unpaired eigenbasis
  hier         bjs plus the condition's deviation from the pooled paired mean, James-Stein shrunk
  map_hier     map plus the same deviation
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


def condition_scale(un, pool, c, x_sd_cond, y_sd_cond):
    return x_sd_cond[c] / pool.sd_x, y_sd_cond[c] / pool.sd_y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--draws", type=int, default=6)
    ap.add_argument("--budgets", default="50,100,200,400,800,1600")
    ap.add_argument("--folds", type=int, default=21)
    ap.add_argument("--k", type=int, default=60)
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
    arms = ["bjs", "bjs_z", "map", "cond_direct", "hier", "map_hier", "paired_all"]
    tot = {a: {b: np.zeros(4) for b in budgets} for a in arms}
    for fi, held in enumerate(folds):
        gs = [g for g in groups_all if held is None or g["target"] == held]
        ks = [k for k in keys if held is None or k.split("|")[0] != held]
        pool, un, X_res, Y_res, cond_res = cm.setup_fold(train, x, y, pop, ks)
        T = np.array([g["T"] for g in gs])
        den = np.einsum("gpq,bpq->gb", np.array([g["TA"] * g["TB"] for g in gs]), masks)
        tn = np.einsum("gpq,bpq->gb", T * T, masks)
        gcond = [g["cond"] for g in gs]
        # unpaired condition-level covariances in pooled-SD units (latent) and condition SDs
        use = np.isin(pop, pool.keys)
        sd_x, sd_y, Sz = {}, {}, {}
        for c in un.conds:
            ksc = [k for k in pool.keys if cm.cond_of(k) == c]
            r = [pool.raw[pool.keys.index(k)] for k in ksc]
            nx = np.array([q["nx"] for q in r], float)
            ny = np.array([q["ny"] for q in r], float)
            vx = sum(n * np.diag(q["Sx"]) for n, q in zip(nx, r)) / nx.sum()
            vy = sum(n * np.diag(q["Sy"]) for n, q in zip(ny, r)) / ny.sum()
            sd_x[c], sd_y[c] = np.sqrt(vx), np.sqrt(vy)
            d = sd_x[c] / pool.sd_x
            Sz[c] = d[:, None] * un.R[c]["Rz"] * d[None, :]
        Rz = un.R["__all__"]["Rz"]
        Ux, _ = es.eigenbasis(un.R["__all__"]["Rx"])
        Uz, lz = es.eigenbasis(Rz)
        blocks = es.eigen_blocks(Ux.shape[0])
        k = args.k
        Mk = {c: Sz[c] @ Uz[:, :k] @ np.diag(1 / lz[:k]) @ Uz[:, :k].T + (np.eye(len(lz)) - Uz[:, :k] @ Uz[:, :k].T)
              for c in un.conds}
        scale = {c: (1 / (sd_x[c] / pool.sd_x))[:, None] * (1 / (sd_y[c] / pool.sd_y))[None, :] for c in un.conds}
        cond_bases = {c: es.eigenbasis(un.R[c]["Rx"])[0] for c in un.conds}
        allm = np.isin(pop, pool.keys)
        paired_all = {}
        for c in un.conds:
            acc_, n_ = 0.0, 0
            for kk in [kk for kk in pool.keys if cm.cond_of(kk) == c]:
                m = pop == kk
                xs, _ = pm.standardize(x[m])
                ys, _ = pm.standardize(y[m])
                acc_ = acc_ + xs.T @ ys
                n_ += int(m.sum())
            paired_all[c] = acc_ / n_
        print(f"[{time.time()-t0:.0f}s] fold {held}: {len(gs)} groups, reservoir {len(X_res)}", flush=True)
        for B in budgets:
            if B > len(X_res):
                continue
            for dr in range(args.draws):
                rng = np.random.default_rng([20261032, fi, B, dr])
                pick = rng.choice(len(X_res), size=B, replace=False)
                X, Y, cc = X_res[pick], Y_res[pick], cond_res[pick]
                preds = {}
                preds["bjs"], _ = es.block_js(X, Y, Ux, blocks)
                Cz, _ = es.block_js(X, Y, Uz, blocks)
                preds["bjs_z"] = Cz
                per_map = {c: scale[c] * (Mk[c] @ Cz) for c in un.conds}
                preds["map"] = np.array([per_map[c] for c in gcond])
                pooled_mean = X.T @ Y / B
                dev, direct = {}, {}
                for c in un.conds:
                    m = cc == c
                    if m.sum() >= 3:
                        direct[c], _ = es.block_js(X[m], Y[m], cond_bases[c], blocks)
                        # deviation of the condition's paired mean from the pooled mean, James-Stein shrunk
                        Zc = (X[m][:, :, None] * Y[m][:, None, :]).reshape(int(m.sum()), -1)
                        mc, trc, _ = es.noise_terms(Zc)
                        dvec = mc - pooled_mean.ravel()
                        f = es.js_factor(float(dvec @ dvec), trc * (1 - m.sum() / B))
                        dev[c] = f * dvec.reshape(pooled_mean.shape)
                    else:
                        direct[c] = np.zeros_like(pooled_mean)
                        dev[c] = np.zeros_like(pooled_mean)
                preds["cond_direct"] = np.array([direct[c] for c in gcond])
                preds["hier"] = np.array([preds["bjs"] + dev[c] for c in gcond])
                preds["map_hier"] = np.array([per_map[c] + scale[c] * dev[c] for c in gcond])
                preds["paired_all"] = np.array([paired_all[c] for c in gcond])
                for a in arms:
                    num, loss = cm.scores(preds[a], T, None, None, masks)
                    tot[a][B] += [num[:, 0].sum(), den[:, 0].sum(), loss[:, 0].sum(), tn[:, 0].sum()]
            if held is None or fi == len(folds) - 1:
                print(f"[{time.time()-t0:.0f}s]  B={B}: " + "  ".join(
                    f"{a} {100*tot[a][B][0]/tot[a][B][1]:.1f}/{tot[a][B][2]/tot[a][B][3]:.3f}" for a in arms), flush=True)
    out = {a: {b: {"rf_all": float(v[0] / v[1]), "loss_rel": float(v[2] / v[3])} for b, v in d.items() if v[1] > 0}
           for a, d in tot.items()}
    (cm.HERE / "logs" / f"explore2_{args.dataset}_k{args.k}.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
