#!/usr/bin/env python3
"""Exploration 5 (development data): shrinkage profiles in the unpaired latent eigenbasis, all with the map.

    python explore5.py frangieh|papalexi [--draws 6] [--folds 21]

Arms (each mapped to the condition): bjs (block James-Stein), ridge (spectral ridge profile lambda/(lambda+a),
a by two-fold cross-validation, = reference regression with unpaired Gram), sure (profile
lambda^g/(lambda^g + k) with (g, k) minimizing Stein's unbiased risk estimate over all rows), avg (mean of bjs
and ridge).
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
from threadpoolctl import threadpool_limits

import common as cm
import competitors as cp
import run_study as rs
import sp_estimators as es

pm = cm.pm
GAMMAS = (0.5, 1.0, 1.5, 2.0, 3.0)
KAPPAS = 10.0 ** np.arange(-3, 2.01, 0.25)


def sure_profile(X, Y, U, lz):
    """Row shrinkage w_i = l_i^g / (l_i^g + k) in the eigenbasis U; (g, k) by SURE over all coefficients."""
    B = len(X)
    XU = X @ U
    A = XU.T @ Y / B                                   # p x q coefficients
    V = ((XU ** 2).T @ (Y ** 2) / B - A ** 2) / B       # noise variance of each coefficient
    a2 = np.sum(A * A, 1)                               # per row ||a_hat||^2
    t = np.sum(V, 1)                                    # per row noise trace
    best, score = None, np.inf
    for g in GAMMAS:
        lg = lz ** g
        for k in KAPPAS * np.median(lg):
            w = lg / (lg + k)
            # SURE for linear shrinkage w a_hat: w^2 ||a_hat||^2 - 2 w (||a_hat||^2 - t) + const
            s = float(np.sum(w * w * a2 - 2 * w * (a2 - t)))
            if s < score:
                best, score = (g, k), s
    g, k = best
    w = lz ** g / (lz ** g + k)
    return U @ (w[:, None] * A), best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--draws", type=int, default=6)
    ap.add_argument("--folds", type=int, default=21)
    args = ap.parse_args()
    st = rs.SETTINGS[args.dataset]
    budgets = st["budgets"]
    t0 = time.time()
    mod, train, ev = cm.hr.load_dataset(args.dataset)
    x, y = pm.features(train)
    pop = pm.keys_of(train)
    keys = pm.eligible_training(pop, train["part"], mod.MIN_TRAIN_HALF)
    names, masks, _ = cm.hr.block_masks(train["genes"], train["proteins"], mod.ENCODING)
    groups_all = ev["test"]
    folds = [None] if args.dataset == "frangieh" else sorted({g["target"] for g in groups_all})[: args.folds]
    arms = ["bjs", "ridge", "sure", "avg"]
    tot = {a: {b: np.zeros(2) for b in budgets} for a in arms}
    for fi, held in enumerate(folds):
        gs = [g for g in groups_all if held is None or g["target"] == held]
        ks = [k for k in keys if held is None or k.split("|")[0] != held]
        pool, un, X_res, Y_res, cond_res = cm.setup_fold(train, x, y, pop, ks)
        ctx = rs.build_context(train, x, y, pop, ks, pool, un)
        T = np.array([g["T"] for g in gs])
        den = np.einsum("gpq,bpq->gb", np.array([g["TA"] * g["TB"] for g in gs]), masks)
        gcond = [g["cond"] for g in gs]
        for B in budgets:
            if B > len(X_res):
                continue
            for dr in range(args.draws):
                rng = np.random.default_rng([rs.DRAW_SEED, {"frangieh": 1, "papalexi": 2}[args.dataset], fi, B, dr])
                pick = rng.choice(len(X_res), size=B, replace=False)
                X, Y = X_res[pick], Y_res[pick]
                crng = np.random.default_rng([rs.DRAW_SEED + 1, fi, B, dr])
                P = {"bjs": es.block_js(X, Y, ctx.Uz, ctx.blocks)[0]}
                Wt, _ = cp.ridge(X, Y, ctx.Rx, True, rng=crng)
                P["ridge"] = ctx.Rx @ Wt
                P["sure"] = sure_profile(X, Y, ctx.Uz, ctx.lz)[0]
                P["avg"] = 0.5 * (P["bjs"] + P["ridge"])
                for a in arms:
                    C = np.array([ctx.mapped(P[a], c) for c in gcond])
                    num, _ = cm.scores(C, T, None, None, masks)
                    tot[a][B] += [num[:, 0].sum(), den[:, 0].sum()]
        if held is None or fi % 5 == 4 or fi == len(folds) - 1:
            print(f"[{time.time()-t0:.0f}s] after fold {fi}: " + " | ".join(
                f"B={b}: " + ", ".join(f"{a} {100*tot[a][b][0]/tot[a][b][1]:.1f}" for a in arms)
                for b in budgets if tot["bjs"][b][1] > 0), flush=True)
    out = {a: {b: float(v[0] / v[1]) for b, v in d.items() if v[1] > 0} for a, d in tot.items()}
    (cm.HERE / "logs" / f"explore5_{args.dataset}.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
