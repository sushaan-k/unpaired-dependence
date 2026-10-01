#!/usr/bin/env python3
"""Exploration 7 (development data): sparse channel with unpaired co-expression (semi-paired lasso).

    python explore7.py frangieh|papalexi [--draws 4] [--folds 21] [--budgets ...]

W minimizes 1/2 tr(W' G W) - tr(W' C_hat) + lambda ||W||_1 (FISTA, all proteins jointly), with C_hat the paired
cross-covariance and G an unpaired RNA correlation (measured Rx: lasso_x; latent Rz: lasso_z); lambda by two-fold
cross-validation on the paired cells; estimate G W, mapped to the condition. Compared with bjs and ridge_u (mapped).
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
ARMS = ["bjs", "ridge", "lasso_x", "lasso_z"]
LAMBDAS = 10.0 ** np.arange(-3, 0.01, 0.25)     # relative to max |C_hat|


def fista(G, C, lam, W0=None, L=None, iters=300, tol=1e-6):
    L = L or float(np.linalg.eigvalsh(G)[-1])
    W = np.zeros_like(C) if W0 is None else W0.copy()
    V, t = W.copy(), 1.0
    for _ in range(iters):
        Wn = V - (G @ V - C) / L
        Wn = np.sign(Wn) * np.maximum(np.abs(Wn) - lam / L, 0)
        tn = (1 + np.sqrt(1 + 4 * t * t)) / 2
        V = Wn + (t - 1) / tn * (Wn - W)
        if np.max(np.abs(Wn - W)) < tol:
            W = Wn
            break
        W, t = Wn, tn
    return W


def lasso_path(G, C, lams, L):
    out, W = [], None
    for lam in sorted(lams, reverse=True):
        W = fista(G, C, lam, W, L)
        out.append((lam, W))
    return out[::-1]


def semi_lasso(X, Y, G, rng):
    L = float(np.linalg.eigvalsh(G)[-1])
    a, b = cp.halves(len(X), rng)
    scale = float(np.max(np.abs(X.T @ Y / len(X))))
    lams = LAMBDAS * scale
    score = np.zeros(len(lams))
    for tr, te in ((a, b), (b, a)):
        C = X[tr].T @ Y[tr] / len(tr)
        for i, (lam, W) in enumerate(lasso_path(G, C, lams, L)):
            score[i] += cp.cv_score(G @ W, X[te], Y[te])
    lam = lams[int(np.argmax(score))]
    W = fista(G, X.T @ Y / len(X), lam, None, L, iters=500)
    return G @ W, lam / scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--draws", type=int, default=4)
    ap.add_argument("--folds", type=int, default=21)
    ap.add_argument("--budgets", type=int, nargs="+")
    args = ap.parse_args()
    budgets = args.budgets or rs.SETTINGS[args.dataset]["budgets"]
    t0 = time.time()
    mod, train, ev = cm.hr.load_dataset(args.dataset)
    x, y = pm.features(train)
    print("p, q =", x.shape[1], y.shape[1], flush=True)
    pop = pm.keys_of(train)
    keys = pm.eligible_training(pop, train["part"], mod.MIN_TRAIN_HALF)
    names, masks, _ = cm.hr.block_masks(train["genes"], train["proteins"], mod.ENCODING)
    groups_all = ev["test"]
    folds = [None] if args.dataset == "frangieh" else sorted({g["target"] for g in groups_all})[: args.folds]
    tot = {a: {b: np.zeros(2) for b in budgets} for a in ARMS}
    lam_log = {b: [] for b in budgets}
    for fi, held in enumerate(folds):
        gs = [g for g in groups_all if held is None or g["target"] == held]
        ks = [k for k in keys if held is None or k.split("|")[0] != held]
        pool, un, X_res, Y_res, cond_res = cm.setup_fold(train, x, y, pop, ks)
        ctx = rs.build_context(train, x, y, pop, ks, pool, un)
        Rz = (ctx.Uz * ctx.lz) @ ctx.Uz.T
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
                ts = time.time()
                P["lasso_x"], lx = semi_lasso(X, Y, ctx.Rx, np.random.default_rng([rs.DRAW_SEED + 2, fi, B, dr]))
                P["lasso_z"], lz_ = semi_lasso(X, Y, Rz, np.random.default_rng([rs.DRAW_SEED + 2, fi, B, dr]))
                lam_log[B].append((lx, lz_, time.time() - ts))
                for a in ARMS:
                    C = np.array([ctx.mapped(P[a], c) for c in gcond])
                    num, _ = cm.scores(C, T, None, None, masks)
                    tot[a][B] += [num[:, 0].sum(), den[:, 0].sum()]
            if held is None:
                print(f"[{time.time()-t0:.0f}s] B={B}: " + ", ".join(
                    f"{a} {100*tot[a][B][0]/tot[a][B][1]:.1f}" for a in ARMS) + f" | lambdas {lam_log[B][-args.draws:]}",
                    flush=True)
        if held is not None and (fi % 5 == 4 or fi == len(folds) - 1):
            print(f"[{time.time()-t0:.0f}s] after fold {fi}: " + " | ".join(
                f"B={b}: " + ", ".join(f"{a} {100*tot[a][b][0]/tot[a][b][1]:.1f}" for a in ARMS)
                for b in budgets if tot["bjs"][b][1] > 0), flush=True)
    out = {a: {b: float(v[0] / v[1]) for b, v in d.items() if v[1] > 0} for a, d in tot.items()}
    (cm.HERE / "logs" / f"explore7_{args.dataset}.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
