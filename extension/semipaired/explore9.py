#!/usr/bin/env python3
"""Exploration 9 (development data, after Champollion's development runs): does propagating each protein's
covariance with its own encoding genes through unpaired co-expression help at small budgets?

    python explore9.py frangieh|papalexi [--draws 3] [--folds 7]

Arms, all mapped: bjs2 (proposed), cog (cognate propagation, James-Stein shrunk, plus two-sided block
James-Stein of the residual products).
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
from threadpoolctl import threadpool_limits

import common as cm
import run_study as rs
import sp_estimators as es

pm = cm.pm
ARMS = ["bjs2", "cog"]


def cognate_two_sided(X, Y, ops, U, V, rblocks, pblocks):
    B, q = Y.shape
    p = X.shape[1]
    Zs = np.stack([(X @ ops[j].T) * Y[:, [j]] for j in range(q)], axis=2)          # B x p x q
    m, tr, lam = es.noise_terms(Zs.reshape(B, -1))
    f = es.js_factor(float(m @ m), tr, lam)
    Cprop = f * m.reshape(p, q)
    R = np.stack([(X - X @ ops[j].T) * Y[:, [j]] for j in range(q)], axis=2)      # residual products
    A = np.einsum("bpq,pk,ql->bkl", R, U, V)
    out = np.zeros((U.shape[1], V.shape[1]))
    for rows in rblocks:
        for cols in pblocks:
            Z = A[:, rows][:, :, cols].reshape(B, -1)
            mb, trb, lamb = es.noise_terms(Z)
            out[np.ix_(rows, cols)] = es.js_factor(float(mb @ mb), trb, lamb) * mb.reshape(len(rows), len(cols))
    return Cprop + U @ out @ V.T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--folds", type=int, default=7)
    args = ap.parse_args()
    budgets = (50, 100, 200, 400)
    t0 = time.time()
    mod, train, ev = cm.hr.load_dataset(args.dataset)
    x, y = pm.features(train)
    pop = pm.keys_of(train)
    keys = pm.eligible_training(pop, train["part"], mod.MIN_TRAIN_HALF)
    names, masks, _ = cm.hr.block_masks(train["genes"], train["proteins"], mod.ENCODING)
    sets = cm.encoding_sets(train["genes"], train["proteins"], mod.ENCODING)
    print("encoding genes per protein:", [len(s) for s in sets], flush=True)
    groups_all = ev["test"]
    folds = [None] if args.dataset == "frangieh" else sorted({g["target"] for g in groups_all})[: args.folds]
    tot = {a: {b: np.zeros(2) for b in budgets} for a in ARMS}
    for fi, held in enumerate(folds):
        gs = [g for g in groups_all if held is None or g["target"] == held]
        ks = [k for k in keys if held is None or k.split("|")[0] != held]
        pool, un, X_res, Y_res, cond_res = cm.setup_fold(train, x, y, pop, ks)
        ctx = rs.build_context(train, x, y, pop, ks, pool, un)
        Rz = (ctx.Uz * ctx.lz) @ ctx.Uz.T
        ops = es.cognate_operator(Rz, sets, union=False)
        V, _ = es.eigenbasis(ctx.Ry)
        pb = es.eigen_blocks(V.shape[1], es.PROTEIN_EDGES)
        T = np.array([g["T"] for g in gs])
        den = np.einsum("gpq,bpq->gb", np.array([g["TA"] * g["TB"] for g in gs]), masks)
        gcond = [g["cond"] for g in gs]
        for B in budgets:
            for dr in range(args.draws):
                rng = np.random.default_rng([rs.DRAW_SEED, {"frangieh": 1, "papalexi": 2}[args.dataset], fi, B, dr])
                pick = rng.choice(len(X_res), size=B, replace=False)
                X, Y = X_res[pick], Y_res[pick]
                P = {"bjs2": es.two_sided_js(X, Y, ctx.Uz, ctx.Ry, ctx.blocks)[0],
                     "cog": cognate_two_sided(X, Y, ops, ctx.Uz, V, ctx.blocks, pb)}
                for a in ARMS:
                    C = np.array([ctx.mapped(P[a], c) for c in gcond])
                    num, _ = cm.scores(C, T, None, None, masks)
                    tot[a][B] += [num[:, 0].sum(), den[:, 0].sum()]
        print(f"[{time.time()-t0:.0f}s] after fold {fi}: " + " | ".join(
            f"B={b}: " + ", ".join(f"{a} {100*tot[a][b][0]/tot[a][b][1]:.1f}" for a in ARMS) for b in budgets), flush=True)
    out = {a: {b: float(v[0] / v[1]) for b, v in d.items() if v[1] > 0} for a, d in tot.items()}
    (cm.HERE / "logs" / f"explore9_{args.dataset}.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
