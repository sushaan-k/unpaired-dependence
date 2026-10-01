#!/usr/bin/env python3
"""Exploration 8 (development data): block James-Stein in a two-sided eigenbasis (RNA latent eigenbasis and protein
eigenbasis of unpaired cells), blocks = RNA eigen-block x protein eigen-block, all mapped.

    python explore8.py frangieh|papalexi [--draws 6] [--folds 21]
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
ARMS = ["bjs", "bjs2_p3", "bjs2_p2_6", "bjs2_p1"]


def variants(X, Y, ctx):
    V, _ = es.eigenbasis(ctx.Ry)
    q = V.shape[1]
    P = {"bjs": es.block_js(X, Y, ctx.Uz, ctx.blocks)[0]}
    for name, edges in (("bjs2_p3", (3,)), ("bjs2_p2_6", (2, 6)), ("bjs2_p1", (1,))):
        pb = es.eigen_blocks(q, edges)
        P[name] = es.block_js2(X, Y, ctx.Uz, V, ctx.blocks, pb)[0]
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--draws", type=int, default=6)
    ap.add_argument("--folds", type=int, default=21)
    args = ap.parse_args()
    budgets = rs.SETTINGS[args.dataset]["budgets"]
    t0 = time.time()
    mod, train, ev = cm.hr.load_dataset(args.dataset)
    x, y = pm.features(train)
    pop = pm.keys_of(train)
    keys = pm.eligible_training(pop, train["part"], mod.MIN_TRAIN_HALF)
    names, masks, _ = cm.hr.block_masks(train["genes"], train["proteins"], mod.ENCODING)
    groups_all = ev["test"]
    folds = [None] if args.dataset == "frangieh" else sorted({g["target"] for g in groups_all})[: args.folds]
    tot = {a: {b: np.zeros(2) for b in budgets} for a in ARMS}
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
                P = variants(X_res[pick], Y_res[pick], ctx)
                for a in ARMS:
                    C = np.array([ctx.mapped(P[a], c) for c in gcond])
                    num, _ = cm.scores(C, T, None, None, masks)
                    tot[a][B] += [num[:, 0].sum(), den[:, 0].sum()]
        if held is None or fi % 5 == 4 or fi == len(folds) - 1:
            print(f"[{time.time()-t0:.0f}s] after fold {fi}: " + " | ".join(
                f"B={b}: " + ", ".join(f"{a} {100*tot[a][b][0]/tot[a][b][1]:.1f}" for a in ARMS)
                for b in budgets if tot["bjs"][b][1] > 0), flush=True)
    out = {a: {b: float(v[0] / v[1]) for b, v in d.items() if v[1] > 0} for a, d in tot.items()}
    (cm.HERE / "logs" / f"explore8_{args.dataset}.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
