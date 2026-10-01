#!/usr/bin/env python3
"""Exploration (development data): transcript-anchored estimation against block James-Stein.

    python explore.py frangieh|papalexi [--draws 5] [--budgets 50,100,200,400,800] [--shape pooled|condition]

Prints recovered fractions (all genes and proteins) and uncorrected relative loss per arm and budget, and
writes logs/explore_<dataset>_<shape>.json.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--budgets", default="50,100,200,400,800")
    ap.add_argument("--shape", default="pooled")
    ap.add_argument("--folds", type=int, default=4)
    args = ap.parse_args()
    budgets = [int(b) for b in args.budgets.split(",")]
    t0 = time.time()
    mod, train, ev = cm.hr.load_dataset(args.dataset)
    x, y = pm.features(train)
    pop = pm.keys_of(train)
    keys = pm.eligible_training(pop, train["part"], mod.MIN_TRAIN_HALF)
    names, masks, _ = cm.hr.block_masks(train["genes"], train["proteins"], mod.ENCODING)
    sets = cm.encoding_sets(train["genes"], train["proteins"], mod.ENCODING)
    print(f"[{time.time()-t0:.0f}s] loaded; encoding genes per protein: {[len(s) for s in sets]}", flush=True)
    groups_all = ev["test"]
    folds = [None] if args.dataset == "frangieh" else sorted({g["target"] for g in groups_all})[: args.folds]
    arms = ["paired_js", "bjs", "cp_sparse", "cp_sparse_res", "cp_union_res"]
    acc = {a: {b: [] for b in budgets} for a in arms}
    for fi, held in enumerate(folds):
        gs = [g for g in groups_all if held is None or g["target"] == held]
        ks = [k for k in keys if held is None or k.split("|")[0] != held]
        pool, un, X_res, Y_res, cond_res = cm.setup_fold(train, x, y, pop, ks)
        T = np.array([g["T"] for g in gs])
        den = np.einsum("gpq,bpq->gb", np.array([g["TA"] * g["TB"] for g in gs]), masks)
        tn = np.einsum("gpq,bpq->gb", T * T, masks)
        gcond = [g["cond"] for g in gs]
        R = un.R["__all__"]
        U0, _ = es.eigenbasis(R["Rx"])
        blocks = es.eigen_blocks(U0.shape[0])
        ops = {"sparse": es.cognate_operator(R["Rz"], sets, union=False),
               "union": es.cognate_operator(R["Rz"], sets, union=True)}
        cops = {c: {"sparse": es.cognate_operator(un.R[c]["Rz"], sets, union=False),
                    "union": es.cognate_operator(un.R[c]["Rz"], sets, union=True)} for c in un.conds}
        print(f"[{time.time()-t0:.0f}s] fold {held}: {len(gs)} groups, reservoir {len(X_res)}", flush=True)
        for B in budgets:
            for dr in range(args.draws):
                rng = np.random.default_rng([20261031, fi, B, dr])
                pick = rng.choice(len(X_res), size=min(B, len(X_res)), replace=False)
                X, Y = X_res[pick], Y_res[pick]
                preds = {}
                preds["paired_js"], _ = es.js_matrix(X, Y)
                preds["bjs"], _ = es.block_js(X, Y, U0, blocks)
                if args.shape == "pooled":
                    preds["cp_sparse"], _ = es.cognate_estimate(X, Y, ops["sparse"], U0, blocks, residual=False)
                    preds["cp_sparse_res"], _ = es.cognate_estimate(X, Y, ops["sparse"], U0, blocks)
                    preds["cp_union_res"], _ = es.cognate_estimate(X, Y, ops["union"], U0, blocks)
                else:
                    # condition shapes: propagate the pooled cognate covariances through each condition's Rz
                    for a, kind, res in (("cp_sparse", "sparse", False), ("cp_sparse_res", "sparse", True),
                                         ("cp_union_res", "union", True)):
                        per = {c: es.cognate_estimate(X, Y, cops[c][kind], U0, blocks, residual=res)[0]
                               for c in un.conds}
                        preds[a] = np.array([per[c] for c in gcond])
                for a in arms:
                    num, loss = cm.scores(preds[a], T, None, None, masks)
                    acc[a][B].append((num[:, 0].sum(), den[:, 0].sum(), loss[:, 0].sum(), tn[:, 0].sum(),
                                      num[:, 1].sum(), den[:, 1].sum()))
            line = []
            for a in arms:
                v = np.array(acc[a][B][-args.draws:])
                line.append(f"{a} {100*v[:,0].sum()/v[:,1].sum():.1f}/{v[:,2].sum()/v[:,3].sum():.3f}")
            print(f"[{time.time()-t0:.0f}s]  B={B}: " + "  ".join(line), flush=True)
    out = {}
    for a in arms:
        out[a] = {}
        for b in budgets:
            v = np.array(acc[a][b])
            out[a][b] = {"rf_all": float(v[:, 0].sum() / v[:, 1].sum()), "loss_rel": float(v[:, 2].sum() / v[:, 3].sum()),
                         "rf_ifn": float(v[:, 4].sum() / v[:, 5].sum())}
    (cm.HERE / "logs" / f"explore_{args.dataset}_{args.shape}.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out))


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
