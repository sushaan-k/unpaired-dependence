#!/usr/bin/env python3
"""Pilot on the training file only (no held-out cell is read): three folds of training ORFs, each held out in turn,
with every arm fitted on the other ORFs, to check that the pipeline runs on this panel and to see the range of
recovered fractions before the external plan is frozen.

    python pilot.py [--draws 5]

Targets: the within-condition cross-correlation of the held-out ORFs' cells, each cell centred on its ORF x
condition mean and pooled within condition (sub-halves: the assay-half labels). Writes results/pilot.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(HERE.parent))

import odata as od  # noqa: E402
import common as cm  # noqa: E402
import run_study as rs  # noqa: E402
import targets as tg  # noqa: E402

pm = cm.pm
BUDGETS = (25, 50, 100, 200)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--explore", choices=("explore6", "explore8"), help="only the variants of that exploration")
    args = ap.parse_args()
    t0 = time.time()
    genes = json.loads((HERE / "results/panel.json").read_text())["genes"]
    train = od.load("train", genes)
    # the pilot drops panel genes detected in fewer than 5% of training cells (CD4, ITGAX), which can be constant in a
    # fold's pool; the external run uses the full panel
    # and genes undetected in the training cells of any fold (the transgene of a held-out training ORF)
    pop = pm.keys_of(train)
    keys = pm.eligible_training(pop, train["part"], od.MIN_TRAIN_HALF)
    orfs = sorted({k.split("|")[0] for k in keys},
                  key=lambda t: hashlib.sha256(f"overcite-pilot-v1|{t}".encode()).hexdigest())
    folds = [orfs[i::3] for i in range(3)]
    keep = (train["counts"] > 0).mean(0) >= 0.05
    target = np.array([p.split("|")[0] for p in pop])
    for held in folds:
        keep &= (train["counts"][~np.isin(target, held)] > 0).sum(0) >= 10
    genes = [g for g, k in zip(genes, keep) if k]
    train["counts"], train["genes"] = train["counts"][:, keep], genes
    print("pilot genes:", len(genes), flush=True)
    x, y = pm.features(train)
    names, masks, _ = cm.hr.block_masks(genes, train["proteins"], od.ENCODING)
    arms = [f"{d}/{m}" for d in rs.DENOISERS for m in rs.MODES]
    if args.explore:
        import importlib
        ex = importlib.import_module(args.explore)
        arms = [f"{a}/mapped" for a in ex.ARMS]
    tot = {a: {b: np.zeros(2) for b in BUDGETS} for a in arms}
    loss = {a: {b: 0.0 for b in BUDGETS} for a in arms}
    tn_tot = 0.0
    for fi, held in enumerate(folds):
        ks = [k for k in keys if k.split("|")[0] not in held]
        pool, un, X_res, Y_res, cond_res = cm.setup_fold(train, x, y, pop, ks)
        ctx = rs.build_context(train, x, y, pop, ks, pool, un)
        hm_ = np.isin(np.array([p.split("|")[0] for p in pop]), held)
        stats = tg.group_stats(train["counts"][hm_], train["library"][hm_], train["y"][hm_], pop[hm_],
                               train["part"][hm_])
        T = tg.pooled(stats)
        conds = sorted(T)
        den = sum(float(np.sum(T[c]["TA"] * T[c]["TB"])) for c in conds)
        tn_tot += sum(float(np.sum(T[c]["T"] ** 2)) for c in conds)
        print(f"fold {fi}: held {held}; reservoir {len(X_res)}; pool keys {len(pool.keys)}; "
              f"reliability {ctx.reliability}", flush=True)
        for B in BUDGETS:
            if B > len(X_res):
                continue
            for dr in range(args.draws):
                rng = np.random.default_rng([20261043, fi, B, dr])
                pick = rng.choice(len(X_res), size=B, replace=False)
                X, Y, cc = X_res[pick], Y_res[pick], cond_res[pick]
                crng = np.random.default_rng([20261044, fi, B, dr])
                if args.explore:
                    P = ex.variants(X, Y, ctx, crng) if args.explore == "explore6" else ex.variants(X, Y, ctx)
                    for a in ex.ARMS:
                        for c in conds:
                            C = ctx.mapped(P[a], c)
                            tot[f"{a}/mapped"][B] += [2 * np.sum(C * T[c]["T"]) - np.sum(C * C), 0.0]
                            loss[f"{a}/mapped"][B] += float(np.sum((C - T[c]["T"]) ** 2))
                    continue
                for name in rs.DENOISERS:
                    P = rs.denoise(name, X, Y, ctx.Rx, ctx.Ry, ctx.n_x, ctx.n_y, ctx.Uz, ctx.blocks, crng)
                    for c in conds:
                        m = cc == c
                        d = ctx.cond[c]
                        per = (rs.denoise(name, X[m], Y[m], d["Rx"], d["Ry"], d["n_x"], d["n_y"], d["Uz"], ctx.blocks,
                                          crng) if m.sum() >= 6 else np.zeros_like(P))
                        for mode, C in (("pooled", P), ("mapped", ctx.mapped(P, c)), ("percond", per)):
                            a = f"{name}/{mode}"
                            tot[a][B] += [2 * np.sum(C * T[c]["T"]) - np.sum(C * C), 0.0]
                            loss[a][B] += float(np.sum((C - T[c]["T"]) ** 2))
            for a in arms:
                tot[a][B][1] += den * args.draws
        print(f"[{time.time() - t0:.0f}s] fold {fi} done", flush=True)
    out = {"budgets": list(BUDGETS), "draws": args.draws, "folds": folds,
           "rf": {a: [float(tot[a][b][0] / tot[a][b][1]) if tot[a][b][1] else None for b in BUDGETS] for a in arms},
           "loss_rel": {a: [loss[a][b] / args.draws / tn_tot for b in BUDGETS] for a in arms}}
    (HERE / f"results/pilot{'_' + args.explore if args.explore else ''}.json").write_text(json.dumps(out, indent=1) + "\n")
    for a in arms:
        print(f"{a:18s} " + " ".join(f"{100 * v:6.1f}" if v is not None else "   -  " for v in out["rf"][a]))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
