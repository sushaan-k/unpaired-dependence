#!/usr/bin/env python3
"""Champollion in the development comparison, on the same paired draws as run_study.py.

    python champ_dev.py frangieh|papalexi [--draws 3] [--folds 21] [--budgets ...] [--cs 1.0 1.5 2.25]

Champollion (epsilon 1, the default; 2,000 iterations) is fitted on the B paired cells with lasso weight
gamma = c / sqrt(B) for each c in the grid, and transports between the unpaired RNA and protein cells of each
condition's pool (up to 2,000 of each); the plan's cross-correlation is the estimate for that condition. The best c
at each budget is chosen on the evaluation groups themselves, which favours Champollion. Writes
results/champ_<dataset>.json (per-group recovered-fraction terms, as in results/<dataset>.json).
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

import champ_tune as ct
import common as cm
import run_study as rs

pm = cm.pm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--folds", type=int, default=21)
    ap.add_argument("--budgets", type=int, nargs="+", default=[50, 100, 200, 400, 800])
    ap.add_argument("--cs", type=float, nargs="+", default=[1.0, 1.5, 2.25])
    args = ap.parse_args()
    t0 = time.time()
    mod, train, ev = cm.hr.load_dataset(args.dataset)
    x, y = pm.features(train)
    pop = pm.keys_of(train)
    keys = pm.eligible_training(pop, train["part"], mod.MIN_TRAIN_HALF)
    names, masks, _ = cm.hr.block_masks(train["genes"], train["proteins"], mod.ENCODING)
    groups_all = ev["test"]
    folds = [None] if args.dataset == "frangieh" else sorted({g["target"] for g in groups_all})[: args.folds]
    rows = []
    for fi, held in enumerate(folds):
        gs = [g for g in groups_all if held is None or g["target"] == held]
        ks = [k for k in keys if held is None or k.split("|")[0] != held]
        pool, un, X_res, Y_res, cond_res = cm.setup_fold(train, x, y, pop, ks)
        use = np.isin(pop, pool.keys) & ~cm.hm.reservoir(train["cell"])
        pools = ct.condition_pools(pop[use], train["part"][use], x[use], y[use], pool, un)
        conds = sorted(pools)
        rec = {c: (pools[c][0], pools[c][1]) for c in conds}
        T = np.array([g["T"] for g in gs])
        den = np.einsum("gpq,bpq->gb", np.array([g["TA"] * g["TB"] for g in gs]), masks)
        tn = np.einsum("gpq,bpq->gb", T * T, masks)
        num = {B: {c: np.zeros_like(den) for c in args.cs} for B in args.budgets}
        loss = {B: {c: np.zeros_like(den) for c in args.cs} for B in args.budgets}
        done = []
        for B in args.budgets:
            if B > len(X_res):
                continue
            done.append(B)
            for dr in range(args.draws):
                rng = np.random.default_rng([rs.DRAW_SEED, {"frangieh": 1, "papalexi": 2}[args.dataset], fi, B, dr])
                pick = rng.choice(len(X_res), size=B, replace=False)
                for c in args.cs:
                    Cc, _ = ct.run_champ(X_res[pick], Y_res[pick], rec, conds, 1.0, c / np.sqrt(B), 2000, dr,
                                         f"dev_{args.dataset}")
                    C = np.array([Cc[conds.index(g["cond"])] for g in gs])
                    n_, l_ = cm.scores(C, T, None, None, masks)
                    num[B][c] += n_ / args.draws
                    loss[B][c] += l_ / args.draws
            print(f"[{time.time() - t0:.0f}s] fold {held} B={B}: " + ", ".join(
                f"c={c}: {100 * num[B][c][:, 0].sum() / den[:, 0].sum():.1f}" for c in args.cs), flush=True)
            partial = rows + [{"key": g["key"], "target": g["target"], "cond": g["cond"], "den": den[i].tolist(),
                               "tn": tn[i].tolist(),
                               "num": {str(b): {str(c): num[b][c][i].tolist() for c in args.cs} for b in done},
                               "loss": {str(b): {str(c): loss[b][c][i].tolist() for c in args.cs} for b in done}}
                              for i, g in enumerate(gs)]
            (cm.HERE / "results" / f"champ_{args.dataset}_partial.json").write_text(json.dumps(
                {"dataset": args.dataset, "budgets_done": done, "fold": fi, "rows": partial}) + "\n")
        for i, g in enumerate(gs):
            rows.append({"key": g["key"], "target": g["target"], "cond": g["cond"], "den": den[i].tolist(),
                         "tn": tn[i].tolist(),
                         "num": {str(B): {str(c): num[B][c][i].tolist() for c in args.cs} for B in done},
                         "loss": {str(B): {str(c): loss[B][c][i].tolist() for c in args.cs} for B in done}})
    out = {"dataset": args.dataset, "budgets": args.budgets, "cs": args.cs, "draws": args.draws, "folds": folds,
           "epsilon": 1.0, "max_iter": 2000, "rows": rows, "seconds": round(time.time() - t0)}
    (cm.HERE / "results" / f"champ_{args.dataset}.json").write_text(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
