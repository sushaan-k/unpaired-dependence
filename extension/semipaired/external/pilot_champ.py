#!/usr/bin/env python3
"""Champollion's lasso weight for each budget, chosen on the training ORFs only (the pilot folds of pilot.py; no
held-out cell is read). The setting most favourable to Champollion on this panel is carried into the external
plan: gamma = c / sqrt(B), with c chosen for each budget from the grid.

    python pilot_champ.py [--draws 1] [--cs ...] [--budgets ...]

Writes results/pilot_champ.json: recovered fraction (pooled within-condition targets of the held-out training
ORFs, summed over folds) for every budget and c.
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
import champ_tune as ct  # noqa: E402
import targets as tg  # noqa: E402

pm = cm.pm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=1)
    ap.add_argument("--cs", type=float, nargs="+", default=[0.67, 1.0, 1.5, 2.25, 3.4])
    ap.add_argument("--budgets", type=int, nargs="+", default=[25, 50, 100, 200])
    ap.add_argument("--out", default="pilot_champ.json")
    args = ap.parse_args()
    t0 = time.time()
    genes = json.loads((HERE / "results/panel.json").read_text())["genes"]
    train = od.load("train", genes)
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
    x, y = pm.features(train)
    tot = {(B, g): np.zeros(2) for B in args.budgets for g in args.cs}
    for fi, held in enumerate(folds):
        ks = [k for k in keys if k.split("|")[0] not in held]
        pool, un, X_res, Y_res, cond_res = cm.setup_fold(train, x, y, pop, ks)
        use = np.isin(pop, pool.keys) & ~cm.hm.reservoir(train["cell"])
        pools = ct.condition_pools(pop[use], train["part"][use], x[use], y[use], pool, un)
        conds = sorted(pools)
        hm_ = np.isin(target, held)
        T = tg.pooled(tg.group_stats(train["counts"][hm_], train["library"][hm_], train["y"][hm_], pop[hm_],
                                     train["part"][hm_]))
        den = sum(float(np.sum(T[c]["TA"] * T[c]["TB"])) for c in conds)
        rec = {c: (pools[c][0], pools[c][1]) for c in conds}
        for B in args.budgets:
            if B > len(X_res):
                continue
            for dr in range(args.draws):
                rng = np.random.default_rng([20261043, fi, B, dr])
                pick = rng.choice(len(X_res), size=B, replace=False)
                for g in args.cs:
                    Cc, _ = ct.run_champ(X_res[pick], Y_res[pick], rec, conds, 1.0, g / np.sqrt(B), 2000, dr, "pilot")
                    num = sum(2 * np.sum(C * T[c]["T"]) - np.sum(C * C) for C, c in zip(Cc, conds))
                    tot[(B, g)] += [num, den]
            print(f"[{time.time() - t0:.0f}s] fold {fi} B={B}: " + ", ".join(
                f"c={g}: {100 * tot[(B, g)][0] / tot[(B, g)][1]:.1f}" for g in args.cs), flush=True)
    out = {str(B): {str(g): float(tot[(B, g)][0] / tot[(B, g)][1]) for g in args.cs} for B in args.budgets}
    (HERE / "results" / args.out).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
