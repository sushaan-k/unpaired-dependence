#!/usr/bin/env python3
"""Development tuning of Champollion's entropic epsilon on Frangieh (budget 400); transport between unpaired
RNA and protein cells of each condition's pool (2,000 of each), cross-covariance of the plan as the prediction.

    python champ_tune.py [--draws 2]

Writes logs/champ_tune.json with the recovered fraction for every setting.
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import time
from pathlib import Path

import numpy as np

import common as cm

pm = cm.pm
VENV = Path("/tmp/claude-0/-home-claude/fa2f5e47-7e0a-544e-ba47-35b58380f98c/scratchpad/champ-venv/bin/python")
SCR = Path("/tmp/claude-0/-home-claude/fa2f5e47-7e0a-544e-ba47-35b58380f98c/scratchpad")


def condition_pools(pop, part, x, y, pool, un, n=2000, seed=20261042):
    """Per condition: up to n unpaired RNA cells and n unpaired protein cells of the pool, centred on their
    population means (the same unpaired observations the other methods use)."""
    rng = np.random.default_rng(seed)
    out = {}
    for c in un.conds:
        ks = [k for k in pool.keys if cm.cond_of(k) == c]
        res = {}
        for half, (arr, sd, means) in ((0, (x, pool.sd_x, pool.mx)), (1, (y, pool.sd_y, pool.my))):
            idx = np.flatnonzero(np.isin(pop, ks) & (part == half))
            idx = rng.choice(idx, size=min(n, len(idx)), replace=False)
            res[half] = (arr[idx] - np.array([means[k] for k in pop[idx]])) / sd
        out[c] = res
    return out


def run_champ(X, Y, rec, keys, epsilon, gamma, max_iter, seed, tag):
    inp, outp = SCR / f"champ_{tag}_in.npz", SCR / f"champ_{tag}_out.npz"
    arrays = {"Xb": X, "Yb": Y, "epsilon": epsilon, "gamma": gamma, "max_iter": max_iter, "seed": seed}
    for i, k in enumerate(keys):
        arrays[f"Xg_{i:04d}"], arrays[f"Yg_{i:04d}"] = rec[k]
    np.savez(inp, **arrays)
    subprocess.run([str(VENV), str(cm.HERE / "champ_worker.py"), str(inp), str(outp)], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    z = np.load(outp)
    return np.array([z[f"C_{i:04d}"] for i in range(len(keys))]), float(z["fit_seconds"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=2)
    ap.add_argument("--budget", type=int, default=400)
    ap.add_argument("--eps", type=float, nargs="+", default=[0.3, 1.0, 3.0])
    ap.add_argument("--gamma", type=float, nargs="+", default=[0.001])
    ap.add_argument("--max-iter", type=int, nargs="+", default=[2000])
    ap.add_argument("--out", default="champ_tune.json")
    args = ap.parse_args()
    mod, train, ev = cm.hr.load_dataset("frangieh")
    x, y = pm.features(train)
    pop = pm.keys_of(train)
    keys = pm.eligible_training(pop, train["part"], mod.MIN_TRAIN_HALF)
    names, masks, _ = cm.hr.block_masks(train["genes"], train["proteins"], mod.ENCODING)
    pool, un, X_res, Y_res, cond_res = cm.setup_fold(train, x, y, pop, keys)
    gs = ev["test"]
    use = np.isin(pop, pool.keys) & ~cm.hm.reservoir(train["cell"])
    pools = condition_pools(pop[use], train["part"][use], x[use], y[use], pool, un)
    conds = sorted(pools)
    rec = {c: (pools[c][0], pools[c][1]) for c in conds}
    T = np.array([g["T"] for g in gs])
    den = np.einsum("gpq,bpq->gb", np.array([g["TA"] * g["TB"] for g in gs]), masks)[:, 0].sum()
    results = []
    for eps, gam, it in itertools.product(args.eps, args.gamma, args.max_iter):
        rf, secs = [], []
        for dr in range(args.draws):
            rng = np.random.default_rng([20261034, args.budget, dr])
            pick = rng.choice(len(X_res), size=args.budget, replace=False)
            Cc, s = run_champ(X_res[pick], Y_res[pick], rec, conds, eps, gam, it, dr, "tune")
            C = np.array([Cc[conds.index(g["cond"])] for g in gs])
            num, _ = cm.scores(C, T, None, None, masks)
            rf.append(float(num[:, 0].sum() / den))
            secs.append(s)
        results.append({"epsilon": eps, "gamma": gam, "max_iter": it, "rf": rf, "fit_seconds": secs})
        print(json.dumps(results[-1]), flush=True)
    (cm.HERE / "logs" / args.out).write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    t0 = time.time()
    main()
