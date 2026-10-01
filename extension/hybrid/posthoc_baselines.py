#!/usr/bin/env python3
"""Post hoc (added after the main comparison had been scored): reference-regression baselines.

    python posthoc_baselines.py frangieh|papalexi

Same folds, reservoir and draws (seeds) as run.py. Two arms that also use the
recipient's RNA covariance, as reference mapping does:

  ridge_paired    protein on RNA by ridge regression in the B paired cells
                  (penalty by two-fold cross-validation within them); prediction
                  Rx_g W' for each recipient g.
  ridge_unpaired  the same normal equations with the pooled unpaired RNA
                  correlation in place of the paired cells' RNA covariance.

and two diagnostic variants of the hybrid and its control whose program part
uses the pooled unpaired RNA correlation instead of each recipient's own
(noisier) RNA correlation: hybrid_pooled and hybrid_pc_pooled.

Writes results/<dataset>_posthoc.json with per-group endpoint terms.
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

import hmethods as hm
import run

pm = hm.pm
GRID = 10.0 ** np.arange(-3, 2.01, 0.5)   # penalty in units of the mean RNA variance (1)


def cv_ridge(X, Y, Rx_pool, rng, unpaired):
    n = len(X)
    idx = rng.permutation(n)
    halves = [idx[: n // 2], idx[n // 2:]]
    score = np.zeros(len(GRID))
    for a, b in ((0, 1), (1, 0)):
        Xa, Ya = X[halves[a]], Y[halves[a]]
        Gram = Rx_pool if unpaired else Xa.T @ Xa / len(Xa)
        Cb = X[halves[b]].T @ Y[halves[b]] / len(halves[b])
        Ca = Xa.T @ Ya / len(Xa)
        for i, lam in enumerate(GRID):
            Wt = np.linalg.solve(Gram + lam * np.eye(len(Gram)), Ca)
            P = Rx_pool @ Wt
            score[i] += 2 * np.sum(P * Cb) - np.sum(P * P)
    lam = GRID[int(np.argmax(score))]
    Gram = Rx_pool if unpaired else X.T @ X / n
    return np.linalg.solve(Gram + lam * np.eye(len(Gram)), X.T @ Y / n), float(lam)


def main():
    dataset = sys.argv[1]
    t0 = time.time()
    mod, train, ev = run.load_dataset(dataset)
    x, y = pm.features(train)
    pop = pm.keys_of(train)
    keys = pm.eligible_training(pop, train["part"], mod.MIN_TRAIN_HALF)
    names, masks, _ = run.block_masks(train["genes"], train["proteins"], mod.ENCODING)
    settings = run.SETTINGS[dataset]
    result = {"dataset": dataset, "blocks": names, "note": "post hoc; added after the main comparison was scored",
              "parts": {}}
    for part in ("test", "nt"):
        groups_all = ev[part]
        folds = sorted({g["target"] for g in groups_all}) if (dataset == "papalexi" and part == "test") else [None]
        rows = []
        for fi, held in enumerate(folds):
            fold_id = fi + (100 if part == "nt" else 0)
            gs = [g for g in groups_all if held is None or g["target"] == held]
            ks = [k for k in keys if k.split("|")[0] != held] if held is not None else keys
            use = np.isin(pop, ks)
            res = hm.reservoir(train["cell"]) & use
            pl = use & ~res
            pool = hm.Pool(x[pl], y[pl], pop[pl], train["part"][pl], train["guide"][pl], ks)
            rmask = res & np.isin(pop, pool.keys)
            X_res, Y_res = pool.centre_cells(x[rmask], y[rmask], pop[rmask])
            T = np.array([g["T"] for g in gs])
            Rxg = np.array([g["Rx"] for g in gs])
            na = hm.noise_aware(pool)
            U0, _ = hm.eigenbasis(pool.Rx)
            V, _ = hm.eigenbasis(pool.Ry)
            blocks = hm.eigen_blocks(U0.shape[0])
            progs = {"hybrid_pooled": (na["U"], na["W"] @ na["U"]),
                     "hybrid_pc_pooled": (U0[:, :na["dim"]], np.zeros((V.shape[0], na["dim"])))}
            acc = {}
            for B in settings["budgets"]:
                if B > len(X_res):
                    continue
                for arm in ("ridge_paired", "ridge_unpaired", "hybrid_pooled", "hybrid_pc_pooled"):
                    acc.setdefault(arm, {})[B] = np.zeros((len(gs), len(names)))
                for dr in range(settings["draws"]):
                    rng = np.random.default_rng([run.DRAW_SEED, {"frangieh": 1, "papalexi": 2}[dataset], fold_id, B, dr])
                    pick = rng.choice(len(X_res), size=B, replace=False)
                    X, Y = X_res[pick], Y_res[pick]
                    rng2 = np.random.default_rng([run.DRAW_SEED + 1, fold_id, B, dr])
                    for arm, unp in (("ridge_paired", False), ("ridge_unpaired", True)):
                        Wt, _ = cv_ridge(X, Y, pool.Rx, rng2, unp)
                        acc[arm][B] += run.score(Rxg @ Wt, gs, masks, T, None)
                    for arm, (U, Wu) in progs.items():
                        Wpost, Yres, _ = hm.program_regression(X, Y, U, Wu)
                        Ar, vr = hm.basis_coefficients(X, Yres, U0, V)
                        post_r, _ = hm.block_js(Ar, vr, np.zeros_like(Ar), blocks)
                        acc[arm][B] += run.score(pool.Rx @ U @ Wpost.T + U0 @ post_r @ V.T, gs, masks, T, None)
                for arm in acc:
                    acc[arm][B] /= settings["draws"]
            for i, g in enumerate(gs):
                rows.append({"key": g["key"], "target": g["target"], "fold": held,
                             "num": {a: {str(B): v[i].tolist() for B, v in d.items()} for a, d in acc.items()}})
            print(f"[{time.time() - t0:5.0f}s] {part} {held}", flush=True)
        result["parts"][part] = {"rows": rows}
    (hm.HERE / "results" / f"{dataset}_posthoc.json").write_text(json.dumps(result) + "\n")


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
