#!/usr/bin/env python3
"""Does the risk bound of Proposition S10 cover the implemented estimator? A finite-population check.

    python bound_check.py frangieh|papalexi|overcite [--draws 200]

The paired calibration reservoir of a data set is treated as the population: theta is the mean per-cell product
of all reservoir cells (centred and scaled as in the study), and the covariance of the per-cell products in each
block gives tau_b and rho_b. The bases come from the unpaired pool, which is disjoint from the reservoir, as in the
estimator. For each budget B, B cells are drawn with replacement (so that draws are independent, as the
proposition assumes). Three rules are applied to every block: the implemented one (positive-part factor, trace
estimated from the drawn cells' products), the proposition's idealized one (known trace, no positive part) and
the unshrunk mean. Their mean squared errors over draws are compared with the bound
sum_b ||a_b||^2 t_b / (||a_b||^2 + t_b) + 4 l_b, with t_b = tau_b / B and l_b = rho_b / B.
Writes logs/bound_check_<dataset>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import common as cm
import run_study as rs
import sp_estimators as es

pm = cm.pm
HERE = Path(__file__).resolve().parent


def load(dataset):
    """Reservoir (X_res, Y_res), bases and blocks exactly as the estimator uses them."""
    if dataset == "overcite":
        sys.path.insert(0, str(HERE / "external"))
        import odata as od
        genes = json.loads((HERE / "external/results/panel.json").read_text())["genes"]
        train = od.load("train", genes)
        minimum = od.MIN_TRAIN_HALF
    else:
        mod, train, _ = cm.hr.load_dataset(dataset)
        minimum = mod.MIN_TRAIN_HALF
    x, y = pm.features(train)
    pop = pm.keys_of(train)
    keys = pm.eligible_training(pop, train["part"], minimum)
    pool, un, X_res, Y_res, _ = cm.setup_fold(train, x, y, pop, keys)
    ctx = rs.build_context(train, x, y, pop, keys, pool, un)
    V, _ = es.eigenbasis(ctx.Ry)
    return X_res, Y_res, ctx.Uz, V, ctx.blocks, es.eigen_blocks(V.shape[1], es.PROTEIN_EDGES)


def check(X, Y, U, V, rblocks, pblocks, budgets, draws, seed=20261050):
    N = len(X)
    XU, YV = X @ U, Y @ V
    blocks = [(r, c) for r in rblocks for c in pblocks]
    pops = []
    for rows, cols in blocks:
        Z = (XU[:, rows][:, :, None] * YV[:, cols][:, None, :]).reshape(N, -1)
        a = Z.mean(0)
        Zc = Z - a
        ev = np.linalg.eigvalsh(Zc.T @ Zc / N)
        pops.append({"a": a, "a2": float(a @ a), "tau": float(ev.sum()), "rho": float(max(ev[-1], 1e-300))})
    theta2 = sum(p["a2"] for p in pops)
    tau_all = sum(q["tau"] for q in pops)
    # a block formed with the protein direction that centred log-ratios leave without variance has signal and
    # noise at rounding level (Papalexi: RNA blocks x the fourth protein direction); its ratio tau/rho is not
    # meaningful and it is excluded from the effective-dimension condition (as in effective_dimension.py)
    res = {"reservoir_cells": N, "theta_norm2": theta2,
           "blocks": [{"rows": len(r), "cols": len(c), "signal_share": p["a2"] / theta2,
                       "noise_share": p["tau"] / tau_all, "effective_dimension": p["tau"] / p["rho"],
                       "null_direction": bool(p["tau"] <= 1e-12 * tau_all)} for (r, c), p in zip(blocks, pops)],
           "budgets": {}}
    rng = np.random.default_rng(seed)
    for B in budgets:
        bound = sum(p["a2"] * (p["tau"] / B) / (p["a2"] + p["tau"] / B) + 4 * p["rho"] / B for p in pops)
        err = {"implemented": 0.0, "idealized": 0.0, "unshrunk": 0.0}
        for _ in range(draws):
            pick = rng.integers(0, N, size=B)
            xu, yv = XU[pick], YV[pick]
            for (rows, cols), p in zip(blocks, pops):
                Z = (xu[:, rows][:, :, None] * yv[:, cols][:, None, :]).reshape(B, -1)
                m, tr, lam = es.noise_terms(Z)
                n2 = max(float(m @ m), 1e-300)
                for name, f in (("implemented", es.js_factor(n2, tr, lam)), ("idealized", 1 - (p["tau"] / B) / n2),
                                ("unshrunk", 1.0)):
                    d = f * m - p["a"]
                    err[name] += float(d @ d)
        risk = {k: v / draws for k, v in err.items()}
        res["budgets"][str(B)] = {"bound": bound, **{f"risk_{k}": v for k, v in risk.items()},
                                  "bound_covers_implemented": bool(risk["implemented"] <= bound),
                                  "bound_covers_idealized": bool(risk["idealized"] <= bound),
                                  "relative_to_theta": {"bound": bound / theta2,
                                                        **{k: v / theta2 for k, v in risk.items()}}}
    res["condition_tau_ge_4rho"] = bool(all(b["effective_dimension"] >= 4 for b in res["blocks"]
                                            if not b["null_direction"]))
    res["min_effective_dimension"] = min(b["effective_dimension"] for b in res["blocks"] if not b["null_direction"])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--budgets", type=int, nargs="+", default=[25, 50, 100, 200, 400, 800])
    args = ap.parse_args()
    X, Y, U, V, rb, pb = load(args.dataset)
    budgets = [b for b in args.budgets if b <= len(X)]
    res = check(X, Y, U, V, rb, pb, budgets, args.draws)
    res.update({"dataset": args.dataset, "draws": args.draws})
    for B in budgets:
        r = res["budgets"][str(B)]["relative_to_theta"]
        print(f"B={B}: bound {r['bound']:.3f}, implemented {r['implemented']:.3f}, idealized {r['idealized']:.3f}, "
              f"unshrunk {r['unshrunk']:.3f} (relative to ||theta||^2)", flush=True)
    print("effective dimensions:", [round(b["effective_dimension"], 1) for b in res["blocks"] if not b["null_direction"]],
          "; condition tau >= 4 rho:", res["condition_tau_ge_4rho"])
    (HERE / "logs" / f"bound_check_{args.dataset}.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
