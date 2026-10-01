#!/usr/bin/env python3
"""Structural law (PLAN.md, G4): does the concentration of dependence in the unpaired eigen-blocks predict the
saving of paired cells?

    python glaw.py predict      # predicted savings from training cells only -> results/law_predicted.json
    python glaw.py test         # with the observed savings -> results/law.json

Predicted saving, from a reservoir of N paired training cells treated as the population: the mean per-cell product
in the unpaired two-sided eigenbasis gives, per block b, a_b^2 = max(||theta_b||^2 - tau_b / N, 0), with tau_b the
trace of the covariance of the per-cell products. The oracle linear shrinkage risk of the blocks,
sum_b a_b^2 tau_b / (B a_b^2 + tau_b), and that of one block of all coefficients, A tau / (B A + tau)
(A = sum_b a_b^2, tau = sum_b tau_b), fall to A / 2 at B_blocks and B_single = tau / A; the prediction is
B_single / B_blocks. Fold 0 of every benchmark data set; for Frangieh, Papalexi and OverCITE-seq the reservoirs of
semipaired/bound_check.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import grun as gr  # noqa: E402

cm, rs, es, pm = gr.cm, gr.rs, gr.es, gr.pm
EARLIER = ("frangieh", "papalexi", "overcite")
PERMUTATIONS, PERM_SEED = 10000, 20261062


def block_moments(X, Y, U, V, rblocks, pblocks):
    """a_b^2 (noise-corrected) and tau_b of every block, and the leading block's shares."""
    N = len(X)
    XU, YV = X @ U, Y @ V
    a2, tau = [], []
    for rows in rblocks:
        for cols in pblocks:
            Z = (XU[:, rows][:, :, None] * YV[:, cols][:, None, :]).reshape(N, -1)
            m = Z.mean(0)
            t = float(np.sum((Z - m) ** 2)) / (N - 1)
            tau.append(t)
            a2.append(max(float(m @ m) - t / N, 0.0))
    return np.array(a2), np.array(tau)


def predicted_saving(a2, tau):
    A, T = a2.sum(), tau.sum()
    if A <= 0:
        return float("nan"), {}
    risk = lambda B: float(np.sum(np.where(a2 > 0, a2 * tau / np.maximum(B * a2 + tau, 1e-300), 0.0)))
    b_blocks = brentq(lambda B: risk(B) - A / 2, 1e-9, 1e12)
    b_single = T / A
    lead = 0
    return b_single / b_blocks, {"B_single": b_single, "B_blocks": b_blocks, "signal_share_leading_block": a2[lead] / A,
                                 "noise_share_leading_block": tau[lead] / T}


def benchmark_fold0(name):
    d = gr.load(gr.source(name))
    fold, part, _ = gr.roles(gr.source(name), d)
    prep = gr.prepare_crossanimal if name == "banc_crossanimal" else gr.prepare_fold
    train, x, y, pop, _, counts = prep(d, 0, fold, part)
    keys = pm.eligible_training(pop, train["part"], gr.MIN_TRAIN_HALF)
    mgr = None if counts else gr.NoCounts()
    if mgr:
        mgr.__enter__()
    try:
        pool, un, X_res, Y_res, _ = cm.setup_fold(train, x, y, pop, keys)
        ctx = rs.build_context(train, x, y, pop, keys, pool, un)
    finally:
        if mgr:
            mgr.__exit__()
    V, _ = es.eigenbasis(ctx.Ry)
    return X_res, Y_res, ctx.Uz, V, ctx.blocks, es.eigen_blocks(V.shape[1], es.PROTEIN_EDGES)


def predict():
    sys.path.insert(0, str(gr.SP))
    import bound_check as bc
    out = {}
    for name in gr.PRIMARY + EARLIER:
        X, Y, U, V, rb, pb = bc.load(name) if name in EARLIER else benchmark_fold0(name)
        a2, tau = block_moments(X, Y, U, V, rb, pb)
        s, det = predicted_saving(a2, tau)
        out[name] = {"predicted_saving": s, "reservoir_cells": int(len(X)), **det}
        print(name, round(s, 2), {k: round(v, 3) for k, v in det.items()}, flush=True)
    (gr.RES / "law_predicted.json").write_text(json.dumps(out, indent=1) + "\n")


def needed_curve(rf, budgets, target):
    return gr.needed(rf, target, budgets)[0]


def observed_earlier(name):
    """Saving against the paired-only envelope at half the reference's recovered fraction, from stored curves."""
    SPR = gr.SP / "results"
    if name == "overcite":
        o = json.loads((gr.SP / "external/results/overcite_plan_estimator.json").read_text())
        rf, budgets, ref = o["rf"], o["budgets"], o["reference_rf"]["paired_all/all"]
    else:
        res = json.loads((SPR / f"{name}.json").read_text())
        budgets = res["budgets"]
        den = sum(r["den"][0] for r in res["rows"])
        arms = sorted({a for r in res["rows"] for a in r["num"]})
        rf = {}
        for a in arms:
            if all(str(B) in res["rows"][0]["num"][a] for B in budgets):
                rf[a] = [sum(r["num"][a][str(B)][0] for r in res["rows"]) / den for B in budgets]
        ref = sum(r["num"]["paired_all"]["0"][0] for r in res["rows"]) / den
    env = np.nanmax(np.array([rf[a] for a in gr.PAIRED_ONLY if a in rf]), axis=0)
    t = 0.5 * ref
    return needed_curve(env, budgets, t) / needed_curve(rf["bjs2/mapped"], budgets, t)


def test():
    pred = json.loads((gr.RES / "law_predicted.json").read_text())
    obs = {}
    for name in gr.PRIMARY:
        p = gr.RES / f"{name}.json"
        if p.exists():
            obs[name] = json.loads(p.read_text())["savings"]["paired_only"]["0.5"]["factor"]
    for name in EARLIER:
        obs[name] = observed_earlier(name)
    names = [n for n in obs if n in pred and np.isfinite(pred[n]["predicted_saving"])]
    xp = np.array([pred[n]["predicted_saving"] for n in names])
    yo = np.array([obs[n] for n in names])
    rho = float(spearmanr(xp, yo).correlation)
    rng = np.random.default_rng(PERM_SEED)
    null = np.array([spearmanr(xp, rng.permutation(yo)).correlation for _ in range(PERMUTATIONS)])
    p = float((1 + np.sum(null >= rho)) / (1 + PERMUTATIONS))
    out = {"datasets": names, "predicted": dict(zip(names, xp.tolist())), "observed": dict(zip(names, yo.tolist())),
           "spearman": rho, "p_one_sided": p, "permutations": PERMUTATIONS, "G4_met": bool(p < 0.05),
           "complete": len(names) == len(gr.PRIMARY) + len(EARLIER)}
    (gr.RES / "law.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        if sys.argv[1] == "predict":
            gr.check_freeze()
            predict()
        else:
            gr.check_freeze()
            test()
