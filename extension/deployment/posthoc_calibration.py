#!/usr/bin/env python3
"""Post hoc (after the deployment test was scored): calibration of the noise term under the two centrings of the
paired cells, in the bone-marrow test.

For every fold, budgets of 25 to 200 paired cells are drawn 20 times from the frozen reservoir (seeds
[SEED + 17, fold, budget, draw], distinct from the test's draws). The drawn cells are centred either on their own
population means with the factor sqrt(n/(n-1)) (the test's prespecified centring, drun.own_std) or by
within-population Helmert contrasts (posthoc_contrasts.contrasts). In the other-site reference's bases (latent RNA
and protein eigenbases, the estimator's blocks), each block's mean product and its estimated noise trace
(sp_estimators.noise_terms, as the estimator uses it) are recorded. Calibration is the ratio of the mean estimated
noise trace to the variance of the block mean across draws, the latter divided by the finite-reservoir factor
1 - B/N; summed over blocks and folds. A ratio below one means the estimated noise is too small. Held-out cells are
not read.

    python posthoc_calibration.py    # results_posthoc/calibration.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import drun  # noqa: E402
import posthoc_contrasts as pc  # noqa: E402

es = drun.es
BUDGETS = (25, 50, 100, 200)
DRAWS = 20


def other_context(cite, fd, part, genes, yp):
    r = np.flatnonzero(np.isin(cite["pop"], fd["other"]))
    x, C, lib = drun.rna(cite, r, genes)
    y = drun.prot(cite, r, yp)
    pop = np.array([f"{p}|{c}" for p, c in zip(cite["pop"][r], cite["cond"][r])])
    return drun.context(x, y, C, lib, pop, part[r], drun.outside("deploy-other", cite["cell"][r]), cite["unit"][r])


def block_moments(X, Y, U, V, rblocks, pblocks):
    XU, YV = X @ U, Y @ V
    out = []
    for rows in rblocks:
        for cols in pblocks:
            Z = (XU[:, rows][:, :, None] * YV[:, cols][:, None, :]).reshape(len(X), -1)
            m, tr, _ = es.noise_terms(Z, iters=1)
            out.append((m, tr))
    return out


def main():
    cite, mult = drun.load("bmmc_cite"), drun.load("bmmc_multiome")
    part, half, res = drun.roles(cite)
    acc = {}
    t0 = time.time()
    for fd in drun.folds(cite, mult):
        f = fd["fold"]
        rr = np.flatnonzero(np.isin(cite["pop"], fd["other"]) & (part == 0))
        rp = np.flatnonzero(np.isin(cite["pop"], fd["other"]) & (part == 1))
        genes, yp = drun.panels(cite, mult, rr, rp)
        _, ctx = other_context(cite, fd, part, genes, yp)
        V, _ = es.eigenbasis(ctx.Ry)
        pblocks = es.eigen_blocks(V.shape[1], es.PROTEIN_EDGES)
        rres = np.flatnonzero(np.isin(cite["pop"], fd["paired"]) & res)
        xr, _, _ = drun.rna(cite, rres, genes)
        yr = drun.prot(cite, rres, yp)
        popr = np.array([f"{p}|{c}" for p, c in zip(cite["pop"][rres], cite["cond"][rres])])
        for B in BUDGETS:
            per = {"contrasts": [], "dfcentre": []}
            for dr in range(DRAWS):
                rng = np.random.default_rng([drun.SEED + 17, f, B, dr])
                pick = rng.choice(len(rres), size=B, replace=False)
                x, y, pp = xr[pick], yr[pick], popr[pick]
                hc = pc.contrasts(x, y, pp)
                o = drun.own_std(x, y, pp)
                if hc is None or o is None:
                    continue
                per["contrasts"].append(block_moments(hc[0], hc[1], ctx.Uz, V, ctx.blocks, pblocks))
                per["dfcentre"].append(block_moments(o[0], o[1], ctx.Uz, V, ctx.blocks, pblocks))
            fpc = 1 - B / len(rres)
            for meth, draws in per.items():
                n, nb = len(draws), len(draws[0])
                t = np.array([np.mean([d[b][1] for d in draws]) for b in range(nb)])
                v = np.array([(np.sum(np.stack([d[b][0] for d in draws]) ** 2)
                               - np.sum(np.stack([d[b][0] for d in draws]).sum(0) ** 2) / n) / (n - 1)
                              for b in range(nb)]) / fpc
                a = acc.setdefault(meth, {}).setdefault(str(B), {"t": np.zeros(nb), "v": np.zeros(nb)})
                a["t"] += t
                a["v"] += v
        print(f"fold {f} [{time.time() - t0:.0f}s]", flush=True)
    out = {"draws": DRAWS, "budgets": list(BUDGETS), "calibration": {}}
    for meth, byB in acc.items():
        out["calibration"][meth] = {B: {"ratio_estimated_to_actual": float(a["t"].sum() / a["v"].sum()),
                                        "ratio_by_block": [float(x) for x in a["t"] / a["v"]]}
                                    for B, a in byB.items()}
    out["written"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    (HERE / "results_posthoc" / "calibration.json").write_text(json.dumps(out, indent=1))
    for meth, byB in out["calibration"].items():
        print(meth, {B: round(v["ratio_estimated_to_actual"], 3) for B, v in byB.items()})


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        drun.check_freeze()
        main()
