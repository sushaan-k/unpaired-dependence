#!/usr/bin/env python3
"""Checks the condition of the risk bound (Proposition 1, THEORY.md): the effective dimension tr(S_b)/lambda_max(S_b)
of the per-cell product noise in every block of the proposed estimator (latent RNA eigen-block x protein
eigen-block of unpaired cells), at several budgets.

    python effective_dimension.py frangieh|papalexi

Writes logs/effective_dimension_<dataset>.json.
"""

from __future__ import annotations

import json
import sys

import numpy as np
from threadpoolctl import threadpool_limits

import common as cm
import run_study as rs
import sp_estimators as es

pm = cm.pm


def main():
    ds = sys.argv[1]
    mod, train, ev = cm.hr.load_dataset(ds)
    x, y = pm.features(train)
    pop = pm.keys_of(train)
    keys = pm.eligible_training(pop, train["part"], mod.MIN_TRAIN_HALF)
    held = None if ds == "frangieh" else sorted({g["target"] for g in ev["test"]})[0]
    ks = [k for k in keys if held is None or k.split("|")[0] != held]
    pool, un, X_res, Y_res, cond_res = cm.setup_fold(train, x, y, pop, ks)
    Uz, _ = es.eigenbasis(un.R["__all__"]["Rz"])
    V, _ = es.eigenbasis(un.R["__all__"]["Ry"])
    var = (Y_res @ V).var(0)
    V = V[:, var > 1e-8 * var.max()]     # centred log-ratios leave one protein direction without variance
    blocks = [(r, c) for r in es.eigen_blocks(len(Uz)) for c in es.eigen_blocks(V.shape[1], es.PROTEIN_EDGES)]
    out = {}
    for B in (50, 100, 400, 1600):
        if B > len(X_res):
            continue
        ratios = []
        for dr in range(5):
            rng = np.random.default_rng([20261041, B, dr])
            pick = rng.choice(len(X_res), size=B, replace=False)
            XU, YV = X_res[pick] @ Uz, Y_res[pick] @ V
            row = []
            for rows, cols in blocks:
                Z = (XU[:, rows][:, :, None] * YV[:, cols][:, None, :]).reshape(B, -1)
                _, tr, lam = es.noise_terms(Z, iters=60)
                row.append(tr / lam)
            ratios.append(row)
        r = np.array(ratios)
        out[B] = {"blocks": [[len(a), len(b)] for a, b in blocks], "effective_dimension_median": np.median(r, 0).tolist(),
                  "effective_dimension_min": r.min(0).tolist()}
        print(B, out[B], flush=True)
    (cm.HERE / "logs" / f"effective_dimension_{ds}.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
