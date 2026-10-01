#!/usr/bin/env python3
"""Design D3 of PLAN.md (amendments 1 and 3): pairing-free channel estimation.

Shared-channel model across biopsy samples s:  y | x ~ N(W x + b, diag(psi)).
Only within-assay moments of each training sample are used; cell pairing is
never used.

pf_means   ecological (between-sample) ridge regression of protein means on
           RNA means; closed form.
pf_lowrank Gaussian likelihood of each sample's protein moments given its RNA
           moments, S_s = W Sx_s W^T + diag(psi), W = U V^T of rank 8.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits

import colon_scale as cs
from gaussian_transfer import Marginal, transfer

MEAN_RIDGES = (1e-3, 1e-2, 1e-1, 1.0)
RANK = 8


def sample_moments(data, samples):
    out = []
    for sample in samples:
        m = data["sample"] == sample
        n, mx, my, Sx, Sy, _ = cs.moments(data["x"][m], data["y"][m])   # cross-moment discarded
        out.append((n, mx, my, Sx, Sy))
    return out


def ecological(units, ridge_scale):
    n = np.array([u[0] for u in units], float)
    MX = np.array([u[1] for u in units])
    MY = np.array([u[2] for u in units])
    w = n / n.sum()
    xbar, ybar = w @ MX, w @ MY
    cx, cy = MX - xbar, MY - ybar
    Gx = (cx * w[:, None]).T @ cx
    ridge = ridge_scale * np.trace(Gx)
    W = np.linalg.solve(Gx + ridge * np.eye(len(xbar)), (cx * w[:, None]).T @ cy).T   # q x p
    b = ybar - W @ xbar
    resid = sum(u[0] * np.diag(u[4] - W @ u[3] @ W.T) for u in units) / n.sum()
    floor = 0.05 * np.mean([np.mean(np.diag(u[4])) for u in units])
    return W, b, np.maximum(resid, floor), ridge


def choose_mean_ridge(units):
    folds = [list(range(k, len(units), 3)) for k in range(3)]
    errors = []
    for scale in MEAN_RIDGES:
        total = 0.0
        for fold in folds:
            train = [units[i] for i in range(len(units)) if i not in fold]
            W, b, _, _ = ecological(train, scale)
            total += sum(u[0] * np.sum((u[2] - W @ u[1] - b) ** 2) for i, u in enumerate(units) if i in fold)
        errors.append(total)
    return MEAN_RIDGES[int(np.argmin(errors))], errors


def lowrank_negloglik(z, units, p, q, ridge):
    U = z[: q * RANK].reshape(q, RANK)
    V = z[q * RANK: q * RANK + p * RANK].reshape(p, RANK)
    b = z[q * RANK + p * RANK: q * RANK + p * RANK + q]
    logpsi = z[q * RANK + p * RANK + q:]
    psi = np.exp(logpsi)
    total = sum(u[0] for u in units)
    value = 0.5 * ridge * total * (np.sum(U * U) + np.sum(V * V))
    gU, gV, gb, glog = ridge * total * U, ridge * total * V, np.zeros(q), np.zeros(q)
    for n, mx, my, Sx, Sy in units:
        SxV = Sx @ V
        M = V.T @ SxV
        S = U @ M @ U.T + np.diag(psi)
        L = np.linalg.cholesky(S)
        Sinv = np.linalg.inv(S)
        mproj = V.T @ mx
        d = my - U @ mproj - b
        A = Sy + np.outer(d, d)
        value += 0.5 * n * (2 * np.sum(np.log(np.diag(L))) + np.sum(Sinv * A))
        G = 0.5 * (Sinv - Sinv @ A @ Sinv)
        GU = G @ U
        Sd = Sinv @ d
        gU += n * (2 * GU @ M - np.outer(Sd, mproj))
        gV += n * (2 * SxV @ (U.T @ GU) - np.outer(mx, U.T @ Sd))
        gb += n * (-Sd)
        glog += n * np.diag(G) * psi
    grad = np.concatenate([gU.ravel(), gV.ravel(), gb, glog]) / total
    return value / total, grad


def lowrank(units, W0, b0, psi0, ridge_scale, maxiter=300):
    p, q = W0.shape[1], W0.shape[0]
    Uw, s, Vt = np.linalg.svd(W0, full_matrices=False)
    U0 = Uw[:, :RANK] * np.sqrt(s[:RANK]); V0 = Vt[:RANK].T * np.sqrt(s[:RANK])
    z0 = np.concatenate([U0.ravel(), V0.ravel(), b0, np.log(psi0)])
    ridge = ridge_scale * 1e-3
    res = minimize(lowrank_negloglik, z0, args=(units, p, q, ridge), jac=True, method="L-BFGS-B",
                   options={"maxiter": maxiter, "maxfun": 2 * maxiter})
    z = res.x
    U = z[: q * RANK].reshape(q, RANK); V = z[q * RANK: q * RANK + p * RANK].reshape(p, RANK)
    psi = np.exp(z[q * RANK + p * RANK + q:])
    return U @ V.T, psi, res


def run(genes=200):
    data = cs.load(genes)
    donors = np.array(sorted(set(data["donor"])))
    results = {}
    for donor in donors:
        started = time.time()
        samples = sorted(set(data["sample"][data["donor"] != donor]))
        units = sample_moments(data, samples)
        scale, errors = choose_mean_ridge(units)
        W, b, psi, ridge = ecological(units, scale)
        W_lr, psi_lr, res = lowrank(units, W, b, psi, scale)
        m = data["donor"] == donor
        adapt, score = m & (data["half"] == 0), m & (data["half"] == 1)
        n, mx, my, Sx, Sy, _ = cs.moments(data["x"][adapt], data["y"][adapt])
        marginal = Marginal(Sx, Sy)
        covs = {"pf_means_regression": Sx @ W.T,
                "pf_means_closed_form": transfer(marginal, W.T / psi[None, :])[0],
                "pf_lowrank_regression": Sx @ W_lr.T,
                "pf_lowrank_closed_form": transfer(marginal, W_lr.T / psi_lr[None, :])[0]}
        xs, ys = data["x"][score], data["y"][score]
        results[str(donor)] = {
            "training_samples": len(samples), "ridge_scale": scale, "ridge_errors": errors,
            "lowrank_message": str(res.message),
            "arms": {arm: cs.evaluate(C, Sx, mx, my, xs, ys) for arm, C in covs.items()},
            "seconds": time.time() - started}
        print(donor, json.dumps({a: round(v["cov_rel_error"], 4) for a, v in results[str(donor)]["arms"].items()}),
              f"scale={scale} {time.time() - started:.0f}s", flush=True)
    return {"genes": int(data["x"].shape[1]), "proteins": int(data["y"].shape[1]), "donors": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--genes", type=int, default=200)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        out = run(args.genes)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
