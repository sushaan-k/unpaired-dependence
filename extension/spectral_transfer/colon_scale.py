#!/usr/bin/env python3
"""Large-panel transfer within the colon study (leave one donor out).

For each recipient donor, the reference is every other donor's paired cells.
The recipient's cells are split by a fixed hash into an adaptation half, used
only as two separate assays (pairing discarded), and a scoring half whose
pairings supply the endpoint. The shared interaction is fitted on the
reference only; its ridge penalty is chosen by inner leave-donors-out
likelihood within the reference.

Arms (cross-covariance C used to predict protein from RNA in scoring cells):
  independence       C = 0
  reference_cov      pooled within-donor reference cross-covariance
  reference_corr     reference cross-correlation rescaled by recipient SDs
  regression         reference RNA->protein regression applied to recipient RNA
  variances_only     closed-form transfer with diagonal recipient covariances
  closed_form        closed-form transfer with full recipient covariances
  oracle             recipient's own paired adaptation cross-covariance (not allowed; ceiling)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits

from gaussian_transfer import Marginal, first_order, fit_interaction, profile_loglik, transfer

HERE = Path(__file__).resolve().parent
DATA = Path("/home/claude/cbio/rawdata/colon_paired.npz")
ARMS = ("independence", "reference_cov", "reference_corr", "regression",
        "first_order", "variances_only", "closed_form", "cell_coupling", "oracle")
SALT = "spectral-transfer-v1"
SHRINK = 0.1
RIDGE_FLOOR = 1e-3
RIDGES = (1e-3, 1e-2, 1e-1)
PANEL = ("CD4", "CD7", "CD14", "CD19", "CD33", "CD38", "CD44", "CD47", "CD52")


def load(genes: int):
    d = np.load(DATA, allow_pickle=False)
    rna = sp.csr_matrix((d["rna_data"], d["rna_indices"], d["rna_indptr"]))
    size = np.asarray(rna.sum(axis=1)).ravel()
    norm = sp.diags(1e4 / size) @ rna
    norm.data = np.log1p(norm.data)
    mean = np.asarray(norm.mean(axis=0)).ravel()
    var = np.asarray(norm.multiply(norm).mean(axis=0)).ravel() - mean ** 2
    names = d["rna_names"]
    keep = mean > 0.05
    score = np.where(keep, var / np.maximum(mean, 1e-12), -np.inf)
    chosen = set(np.argsort(-score)[:genes].tolist())
    chosen |= {int(np.flatnonzero(names == g)[0]) for g in PANEL if (names == g).any()}
    chosen = np.array(sorted(chosen))
    x = norm[:, chosen].toarray()
    adt = np.log1p(d["adt"].astype(float))
    y = adt - adt.mean(axis=1, keepdims=True)          # centred log ratio per cell
    donors = d["CoLabs_patient"]
    half = np.array([int(hashlib.sha256(f"{SALT}|{p}|{b}".encode()).hexdigest(), 16) % 2
                     for p, b in zip(donors, d["barcodes"])])
    return {"x": x, "y": y, "donor": donors, "half": half, "genes": names[chosen],
            "proteins": d["adt_names"], "condition": d["condition"], "sample": d["CoLabs_sample"]}


def moments(x, y):
    mx, my = x.mean(0), y.mean(0)
    xc, yc = x - mx, y - my
    n = len(x)
    Sx, Sy, Sxy = xc.T @ xc / n, yc.T @ yc / n, xc.T @ yc / n
    Sx = (1 - SHRINK) * Sx + SHRINK * np.diag(np.diag(Sx))
    Sy = (1 - SHRINK) * Sy + SHRINK * np.diag(np.diag(Sy))
    Sx = Sx + RIDGE_FLOOR * np.mean(np.diag(Sx)) * np.eye(len(Sx))
    Sy = Sy + RIDGE_FLOOR * np.mean(np.diag(Sy)) * np.eye(len(Sy))
    return n, mx, my, Sx, Sy, Sxy


def reference_people(data, donors):
    out = []
    for donor in donors:
        m = data["donor"] == donor
        n, _, _, Sx, Sy, Sxy = moments(data["x"][m], data["y"][m])
        out.append((n, Marginal(Sx, Sy), Sxy))
    return out


def choose_ridge(people):
    """Inner 3-fold leave-donors-out profile likelihood within the reference."""
    folds = [list(range(k, len(people), 3)) for k in range(3)]
    scores = []
    for ridge in RIDGES:
        total = 0.0
        for fold in folds:
            train = [people[i] for i in range(len(people)) if i not in fold]
            B, _ = fit_interaction(train, ridge, maxiter=300)
            total += profile_loglik(B, [people[i] for i in fold])[0]
        scores.append(total)
    return RIDGES[int(np.argmax(scores))], scores


def cell_coupling(B, x, y, tol=1e-10, maxit=20000):
    """Log-domain Sinkhorn between unpaired RNA and protein clouds (uniform weights)."""
    xc, yc = x - x.mean(0), y - y.mean(0)
    logK = xc @ B @ yc.T
    n, m = logK.shape
    loga, logb = np.full(n, -np.log(n)), np.full(m, -np.log(m))
    f, g = np.zeros(n), np.zeros(m)
    for it in range(maxit):
        f = loga - logsumexp(logK + g[None, :], axis=1)
        g = logb - logsumexp(logK + f[:, None], axis=0)
        if it % 10 == 9:
            rows = logsumexp(logK + f[:, None] + g[None, :], axis=1)
            if np.max(np.abs(np.exp(rows) - np.exp(loga))) < tol:
                break
    Q = np.exp(logK + f[:, None] + g[None, :])
    return xc.T @ Q @ yc, it


def arm_covariances(ref, B, rec_marginal, Sx_rec, Sy_rec, oracle_Sxy, cells=None):
    n_ref = sum(n for n, _, _ in ref)
    ref_Sxy = sum(n * S for n, _, S in ref) / n_ref
    ref_Sx = sum(n * m.Sx for n, m, _ in ref) / n_ref
    ref_Sy = sum(n * m.Sy for n, m, _ in ref) / n_ref
    dx, dy = np.sqrt(np.diag(ref_Sx)), np.sqrt(np.diag(ref_Sy))
    ref_corr = ref_Sxy / np.outer(dx, dy)
    rx, ry = np.sqrt(np.diag(Sx_rec)), np.sqrt(np.diag(Sy_rec))
    W_ref = np.linalg.solve(ref_Sx, ref_Sxy)            # p x q regression coefficients
    diag_marginal = Marginal(np.diag(np.diag(Sx_rec)), np.diag(np.diag(Sy_rec)))
    return {
        "independence": np.zeros_like(ref_Sxy),
        "reference_cov": ref_Sxy,
        "reference_corr": ref_corr * np.outer(rx, ry),
        "regression": Sx_rec @ W_ref,
        "first_order": first_order(rec_marginal, B),
        "variances_only": transfer(diag_marginal, B)[0],
        "closed_form": transfer(rec_marginal, B)[0],
        "cell_coupling": cell_coupling(B, *cells)[0],
        "oracle": oracle_Sxy,
    }


def evaluate(C, Sx, mx, my, xs, ys):
    """Scoring-cell protein prediction from RNA with coefficient Sx^{-1} C."""
    W = np.linalg.solve(Sx, C)
    pred = my + (xs - mx) @ W
    resid = ys - pred
    base = ys - my
    ysc, xsc = ys - ys.mean(0), xs - xs.mean(0)
    held = xsc.T @ ysc / len(xs)
    return {
        "mse": float(np.mean(resid ** 2)),
        "r2": float(1 - np.sum(resid ** 2) / np.sum(base ** 2)),
        "cov_corr": float(np.corrcoef(C.ravel(), held.ravel())[0, 1]) if np.any(C) else 0.0,
        "cov_rel_error": float(np.linalg.norm(C - held) / np.linalg.norm(held)),
    }


def run(genes: int, recipients=None, reference=None):
    data = load(genes)
    donors = np.array(sorted(set(data["donor"])))
    if reference == "HC":
        reference_donors = sorted(set(data["donor"][data["condition"] == "HC"]))
        recipients = [d for d in donors if d not in reference_donors]
    recipients = donors if recipients is None else recipients
    results, cache = {}, {}
    for donor in recipients:
        started = time.time()
        pool = reference_donors if reference == "HC" else [d for d in donors if d != donor]
        ref = reference_people(data, pool)
        key = tuple(pool)
        if key not in cache:                       # identical reference pool: fit once
            ridge, ridge_scores = choose_ridge(ref)
            B, fit = fit_interaction(ref, ridge, maxiter=500)
            cache[key] = (ridge, ridge_scores, B, fit)
        ridge, ridge_scores, B, fit = cache[key]
        m = data["donor"] == donor
        adapt, score = m & (data["half"] == 0), m & (data["half"] == 1)
        n, mx, my, Sx, Sy, Sxy_oracle = moments(data["x"][adapt], data["y"][adapt])
        covs = arm_covariances(ref, B, Marginal(Sx, Sy), Sx, Sy, Sxy_oracle,
                               cells=(data["x"][adapt], data["y"][adapt]))
        xs, ys = data["x"][score], data["y"][score]
        results[str(donor)] = {
            "condition": str(data["condition"][m][0]), "adaptation_cells": int(adapt.sum()),
            "scoring_cells": int(score.sum()), "ridge": ridge, "ridge_scores": ridge_scores,
            "fit_message": str(fit.message),
            "arms": {arm: evaluate(covs[arm], Sx, mx, my, xs, ys) for arm in ARMS},
            "seconds": time.time() - started,
        }
        print(donor, json.dumps({a: round(v["r2"], 4) for a, v in results[str(donor)]["arms"].items()}),
              f"ridge={ridge} {time.time() - started:.0f}s", flush=True)
    return {"genes": int(data["x"].shape[1]), "proteins": int(data["y"].shape[1]),
            "donors": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--genes", type=int, default=200)
    parser.add_argument("--recipients", nargs="*")
    parser.add_argument("--reference", choices=("HC",))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        out = run(args.genes, args.recipients, args.reference)
    if args.out:
        args.out.write_text(json.dumps(out, indent=2) + "\n")
