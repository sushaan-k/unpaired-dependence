#!/usr/bin/env python3
"""Post hoc validation of the noise correction against known latent covariance (added in response to review).

Semi-synthetic design on the bone-marrow adaptation cells (12 samples):

  truth    each cell's expected expression proportions are the average of the
           observed proportions (count / library) over the cell and its 14
           nearest neighbours (30 principal components of standardized log
           expression); the rest of the transcriptome keeps the remaining share.
           With --gene, independent gene-specific lognormal variation is added so
           that each gene's latent within-type variance matches the real data's
           (within-type variance times one minus its count-splitting noise share).
           Expected library = observed library x depth factor (0.5, 1, 2).
  counts   Poisson draws for the panel genes and for the rest of the
           transcriptome; the library is their sum, so normalization couples genes
           exactly as in real data.
  exact    40 independent replicate draws per cell give the noise covariance
           N = E[Cov(x | truth)] (full matrix, including coupling through the
           library) and the signal covariance Cov(E[x | truth]) = Cov(mean of
           replicates) - N / 40, for x = log1p(1e4 count / library).
  pipeline the first replicate is analysed exactly as the real data: binomial count
           splitting with thinned libraries, split-half reliability, Spearman-Brown,
           diagonal subtraction from the shrunk correlation matrix, eigenvalue floor.

Compared quantities, on the scale of the observed standard deviations: the
per-gene noise share (estimated against true), off-diagonal noise coupling, and
the compatibility statistic ||F|| of the frozen, P1 and colon channels with the
real protein correlation of the sample, computed from the measured, the
corrected and the true latent covariance (the true latent covariance receives
the same shrinkage as the estimate: shrink(S) with S the true signal
covariance). Analyses: all cells, and within coarse types.
Writes results/noise_simulation.json (smoothed truth) or results/noise_simulation_gene.json (--gene).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "cross_study"))

from data import shrink, standardize, type_centre  # noqa: E402

from bmmc import load  # noqa: E402
from compat import opnorm, psd_power  # noqa: E402
from noise import lognorm, noise_fraction, signal_correlation, split_halves  # noqa: E402
from predict_bmmc import load_channels  # noqa: E402

SEED = 20261012
REPLICATES = 40
NEIGHBOURS = 15
DEPTHS = (0.5, 1.0, 2.0)
CHANNELS = ("frozen", "P1", "colon")
TRUTH = "gene" if "--gene" in sys.argv else "smoothed"


def truth(counts, library, labels, keep, kind, rng):
    """Smoothed truth; with kind == "gene", plus independent gene-specific lognormal variation whose
    size is chosen so that each gene's latent within-type variance matches the real data's
    (within-type variance times one minus the count-splitting noise share)."""
    x = lognorm(counts, library)
    xs, _ = standardize(x)
    U, s, _ = np.linalg.svd(xs - xs.mean(0), full_matrices=False)
    pcs = U[:, :30] * s[:30]
    idx = NearestNeighbors(n_neighbors=NEIGHBOURS).fit(pcs).kneighbors(pcs, return_distance=False)
    prop = counts / np.maximum(library, 1)[:, None]
    p = prop[idx].mean(1)
    if kind == "gene":
        kept = np.isin(labels, keep)
        h1, h2 = split_halves(counts[kept], library[kept])
        nu = noise_fraction(h1, h2, labels[kept], keep)
        real = type_centre(x[kept], labels[kept], keep).var(0) * (1 - nu)
        smooth = np.log1p(1e4 * p)
        have = type_centre(smooth[kept], labels[kept], keep).var(0)
        slope = np.mean(1e4 * p / (1 + 1e4 * p), 0)
        sigma2 = np.clip((real - have) / np.maximum(slope, 1e-3) ** 2, 0, 4)
        p = p * np.exp(np.sqrt(sigma2) * rng.standard_normal(p.shape) - sigma2 / 2)
        scale = np.maximum(p.sum(1), 1)
        p = p / scale[:, None]
    rest = np.clip(1 - p.sum(1), 0, None)
    return p, rest


def replicates(p, rest, depth, rng, K):
    """K replicate count draws per cell; returns the first replicate's counts and library, and all log values."""
    n, g = p.shape
    X = np.empty((K, n, g))
    first = None
    for k in range(K):
        c = rng.poisson(depth[:, None] * p)
        L = c.sum(1) + rng.poisson(depth * rest)
        X[k] = lognorm(c, L)
        if k == 0:
            first = (c.astype(float), L.astype(float))
    return first, X


def cov_parts(X, labels=None, keep=None):
    """Signal and noise covariance from replicates (K, n, g); within types if labels are given."""
    K, n, g = X.shape
    mean = X.mean(0)
    dev = (X - mean).reshape(K * n, g)
    N = dev.T @ dev / (n * (K - 1))
    m = mean if labels is None else type_centre(mean, labels, keep)
    m = m - m.mean(0)
    return m.T @ m / n - N / K, N


def F_stat(W, VS, R, Ryih):
    if VS.shape[1] == 0:
        return 0.0
    QN, _ = np.linalg.qr(psd_power(R, -0.5) @ VS)
    return opnorm(QN.T @ psd_power(R, 0.5) @ W.T @ Ryih)


def analyse(c, L, X, y, labels, keep, chans):
    """Pipeline on the first replicate versus the exact latent covariance."""
    x = lognorm(c, L)
    h1, h2 = split_halves(c, L)
    if labels is None:
        xa, ya, nu = x, y, noise_fraction(h1, h2)
    else:
        xa, ya = type_centre(x, labels, keep), type_centre(y, labels, keep)
        nu = noise_fraction(h1, h2, labels, keep)
    S, N = cov_parts(X, labels, keep)
    xs, sd = standardize(xa)
    ys, _ = standardize(ya)
    n = len(xa)
    Rx = shrink(xs.T @ xs / n)
    Rz = signal_correlation(Rx, nu)
    D = np.where(sd > 0, sd, 1.0)
    St, Nt = S / np.outer(D, D), N / np.outer(D, D)
    target = shrink(St)
    oracle = signal_correlation(Rx, np.clip(np.diag(Nt) / np.diag(Rx), 0, 1))
    total = np.diag(S) + np.diag(N)
    ok = total > 0
    nu_true = np.where(ok, np.diag(N) / np.where(ok, total, 1), np.nan)
    off = ~np.eye(len(S), dtype=bool)
    dn = np.sqrt(np.clip(np.diag(Nt), 1e-12, None))
    noise_corr = Nt / np.outer(dn, dn)
    row = {"cells": int(n), "nu_true_median": float(np.median(nu_true[ok])), "nu_hat_median": float(np.median(nu[ok])),
           "nu_bias_mean": float(np.mean(nu[ok] - nu_true[ok])),
           "nu_abs_error_mean": float(np.mean(np.abs(nu[ok] - nu_true[ok]))),
           "nu_corr": float(np.corrcoef(nu[ok], nu_true[ok])[0, 1]),
           "coupling_ratio": float(np.linalg.norm(Nt[off]) / np.linalg.norm(St[off])),
           "noise_corr_mean_abs": float(np.mean(np.abs(noise_corr[off]))),
           "noise_corr_max_abs": float(np.max(np.abs(noise_corr[off]))),
           "offdiag_error_measured": float(np.linalg.norm(Rx[off] - target[off]) / np.linalg.norm(target[off])),
           "offdiag_error_corrected": float(np.linalg.norm(Rz[off] - target[off]) / np.linalg.norm(target[off])),
           "error_measured": float(np.linalg.norm(Rx - target) / np.linalg.norm(target)),
           "error_corrected": float(np.linalg.norm(Rz - target) / np.linalg.norm(target)),
           "error_oracle_diagonal": float(np.linalg.norm(oracle - target) / np.linalg.norm(target)),
           "F": {}}
    Ryih = psd_power(shrink(ys.T @ ys / n), -0.5)
    for name in CHANNELS:
        ch = chans[name]
        row["F"][name] = {k: F_stat(ch["W"], ch["VS"], R, Ryih)
                          for k, R in (("measured", Rx), ("corrected", Rz), ("true", target), ("oracle", oracle))}
    return row


def main():
    t0 = time.time()
    rec = load("adaptation")
    chans = load_channels()
    units = json.loads((HERE / "results/bmmc_units.json").read_text())
    rng = np.random.default_rng(SEED)
    rows = []
    for batch, unit in units.items():
        m = rec["batch"] == batch
        counts, library, y = rec["counts"][m], rec["library"][m], rec["y"][m]
        coarse = rec["labels"]["coarse"][m]
        keep = unit["types"]["coarse"]
        p, rest = truth(counts, library, coarse, keep, TRUTH, rng)
        kept = np.isin(coarse, keep)
        for factor in DEPTHS:
            (c, L), X = replicates(p, rest, factor * library, rng, REPLICATES)
            for analysis in ("total", "within_coarse"):
                if analysis == "total":
                    row = analyse(c, L, X, y, None, None, chans)
                else:
                    row = analyse(c[kept], L[kept], X[:, kept], y[kept], coarse[kept], keep, chans)
                row.update(batch=batch, depth=factor, analysis=analysis,
                           median_library=float(np.median(L)))
                rows.append(row)
                print(batch, factor, analysis, {k: round(v["corrected"], 2) for k, v in row["F"].items()},
                      {k: round(v["true"], 2) for k, v in row["F"].items()},
                      "nu", round(row["nu_hat_median"], 2), round(row["nu_true_median"], 2),
                      f"({time.time() - t0:.0f}s)", flush=True)
    summary = {}
    for factor in DEPTHS:
        for analysis in ("total", "within_coarse"):
            rs = [r for r in rows if r["depth"] == factor and r["analysis"] == analysis]
            s = {k: float(np.median([r[k] for r in rs])) for k in
                 ("nu_true_median", "nu_hat_median", "nu_bias_mean", "nu_abs_error_mean", "nu_corr", "coupling_ratio",
                  "noise_corr_mean_abs", "noise_corr_max_abs", "offdiag_error_measured", "offdiag_error_corrected",
                  "error_measured", "error_corrected", "error_oracle_diagonal", "median_library")}
            for name in CHANNELS:
                F = {k: np.array([r["F"][name][k] for r in rs]) for k in ("measured", "corrected", "true", "oracle")}
                s[name] = {"F_true_median": float(np.median(F["true"])),
                           "F_measured_median": float(np.median(F["measured"])),
                           "F_corrected_median": float(np.median(F["corrected"])),
                           "relative_error_corrected_median": float(np.median(np.abs(F["corrected"] / F["true"] - 1))),
                           "relative_error_corrected_max": float(np.max(np.abs(F["corrected"] / F["true"] - 1))),
                           "relative_error_measured_median": float(np.median(np.abs(F["measured"] / F["true"] - 1))),
                           "same_side_of_one_corrected": int(np.sum((F["corrected"] > 1) == (F["true"] > 1))),
                           "same_side_of_one_measured": int(np.sum((F["measured"] > 1) == (F["true"] > 1))),
                           "true_above_one": int(np.sum(F["true"] > 1)), "analyses": len(rs)}
            summary[f"{analysis}/{factor}"] = s
    out = {"design": {"replicates": REPLICATES, "neighbours": NEIGHBOURS, "depths": DEPTHS, "seed": SEED,
                      "truth": TRUTH},
           "summary": summary, "rows": rows, "seconds": time.time() - t0}
    name = "noise_simulation_gene.json" if TRUTH == "gene" else "noise_simulation.json"
    (HERE / "results" / name).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
