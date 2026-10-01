"""Small statistics used by the failure analyses (no external dependencies beyond numpy/scipy)."""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def auc(y, score):
    """Area under the ROC curve (probability a failure scores higher than a success; ties count half)."""
    y = np.asarray(y, bool)
    score = np.asarray(score, float)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def standardizer(X):
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    return mu, sd


def logistic_fit(X, y, ridge=1e-2, iters=100):
    """Logistic regression on standardized columns with a small ridge on the slopes (Newton-Raphson).

    Returns (intercept, slopes, mu, sd); the linear score is intercept + ((X - mu) / sd) @ slopes.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    mu, sd = standardizer(X)
    Z = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    beta = np.zeros(Z.shape[1])
    pen = np.full(Z.shape[1], ridge)
    pen[0] = 0.0
    for _ in range(iters):
        eta = Z @ beta
        p = 1 / (1 + np.exp(-eta))
        grad = Z.T @ (y - p) - pen * beta
        H = (Z * (p * (1 - p))[:, None]).T @ Z + np.diag(pen)
        step = np.linalg.solve(H, grad)
        beta += step
        if np.max(np.abs(step)) < 1e-10:
            break
    return float(beta[0]), beta[1:], mu, sd


def logistic_score(model, X):
    b0, b, mu, sd = model
    return b0 + ((np.asarray(X, float) - mu) / sd) @ b


def cluster_bootstrap(clusters, statistic, replicates=2000, seed=0):
    """Resample clusters with replacement; statistic(indices) -> float or array. Returns replicate values."""
    clusters = np.asarray(clusters)
    ids = sorted(set(clusters))
    members = {c: np.flatnonzero(clusters == c) for c in ids}
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(replicates):
        pick = rng.choice(len(ids), len(ids), replace=True)
        idx = np.concatenate([members[ids[i]] for i in pick])
        out.append(statistic(idx))
    return np.array(out, float)
