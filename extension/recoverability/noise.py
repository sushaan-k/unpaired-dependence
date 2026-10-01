"""RNA-only estimates of technical (sampling) noise by binomial count splitting.

Each RNA count c ~ Poisson(lambda) is split into c1 ~ Binomial(c, 1/2) and
c2 = c - c1, which are independent Poisson(lambda/2) draws given lambda
(molecular cross-validation; Batson et al. 2019; data thinning, Neufeld et al.
2024). Log-normalized expression of the two halves shares the cell's latent
expression but not its sampling noise, so their correlation across cells is the
split-half reliability of each gene. The Spearman-Brown formula gives the
reliability at full depth, and one minus it is the share of the gene's
standardized variance that is sampling noise. Sampling noise is independent
across genes, so it inflates only the diagonal of the RNA correlation matrix.
"""

from __future__ import annotations

import numpy as np

SEED = 20260926


def lognorm(counts, library):
    return np.log1p(1e4 * counts / np.maximum(library, 1)[:, None])


def split_halves(counts, library, seed=SEED):
    """Binomial thinning of every count; libraries are thinned consistently."""
    rng = np.random.default_rng(seed)
    c = np.asarray(counts, dtype=np.int64)
    c1 = rng.binomial(c, 0.5)
    other = np.maximum(np.rint(library).astype(np.int64) - c.sum(1), 0)
    o1 = rng.binomial(other, 0.5)
    L1 = c1.sum(1) + o1
    L2 = (c - c1).sum(1) + (other - o1)
    return lognorm(c1, L1), lognorm(c - c1, L2)


def noise_fraction(x1, x2, labels=None, keep=None):
    """Per-gene share of standardized full-depth variance that is sampling noise.

    With labels, both halves are centred within the kept cell types first.
    """
    if labels is not None:
        x1, x2 = x1.copy(), x2.copy()
        for t in keep:
            m = labels == t
            x1[m] -= x1[m].mean(0)
            x2[m] -= x2[m].mean(0)
    a, b = x1 - x1.mean(0), x2 - x2.mean(0)
    va, vb = np.mean(a * a, 0), np.mean(b * b, 0)
    cov = np.mean(a * b, 0)
    ok = (va > 0) & (vb > 0)
    r = np.zeros(x1.shape[1])
    r[ok] = cov[ok] / np.sqrt(va[ok] * vb[ok])
    r = np.clip(r, 0.0, 1.0)
    reliability = 2 * r / (1 + r)
    return 1.0 - reliability


def signal_correlation(Rx, nu, floor=1e-3):
    """R_x minus the noise share on its diagonal, projected to be positive semidefinite."""
    S = Rx - np.diag(nu * np.diag(Rx))
    w, V = np.linalg.eigh((S + S.T) / 2)
    return (V * np.clip(w, floor * np.mean(np.diag(Rx)), None)) @ V.T
