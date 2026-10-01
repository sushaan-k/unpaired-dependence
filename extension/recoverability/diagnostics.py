"""Unpaired diagnostics of cross-assay recoverability.

Every quantity here is computed from within-assay summaries only: training
populations' RNA and protein moments (from disjoint cells) and a recipient's
RNA and protein covariance matrices at the target level (all cells, or cells
centred on their cell type). No cross-assay moment of any recipient is used.

Coverage asks whether the training populations vary along the recipient's RNA
axes: the share of recipient RNA variance inside the effective supported
subspace S of the channel estimate (for the ridge estimator, the directions it
retains by at least one half; S depends on the penalty).

Compatibility asks whether the between-population channel can be a
within-cell channel for this recipient: under a shared channel
y | x ~ N(W x + b, Psi) with Psi >= 0, the recipient's protein covariance must
dominate W R_x W^T. The implied explained variance of protein j,
r_j = (W_S R_x W_S^T)_jj / (R_y)_jj, must not exceed one, and the implied
squared canonical correlations (eigenvalues of R_y^-1/2 W_S R_x W_S^T R_y^-1/2)
must not exceed one.
"""

from __future__ import annotations

import numpy as np


def psd_power(S, power, floor=1e-12):
    w, V = np.linalg.eigh((S + S.T) / 2)
    return (V * np.clip(w, floor, None) ** power) @ V.T


def supported_projection(G, ridge, threshold=0.5):
    """Projection onto directions the ridge estimator retains by at least `threshold`."""
    g, V = np.linalg.eigh((G + G.T) / 2)
    h = np.clip(g, 0, None) / (np.clip(g, 0, None) + ridge)
    keep = V[:, h >= threshold]
    return keep @ keep.T, int(keep.shape[1]), float(h.sum())


def rowspace_projection(W, tol=1e-8):
    """Projection onto the row space of a low-rank channel W (q x p)."""
    _, s, Vt = np.linalg.svd(W, full_matrices=False)
    keep = Vt[s > tol * s.max()]
    return keep.T @ keep, int(keep.shape[0])


def coverage(P, Rx):
    return float(np.trace(P @ Rx @ P) / np.trace(Rx))


def compatibility(W, P, Rx, Ry):
    """Implied dependence of the supported channel W P against what R_y permits."""
    WS = W @ P
    implied = WS @ Rx @ WS.T
    Ryih = psd_power(Ry, -0.5)
    lam = np.linalg.eigvalsh(Ryih @ implied @ Ryih)
    r = np.diag(implied) / np.diag(Ry)
    return {"lam_max": float(lam.max()),
            "lam_excess": float(np.sum(np.clip(lam - 1, 0, None)) / max(np.sum(np.clip(lam, 0, None)), 1e-12)),
            "mean_r": float(r.mean()),
            "median_r": float(np.median(r)),
            "frac_over": float(np.mean(r > 1)),
            "excess": float(np.sum(np.clip(r - 1, 0, None)) / max(np.sum(r), 1e-12))}


def between_share(v, labels, keep):
    """Share of the (standardized) variance of v that lies between the kept cell types."""
    cells = np.isin(labels, keep)
    v = v[cells]
    labels = labels[cells]
    sd = v.std(0)
    v = (v - v.mean(0)) / np.where(sd > 0, sd, 1.0)
    total = np.sum(v ** 2)
    between = sum(np.sum(labels == t) * np.sum(v[labels == t].mean(0) ** 2) for t in keep)
    return float(between / total)


def covariance_shift(Rx, R_train):
    """One minus the cosine similarity of two correlation matrices (off-diagonal part)."""
    off = ~np.eye(len(Rx), dtype=bool)
    a, b = Rx[off], R_train[off]
    return float(1 - a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
