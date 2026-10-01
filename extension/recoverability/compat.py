"""Compatibility of a between-population channel with a recipient's within-assay covariances.

Setting (Supplementary Note 6). Population means identify the channel W on
the span M of centred population RNA means; the estimator retains the
effective supported subspace S (ridge factor >= 1/2), spanned by the columns
of VS. A recipient cross-covariance C is compatible with W on S when the joint
matrix [[R, C], [C^T, R_y]] is positive semidefinite and C = R W^T holds on S,
where R is the recipient's RNA covariance of the latent state (the measured
correlation matrix with RNA sampling noise removed from its diagonal; noise.py)
or, as originally reported, the measured matrix. The set of such C is nonempty
if and only if the whitened identified block
    F = Q_N^T R^1/2 W^T R_y^-1/2,   Q_N an orthonormal basis of R^-1/2 S,
has operator norm at most one. ||F|| is the compatibility statistic.

Training uncertainty is propagated by resampling training populations and
refitting W with the penalty scale fixed; recipient uncertainty by resampling
cells. The basic bootstrap lower bound 2 F_hat - q_0.95(F*) corrects the upward
bias of a largest singular value.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg


def svd(M, compute_uv=True):
    """numpy's SVD (LAPACK gesdd); if it does not converge, LAPACK gesvd (amendment 1)."""
    try:
        return np.linalg.svd(M, full_matrices=False, compute_uv=compute_uv)
    except np.linalg.LinAlgError:
        return scipy.linalg.svd(M, full_matrices=False, compute_uv=compute_uv, lapack_driver="gesvd")


def opnorm(M):
    return float(svd(M, compute_uv=False)[0])


def psd_power(S, power, floor=1e-12):
    w, V = np.linalg.eigh((S + S.T) / 2)
    return (V * np.clip(w, floor, None) ** power) @ V.T


def supported_basis(G, ridge, threshold=0.5):
    g, V = np.linalg.eigh((G + G.T) / 2)
    h = np.clip(g, 0, None) / (np.clip(g, 0, None) + ridge)
    order = np.argsort(-h)
    h, V = h[order], V[:, order]
    return V[:, h >= threshold], h


def whitened_block(W, VS, R, Ry):
    Rh, Rih, Ryih = psd_power(R, 0.5), psd_power(R, -0.5), psd_power(Ry, -0.5)
    QN, _ = np.linalg.qr(Rih @ VS)
    return QN.T @ Rh @ W.T @ Ryih


def compatibility_statistic(W, VS, R, Ry):
    if VS.shape[1] == 0:
        return 0.0
    return opnorm(whitened_block(W, VS, R, Ry))


def per_protein(W, VS, R):
    """Implied explained variance of each protein from the channel on S (correlation scale)."""
    WS = W @ VS @ VS.T
    return np.diag(WS @ R @ WS.T)


def completion_bounds(W, VS, R, Ry):
    """Entrywise sharp bounds of the compatible set {C} (None if it is empty).

    K = R^-1/2 C R_y^-1/2 = Q_N F + Q_perp Gamma D with D = (I - F^T F)^1/2 and any
    contraction Gamma; entry (i, j) of C ranges over c_ij +- ||a_i|| ||b_j|| with
    a_i = Q_perp^T R^1/2 e_i and b_j = D R_y^1/2 e_j (each attained by a rank-one Gamma).
    """
    Rh, Rih, Ryh, Ryih = psd_power(R, 0.5), psd_power(R, -0.5), psd_power(Ry, 0.5), psd_power(Ry, -0.5)
    QN, _ = np.linalg.qr(Rih @ VS)
    F = QN.T @ Rh @ W.T @ Ryih
    if opnorm(F) > 1:
        return None
    full, _ = np.linalg.qr(np.hstack([QN, np.eye(len(R))]))
    Qp = full[:, QN.shape[1]:len(R)]
    Qp = Qp - QN @ (QN.T @ Qp)
    w, V = np.linalg.eigh(np.eye(len(Ry)) - F.T @ F)
    D = (V * np.sqrt(np.clip(w, 0, None))) @ V.T
    centre = Rh @ QN @ F @ Ryh
    a = np.linalg.norm(Qp.T @ Rh, axis=0)
    b = np.linalg.norm(D @ Ryh, axis=0)
    half = np.outer(a, b)
    return centre - half, centre + half, centre


def channel_departure(C, R, W, VS):
    """Relative change, along S, between W and the channel implied by a prediction C (C = R W~^T)."""
    Wt = np.linalg.solve(R, C).T
    SS = psd_power(VS.T @ R @ VS, 0.5)
    ref = W @ VS @ SS
    return float(np.linalg.norm(Wt @ VS @ SS - ref) / np.linalg.norm(ref))
