"""Semi-paired estimators of the within-cell gene-protein cross-correlation.

Paired calibration cells X (B x p) and Y (B x q) are centred on unpaired population
means and scaled by pooled within-population standard deviations. Unpaired
summaries: measured and latent RNA correlation (Rx, Rz) and protein correlation
(Ry). Every shrinkage step is positive-part James-Stein with c = tr(S), where S is the
noise covariance of the shrunk coefficients estimated from per-cell products;
the largest eigenvalue of S is recorded to check the effective-dimension
condition tr(S) >= 4 lambda_max(S) of the risk bound (Supplementary Note 10).
"""

from __future__ import annotations

import numpy as np

EIGEN_EDGES = (5, 20, 60)
PROTEIN_EDGES = (3,)


def eigenbasis(R):
    ev, vec = np.linalg.eigh(R)
    order = np.argsort(ev)[::-1]
    return vec[:, order], np.maximum(ev[order], 1e-9)


def eigen_blocks(p, edges=EIGEN_EDGES):
    e = [0] + [x for x in edges if x < p] + [p]
    return [list(range(a, b)) for a, b in zip(e[:-1], e[1:])]


def noise_terms(Z, iters=30, seed=0):
    """For per-cell product vectors Z (B x d): the mean, and the trace and (power-iteration) largest eigenvalue
    of the covariance of the mean."""
    n = len(Z)
    m = Z.mean(0)
    Zc = Z - m
    tr = float(np.einsum("ij,ij->", Zc, Zc)) / ((n - 1) * n)
    v = np.random.default_rng(seed).standard_normal(Z.shape[1])
    v /= np.linalg.norm(v)
    for _ in range(iters):
        w = Zc.T @ (Zc @ v)
        nw = np.linalg.norm(w)
        if nw == 0:
            break
        v = w / nw
    lam = float(np.sum((Zc @ v) ** 2)) / ((n - 1) * n)
    return m, tr, lam


def js_factor(norm2, tr, lam=None, positive=True):
    """James-Stein factor 1 - tr(S)/||m||^2 (positive part); the risk bound of Supplementary Note 10 holds for
    c = tr(S) when tr(S) >= 4 lambda_max(S)."""
    f = 1.0 - tr / max(norm2, 1e-15)
    return max(f, 0.0) if positive else f


def js_matrix(X, Y, Wx=None):
    """Positive-part James-Stein (Bock constant) of the mean cross-product of (X Wx) and Y towards zero."""
    Xw = X if Wx is None else X @ Wx
    Z = (Xw[:, :, None] * Y[:, None, :]).reshape(len(X), -1)
    m, tr, lam = noise_terms(Z)
    f = js_factor(float(m @ m), tr, lam)
    return f * m.reshape(Xw.shape[1], Y.shape[1]), f


def block_js(X, Y, U, blocks):
    """Block James-Stein in the RNA basis U (all protein coordinates in every block); returns p x q estimate."""
    XU = X @ U
    A = np.zeros((U.shape[1], Y.shape[1]))
    factors = []
    for rows in blocks:
        Ab, f = js_matrix(XU[:, rows], Y)
        A[rows] = Ab
        factors.append(f)
    return U @ A, factors


def block_js_cols(X, Y, U, blocks):
    """Block James-Stein in the RNA basis U with one block per eigen-block and protein; returns p x q estimate."""
    XU = X @ U
    A = np.zeros((U.shape[1], Y.shape[1]))
    factors = []
    for rows in blocks:
        for j in range(Y.shape[1]):
            Ab, f = js_matrix(XU[:, rows], Y[:, [j]])
            A[rows, j] = Ab[:, 0]
            factors.append(f)
    return U @ A, factors


def two_sided_js(X, Y, U, Ry, rblocks, edges=PROTEIN_EDGES):
    """The proposed denoiser: block James-Stein in the RNA eigenbasis U and the eigenbasis of the unpaired protein
    correlation Ry, blocks = RNA eigen-block x protein eigen-block (protein edges 3: leading three, rest)."""
    V, _ = eigenbasis(Ry)
    return block_js2(X, Y, U, V, rblocks, eigen_blocks(V.shape[1], edges))


def block_js2(X, Y, U, V, rblocks, pblocks):
    """Block James-Stein in a two-sided basis: RNA eigenbasis U and protein eigenbasis V (both from unpaired
    cells), one block per RNA eigen-block and protein eigen-block; returns the p x q estimate."""
    XU, YV = X @ U, Y @ V
    A = np.zeros((U.shape[1], V.shape[1]))
    factors = []
    for rows in rblocks:
        for cols in pblocks:
            Ab, f = js_matrix(XU[:, rows], YV[:, cols])
            A[np.ix_(rows, cols)] = Ab
            factors.append(f)
    return U @ A @ V.T, factors


def cognate_operator(Rz, sets, union):
    """Linear map from a cell's RNA vector to its projection on the encoding genes, propagated through Rz.

    Returns a list of p x p operators (one per protein; the same for all proteins when union=True)."""
    p = Rz.shape[0]
    if union:
        S = sorted({i for s in sets for i in s})
        P = np.zeros((p, p))
        if S:
            P[:, S] = Rz[:, S] @ np.linalg.inv(Rz[np.ix_(S, S)])
        return [P] * len(sets)
    out = []
    for S in sets:
        P = np.zeros((p, p))
        if S:
            P[:, S] = Rz[:, S] @ np.linalg.inv(Rz[np.ix_(S, S)])
        out.append(P)
    return out


def cognate_estimate(X, Y, ops, U, blocks, residual=True, shrink=True):
    """Transcript-anchored estimate: each protein's cross-covariance is the propagation of its covariance with
    its encoding genes through unpaired latent RNA co-expression (James-Stein shrunk), plus the block
    James-Stein estimate of what the propagation leaves."""
    B, q = Y.shape
    p = X.shape[1]
    # propagated part: column j = mean_i (P_j x_i) y_ij
    Zs = np.stack([(X @ ops[j].T) * Y[:, [j]] for j in range(q)], axis=2)   # B x p x q
    Zf = Zs.reshape(B, -1)
    m, tr, lam = noise_terms(Zf)
    f = js_factor(float(m @ m), tr, lam) if shrink else 1.0
    Cprop = f * m.reshape(p, q)
    if not residual:
        return Cprop, {"prop": f}
    # residual: column j = mean_i ((I - P_j) x_i) y_ij, block James-Stein in basis U
    R = np.stack([(X - X @ ops[j].T) * Y[:, [j]] for j in range(q)], axis=2)   # B x p x q
    RU = np.einsum("bpq,pk->bkq", R, U)
    A = np.zeros((p, q))
    factors = []
    for rows in blocks:
        Zb = RU[:, rows, :].reshape(B, -1)
        mb, trb, lamb = noise_terms(Zb)
        fb = js_factor(float(mb @ mb), trb, lamb)
        A[rows] = fb * mb.reshape(len(rows), q)
        factors.append(fb)
    return Cprop + U @ A, {"prop": f, "residual": factors}
