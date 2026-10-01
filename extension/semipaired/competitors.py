"""Comparators for the semi-paired study: direct cross-covariance shrinkage (SCOSE, FCOSE), SemiCCA, a
partially paired Gaussian factor model, reference ridge regression and cross-validated low rank.

All take paired calibration cells X (B x p), Y (B x q) centred on unpaired population means in pooled-SD units;
the semi-supervised ones also take unpaired covariances Sx (p x p) and Sy (q x q) in the same units and the
numbers of unpaired cells behind them.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as sla

CV_SEED = 20261035


def halves(n, rng):
    idx = rng.permutation(n)
    return idx[: n // 2], idx[n // 2:]


def cv_score(C_pred, Xt, Yt):
    """Unbiased (up to a constant) held-out score of a cross-covariance prediction: 2<C, S_test> - ||C||^2."""
    S = Xt.T @ Yt / len(Xt)
    return 2 * float(np.sum(C_pred * S)) - float(np.sum(C_pred * C_pred))


# ---------------------------------------------------------------- direct shrinkage (Muandet-style SCOSE/FCOSE)

def scose(X, Y):
    """Simple cross-covariance shrinkage towards zero with the leave-one-out optimal intensity."""
    n = len(X)
    S = X.T @ Y / n
    zs = np.einsum("ip,pq,iq->i", X, S, Y)            # <z_i, S>
    zz = np.sum(X * X, 1) * np.sum(Y * Y, 1)            # ||z_i||^2
    SS = float(np.sum(S * S))
    num = float(np.sum((n * zs - zz) / (n - 1)))
    den = float(np.sum((n * n * SS - 2 * n * zs + zz) / (n - 1) ** 2))
    return float(np.clip(num / den, 0, 1)) * S


def fcose(X, Y, grid=10.0 ** np.arange(-3, 2.01, 0.5), folds=5, rng=None):
    """Flexible shrinkage: S = X' diag(beta) Y / n with beta = (M + lambda I)^-1 M 1, M = (XX') o (YY');
    lambda (relative to the mean eigenvalue of M) by K-fold cross-validation of the held-out cross-product loss."""
    rng = rng or np.random.default_rng(CV_SEED)
    n = len(X)
    M = (X @ X.T) * (Y @ Y.T)
    fold = rng.permutation(n) % folds
    loss = np.zeros(len(grid))
    for f in range(folds):
        tr, te = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
        Mt = M[np.ix_(tr, tr)]
        d, V = np.linalg.eigh(Mt)
        d = np.maximum(d, 0)
        scale = max(d.mean(), 1e-12)
        V1 = V.T @ np.ones(len(tr))
        Mx = M[np.ix_(te, tr)]
        for i, lam in enumerate(grid):
            beta = V @ ((d / (d + lam * scale)) * V1)
            cross = float(np.sum(Mx @ beta)) / (len(te) * len(tr))
            norm2 = float(beta @ Mt @ beta) / len(tr) ** 2
            loss[i] += -2 * cross + norm2
    lam = grid[int(np.argmin(loss))]
    d, V = np.linalg.eigh(M)
    d = np.maximum(d, 0)
    beta = V @ ((d / (d + lam * max(d.mean(), 1e-12))) * (V.T @ np.ones(n)))
    return X.T @ (beta[:, None] * Y) / n, float(lam)


# ---------------------------------------------------------------- SemiCCA (Kimura et al. 2013)

def _inv_sqrt(M):
    w, V = np.linalg.eigh((M + M.T) / 2)
    return (V / np.sqrt(np.maximum(w, 1e-10))) @ V.T


def semicca_directions(Sxy, Sxx_p, Syy_p, Sxx_u, Syy_u, beta, k):
    """Leading generalized eigenvectors of SemiCCA: L w = mu R w with L = beta [[0, Sxy], [Syx, 0]] +
    (1 - beta) diag(Sxx_u, Syy_u) and R = beta diag(Sxx_p, Syy_p) + (1 - beta) I (R is block diagonal, so it is
    whitened block by block)."""
    p, q = Sxy.shape
    Rxi = _inv_sqrt(beta * Sxx_p + (1 - beta) * np.eye(p) + 1e-6 * np.eye(p))
    Ryi = _inv_sqrt(beta * Syy_p + (1 - beta) * np.eye(q) + 1e-6 * np.eye(q))
    L = np.zeros((p + q, p + q))
    L[:p, :p] = (1 - beta) * Rxi @ Sxx_u @ Rxi
    L[p:, p:] = (1 - beta) * Ryi @ Syy_u @ Ryi
    L[:p, p:] = beta * Rxi @ Sxy @ Ryi
    L[p:, :p] = L[:p, p:].T
    w, V = np.linalg.eigh(L)
    V = V[:, np.argsort(w)[::-1][:k]]
    return Rxi @ V[:p], Ryi @ V[p:]


def semicca_projection(Sxy, Wx, Wy, Sxx_u, Syy_u):
    """The paired cross-covariance projected on the SemiCCA subspaces (metric: unpaired covariances)."""
    Px = Wx @ np.linalg.pinv(Wx.T @ Sxx_u @ Wx) @ Wx.T
    Py = Wy @ np.linalg.pinv(Wy.T @ Syy_u @ Wy) @ Wy.T
    return Sxx_u @ Px @ Sxy @ Py @ Syy_u


def semicca(X, Y, Sxx_u, Syy_u, betas=(0.2, 0.5, 0.8), ks=None, rng=None):
    rng = rng or np.random.default_rng(CV_SEED)
    q = Y.shape[1]
    ks = ks or sorted({k for k in (1, 2, 4, 8, q) if k <= q})
    a, b = halves(len(X), rng)
    best, score = None, -np.inf
    for beta in betas:
        for k in ks:
            s = 0.0
            for tr, te in ((a, b), (b, a)):
                Xt, Yt = X[tr], Y[tr]
                n = len(tr)
                Wx, Wy = semicca_directions(Xt.T @ Yt / n, Xt.T @ Xt / n, Yt.T @ Yt / n, Sxx_u, Syy_u, beta, k)
                s += cv_score(semicca_projection(Xt.T @ Yt / n, Wx, Wy, Sxx_u, Syy_u), X[te], Y[te])
            if s > score:
                best, score = (beta, k), s
    n = len(X)
    Wx, Wy = semicca_directions(X.T @ Y / n, X.T @ X / n, Y.T @ Y / n, Sxx_u, Syy_u, *best)
    return semicca_projection(X.T @ Y / n, Wx, Wy, Sxx_u, Syy_u), best


# ---------------------------------------------------------------- partially paired Gaussian factor model

def factor_em(X, Y, Sxx_u, n_x, Syy_u, n_y, k, iters=60, tol=1e-7, init=None):
    """Maximum likelihood (EM) for z = (x, y) = L f + e, f ~ N(0, I_k), e ~ N(0, diag(psi)), from RNA-only cells
    (covariance Sxx_u, n_x cells), protein-only cells (Syy_u, n_y) and B paired cells; returns L and psi."""
    p, q = X.shape[1], Y.shape[1]
    B = len(X)
    Z = np.hstack([X, Y])
    Sp = Z.T @ Z / B
    pats = [(np.arange(p), Sxx_u, float(n_x)), (np.arange(p, p + q), Syy_u, float(n_y)), (np.arange(p + q), Sp, float(B))]
    N = n_x + n_y + B
    if init is None:
        ev, V = np.linalg.eigh(Sxx_u)
        o = np.argsort(ev)[::-1][:k]
        Lx = V[:, o] * np.sqrt(np.maximum(ev[o] - np.median(ev), 1e-3))
        # protein loadings through the paired regression on the RNA factor scores
        F = X @ Lx @ np.linalg.pinv(Lx.T @ Lx)
        Ly = np.linalg.lstsq(F, Y, rcond=None)[0].T
        L = np.vstack([Lx, Ly])
        psi = np.maximum(np.concatenate([np.diag(Sxx_u), np.diag(Syy_u)]) - np.sum(L * L, 1), 0.05)
    else:
        L, psi = init
    prev = -np.inf
    for it in range(iters):
        Szf = np.zeros((p + q, k))
        Szz = np.zeros(p + q)
        Sff_tot = np.zeros((k, k))
        ll = 0.0
        for O, S, n in pats:
            LO, pO = L[O], psi[O]
            A = np.eye(k) + LO.T @ (LO / pO[:, None])
            Ai = np.linalg.inv(A)
            G = Ai @ (LO / pO[:, None]).T                 # k x |O|  (posterior mean map)
            Vf = Ai                                        # posterior covariance of f
            Eff = G @ S @ G.T + Vf                         # E[f f'] per cell
            Sff_tot += n * Eff
            m = np.setdiff1d(np.arange(p + q), O)
            Szf[O] += n * (S @ G.T)
            Szz[O] += n * np.diag(S)
            if len(m):
                Lm = L[m]
                Szf[m] += n * (Lm @ Eff)
                Szz[m] += n * (np.einsum("ik,kl,il->i", Lm, Eff, Lm) + psi[m])
            # log-likelihood of the observed block (for convergence)
            sign, logdet = np.linalg.slogdet(A)
            SigInv_S = S / pO[:, None] - (LO / pO[:, None]) @ (Ai @ ((LO / pO[:, None]).T @ S))
            ll += -0.5 * n * (np.sum(np.log(pO)) + logdet + np.trace(SigInv_S))
        L = Szf @ np.linalg.inv(Sff_tot)
        psi = np.maximum((Szz - np.sum(L * Szf, 1)) / N, 1e-3)
        if ll - prev < tol * abs(ll):
            break
        prev = ll
    return L, psi


def factor_model(X, Y, Sxx_u, n_x, Syy_u, n_y, ks=(2, 5, 10, 20, 40), rng=None):
    rng = rng or np.random.default_rng(CV_SEED)
    p = X.shape[1]
    a, b = halves(len(X), rng)
    best, score = None, -np.inf
    for k in ks:
        s = 0.0
        for tr, te in ((a, b), (b, a)):
            L, _ = factor_em(X[tr], Y[tr], Sxx_u, n_x, Syy_u, n_y, k)
            s += cv_score(L[:p] @ L[p:].T, X[te], Y[te])
        if s > score:
            best, score = k, s
    L, _ = factor_em(X, Y, Sxx_u, n_x, Syy_u, n_y, best)
    return L[:p] @ L[p:].T, best


# ---------------------------------------------------------------- reference regression (ridge)

RIDGE_GRID = 10.0 ** np.arange(-3, 2.01, 0.5)


def ridge(X, Y, Rx_pool, unpaired_gram, rng=None):
    """Protein on RNA by ridge regression in the paired cells (Gram from the paired cells or from unpaired cells);
    returns W' (p x q) so that a recipient's prediction is R_x,recipient W'."""
    rng = rng or np.random.default_rng(CV_SEED)
    a, b = halves(len(X), rng)
    score = np.zeros(len(RIDGE_GRID))
    for tr, te in ((a, b), (b, a)):
        Xa, Ya = X[tr], Y[tr]
        Gram = Rx_pool if unpaired_gram else Xa.T @ Xa / len(Xa)
        Ca = Xa.T @ Ya / len(Xa)
        for i, lam in enumerate(RIDGE_GRID):
            Wt = np.linalg.solve(Gram + lam * np.eye(len(Gram)), Ca)
            score[i] += cv_score(Rx_pool @ Wt, X[te], Y[te])
    lam = RIDGE_GRID[int(np.argmax(score))]
    Gram = Rx_pool if unpaired_gram else X.T @ X / len(X)
    return np.linalg.solve(Gram + lam * np.eye(len(Gram)), X.T @ Y / len(X)), float(lam)


# ---------------------------------------------------------------- cross-validated low rank (paired only)

def low_rank(X, Y, rng=None):
    rng = rng or np.random.default_rng(CV_SEED)
    a, b = halves(len(X), rng)
    rmax = min(X.shape[1], Y.shape[1])
    score = np.zeros(rmax + 1)
    for tr, te in ((a, b), (b, a)):
        U, s, Vt = np.linalg.svd(X[tr].T @ Y[tr] / len(tr), full_matrices=False)
        for r in range(rmax + 1):
            score[r] += cv_score((U[:, :r] * s[:r]) @ Vt[:r], X[te], Y[te])
    r = int(np.argmax(score))
    U, s, Vt = np.linalg.svd(X.T @ Y / len(X), full_matrices=False)
    return (U[:, :r] * s[:r]) @ Vt[:r], r
