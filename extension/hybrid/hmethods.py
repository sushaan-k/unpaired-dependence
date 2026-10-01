"""Estimators of the hybrid development comparison (PLAN.md).

Units: features divided by pooled within-population standard deviations of the
unpaired pool. Every estimate is a p x q gene-protein cross-correlation.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import warnings

from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import lasso_path

warnings.filterwarnings("ignore", category=ConvergenceWarning)

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE.parent / "perturbation"))

import pmethods as pm  # noqa: E402

RESERVOIR_SALT = "hybrid-reservoir-v1"
RESERVOIR_FRACTION = 0.25
MIN_POOL_HALF = 10
MIN_GUIDE_CELLS = 3
FLIPS, FLIP_SEED = 50, 20261021
LAMBDA_STEPS = 12
KAPPA_MAX = 0.95
PSI_FLOOR = 0.05


def reservoir(cells):
    """True for the 25% of cells, by barcode hash, that form the paired calibration reservoir."""
    return np.array([int(hashlib.sha256(f"{RESERVOIR_SALT}|{c}".encode()).hexdigest()[:12], 16) / 16 ** 12
                     < RESERVOIR_FRACTION for c in cells])


def first_guide(guides):
    return np.array([g.split(";")[0] for g in guides])


def cond_of(key):
    return key.split("|")[1]


class Pool:
    """Unpaired summaries of the training populations from pool cells (RNA: part 0; protein: part 1)."""

    def __init__(self, x, y, pop, part, guide, keys):
        keys = [k for k in keys if np.sum((pop == k) & (part == 0)) >= MIN_POOL_HALF
                and np.sum((pop == k) & (part == 1)) >= MIN_POOL_HALF]
        self.keys = keys
        self.raw = pm.moments(x, y, pop, part, keys)
        self.sd_x, self.sd_y = pm.est.pooled_sd(self.raw)
        self.mx = {r["patient"]: r["mx"] for r in self.raw}
        self.my = {r["patient"]: r["my"] for r in self.raw}
        self.nx = {r["patient"]: r["nx"] for r in self.raw}
        self.ny = {r["patient"]: r["ny"] for r in self.raw}
        nx = np.array([r["nx"] for r in self.raw], float)
        ny = np.array([r["ny"] for r in self.raw], float)
        Sx = sum(n * r["Sx"] for n, r in zip(nx, self.raw)) / nx.sum() / np.outer(self.sd_x, self.sd_x)
        Sy = sum(n * r["Sy"] for n, r in zip(ny, self.raw)) / ny.sum() / np.outer(self.sd_y, self.sd_y)
        self.Sx_raw, self.Sy_raw = Sx, Sy
        self.Rx, self.Ry = pm.shrink(Sx), pm.shrink(Sy)
        self.marginal = pm.Marginal(self.Rx, self.Ry)
        # standardized, condition-centred population means (weights: cells)
        self.conds = sorted({cond_of(k) for k in keys})
        self.cx, self.cy = {}, {}
        for c in self.conds:
            ks = [k for k in keys if cond_of(k) == c]
            wx = np.array([self.nx[k] for k in ks], float)
            wy = np.array([self.ny[k] for k in ks], float)
            self.cx[c] = sum(w * self.mx[k] for w, k in zip(wx, ks)) / wx.sum() / self.sd_x
            self.cy[c] = sum(w * self.my[k] for w, k in zip(wy, ks)) / wy.sum() / self.sd_y
        self.dx = {k: self.mx[k] / self.sd_x - self.cx[cond_of(k)] for k in keys}
        self.dy = {k: self.my[k] / self.sd_y - self.cy[cond_of(k)] for k in keys}
        # guide-level RNA means (standardized), RNA pool cells only
        g = first_guide(guide)
        self.guide_means = {}
        for k in keys:
            m = (pop == k) & (part == 0)
            xs = x[m] / self.sd_x
            gs = g[m]
            rows = []
            for u in sorted(set(gs)):
                sel = gs == u
                if sel.sum() >= MIN_GUIDE_CELLS:
                    rows.append(xs[sel].mean(0))
            self.guide_means[k] = np.array(rows)

    def centre_cells(self, x, y, pop):
        """Paired calibration cells centred on their population's unpaired means, in pool SD units."""
        mx = np.array([self.mx[k] for k in pop])
        my = np.array([self.my[k] for k in pop])
        return (x - mx) / self.sd_x, (y - my) / self.sd_y

    def psi_and_B(self, W):
        psi = np.diag(self.Sy_raw) - np.diag(W @ self.Sx_raw @ W.T)
        psi = np.maximum(psi, PSI_FLOOR * np.mean(np.diag(self.Sy_raw)))
        return psi, W.T / psi[None, :]


# ---------------------------------------------------------------- unpaired estimators

def current(pool):
    """The frozen estimator of extension/perturbation on the pool."""
    ch = pm.fit_channel(pm.centre_within_condition(pool.raw))
    return {"B": ch["B"], "W": ch["W"], "U": ch["VS"], "dim": int(ch["VS"].shape[1]), "scale": ch["scale"]}


def crossfold_gram(pool, keys, signs=None):
    """Between-population RNA Gram matrix from products of means of different guides (unbiased for sampling noise)."""
    p = len(pool.sd_x)
    used = [k for k in keys if len(pool.guide_means[k]) >= 2]
    w = np.array([pool.nx[k] for k in used], float)
    w /= w.sum()
    G = np.zeros((p, p))
    for wk, k in zip(w, used):
        D = pool.guide_means[k] - pool.cx[cond_of(k)]
        if signs is not None:
            D = D * signs[k][:, None]
        S = D.sum(0)
        K = len(D)
        G += wk * (np.outer(S, S) - D.T @ D) / (K * (K - 1))
    return G, used, w


def noise_aware(pool, keys=None, flips=FLIPS, seed=FLIP_SEED):
    keys = keys or pool.keys
    G, used, w = crossfold_gram(pool, keys)
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(flips):
        signs = {k: rng.choice([-1.0, 1.0], size=len(pool.guide_means[k])) for k in used}
        Gf, _, _ = crossfold_gram(pool, keys, signs)
        null.append(np.linalg.eigvalsh(Gf)[-1])
    threshold = float(np.quantile(null, 0.95))
    ev, vec = np.linalg.eigh(G)
    order = np.argsort(ev)[::-1]
    ev, vec = ev[order], vec[:, order]
    k = int(np.sum(ev > threshold))
    U = vec[:, :k]
    C_yx = sum(wk * np.outer(pool.dy[kk], pool.dx[kk]) for wk, kk in zip(w, used))
    W = (C_yx @ U) / ev[:k][None, :] @ U.T if k else np.zeros((len(pool.sd_y), len(pool.sd_x)))
    psi, B = pool.psi_and_B(W)
    return {"B": B, "W": W, "U": U, "dim": k, "eigenvalues": ev[:10].tolist(), "threshold": threshold}


def debiased_moments(pool, keys):
    """Infinite-split cross-fold moments (SplitUP): plug-in Gram minus the sampling-noise term, and the cross-moment."""
    p, q = len(pool.sd_x), len(pool.sd_y)
    N = float(sum(pool.nx[k] for k in keys))
    G, C = np.zeros((p, p)), np.zeros((p, q))
    for c in sorted({cond_of(k) for k in keys}):
        ks = [k for k in keys if cond_of(k) == c]
        n = np.array([pool.nx[k] for k in ks], float)
        ny = np.array([pool.ny[k] for k in ks], float)
        mx = np.array([pool.mx[k] / pool.sd_x for k in ks])
        my = np.array([pool.my[k] / pool.sd_y for k in ks])
        cx, cy = (n @ mx) / n.sum(), (ny @ my) / ny.sum()
        for i, k in enumerate(ks):
            r = pool.raw[pool.keys.index(k)]
            Sx = r["Sx"] / np.outer(pool.sd_x, pool.sd_x)
            d = mx[i] - cx
            G += (n[i] / N) * (np.outer(d, d) - (1 - n[i] / n.sum()) * Sx / n[i])
            C += (n[i] / N) * np.outer(d, my[i] - cy)
    return G, C


def lasso_coefs(G, c, lambdas):
    p = G.shape[0]
    _, coefs, _ = lasso_path(G, c, alphas=lambdas / p, max_iter=10000, tol=1e-4)
    return coefs    # p x len(lambdas)


def splitup(pool, keys=None):
    keys = keys or pool.keys
    folds = pm.target_folds(keys)
    G, C = debiased_moments(pool, keys)
    parts = []
    for f in range(3):
        tr = [k for k in keys if folds[k] != f]
        te = [k for k in keys if folds[k] == f]
        parts.append((debiased_moments(pool, tr), debiased_moments(pool, te)))
    p, q = C.shape
    W = np.zeros((q, p))
    chosen = []
    for j in range(q):
        lam_max = float(np.max(np.abs(G.T @ C[:, j])))
        lambdas = lam_max * 2.0 ** -np.arange(LAMBDA_STEPS, dtype=float)
        loss = np.zeros(LAMBDA_STEPS)
        for (Gtr, Ctr), (Gte, Cte) in parts:
            betas = lasso_coefs(Gtr, Ctr[:, j], lambdas)
            loss += np.sum((Cte[:, [j]] - Gte @ betas) ** 2, axis=0)
        i = int(np.argmin(loss))
        beta = lasso_coefs(G, C[:, j], lambdas[: i + 1])[:, -1]
        S = np.flatnonzero(np.abs(beta) > 1e-10)
        if len(S):
            W[j, S] = np.linalg.lstsq(G[:, S], C[:, j], rcond=None)[0]
        chosen.append({"lambda_step": i, "support": int(len(S))})
    psi, B = pool.psi_and_B(W)
    return {"B": B, "W": W, "tuning": chosen}


# ---------------------------------------------------------------- paired estimators

def products(X, Y):
    """Mean cross-product and the noise variance of each entry (from per-cell products)."""
    n = len(X)
    C = X.T @ Y / n
    var = np.maximum(((X ** 2).T @ (Y ** 2) / n - C ** 2) / n, 1e-12)
    return C, var


def james_stein(C, var):
    s = max(0.0, 1.0 - float(var.sum()) / max(float(np.sum(C ** 2)), 1e-12))
    return s * C


def low_rank(X, Y, rng):
    n = len(X)
    idx = rng.permutation(n)
    h = [idx[: n // 2], idx[n // 2:]]
    halves = [X[i].T @ Y[i] / len(i) for i in h]
    rmax = min(X.shape[1], Y.shape[1])
    score = np.zeros(rmax + 1)
    for a, b in ((0, 1), (1, 0)):
        U, s, Vt = np.linalg.svd(halves[a], full_matrices=False)
        for r in range(rmax + 1):
            M = (U[:, :r] * s[:r]) @ Vt[:r]
            score[r] += 2 * np.sum(M * halves[b]) - np.sum(M * M)
    r = int(np.argmax(score))
    U, s, Vt = np.linalg.svd(X.T @ Y / n, full_matrices=False)
    return (U[:, :r] * s[:r]) @ Vt[:r], r


def eot_interaction(C, pool):
    """Gaussian entropic interaction that reproduces C on the pooled marginals (closed form inverted)."""
    Rxi, Ryi = pm.psd_power(pool.Rx, -0.5), pm.psd_power(pool.Ry, -0.5)
    U, kappa, Vt = np.linalg.svd(Rxi @ C @ Ryi, full_matrices=False)
    kappa = np.minimum(kappa, KAPPA_MAX)
    beta = kappa / (1 - kappa ** 2)
    return Rxi @ (U * beta) @ Vt @ Ryi


EIGEN_EDGES = (5, 20, 60)   # RNA eigen-coordinate blocks: 1-5, 6-20, 21-60, the rest


def eigen_blocks(p):
    edges = [0] + [e for e in EIGEN_EDGES if e < p] + [p]
    return [list(range(a, b)) for a, b in zip(edges[:-1], edges[1:])]


def block_js(Ahat, var, M, blocks):
    """Positive-part James-Stein towards M, separately within each row block (all columns)."""
    D = Ahat - M
    out = M.copy()
    factors = []
    for rows in blocks:
        d, v = D[rows], var[rows]
        f = max(0.0, 1.0 - float(v.sum()) / max(float(np.sum(d * d)), 1e-12))
        out[rows] = M[rows] + f * d
        factors.append(f)
    return out, factors


def program_regression(X, Y, U, Wu):
    """Protein on program scores S = XU in the paired cells, each program's slopes shrunk (positive-part
    James-Stein) towards its unpaired slopes Wu; returns the slopes, the residual protein and the factors."""
    k = U.shape[1]
    if k == 0:
        return np.zeros((Y.shape[1], 0)), Y, []
    S = X @ U
    SSi = np.linalg.inv(S.T @ S)
    What = Y.T @ S @ SSi
    R = Y - S @ What.T
    s2 = np.sum(R * R, axis=0) / max(len(Y) - k, 1)
    var = np.outer(s2, np.diag(SSi))
    Wpost = Wu.copy()
    factors = []
    for j in range(k):
        d = What[:, j] - Wu[:, j]
        f = max(0.0, 1.0 - float(var[:, j].sum()) / max(float(np.sum(d * d)), 1e-12))
        Wpost[:, j] = Wu[:, j] + f * d
        factors.append(f)
    return Wpost, Y - S @ Wpost.T, factors


def eigenbasis(R):
    ev, vec = np.linalg.eigh(R)
    order = np.argsort(ev)[::-1]
    return vec[:, order], np.maximum(ev[order], 1e-9)


def basis_coefficients(X, Y, U, V):
    return products(X @ U, Y @ V)
