"""Certified inner bounds on the individual-frequency identified interval.

For a fixed bilinear interaction S(x, y) = x^T B y on binary RNA and protein
profiles, a recipient described only by its 18 marker frequencies identifies
the query cell t = Pr(x_i = 1, y_j = 1) up to an interval [l, h]
(Additional file 1, Proposition S1). This module constructs explicit,
strictly positive assay laws that match every marker frequency, reconstructs
their joint law with the fixed interaction, and records the query value. Each
recorded value is attained, so the reported [l_hat, h_hat] lies inside [l, h].
Nothing here certifies global optimality.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.linalg.blas import dsyrk
from scipy.optimize import minimize

MARKERS = 9
SUPPORT = ((np.arange(2**MARKERS)[:, None] >> np.arange(MARKERS)) & 1).astype(float)


class ScalingError(RuntimeError):
    """Matrix scaling did not reach the requested marginal tolerance."""


def kernel(interaction: np.ndarray) -> np.ndarray:
    """Positive kernel exp(S) on the full binary support, scaled to max one."""
    score = SUPPORT @ interaction @ SUPPORT.T
    return np.exp(score - score.max())


def scale(K, r, c, u=None, v=None, tol=1e-12, maxit=50000):
    """Sinkhorn scaling of positive K to positive marginals r and c."""
    u = np.ones(len(r)) if u is None else u.copy()
    v = np.ones(len(c)) if v is None else v.copy()
    error = np.inf
    for iteration in range(maxit):
        u = r / (K @ v)
        v = c / (K.T @ u)
        if iteration % 5 == 4:
            row = u * (K @ v)
            error = float(np.max(np.abs(row - r)))  # columns are exact after the v step
            if error < tol:
                return u[:, None] * K * v[None, :], u, v
    raise ScalingError(f"marginal error {error:.3g} after {maxit} iterations")


def log_odds(t, m, n):
    """Log odds ratio of the 2x2 table with margins m, n and cell t."""
    t, m, n = np.broadcast_arrays(np.asarray(t, float), m, n)
    return np.log(t) + np.log1p(-m - n + t) - np.log(m - t) - np.log(n - t)


def cell_from_log_odds(theta, m, n):
    """Cell probability with margins m, n in (0, 1) and log odds theta."""
    e = np.exp(theta)
    a = 1.0 - e
    b = 1.0 - m - n + e * (m + n)
    c = -e * m * n
    if abs(a) < 1e-14:
        return -c / b
    disc = np.sqrt(max(b * b - 4 * a * c, 0.0))
    q = -0.5 * (b + np.copysign(disc, b))
    low, high = max(0.0, m + n - 1), min(m, n)
    roots = [q / a, c / q] if q != 0 else [-b / a]
    feasible = [z for z in roots if low - 1e-15 <= z <= high + 1e-15]
    if not feasible:
        raise ValueError("no feasible cell probability")
    return float(np.clip(feasible[0], low, high))


def table(t, m, n):
    return np.array([[1 - m - n + t, n - t], [m - t, t]])


def minimax(t_low, t_high, m, n):
    """Equalizing forward-KL prediction and its risk (Proposition S1).

    Returns (theta_star, risk) with risk = KL(P(t_low) || P_star), in nats.
    Applied to an inner interval, risk is a lower bound on the minimax risk
    over the sharp interval, because the sharp interval contains it.
    """
    if t_high - t_low < 1e-15:
        return float(log_odds(t_low, m, n)), 0.0
    low, high = table(t_low, m, n), table(t_high, m, n)

    def entropy(p):
        p = p[p > 0]
        return -float(np.sum(p * np.log(p)))

    theta = (entropy(low) - entropy(high)) / (t_high - t_low)
    star = table(cell_from_log_odds(theta, m, n), m, n)
    risk = float(np.sum(np.where(low > 0, low * np.log(low / star), 0.0)))
    return float(theta), risk


def frechet(mu, centre):
    """Frechet bounds for Pr(x_centre = 1, x_k = 1), k != centre."""
    others = [k for k in range(MARKERS) if k != centre]
    low = np.maximum(0.0, mu[centre] + mu[others] - 1.0)
    high = np.minimum(mu[centre], mu[others])
    return others, low, high


def star_law(mu, centre, s):
    """Star law: markers conditionally independent given the centre marker.

    s in [0, 1]^8 places each pairwise overlap within its Frechet bounds.
    Returns the law on the 512 states and its derivative with respect to s.
    All marker means equal mu exactly, for every s.
    """
    others, low, high = frechet(mu, centre)
    span = high - low
    overlap = low + span * s
    above = overlap / mu[centre]                          # Pr(x_k=1 | x_c=1)
    below = (mu[others] - overlap) / (1.0 - mu[centre])   # Pr(x_k=1 | x_c=0)
    xc = SUPPORT[:, centre][:, None] == 1
    xk = SUPPORT[:, others] == 1
    factors = np.where(xc, np.where(xk, above, 1 - above), np.where(xk, below, 1 - below))
    base = np.where(SUPPORT[:, centre] == 1, mu[centre], 1.0 - mu[centre])
    law = base * factors.prod(axis=1)
    d_above = span / mu[centre]
    d_below = -span / (1.0 - mu[centre])
    dlog = np.where(
        xc,
        np.where(xk, d_above / above, -d_above / (1 - above)),
        np.where(xk, d_below / below, -d_below / (1 - below)),
    )
    return law, law[:, None] * dlog


def independence_start(mu, centre):
    others, low, high = frechet(mu, centre)
    return np.clip((mu[centre] * mu[others] - low) / (high - low), 0.0, 1.0)


def first_order_vertices(interaction, mu, nu, i, j):
    """Exact weak-interaction range of (Sigma_x B Sigma_y)_ij at fixed means.

    Only the covariances of x_i with each RNA marker and of y_j with each
    protein marker enter this coefficient. A star law realizes any combination
    of those pairwise tables, so the range is a bilinear program over a product
    of Frechet boxes; it is attained at box vertices. Returns
    ((k_min, k_max), (s_min, s_max)) with s the 16 star parameters.
    """
    ox, lox, hix = frechet(mu, i)
    oy, loy, hiy = frechet(nu, j)
    lox, hix = lox - mu[i] * mu[ox], hix - mu[i] * mu[ox]
    loy, hiy = loy - nu[j] * nu[oy], hiy - nu[j] * nu[oy]
    bits = ((np.arange(256)[:, None] >> np.arange(8)) & 1).astype(float)
    sigma_y = np.empty((256, MARKERS))
    sigma_y[:, j] = nu[j] * (1 - nu[j])
    sigma_y[:, oy] = np.where(bits == 1, hiy, loy)
    weights = sigma_y @ interaction.T                     # rows: B sigma_y
    base = weights[:, i] * mu[i] * (1 - mu[i])
    wx = weights[:, ox]
    high = base + np.maximum(wx * lox, wx * hix).sum(axis=1)
    low = base + np.minimum(wx * lox, wx * hix).sum(axis=1)
    kmax, kmin = int(np.argmax(high)), int(np.argmin(low))
    s_max = np.concatenate([(wx[kmax] > 0).astype(float), bits[kmax]])
    s_min = np.concatenate([(wx[kmin] <= 0).astype(float), bits[kmin]])
    return (float(low[kmin]), float(high[kmax])), (s_min, s_max)


def first_order_bruteforce(interaction, mu, nu, i, j):
    """Reference enumeration over all 2^16 vertices (tests only)."""
    ox, lox, hix = frechet(mu, i)
    oy, loy, hiy = frechet(nu, j)
    values = []
    for code in range(2**16):
        z = (code >> np.arange(16)) & 1
        sx = np.empty(MARKERS); sx[i] = mu[i] * (1 - mu[i])
        sx[ox] = np.where(z[:8] == 1, hix, lox) - mu[i] * mu[ox]
        sy = np.empty(MARKERS); sy[j] = nu[j] * (1 - nu[j])
        sy[oy] = np.where(z[8:] == 1, hiy, loy) - nu[j] * nu[oy]
        values.append(sx @ interaction @ sy)
    return min(values), max(values)


class StarQuery:
    """Query value and exact gradient over the 16 star parameters."""

    def __init__(self, K, mu, nu, i, j, margin=1e-6, tol=1e-11):
        self.K, self.mu, self.nu, self.i, self.j = K, mu, nu, i, j
        self.margin, self.tol = margin, tol
        self.u = self.v = None
        self.xi = SUPPORT[:, i]
        self.yj = SUPPORT[:, j]
        self.evaluations = 0
        self.failures = 0
        self.best = {+1: (-np.inf, None), -1: (np.inf, None)}

    def clip(self, z):
        return np.clip(z, self.margin, 1.0 - self.margin)

    def laws(self, z):
        z = self.clip(z)
        r, dr = star_law(self.mu, self.i, z[:8])
        c, dc = star_law(self.nu, self.j, z[8:])
        return z, r, dr, c, dc

    def value(self, z, tol=None):
        z, r, _, c, _ = self.laws(z)
        Q, _, _ = scale(self.K, r, c, tol=self.tol if tol is None else tol)
        return float(self.xi @ Q @ self.yj), Q, r, c

    def value_and_gradient(self, z):
        z, r, dr, c, dc = self.laws(z)
        Q, self.u, self.v = scale(self.K, r, c, self.u, self.v, tol=self.tol)
        self.evaluations += 1
        weighted = Q * np.outer(self.xi, self.yj)
        t = float(weighted.sum())
        ga, gb = weighted.sum(axis=1), weighted.sum(axis=0)
        sr, sc = np.sqrt(r), np.sqrt(c)
        A = Q / sr[:, None] / sc[None, :]
        M = -dsyrk(1.0, A, trans=1)                      # upper triangle of -A^T A
        M[np.diag_indices_from(M)] += 1.0
        M += np.outer(sc, sc)
        rhs = (gb - Q.T @ (ga / r)) / sc
        kappa = cho_solve(cho_factor(M, lower=False, check_finite=False), rhs) / sc
        lam = (ga - Q @ kappa) / r
        gradient = np.concatenate([lam @ dr, kappa @ dc])
        for sign in (+1, -1):
            if sign * t > sign * self.best[sign][0]:
                self.best[sign] = (t, z.copy())
        return t, gradient

    def optimize(self, sign, starts, maxiter=200):
        def objective(z):
            try:
                t, g = self.value_and_gradient(z)
            except ScalingError:
                self.failures += 1
                self.u = self.v = None
                return 1e3, np.zeros(16)
            return -sign * t, -sign * g

        for start in starts:
            self.u = self.v = None
            minimize(objective, self.clip(np.asarray(start, float)), jac=True,
                     method="L-BFGS-B", bounds=[(self.margin, 1 - self.margin)] * 16,
                     options={"maxiter": maxiter})
        return self.best[sign]


def certify(K, mu, nu, i, j, z, margin=1e-6, tol=1e-12):
    """Recompute an attained query value from saved star parameters.

    Checks the 18 marker means of the two laws and the joint marginals.
    Returns the query cell probability.
    """
    problem = StarQuery(K, mu, nu, i, j, margin=margin)
    z = problem.clip(np.asarray(z, float))
    t, Q, r, c = problem.value(z, tol=tol)
    assert np.all(r > 0) and np.all(c > 0)
    assert np.max(np.abs(r @ SUPPORT - mu)) < 1e-10
    assert np.max(np.abs(c @ SUPPORT - nu)) < 1e-10
    assert np.max(np.abs(Q.sum(axis=1) - r)) < 1e-10
    assert np.max(np.abs(Q.sum(axis=0) - c)) < 1e-10
    return t


def inner_interval(K, interaction, mu, nu, i, j, margin=1e-6, maxiter=200):
    """Star-family attained extremes of the query cell for one recipient."""
    _, (s_min, s_max) = first_order_vertices(interaction, mu, nu, i, j)
    independent = np.concatenate([independence_start(mu, i), independence_start(nu, j)])
    problem = StarQuery(K, mu, nu, i, j, margin=margin)
    t_high, z_high = problem.optimize(+1, [s_max, independent], maxiter)
    t_low, z_low = problem.optimize(-1, [s_min, independent], maxiter)
    return {
        "t_low": t_low, "t_high": t_high, "z_low": z_low, "z_high": z_high,
        "evaluations": problem.evaluations, "failures": problem.failures,
    }
