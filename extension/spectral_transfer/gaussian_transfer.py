"""Closed-form transfer of cross-assay dependence for continuous assays.

Model. Within each person k, RNA features x and protein features y have joint
density proportional to exp(x^T B y + a_k(x) + b_k(y)), the continuous analogue
of the paper's Eq. (1). For Gaussian within-assay laws this is a joint Gaussian
whose precision has off-diagonal block -B, shared across people, while the
within-assay covariances Sx_k and Sy_k are person-specific.

Transfer operator. Given Sx, Sy and B, the cross-covariance is

    C = Sx^{1/2} U diag(g(beta)) V^T Sy^{1/2},   Sx^{1/2} B Sy^{1/2} = U diag(beta) V^T,
    g(beta) = (sqrt(1 + 4 beta^2) - 1) / (2 beta),

the closed form of the Gaussian entropic coupling (as in entropic optimal
transport between Gaussian measures). To first order C = Sx B Sy, the paper's
response formula (main text Eq. 5); g saturates below one.

Fitting. With paired reference people, the profile log-likelihood of the shared
B (within-assay blocks profiled out) is concave with gradient
sum_k n_k (Sxy_k - C_k(B)): the fitted interaction makes the transferred
cross-covariances match the observed ones on average.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def psd_sqrt(S):
    w, V = np.linalg.eigh((S + S.T) / 2)
    return (V * np.sqrt(np.clip(w, 0, None))) @ V.T


def saturate(beta):
    beta = np.asarray(beta, float)
    out = beta.copy()
    big = np.abs(beta) > 1e-8
    out[big] = (np.sqrt(1 + 4 * beta[big] ** 2) - 1) / (2 * beta[big])
    return out


class Marginal:
    """Within-assay second moments of one person, with cached square roots."""

    def __init__(self, Sx, Sy):
        self.Sx, self.Sy = Sx, Sy
        self.Rx, self.Ry = psd_sqrt(Sx), psd_sqrt(Sy)


def transfer(marginal: Marginal, B):
    """Closed-form cross-covariance and saturated singular values."""
    U, beta, Vt = np.linalg.svd(marginal.Rx @ B @ marginal.Ry, full_matrices=False)
    gamma = saturate(beta)
    return marginal.Rx @ (U * gamma) @ Vt @ marginal.Ry, gamma


def first_order(marginal: Marginal, B):
    return marginal.Sx @ B @ marginal.Sy


def profile_loglik(B, people, ridge=0.0):
    """Concave profile log-likelihood of a shared B and its gradient.

    people: iterable of (n_k, Marginal_k, Sxy_k) from paired reference data.
    Per cell and up to constants, person k contributes
        <Sxy_k, B> - <C_k(B), B> - 1/2 sum log(1 - gamma_k^2).
    """
    value, grad = -0.5 * ridge * np.sum(B * B), -ridge * B
    for n, marginal, Sxy in people:
        C, gamma = transfer(marginal, B)
        value += n * (np.sum(Sxy * B) - np.sum(C * B) - 0.5 * np.sum(np.log1p(-gamma ** 2)))
        grad += n * (Sxy - C)
    return value, grad


def fit_interaction(people, ridge, B0=None, maxiter=500):
    """Maximize the concave penalized profile likelihood by L-BFGS."""
    people = list(people)
    p, q = people[0][1].Sx.shape[0], people[0][1].Sy.shape[0]
    total = sum(n for n, _, _ in people)
    shape = (p, q)

    def objective(z):
        value, grad = profile_loglik(z.reshape(shape), people, ridge * total)
        return -value / total, -grad.ravel() / total

    z0 = np.zeros(p * q) if B0 is None else B0.ravel()
    result = minimize(objective, z0, jac=True, method="L-BFGS-B",
                      options={"maxiter": maxiter, "maxfun": 2 * maxiter, "gtol": 1e-9, "ftol": 1e-14})
    return result.x.reshape(shape), result


def joint_precision(Sx, C, Sy):
    return np.linalg.inv(np.block([[Sx, C], [C.T, Sy]]))
