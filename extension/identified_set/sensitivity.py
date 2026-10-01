#!/usr/bin/env python3
"""Planned sensitivity check: unrestricted local search beyond the star family.

For the first colon donor and all 81 queries, start from the certified star
endpoints and continue with L-BFGS over all strictly positive assay laws that
match the 18 marker frequencies. A law is parameterized by free log-weights
alpha on the 512 profiles and exponentially tilted in the nine marker
directions so that its marker means are exact. Every evaluated point is
attained, so the extended endpoints remain inner bounds.

    python sensitivity.py
"""

from __future__ import annotations

import json

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits

from identified import (SUPPORT, ScalingError, certify, cho_factor, cho_solve, dsyrk,
                        kernel, log_odds, scale, star_law)
from run_identified import MARGIN, PARTS, RESULTS, load_recipients

DONOR = 95  # first colon donor in the fixed recipient order


def tilt(alpha, mean, start=None):
    """Law proportional to exp(alpha + X lam) with marker means equal to mean."""
    lam = np.zeros(9) if start is None else start.copy()
    for _ in range(200):
        logits = alpha + SUPPORT @ lam
        law = np.exp(logits - logsumexp(logits))
        gradient = law @ SUPPORT - mean
        if np.max(np.abs(gradient)) < 1e-13:
            return law, lam
        centred = SUPPORT - law @ SUPPORT
        hessian = (centred * law[:, None]).T @ centred
        step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        value = logsumexp(logits) - lam @ mean
        scale_step = 1.0
        while scale_step > 1e-8:
            trial = lam - scale_step * step
            if logsumexp(alpha + SUPPORT @ trial) - trial @ mean <= value + 1e-16:
                break
            scale_step /= 2
        lam = trial
    law = np.exp(alpha + SUPPORT @ lam - logsumexp(alpha + SUPPORT @ lam))
    if np.max(np.abs(law @ SUPPORT - mean)) > 1e-10:
        raise ScalingError("tilt did not converge")
    return law, lam


def residual(law, g):
    """Residual of g after regression on the marker indicators under law."""
    mean = law @ SUPPORT
    centred = SUPPORT - mean
    cov = (centred * law[:, None]).T @ centred
    beta = np.linalg.lstsq(cov, (centred * law[:, None]).T @ (g - law @ g), rcond=None)[0]
    return g - law @ g - centred @ beta


class FreeQuery:
    def __init__(self, K, mu, nu, i, j):
        self.K, self.mu, self.nu, self.i, self.j = K, mu, nu, i, j
        self.lam_r = self.lam_c = None
        self.u = self.v = None
        self.best = {+1: (-np.inf, None), -1: (np.inf, None)}

    def evaluate(self, w):
        r, self.lam_r = tilt(w[:512], self.mu, self.lam_r)
        c, self.lam_c = tilt(w[512:], self.nu, self.lam_c)
        Q, self.u, self.v = scale(self.K, r, c, self.u, self.v, tol=1e-11)
        weighted = Q * np.outer(SUPPORT[:, self.i], SUPPORT[:, self.j])
        t = float(weighted.sum())
        ga, gb = weighted.sum(axis=1), weighted.sum(axis=0)
        sr, sc = np.sqrt(r), np.sqrt(c)
        A = Q / sr[:, None] / sc[None, :]
        M = -dsyrk(1.0, A, trans=1)
        M[np.diag_indices_from(M)] += 1.0
        M += np.outer(sc, sc)
        kappa = cho_solve(cho_factor(M, lower=False, check_finite=False),
                          (gb - Q.T @ (ga / r)) / sc) / sc
        lam = (ga - Q @ kappa) / r
        gradient = np.concatenate([r * residual(r, lam), c * residual(c, kappa)])
        for sign in (+1, -1):
            if sign * t > sign * self.best[sign][0]:
                self.best[sign] = (t, (r.copy(), c.copy()))
        return t, gradient

    def optimize(self, sign, start, maxiter=300):
        def objective(w):
            try:
                t, g = self.evaluate(w)
            except (ScalingError, np.linalg.LinAlgError):
                self.u = self.v = self.lam_r = self.lam_c = None
                return 1e3, np.zeros_like(w)
            return -sign * t, -sign * g
        minimize(objective, start, jac=True, method="L-BFGS-B",
                 options={"maxiter": maxiter, "maxfun": 1000, "gtol": 1e-12, "ftol": 1e-15})
        return self.best[sign]


def certify_laws(K, mu, nu, i, j, r, c):
    assert np.all(r > 0) and np.all(c > 0)
    assert np.max(np.abs(r @ SUPPORT - mu)) < 1e-10
    assert np.max(np.abs(c @ SUPPORT - nu)) < 1e-10
    Q, _, _ = scale(K, r, c, tol=1e-12)
    return float(SUPPORT[:, i] @ Q @ SUPPORT[:, j])


def main():
    data = load_recipients()
    saved = np.load(PARTS / f"{DONOR:03d}.npz")   # same parameters as star_endpoints.npz
    K = kernel(data["interaction"])
    mu, nu = data["rna_means"][DONOR], data["protein_means"][DONOR]
    assert data["cohorts"][DONOR] == "Colon" and data["cohorts"][DONOR - 1] != "Colon"
    rows = []
    for q in range(81):
        i, j = divmod(q, 9)
        result = {}
        for sign, key in ((-1, "low"), (+1, "high")):
            z = np.clip(saved[f"z_{key}"][q], MARGIN, 1 - MARGIN)
            star_value = certify(K, mu, nu, i, j, z, MARGIN)   # as in run_identified.merge
            r0, _ = star_law(mu, i, z[:8])
            c0, _ = star_law(nu, j, z[8:])
            problem = FreeQuery(K, mu, nu, i, j)
            t, (r, c) = problem.optimize(sign, np.concatenate([np.log(r0), np.log(c0)]))
            certified = certify_laws(K, mu, nu, i, j, r, c)
            if sign * certified < sign * star_value:        # search never reports a worse point
                certified = star_value
            result[key] = (star_value, certified)
        m, n = mu[i], nu[j]
        rows.append({
            "rna": i, "protein": j,
            "star_log_odds": [float(log_odds(result["low"][0], m, n)), float(log_odds(result["high"][0], m, n))],
            "free_log_odds": [float(log_odds(result["low"][1], m, n)), float(log_odds(result["high"][1], m, n))],
        })
        print(q, rows[-1], flush=True)
    change_low = np.array([r["star_log_odds"][0] - r["free_log_odds"][0] for r in rows])
    change_high = np.array([r["free_log_odds"][1] - r["star_log_odds"][1] for r in rows])
    report = {
        "donor_index": DONOR, "queries": 81,
        "lower_endpoint_extension_log_odds": {"median": float(np.median(change_low)),
            "max": float(change_low.max()), "min": float(change_low.min())},
        "upper_endpoint_extension_log_odds": {"median": float(np.median(change_high)),
            "max": float(change_high.max()), "min": float(change_high.min())},
        "star_width_median": float(np.median([r["star_log_odds"][1] - r["star_log_odds"][0] for r in rows])),
        "free_width_median": float(np.median([r["free_log_odds"][1] - r["free_log_odds"][0] for r in rows])),
        "rows": rows,
    }
    (RESULTS / "sensitivity_free_search.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
