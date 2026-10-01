#!/usr/bin/env python3
"""Numerical checks of Propositions S13-S15 (Supplementary Note 13). Not proofs; they guard against algebra slips.

    python check_theory.py      # results/check_theory.json

S13: for per-cell products of non-Gaussian, correlated x and y, the risk of the implemented rule
(1 - t_hat/||m||^2)_+ m, with t_hat the unbiased trace estimate from the same cells, stays within
sqrt(36 l/B + 22 kappa tau/B^2) of sqrt(R_lin) in root-risk. S14: the oracle-linear saving is >= 1, <= 1/pi_min over
signal blocks, depends on the dependence only through the signal shares, and has limits 1/pi(signal blocks) and
1 + chi^2(w || pi). S15: for jointly Gaussian (x, y), tau_b = tau_b(independence) + ||theta_b||^2.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

HERE = Path(__file__).resolve().parent
SEED = 20261070


def sample_xy(rng, n, a, b, rho, kind):
    """Correlated (x, y) with non-Gaussian margins via a Gaussian copula; returns uncentred draws."""
    L = rng.standard_normal((a + b, a + b)) * 0.3
    C = L @ L.T + np.eye(a + b)
    d = np.sqrt(np.diag(C))
    C = C / np.outer(d, d)
    C[:a, a:] *= rho
    C[a:, :a] *= rho
    w, V = np.linalg.eigh(C)
    C = V @ np.diag(np.maximum(w, 1e-3)) @ V.T
    d = np.sqrt(np.diag(C))
    C = C / np.outer(d, d)
    Lc = np.linalg.cholesky(C)
    def draw(m):
        g = rng.standard_normal((m, a + b)) @ Lc.T
        if kind == "lognormal":
            v = np.exp(0.8 * g)
        elif kind == "poisson":
            v = rng.poisson(np.exp(0.5 + 0.7 * g)).astype(float)
        else:
            v = g
        return v[:, :a], v[:, a:]
    return draw


def products(x, y):
    return (x[:, :, None] * y[:, None, :]).reshape(len(x), -1)


def rule(Z):
    B = len(Z)
    m = Z.mean(0)
    t = float(np.sum((Z - m) ** 2)) / (B * (B - 1))
    n2 = float(m @ m)
    f = max(1.0 - t / n2, 0.0) if n2 > 0 else 0.0
    return f * m


def check_s13(rng):
    out = []
    for kind in ("gaussian", "lognormal", "poisson"):
        for a, b in ((5, 3), (12, 10)):
            for rho in (0.0, 0.3, 0.8):
                draw = sample_xy(rng, 0, a, b, rho, kind)
                x, y = draw(400000)
                mx, my = x.mean(0), y.mean(0)
                Zbig = products(x - mx, y - my)
                theta = Zbig.mean(0)
                E = Zbig - theta
                Sig = E.T @ E / len(E)
                tau = float(np.trace(Sig))
                ell = float(np.linalg.eigvalsh(Sig)[-1])
                kappa = float(np.var(np.sum(E * E, 1))) / tau ** 2
                for B in (5, 10, 25, 100):
                    s = tau / B
                    t2 = float(theta @ theta)
                    Rlin = t2 * s / (t2 + s)
                    reps = 20000 if B <= 25 else 6000
                    err = np.empty(reps)
                    for r in range(reps):
                        idx = rng.integers(0, len(x), B)
                        Z = products(x[idx] - mx, y[idx] - my)
                        err[r] = float(np.sum((rule(Z) - theta) ** 2))
                    R = float(err.mean())
                    se = float(err.std() / np.sqrt(reps))
                    Delta = 36 * ell / B + 22 * kappa * tau / B ** 2
                    gap = abs(np.sqrt(R) - np.sqrt(Rlin))
                    out.append({"kind": kind, "d": a * b, "rho": rho, "B": B, "risk": R, "risk_se": se, "R_lin": Rlin,
                                "sqrt_gap": gap, "sqrt_Delta": float(np.sqrt(Delta)), "holds": bool(gap <= np.sqrt(Delta)),
                                "eff_dim": tau / ell, "kappa": kappa, "rel_gap": gap / max(np.sqrt(Rlin), 1e-12)})
    return out


def saving(w, pi, eps):
    """Oracle-linear saving S(eps) from signal shares w and noise shares pi (both sum to one)."""
    phi = lambda beta: float(np.sum(np.where(w > 0, w * pi / (beta * w + pi), 0.0)))
    beta_k = brentq(lambda b: phi(b) - eps, 1e-12, 1e12)
    beta_1 = (1 - eps) / eps
    return beta_1 / beta_k


def check_s14(rng):
    rows, ok = [], True
    for _ in range(2000):
        K = int(rng.integers(2, 9))
        pi = rng.dirichlet(np.ones(K))
        w = rng.dirichlet(np.full(K, float(rng.choice([0.2, 1, 5]))))
        if rng.random() < 0.3:
            w[rng.integers(0, K)] = 0.0
            w /= w.sum()
        eps = float(rng.uniform(0.02, 0.98))
        S = saving(w, pi, eps)
        lo_ok = S >= 1 - 1e-9
        hi = 1 / pi[w > 0].min()
        hi_ok = S <= hi * (1 + 1e-9)
        # scale invariance: multiply all signals by c and all noises by c' leaves S unchanged (shares unchanged)
        ok &= lo_ok and hi_ok
        rows.append((S, hi, lo_ok, hi_ok))
    # limits
    lim = []
    for _ in range(200):
        K = int(rng.integers(2, 8))
        pi = rng.dirichlet(np.ones(K))
        w = rng.dirichlet(np.ones(K))
        if rng.random() < 0.5:
            w[0] = 0.0
            w /= w.sum()
        s_hi = saving(w, pi, 1e-6)
        s_lo = saving(w, pi, 1 - 1e-6)
        lim.append((abs(s_hi - 1 / pi[w > 0].sum()) / s_hi, abs(s_lo - np.sum(w ** 2 / pi)) / s_lo))
    lim = np.array(lim)
    # equality case w = pi
    pi = rng.dirichlet(np.ones(6))
    eq = [saving(pi.copy(), pi, e) for e in (0.1, 0.5, 0.9)]
    return {"random_cases": len(rows), "bounds_hold": bool(ok),
            "limit_high_accuracy_max_rel_err": float(lim[:, 0].max()),
            "limit_low_accuracy_max_rel_err": float(lim[:, 1].max()), "equal_shares_saving": eq}


def check_s15(rng):
    a, b = 6, 4
    draw = sample_xy(rng, 0, a, b, 0.6, "gaussian")
    x, y = draw(1000000)
    x -= x.mean(0)
    y -= y.mean(0)
    Z = products(x, y)
    theta = Z.mean(0)
    tau = float(np.sum(Z.var(0)))
    tau_ind = float(np.sum(x.var(0))) * float(np.sum(y.var(0)))
    return {"tau": tau, "tau_independence_plus_signal": tau_ind + float(theta @ theta),
            "rel_diff": abs(tau - tau_ind - float(theta @ theta)) / tau}


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    res = {"S14": check_s14(rng), "S15": check_s15(rng), "S13": check_s13(rng)}
    s13 = res["S13"]
    res["S13_summary"] = {"cases": len(s13), "all_hold": all(r["holds"] for r in s13),
                          "max_gap_over_bound": max(r["sqrt_gap"] / r["sqrt_Delta"] for r in s13)}
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "check_theory.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "S13"}, indent=1))
