"""Reference-side estimators and tuning rules of PLAN.md ("Arms").

Pairing-free estimators see only per-patient within-assay moments computed from
disjoint cell halves (RNA half, protein half). Paired arms see per-patient
paired moments. Every tuned arm uses the same five-value grid, the same three
patient-grouped folds and the same edge rule.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "spectral_transfer"))
from gaussian_transfer import Marginal, fit_interaction, profile_loglik, transfer  # noqa: E402

from data import shrink, standardize  # noqa: E402

GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
RANKS = (2, 4, 8, 16, 32)
MOMENT_OPTIONS = {"maxiter": 20000, "maxfun": 40000, "gtol": 1e-7, "ftol": 1e-13}


# ----------------------------------------------------------------------------- tuning
def patient_folds(patients, salt="folds-v1", k=3):
    order = sorted(set(patients), key=lambda p: hashlib.sha256(f"{salt}|{p}".encode()).hexdigest())
    return {p: i % k for i, p in enumerate(order)}


def tune(grid, loss):
    """Minimise loss over grid with the edge rule (extend x10 at an edge, at most 3 times)."""
    values = list(grid)
    losses = {v: float(loss(v)) for v in values}
    extensions = 0
    while extensions < 3:
        best = min(values, key=lambda v: losses[v])
        if best == values[0]:
            new = values[0] / 10
            values.insert(0, new)
        elif best == values[-1]:
            new = values[-1] * 10
            values.append(new)
        else:
            break
        losses[new] = float(loss(new))
        extensions += 1
    best = min(values, key=lambda v: losses[v])
    return best, {"values": values, "losses": [losses[v] for v in values], "extensions": extensions,
                  "at_edge": bool(best in (values[0], values[-1]))}


def tune_rank(ranks, loss, rmax):
    """Rank version of the edge rule: an upper edge extends to min(10 r, rmax), a lower edge to 1."""
    values = list(ranks)
    losses = {r: float(loss(r)) for r in values}
    best = min(values, key=lambda r: losses[r])
    extensions = 0
    if best == values[-1] and min(10 * best, rmax) > best:
        new = min(10 * best, rmax)
        values.append(new)
        losses[new] = float(loss(new))
        extensions = 1
    elif best == values[0] and best > 1:
        values.insert(0, 1)
        losses[1] = float(loss(1))
        extensions = 1
    best = min(values, key=lambda r: losses[r])
    return best, {"values": values, "losses": [losses[r] for r in values], "extensions": extensions,
                  "at_edge": bool(best in (values[0], values[-1]) and best not in (1, rmax))}


# ----------------------------------------------------------------------------- pairing-free
def pf_raw_moments(train, patients):
    """Per patient: RNA moments from half 0, protein moments from half 1 (never the same cell)."""
    out = []
    for p in patients:
        m = train["patient"] == p
        xr, yr = train["x"][m & (train["half"] == 0)], train["y"][m & (train["half"] == 1)]
        out.append({"patient": p, "nx": len(xr), "mx": xr.mean(0), "Sx": np.cov(xr.T, bias=True),
                    "ny": len(yr), "my": yr.mean(0), "Sy": np.cov(yr.T, bias=True)})
    return out


def pooled_sd(raw):
    nx = np.array([r["nx"] for r in raw], float)
    ny = np.array([r["ny"] for r in raw], float)
    vx = sum(n * np.diag(r["Sx"]) for n, r in zip(nx, raw)) / nx.sum()
    vy = sum(n * np.diag(r["Sy"]) for n, r in zip(ny, raw)) / ny.sum()
    return np.sqrt(vx), np.sqrt(vy)


def pf_units(raw, sd_x, sd_y):
    """(patient, n, mx, my, Sx, Sy) in pooled within-patient SD units, shrunk as colon_scale."""
    units = []
    for r in raw:
        Sx = r["Sx"] / np.outer(sd_x, sd_x)
        Sy = r["Sy"] / np.outer(sd_y, sd_y)
        units.append((r["patient"], r["ny"], r["mx"] / sd_x, r["my"] / sd_y, shrink(Sx), shrink(Sy)))
    return units


def ecological(units, scale):
    """Weighted ridge regression of protein means on RNA means (as pairing_free.ecological)."""
    n = np.array([u[1] for u in units], float)
    MX = np.array([u[2] for u in units])
    MY = np.array([u[3] for u in units])
    w = n / n.sum()
    xbar, ybar = w @ MX, w @ MY
    cx, cy = MX - xbar, MY - ybar
    G = (cx * w[:, None]).T @ cx
    ridge = scale * np.trace(G)
    W = np.linalg.solve(G + ridge * np.eye(len(xbar)), (cx * w[:, None]).T @ cy).T
    b = ybar - W @ xbar
    resid = sum(u[1] * np.diag(u[5] - W @ u[4] @ W.T) for u in units) / n.sum()
    floor = 0.05 * np.mean([np.mean(np.diag(u[5])) for u in units])
    return W, b, np.maximum(resid, floor), ridge, G


def fit_pf_means(units, fold_of):
    def loss(scale):
        total = 0.0
        for k in range(3):
            train = [u for u in units if fold_of[u[0]] != k]
            W, b, *_ = ecological(train, scale)
            total += sum(u[1] * np.sum((u[3] - W @ u[2] - b) ** 2) for u in units if fold_of[u[0]] == k)
        return total

    scale, report = tune(GRID, loss)
    W, b, psi, ridge, G = ecological(units, scale)
    return {"W": W, "b": b, "psi": psi, "ridge": ridge, "scale": scale, "G": G, "tuning": report,
            "B": W.T / psi[None, :]}


def moment_negloglik(z, units, p, q, r, ridge):
    """Per-cell negative log-likelihood of protein moments given RNA moments, W = U V^T of rank r.

    Same function as pairing_free.lowrank_negloglik with the rank as an argument.
    """
    U = z[: q * r].reshape(q, r)
    V = z[q * r: q * r + p * r].reshape(p, r)
    b = z[q * r + p * r: q * r + p * r + q]
    psi = np.exp(z[q * r + p * r + q:])
    total = sum(u[1] for u in units)
    value = 0.5 * ridge * total * (np.sum(U * U) + np.sum(V * V))
    gU, gV, gb, glog = ridge * total * U, ridge * total * V, np.zeros(q), np.zeros(q)
    for _, n, mx, my, Sx, Sy in units:
        SxV = Sx @ V
        M = V.T @ SxV
        S = U @ M @ U.T + np.diag(psi)
        L = np.linalg.cholesky(S)
        Sinv = np.linalg.inv(S)
        mproj = V.T @ mx
        d = my - U @ mproj - b
        A = Sy + np.outer(d, d)
        value += 0.5 * n * (2 * np.sum(np.log(np.diag(L))) + np.sum(Sinv * A))
        G = 0.5 * (Sinv - Sinv @ A @ Sinv)
        GU = G @ U
        Sd = Sinv @ d
        gU += n * (2 * GU @ M - np.outer(Sd, mproj))
        gV += n * (2 * SxV @ (U.T @ GU) - np.outer(mx, U.T @ Sd))
        gb += n * (-Sd)
        glog += n * np.diag(G) * psi
    grad = np.concatenate([gU.ravel(), gV.ravel(), gb, glog]) / total
    return value / total, grad


def moment_negloglik_fast(z, units, p, q, r, ridge):
    """moment_negloglik computed through the Woodbury identity in O(q^2 r) per unit.

    S = Psi + U M U^T,  S^-1 = Psi^-1 - F K F^T with F = Psi^-1 U, K = M (I + P M)^-1,
    P = U^T Psi^-1 U, and log det S = sum log psi + log det(I + P M). Equal to
    moment_negloglik up to rounding (checked in test_cross_study.py).
    """
    U = z[: q * r].reshape(q, r)
    V = z[q * r: q * r + p * r].reshape(p, r)
    b = z[q * r + p * r: q * r + p * r + q]
    psi = np.exp(z[q * r + p * r + q:])
    inv_psi = 1.0 / psi
    total = sum(u[1] for u in units)
    value = 0.5 * ridge * total * (np.sum(U * U) + np.sum(V * V))
    gU, gV, gb, glog = ridge * total * U, ridge * total * V, np.zeros(q), np.zeros(q)
    F = U * inv_psi[:, None]
    P = U.T @ F
    eye = np.eye(r)
    logpsi = np.sum(np.log(psi))
    for _, n, mx, my, Sx, Sy in units:
        SxV = Sx @ V
        M = V.T @ SxV
        IPM = eye + P @ M
        K = np.linalg.solve(IPM.T, M.T).T
        E = F @ K
        mproj = V.T @ mx
        d = my - U @ mproj - b
        Fd = F.T @ d
        AF = Sy @ F + np.outer(d, Fd)
        T2 = F.T @ AF
        diagA = np.diag(Sy) + d * d
        logdet = logpsi + np.linalg.slogdet(IPM)[1]
        trace = np.sum(inv_psi * diagA) - np.sum(K * T2.T)
        value += 0.5 * n * (logdet + trace)
        SiU = F - E @ P
        ASiU = Sy @ SiU + np.outer(d, d @ SiU)
        SiASiU = ASiU * inv_psi[:, None] - E @ (F.T @ ASiU)
        GU = 0.5 * (SiU - SiASiU)
        diag_Si = inv_psi - np.sum(E * F, axis=1)
        diag_SiASi = inv_psi ** 2 * diagA - 2 * inv_psi * np.sum(AF * E, axis=1) + np.sum((E @ T2) * E, axis=1)
        diagG = 0.5 * (diag_Si - diag_SiASi)
        Sd = inv_psi * d - E @ Fd
        gU += n * (2 * GU @ M - np.outer(Sd, mproj))
        gV += n * (2 * SxV @ (U.T @ GU) - np.outer(mx, U.T @ Sd))
        gb += n * (-Sd)
        glog += n * diagG * psi
    grad = np.concatenate([gU.ravel(), gV.ravel(), gb, glog]) / total
    return value / total, grad


def moment_fit(units, r, ridge, init):
    """init = (W, b, psi) from pf_means; returns W, b, psi and the optimiser result."""
    W0, b0, psi0 = init
    q, p = W0.shape
    Uw, s, Vt = np.linalg.svd(W0, full_matrices=False)
    U0, V0 = Uw[:, :r] * np.sqrt(s[:r]), Vt[:r].T * np.sqrt(s[:r])
    z0 = np.concatenate([U0.ravel(), V0.ravel(), b0, np.log(psi0)])
    res = minimize(moment_negloglik_fast, z0, args=(units, p, q, r, ridge), jac=True, method="L-BFGS-B",
                   options=MOMENT_OPTIONS)
    z = res.x
    U, V = z[: q * r].reshape(q, r), z[q * r: q * r + p * r].reshape(p, r)
    b, psi = z[q * r + p * r: q * r + p * r + q], np.exp(z[q * r + p * r + q:])
    return U @ V.T, b, psi, res


def heldout_negloglik(W, b, psi, units):
    """Unpenalised per-cell negative log-likelihood of held-out units (sum over their cells)."""
    total = 0.0
    for _, n, mx, my, Sx, Sy in units:
        S = W @ Sx @ W.T + np.diag(psi)
        L = np.linalg.cholesky(S)
        d = my - W @ mx - b
        A = Sy + np.outer(d, d)
        total += 0.5 * n * (2 * np.sum(np.log(np.diag(L))) + np.sum(np.linalg.inv(S) * A))
    return total


def fit_pf_moment(units, fold_of, means_scale, log=print, rank=None):
    """Rank from RANKS by held-out likelihood without edge extension (PLAN.md, amendment 1),
    or the given fixed rank (colon re-analysis)."""
    ridge = 1e-3 * means_scale
    reports = {}

    def loss(r):
        total = 0.0
        for k in range(3):
            train = [u for u in units if fold_of[u[0]] != k]
            W0, b0, psi0, *_ = ecological(train, means_scale)
            W, b, psi, res = moment_fit(train, r, ridge, (W0, b0, psi0))
            held = heldout_negloglik(W, b, psi, [u for u in units if fold_of[u[0]] == k])
            reports[(r, k)] = {"converged": bool(res.success), "iterations": int(res.nit),
                               "message": str(res.message), "heldout": float(held)}
            total += held
            log(f"  rank {r} fold {k}: {res.nit} it, {res.message}, held-out {held:.2f}")
        return total

    if rank is None:
        losses = {r: float(loss(r)) for r in RANKS}
        rank = min(RANKS, key=lambda r: losses[r])
        report = {"values": list(RANKS), "losses": [losses[r] for r in RANKS], "extensions": 0,
                  "at_edge": bool(rank in (RANKS[0], RANKS[-1]))}
    else:
        report = {"fixed_rank": rank}
    W0, b0, psi0, *_ = ecological(units, means_scale)
    W, b, psi, res = moment_fit(units, rank, ridge, (W0, b0, psi0))
    report["cv_fits"] = {f"{r}/{k}": v for (r, k), v in reports.items()}
    return {"W": W, "b": b, "psi": psi, "rank": rank, "ridge": ridge, "tuning": report,
            "converged": bool(res.success), "iterations": int(res.nit), "message": str(res.message),
            "grad_norm": float(np.linalg.norm(res.jac)), "B": W.T / psi[None, :]}


# ----------------------------------------------------------------------------- paired reference
def paired_people(train, patients):
    """Per-patient paired moments on the within-patient correlation scale."""
    people = []
    for p in patients:
        m = train["patient"] == p
        x, y = train["x"][m], train["y"][m]
        xs, sx = standardize(x)
        ys, sy = standardize(y)
        n = len(x)
        Rx_raw, Ry_raw, Rxy = xs.T @ xs / n, ys.T @ ys / n, xs.T @ ys / n
        xc, yc = x - x.mean(0), y - y.mean(0)
        people.append({"patient": p, "n": n, "marginal": Marginal(shrink(Rx_raw), shrink(Ry_raw)),
                       "Rx_raw": Rx_raw, "Ry_raw": Ry_raw, "Rxy": Rxy, "Cxy_raw": xc.T @ yc / n})
    return people


def as_triples(people):
    return [(q["n"], q["marginal"], q["Rxy"]) for q in people]


def fit_paired_closed_form(people, fold_of, log=print):
    warm = {}

    def loss(ridge):
        total = 0.0
        for k in range(3):
            train = [q for q in people if fold_of[q["patient"]] != k]
            B, res = fit_interaction(as_triples(train), ridge, B0=warm.get(k), maxiter=2000)
            warm[k] = B
            total -= profile_loglik(B, as_triples([q for q in people if fold_of[q["patient"]] == k]))[0]
            log(f"  ridge {ridge:g} fold {k}: {res.nit} it, {res.message}")
        return total

    ridge, report = tune(GRID, loss)
    B, res = fit_interaction(as_triples(people), ridge, maxiter=2000)
    return {"B": B, "ridge": ridge, "tuning": report, "converged": bool(res.success),
            "iterations": int(res.nit), "message": str(res.message),
            "grad_norm": float(np.linalg.norm(res.jac))}


def pooled(people, key):
    n = np.array([q["n"] for q in people], float)
    return sum(w * q[key] for w, q in zip(n, people)) / n.sum()


def ridge_coefficients(Rx, Rxy, lam):
    c = np.mean(np.diag(Rx))
    return np.linalg.solve(Rx + lam * c * np.eye(len(Rx)), Rxy)


def fit_reference_regression(people, fold_of):
    def loss(lam):
        total = 0.0
        for k in range(3):
            train = [q for q in people if fold_of[q["patient"]] != k]
            W = ridge_coefficients(pooled(train, "Rx_raw"), pooled(train, "Rxy"), lam)
            for q in people:
                if fold_of[q["patient"]] == k:
                    total += q["n"] * (np.trace(q["Ry_raw"]) - 2 * np.sum(W * q["Rxy"])
                                       + np.sum(W * (q["Rx_raw"] @ W)))
        return total

    lam, report = tune(GRID, loss)
    return {"W": ridge_coefficients(pooled(people, "Rx_raw"), pooled(people, "Rxy"), lam), "lambda": lam,
            "tuning": report}


# ----------------------------------------------------------------------------- recipient benchmark
def benchmark(x, y, barcodes):
    """Recipient's own adaptation pairs: tuned ridge regression; returns coefficients and tuning."""
    xs, _ = standardize(x)
    ys, _ = standardize(y)
    order = np.argsort([hashlib.sha256(f"bench-v1|{b}".encode()).hexdigest() for b in barcodes])
    fold = np.empty(len(x), int)
    fold[order] = np.arange(len(x)) % 3

    def loss(lam):
        total = 0.0
        for k in range(3):
            tr, te = fold != k, fold == k
            mx, my = xs[tr].mean(0), ys[tr].mean(0)
            xt, yt = xs[tr] - mx, ys[tr] - my
            W = ridge_coefficients(xt.T @ xt / tr.sum(), xt.T @ yt / tr.sum(), lam)
            total += np.sum((ys[te] - my - (xs[te] - mx) @ W) ** 2)
        return total

    lam, report = tune(GRID, loss)
    n = len(x)
    return ridge_coefficients(xs.T @ xs / n, xs.T @ ys / n, lam), {"lambda": lam, "tuning": report}


def closed_form(Rx, Ry, B):
    return transfer(Marginal(Rx, Ry), B)[0]
