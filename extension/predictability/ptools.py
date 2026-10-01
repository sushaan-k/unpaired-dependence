"""Shared tools of the predictability study (extension/predictability): roles, unpaired summaries, the two
estimators in fast form, observed recovery curves against held-out cells of the same population, and the law's
predicted curves (Supplementary Note 13).

Roles, by SHA-256 of the cell identifier: 25% truth cells (held out, used only to score), 30% reservoir (paired
cells: the budget draws and the pilot) and 45% unpaired pool, whose cells contribute X if a second hash is below one
half and Y otherwise. The unpaired summaries, bases and blocks are those of the proposed estimator
(extension/semipaired, via extension/generality/grun.py, all unchanged): latent RNA correlation by count splitting,
protein correlation, eigen-blocks RNA 1-5/6-20/21-60/rest x protein 1-3/rest. Estimand: the pooled
within-population cross-covariance in pool standard-deviation units (cells centred on their population's unpaired
means), the quantity the pooled estimate targets before the condition map.

The fast estimators compute exactly what sp_estimators.block_js2 and js_matrix compute (checked in dev.py
selfcheck): for a product block the squared norm of a cell's coefficients is ||x~_R||^2 ||y~_C||^2, so the noise
trace needs no per-cell product vectors.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
EXT = HERE.parent
sys.path.insert(0, str(EXT / "generality"))
import grun as gr  # noqa: E402  (frozen; read-only use of its data helpers)

cm, es, pm, hm = gr.cm, gr.es, gr.pm, gr.cm.hm

ROLE_SALT, PART_SALT, HALF_SALT, PILOT_SALT = ("predictability-role-v1", "predictability-part-v1",
                                               "predictability-half-v1", "predictability-pilot-v1")
TRUTH_FRACTION, RESERVOIR_FRACTION = 0.25, 0.30
MIN_POP_HALF = 10
BUDGETS = (25, 50, 100, 200, 400, 800, 1600, 3200)


def h(text):
    return int(hashlib.sha256(text.encode()).hexdigest()[:12], 16) / 16 ** 12


def sha(path):
    d = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            d.update(b)
    return d.hexdigest()


def load(path, prefix, allow_object=False):
    """A data set in the common format of extension/generality (gdata.py). allow_object: the benchmark's
    stephenson.npz stores strings as object arrays (generality Deviation 1); it is our own extraction."""
    z = np.load(path, allow_pickle=allow_object)
    d = {k: z[k] for k in z.files}
    if "x_data" in d:
        d["x_counts"] = sp.csr_matrix((d["x_data"], d["x_indices"], d["x_indptr"]), shape=tuple(d["x_shape"]))
    d["y_kind"] = str(d["y_kind"])
    for k in ("cell", "unit", "cond", "pop"):
        d[k] = d[k].astype(str)
    d["cell"] = np.array([f"{prefix}|{c}" for c in d["cell"]])
    return d


def roles(cells):
    u = np.array([h(f"{ROLE_SALT}|{c}") for c in cells])
    role = np.where(u < TRUTH_FRACTION, 0, np.where(u < TRUTH_FRACTION + RESERVOIR_FRACTION, 1, 2))
    part = np.array([int(h(f"{PART_SALT}|{c}") >= 0.5) for c in cells])
    half = np.array([int(h(f"{HALF_SALT}|{c}") >= 0.5) for c in cells])
    order = np.array([h(f"{PILOT_SALT}|{c}") for c in cells])
    return role, part, half, order


class Problem:
    """One estimation problem: unpaired summaries from the pool, centred reservoir and truth cells."""

    def __init__(self, d, n_x=gr.PANEL_X, n_y=gr.PANEL_Y, keep=None):
        role, part, half, order = roles(d["cell"])
        if keep is not None:                       # sub-problem: a subset of cells (e.g. one condition)
            role = np.where(keep, role, -1)
        pool = role == 2
        # panels from unpaired pool cells only: X features from X-half cells, Y features from Y-half cells
        xp = self._x_panel(d, np.flatnonzero(pool & (part == 0)), n_x)
        yp = self._y_panel(d, np.flatnonzero(pool & (part == 1)), n_y)
        use = np.flatnonzero(role >= 0)
        x, y, C, lib = gr.features(d, use, xp, yp)
        pop = np.array([f"{p}|{c}" for p, c in zip(d["pop"][use], d["cond"][use])])
        r, pt, hv, od = role[use], part[use], half[use], order[use]
        pl = r == 2
        keys = pm.eligible_training(pop[pl], pt[pl], MIN_POP_HALF)
        counts = C is not None
        mgr = None if counts else gr.NoCounts()
        if mgr:
            mgr.__enter__()
        try:
            self.pool = hm.Pool(x[pl], y[pl], pop[pl], pt[pl], d["unit"][use][pl], keys)
            un = cm.Unpaired(self.pool, x[pl], y[pl], C[pl] if counts else x[pl],
                             lib[pl] if counts else np.ones(int(pl.sum())), pop[pl], pt[pl], keys)
        finally:
            if mgr:
                mgr.__exit__()
        R = un.R["__all__"]
        self.U, self.lz = es.eigenbasis(R["Rz"])
        self.V, self.ly = es.eigenbasis(R["Ry"])
        self.rblocks = es.eigen_blocks(self.U.shape[1])
        self.pblocks = es.eigen_blocks(self.V.shape[1], es.PROTEIN_EDGES)
        ok = np.isin(pop, self.pool.keys)
        # measured within-population covariances of the pool (pool SD units), for unpaired noise shares
        xs, ys = x[pl] / self.pool.sd_x, y[pl] / self.pool.sd_y
        mr, mp = np.isin(pop[pl], self.pool.keys) & (pt[pl] == 0), np.isin(pop[pl], self.pool.keys) & (pt[pl] == 1)
        Rx, _ = cm.within_population_corr(xs[mr], pop[pl][mr], self.pool.keys)
        Ry, _ = cm.within_population_corr(ys[mp], pop[pl][mp], self.pool.keys)
        self.Cx, self.Cy = self.U.T @ Rx @ self.U, self.V.T @ Ry @ self.V
        res = np.flatnonzero((r == 1) & ok)
        res = res[np.argsort(od[res], kind="stable")]           # pilot = first cells in a fixed hash order
        self.Xr, self.Yr = self.pool.centre_cells(x[res], y[res], pop[res])
        tru = (r == 0) & ok
        # truth cells are kept aside; no statistic of them is computed before score_setup() (evaluation stage)
        self._truth = (x[tru], y[tru], pop[tru], hv[tru])
        self.n = {"pool_x": int(mr.sum()), "pool_y": int(mp.sum()), "reservoir": int(len(res)),
                  "truth": int(tru.sum()), "populations": len(self.pool.keys), "p": int(len(xp)), "q": int(len(yp))}
        self.panels = (xp, yp)
        self.counts = counts

    def score_setup(self):
        """Evaluation stage only: the truth cells' mean product T (all cells) and the unbiased ||theta||^2 estimate
        <T_A, T_B> from their two hash halves, in the unpaired bases."""
        xt, yt, popt, hvt = self._truth
        Xt, Yt = self.pool.centre_cells(xt, yt, popt)
        XtU, YtV = Xt @ self.U, Yt @ self.V
        self.T = XtU.T @ YtV / len(Xt)
        a, b = hvt == 0, hvt == 1
        TA, TB = XtU[a].T @ YtV[a] / a.sum(), XtU[b].T @ YtV[b] / b.sum()
        self.den = float(np.sum(TA * TB))                        # unbiased for ||theta||^2

    @staticmethod
    def _x_panel(d, rows, n):
        saved = gr.PANEL_X
        gr.PANEL_X = n
        try:
            return gr.x_panel(d, rows)
        finally:
            gr.PANEL_X = saved

    @staticmethod
    def _y_panel(d, rows, n):
        saved = gr.PANEL_Y
        gr.PANEL_Y = n
        try:
            return gr.y_panel(d, rows)
        finally:
            gr.PANEL_Y = saved

    # ---------------------------------------------------------------- estimators (coefficients in U, V)
    def fit(self, X, Y, U=None, V=None):
        """Block James-Stein (proposed, pooled) and one-block James-Stein (paired-only), both in coordinates
        (U, V); U, V default to the unpaired bases (a random orthonormal pair gives the random-basis control)."""
        U = self.U if U is None else U
        V = self.V if V is None else V
        B = len(X)
        XU, YV = X @ U, Y @ V
        M = XU.T @ YV / B
        nx, ny = XU * XU, YV * YV
        A = np.zeros_like(M)
        for rows in self.rblocks:
            sx = nx[:, rows].sum(1)
            for cols in self.pblocks:
                Mb = M[np.ix_(rows, cols)]
                n2 = float(np.sum(Mb * Mb))
                t = (float(np.sum(sx * ny[:, cols].sum(1))) - B * n2) / (B * (B - 1))
                A[np.ix_(rows, cols)] = (max(1.0 - t / n2, 0.0) if n2 > 0 else 0.0) * Mb
        n2 = float(np.sum(M * M))
        t = (float(np.sum(nx.sum(1) * ny.sum(1))) - B * n2) / (B * (B - 1))
        A1 = (max(1.0 - t / n2, 0.0) if n2 > 0 else 0.0) * M
        return A, A1

    def rf(self, A, U=None, V=None):
        """Noise-unbiased recovered fraction of the pooled within-population cross-covariance of the truth cells."""
        if U is not None:            # estimate in other coordinates: rotate into the unpaired bases
            A = (self.U.T @ U) @ A @ (V.T @ self.V)
        return (2 * float(np.sum(A * self.T)) - float(np.sum(A * A))) / self.den

    # ---------------------------------------------------------------- moments for the law
    def moments(self, X, Y, U=None, V=None, ell=True, iters=25, seed=0):
        """Per block: unbiased signal r_b^2 = ||m_b||^2 - t_b, per-cell noise trace tau_b = N t_b and (optionally)
        the largest eigenvalue l_b of the per-cell covariance (power iteration on the centred products)."""
        U = self.U if U is None else U
        V = self.V if V is None else V
        N = len(X)
        XU, YV = X @ U, Y @ V
        M = XU.T @ YV / N
        nx, ny = XU * XU, YV * YV
        out = []
        rng = np.random.default_rng(seed)
        for rows in self.rblocks:
            sx = nx[:, rows].sum(1)
            for cols in self.pblocks:
                Mb = M[np.ix_(rows, cols)]
                n2 = float(np.sum(Mb * Mb))
                t = (float(np.sum(sx * ny[:, cols].sum(1))) - N * n2) / (N * (N - 1))
                rec = {"r2": n2 - t, "tau": N * t, "d": len(rows) * len(cols)}
                if ell:
                    a, b = XU[:, rows], YV[:, cols]
                    v = rng.standard_normal((len(rows), len(cols)))
                    v /= np.linalg.norm(v)
                    lam = 0.0
                    for _ in range(iters):
                        s = np.einsum("ir,rc,ic->i", a, v, b) - float(np.sum(Mb * v))   # centred z_i . v
                        w = (a * s[:, None]).T @ b / (N - 1)
                        lam = float(np.linalg.norm(w))
                        if lam == 0:
                            break
                        v = w / lam
                    rec["ell"] = lam
                out.append(rec)
        return out

    def unpaired_noise(self):
        """Independence noise traces tau_b = tr Cov(x~_R) tr Cov(y~_C) and l_b from the pool's marginal covariances."""
        out = []
        for rows in self.rblocks:
            cx = self.Cx[np.ix_(rows, rows)]
            for cols in self.pblocks:
                cy = self.Cy[np.ix_(cols, cols)]
                out.append({"tau": float(np.trace(cx) * np.trace(cy)),
                            "ell": float(np.linalg.eigvalsh(cx)[-1] * np.linalg.eigvalsh(cy)[-1]),
                            "d": len(rows) * len(cols)})
        return out


# -------------------------------------------------------------------- the law (Supplementary Note 13)

def rho(r2, tau, B):
    return r2 * tau / (B * r2 + tau) if r2 > 0 else 0.0


def sph_risk(r2, s, deff, n=40000, seed=20261071):
    """Risk of (1 - s/||X||^2)_+ X for X ~ N(a, (s/deff) I_deff), ||a||^2 = r2 (Gaussian spherical equivalent of a
    block of effective dimension deff; deff may be fractional)."""
    if s <= 0:
        return 0.0
    deff = max(deff, 1.0)
    rng = np.random.default_rng(seed)
    sig = np.sqrt(s / deff)
    r = np.sqrt(max(r2, 0.0))
    z1 = rng.standard_normal(n)
    q = rng.chisquare(max(deff - 1.0, 1e-6), n) if deff > 1 else np.zeros(n)
    along = r + sig * z1
    X2 = along ** 2 + sig * sig * q
    g = np.clip(1.0 - s / np.maximum(X2, 1e-300), 0.0, None)
    loss = g * g * X2 - 2 * g * r * along + r * r
    return float(loss.mean())


def curves(blocks, budgets, law="lin"):
    """Predicted recovered-fraction curves of block and one-block shrinkage from block moments."""
    r2 = np.array([max(b["r2"], 0.0) for b in blocks])
    tau = np.array([b["tau"] for b in blocks])
    R2, T = r2.sum(), tau.sum()
    if R2 <= 0:
        return None
    out_k, out_1 = [], []
    for B in budgets:
        if law == "lin":
            rk = sum(rho(a, t, B) for a, t in zip(r2, tau))
            r1 = rho(R2, T, B)
        else:
            ell = np.array([b["ell"] for b in blocks])
            rk = sum(sph_risk(a, t / B, t / l) for a, t, l in zip(r2, tau, ell))
            # one block: per-cell covariance of all coefficients; its largest eigenvalue is at least the blocks'
            r1 = sph_risk(R2, T / B, T / max(float(ell.max()), 1e-300))
        out_k.append(1 - rk / R2)
        out_1.append(1 - r1 / R2)
    return np.array(out_k), np.array(out_1)


def shares(blocks):
    r2 = np.array([max(b["r2"], 0.0) for b in blocks])
    tau = np.array([b["tau"] for b in blocks])
    return r2 / max(r2.sum(), 1e-300), tau / tau.sum()


def saving_closed(w, pi, eps):
    """Proposition S14(a): S(eps) from signal shares w and noise shares pi."""
    from scipy.optimize import brentq
    phi = lambda beta: float(np.sum(np.where(w > 0, w * pi / (beta * w + pi), 0.0)))
    beta = brentq(lambda b: phi(b) - eps, 1e-12, 1e12)
    return (1 - eps) / (eps * beta)


def needed(values, target, budgets):
    return gr.needed(values, target, budgets)


def jdump(obj):
    def conv(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    return json.dumps(obj, default=conv)


# -------------------------------------------------------------------- bootstrap law (finite-sample refinement)

def block_means(P, XU, YV):
    """Per-block mean products and unbiased noise traces of the mean for cells (XU, YV) in the unpaired bases."""
    B = len(XU)
    M = XU.T @ YV / B
    nx, ny = XU * XU, YV * YV
    tr = {}
    for i, rows in enumerate(P.rblocks):
        sx = nx[:, rows].sum(1)
        for j, cols in enumerate(P.pblocks):
            Mb = M[np.ix_(rows, cols)]
            tr[(i, j)] = (float(np.sum(sx * ny[:, cols].sum(1))) - B * float(np.sum(Mb * Mb))) / (B * (B - 1))
    return M, tr


def boot_curves(P, X, Y, budgets, draws=20, seed=0, signal="unbiased"):
    """Bootstrap law: B cells are resampled with replacement from (X, Y), whose per-cell products are shifted in every
    block so that the population mean Theta_b has squared norm max(||M_b||^2 - t_b, 0) along M_b (signal="unbiased").
    Loss of block and one-block James-Stein against Theta; returns predicted RF curves (block, one block)."""
    XU, YV = X @ P.U, Y @ P.V
    N = len(X)
    M, tr = block_means(P, XU, YV)
    Theta = np.zeros_like(M)
    for i, rows in enumerate(P.rblocks):
        for j, cols in enumerate(P.pblocks):
            Mb = M[np.ix_(rows, cols)]
            n2 = float(np.sum(Mb * Mb))
            r2 = max(n2 - tr[(i, j)], 0.0)
            Theta[np.ix_(rows, cols)] = Mb * np.sqrt(r2 / n2) if n2 > 0 else 0.0
    shift = Theta - M
    R2 = float(np.sum(Theta * Theta))
    if R2 <= 0:
        return None
    rng = np.random.default_rng(seed)
    out_k, out_1 = [], []
    for B in budgets:
        lk, l1 = [], []
        for _ in range(draws):
            idx = rng.integers(0, N, B)
            Md, trd = block_means(P, XU[idx], YV[idx])
            Mp = Md + shift
            A = np.zeros_like(Mp)
            for i, rows in enumerate(P.rblocks):
                for j, cols in enumerate(P.pblocks):
                    Mb = Mp[np.ix_(rows, cols)]
                    n2 = float(np.sum(Mb * Mb))
                    A[np.ix_(rows, cols)] = (max(1 - trd[(i, j)] / n2, 0.0) if n2 > 0 else 0.0) * Mb
            n2 = float(np.sum(Mp * Mp))
            t_all = sum(trd.values())
            A1 = (max(1 - t_all / n2, 0.0) if n2 > 0 else 0.0) * Mp
            lk.append(float(np.sum((A - Theta) ** 2)))
            l1.append(float(np.sum((A1 - Theta) ** 2)))
        out_k.append(1 - np.mean(lk) / R2)
        out_1.append(1 - np.mean(l1) / R2)
    return np.array(out_k), np.array(out_1)
