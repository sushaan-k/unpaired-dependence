"""Self-tuning semi-paired estimator (DEV_SELFTUNE.md): the block partition and the within-block shrinkage are
chosen for each data set by Stein's unbiased risk estimate (SURE) from the paired cells' own per-cell products,
in the two-sided eigenbasis of the unpaired cells. No held-out data and no cross-validation are used.

    V1  self-tuned partition, positive-part James-Stein in each block
    V2  fixed partition, preconditioned (heteroscedastic, moment-based) shrinkage in each block
    V3  per block the rule with the smaller SURE, partition by the smallest total SURE

All return the p x q pooled estimate U A V'; the condition map is applied afterwards, unchanged.
"""

from __future__ import annotations

import numpy as np

import sp_estimators as es

RNA_EDGES = ((5, 20, 60), (2, 5, 10, 20, 40, 80), (3, 10, 30, 100), (10, 40), (5, 20), (1, 3, 10, 30, 100), ())
PROT_EDGES = ((3,), (), (1, 3), (2, 5, 10), (3, 10, 30), (5,))


def moments(XU, YV):
    """Mean of the per-cell products in the rotated coordinates and the variance of each mean."""
    B = len(XU)
    M = XU.T @ YV / B
    s = ((XU ** 2).T @ (YV ** 2) - B * M ** 2) / ((B - 1) * B)
    return M, np.maximum(s, 0.0)


def block_stats(XU, YV, M, s, rows, cols):
    """||m||^2, t = tr S, m'Sm and Sm of one block (S the covariance of the block mean, from per-cell products)."""
    B = len(XU)
    Mb = M[np.ix_(rows, cols)]
    w = np.einsum("ij,ij->i", XU[:, rows] @ Mb, YV[:, cols])      # per-cell m'z_i
    wc = w - w.mean()
    n2 = float(np.sum(Mb * Mb))
    t = float(np.sum(s[np.ix_(rows, cols)]))
    msm = float(np.sum(wc * wc)) / ((B - 1) * B)
    Sm = (XU[:, rows] * wc[:, None]).T @ YV[:, cols] / ((B - 1) * B)   # (S m) for every coordinate of the block
    return Mb, n2, t, msm, Sm


def sure_js(n2, t, msm):
    """SURE of positive-part James-Stein with c = t = tr S (general S)."""
    if n2 > t:
        return t - t * t / n2 + 4 * t * msm / (n2 * n2)
    return n2 - t


def js(Mb, n2, t):
    return max(1.0 - t / max(n2, 1e-300), 0.0) * Mb


def precond(Mb, sb, n2, t, Sm):
    """Preconditioned (heteroscedastic) shrinkage: m_j tau^2 / (tau^2 + s_j) with the moment estimate
    tau^2 = (||m||^2 - t) / d, zero if negative. With equal s_j it is positive-part James-Stein with c = t. Returns
    (estimate, SURE), the SURE exact for Gaussian block means with covariance S (it includes the dependence of
    tau^2 on m)."""
    d = Mb.size
    tau2 = max(0.0, (n2 - t) / d)
    if tau2 == 0.0:
        return np.zeros_like(Mb), n2 - t
    b = sb / (tau2 + sb)
    val = float(np.sum(b * b * Mb * Mb) + 2 * np.sum(sb * (1 - b))
                + (4 / d) * np.sum(Mb * sb * Sm / (tau2 + sb) ** 2) - t)
    return (1 - b) * Mb, val


def estimate(X, Y, U, V, variant, rna_edges=es.EIGEN_EDGES, prot_edges=es.PROTEIN_EDGES, info=None):
    """Pooled estimate of variant V1, V2 or V3 (V0 is sp_estimators.block_js2)."""
    XU, YV = X @ U, Y @ V
    M, s = moments(XU, YV)
    p, q = M.shape
    cands = [(re, pe) for re in RNA_EDGES for pe in PROT_EDGES] if variant in ("V1", "V3") else \
        [(tuple(rna_edges), tuple(prot_edges))]
    best = None
    for re, pe in cands:
        rb, pb = es.eigen_blocks(p, re), es.eigen_blocks(q, pe)
        total, parts = 0.0, []
        for rows in rb:
            for cols in pb:
                Mb, n2, t, msm, Sm = block_stats(XU, YV, M, s, rows, cols)
                if variant == "V1":
                    parts.append((rows, cols, "js", sure_js(n2, t, msm), Mb, n2, t))
                elif variant == "V2":
                    est, val = precond(Mb, s[np.ix_(rows, cols)], n2, t, Sm)
                    parts.append((rows, cols, "pc", val, est, n2, t))
                else:
                    sj = sure_js(n2, t, msm)
                    est, val = precond(Mb, s[np.ix_(rows, cols)], n2, t, Sm)
                    parts.append((rows, cols, "js", sj, Mb, n2, t) if sj <= val else (rows, cols, "pc", val, est, n2, t))
                total += parts[-1][3]
        if best is None or total < best[0]:
            best = (total, re, pe, parts)
    A = np.zeros((p, q))
    for rows, cols, rule, _, Mb, n2, t in best[3]:
        A[np.ix_(rows, cols)] = js(Mb, n2, t) if rule == "js" else Mb
    if info is not None:
        info.update({"rna_edges": list(best[1]), "prot_edges": list(best[2]), "sure": best[0],
                     "rules": [r[2] for r in best[3]]})
    return U @ A @ V.T
