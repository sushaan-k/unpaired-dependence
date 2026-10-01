"""Choosing a small paired reference with unpaired information.

Setting: a recipient sample has been profiled in separate assays (all cells
give RNA-only and protein-only summaries, with cell-type labels in each assay),
and a budget of N cells can be profiled with paired RNA and protein. The
target is the recipient's within-type cross-correlation (cells centred on
their type), which population means do not identify and which pairing-free
transfer failed to recover.

Estimator (all strategies): stratified within-type cross-covariance with
positive-part James-Stein shrinkage. Paired cells of type t are centred on the
type means from the unpaired summaries, their cross-products are averaged
within type and weighted by the type's share of cells pi_t (unpaired), and the
result is scaled by the pooled within-type standard deviations (unpaired).
Types without paired cells contribute zero. The expected squared norm of the
sampling noise is N_E = sum_t pi_t^2 V_t / n_t (V_t below, from unpaired
within-type variances), and the estimate is multiplied by
max(0, 1 - N_E / ||C_hat||^2). For the total cross-correlation, the
between-type part is formed from type means in each assay (unpaired) and added
to the within part; the paired-only comparator shrinks the paired cells' own
cross-correlation in the same way (N_E = p q / N).

Allocation over types with budget N:
  random    N cells drawn at random (expected n_t proportional to pi_t);
  balanced  equal numbers per type (capped by availability, surplus redistributed);
  guided    Neyman allocation n_t proportional to pi_t * sqrt(V_t), with
            V_t = tr(S_x,t / d_x^2) * tr(S_y,t / d_y^2) the per-cell variance scale
            of the type's standardized cross-products, from unpaired within-type
            variances. This minimises the expected squared error of the stratified
            estimator when cross-products are weakly correlated.
"""

from __future__ import annotations

import numpy as np


def summaries(x, y, labels, types):
    """Unpaired summaries: type shares, type means per assay, pooled within-type and total SDs."""
    cells = np.isin(labels, types)
    x, y, labels = x[cells], y[cells], labels[cells]
    pi = np.array([np.mean(labels == t) for t in types])
    mx = np.array([x[labels == t].mean(0) for t in types])
    my = np.array([y[labels == t].mean(0) for t in types])
    code = np.array([list(types).index(t) for t in labels]) if len(labels) else np.zeros(0, int)
    xc, yc = x - mx[code], y - my[code]
    dx_w, dy_w = xc.std(0), yc.std(0)
    vx = np.array([np.sum(xc[labels == t].var(0) / np.where(dx_w > 0, dx_w, 1) ** 2) for t in types])
    vy = np.array([np.sum(yc[labels == t].var(0) / np.where(dy_w > 0, dy_w, 1) ** 2) for t in types])
    return {"types": list(types), "pi": pi, "mx": mx, "my": my, "dx_w": dx_w, "dy_w": dy_w,
            "dx": x.std(0), "dy": y.std(0), "mx_all": x.mean(0), "my_all": y.mean(0),
            "count": np.array([np.sum(labels == t) for t in types]), "V": vx * vy}


def allocate(strategy, N, s, rng):
    """Number of paired cells per type (sums to N unless availability binds)."""
    count = s["count"]
    T = len(count)
    if strategy == "random":
        draw = rng.choice(np.repeat(np.arange(T), count), size=min(N, count.sum()), replace=False)
        return np.bincount(draw, minlength=T)
    weights = np.ones(T) if strategy == "balanced" else s["pi"] * np.sqrt(s["V"])
    n = np.zeros(T, int)
    remaining, open_ = min(N, count.sum()), np.ones(T, bool)
    while remaining > 0 and open_.any():
        share = np.where(open_, weights, 0)
        share = share / share.sum() * remaining
        add = np.floor(share).astype(int)
        frac = share - add
        extra = remaining - add.sum()
        if extra > 0:
            add[np.argsort(-frac)[:extra]] += 1
        add = np.minimum(add, count - n)
        n += add
        remaining -= add.sum()
        open_ = n < count
        if add.sum() == 0:
            break
    return n


def select(n_per_type, labels, types, rng):
    idx = []
    for t, k in zip(types, n_per_type):
        pool = np.flatnonzero(labels == t)
        if k > 0:
            idx.append(rng.choice(pool, size=k, replace=False))
    return np.concatenate(idx) if idx else np.zeros(0, int)


def js_factor(C, noise):
    norm2 = float(np.sum(C * C))
    return max(0.0, 1.0 - noise / norm2) if norm2 > 0 else 0.0


def within_estimate(x, y, labels, idx, s, shrink=True):
    """Stratified within-type cross-correlation from the paired cells idx (James-Stein shrunk)."""
    C = np.zeros((x.shape[1], y.shape[1]))
    noise = 0.0
    for k, t in enumerate(s["types"]):
        sel = idx[labels[idx] == t]
        if len(sel) == 0:
            continue
        dx = x[sel] - s["mx"][k]
        dy = y[sel] - s["my"][k]
        C += s["pi"][k] * dx.T @ dy / len(sel)
        noise += s["pi"][k] ** 2 * s["V"][k] / len(sel)
    C = C / np.outer(np.where(s["dx_w"] > 0, s["dx_w"], 1), np.where(s["dy_w"] > 0, s["dy_w"], 1))
    return C * js_factor(C, noise) if shrink else C


def within_to_total(C_within, s):
    """Within-type cross-covariance (on the within scale) expressed on the total correlation scale."""
    return C_within * np.outer(s["dx_w"], s["dy_w"]) / np.outer(np.where(s["dx"] > 0, s["dx"], 1),
                                                                   np.where(s["dy"] > 0, s["dy"], 1))


def between_estimate(s):
    """Between-type cross-correlation from unpaired type means (total correlation scale)."""
    dx = s["mx"] - s["mx_all"]
    dy = s["my"] - s["my_all"]
    C = (dx * s["pi"][:, None]).T @ dy
    return C / np.outer(np.where(s["dx"] > 0, s["dx"], 1), np.where(s["dy"] > 0, s["dy"], 1))


def paired_only_total(x, y, idx, s, shrink=True):
    """Cross-correlation of the paired cells alone (centred on their own means), unpaired SDs, shrunk."""
    dx = x[idx] - x[idx].mean(0)
    dy = y[idx] - y[idx].mean(0)
    C = dx.T @ dy / len(idx) / np.outer(np.where(s["dx"] > 0, s["dx"], 1), np.where(s["dy"] > 0, s["dy"], 1))
    return C * js_factor(C, x.shape[1] * y.shape[1] / len(idx)) if shrink else C
