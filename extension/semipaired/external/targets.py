"""Within-condition targets pooled over held-out populations.

Every cell is centred on the mean of its population (ORF x condition) within the subset considered (all cells, or
one scoring sub-half), and the centred cells of all held-out populations in a condition are pooled into one
cross-correlation. Per-population sums are kept so that a bootstrap over held-out ORFs can re-pool them.
"""

from __future__ import annotations

import numpy as np

import odata as od

SUBSETS = ("T", "TA", "TB")


def group_stats(counts, library, y, pop, part):
    """Per population: centred cross-product and variance sums for all cells (T) and each sub-half (TA, TB)."""
    x = od.lognorm(counts, library)
    out = []
    for k in sorted(set(pop)):
        m = pop == k
        rec = {"key": str(k), "target": str(k).split("|")[0], "cond": str(k).split("|")[1], "n": int(m.sum())}
        for s, sel in zip(SUBSETS, (np.ones(len(pop), bool), part == 0, part == 1)):
            mm = m & sel
            if mm.sum() < 2:
                rec[s] = None
                continue
            xc = x[mm] - x[mm].mean(0)
            yc = y[mm] - y[mm].mean(0)
            rec[s] = {"xy": xc.T @ yc, "xx": np.sum(xc * xc, 0), "yy": np.sum(yc * yc, 0), "n": int(mm.sum())}
        out.append(rec)
    return out


def pooled(stats, weights=None):
    """Per condition: pooled within-population cross-correlations T, TA and TB (weights: multiplicity of each
    population, for bootstrap resamples)."""
    w = np.ones(len(stats)) if weights is None else np.asarray(weights, float)
    res = {}
    for c in sorted({r["cond"] for r in stats}):
        res[c] = {}
        for s in SUBSETS:
            xy, xx, yy = 0.0, 0.0, 0.0
            for r, wi in zip(stats, w):
                if r["cond"] != c or wi == 0 or r[s] is None:
                    continue
                xy = xy + wi * r[s]["xy"]
                xx = xx + wi * r[s]["xx"]
                yy = yy + wi * r[s]["yy"]
            if np.isscalar(xy):
                res[c] = None
                break
            res[c][s] = xy / np.sqrt(np.outer(np.maximum(xx, 1e-12), np.maximum(yy, 1e-12)))
    return {c: v for c, v in res.items() if v is not None}
