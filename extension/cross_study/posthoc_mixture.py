#!/usr/bin/env python3
"""Post hoc analysis P2 (PLAN.md; exploratory): between-type and within-type dependence.

For each recipient donor, the scoring cells' cross-covariance over the kept
level-1 (Hao) or coarse (colon) types splits exactly into a between-type part
(type means weighted by type proportions) and a within-type part (pooled
within-type cross-covariance). Both are divided by the scoring cells' total
standard deviations. The frozen pf_means channel W predicts each part from the
donor's adaptation cells as the corresponding RNA covariance part, in the same
units, times W^T (direct channel, no saturation). Writes
results/posthoc_mixture.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from data import load_recipient

HERE = Path(__file__).resolve().parent
PARTS = {"hao": ("within_l1", "l1"), "colon": ("within_coarse", "coarse")}


def decompose(x, y, labels, keep):
    """Between- and within-type cross-covariance of x and y; exact: between + within = total."""
    cells = np.isin(labels, keep)
    x, y, labels = x[cells], y[cells], labels[cells]
    n = len(x)
    mx, my = x.mean(0), y.mean(0)
    between = np.zeros((x.shape[1], y.shape[1]))
    within = np.zeros_like(between)
    for t in keep:
        m = labels == t
        dx, dy = x[m].mean(0) - mx, y[m].mean(0) - my
        between += m.sum() / n * np.outer(dx, dy)
        cx, cy = x[m] - x[m].mean(0), y[m] - y[m].mean(0)
        within += cx.T @ cy / n
    total = (x - mx).T @ (y - my) / n
    assert np.allclose(between + within, total, atol=1e-8 * max(1.0, np.abs(total).max()))
    return between, within, x.std(0), y.std(0), n


def scaled(M, sa, sb):
    inv = lambda s: np.where(s > 0, 1.0 / np.where(s > 0, s, 1.0), 0.0)
    return M * np.outer(inv(sa), inv(sb))


def metrics(P, T):
    return {"E": float(np.linalg.norm(P - T) / np.linalg.norm(T)),
            "entry_correlation": float(np.corrcoef(P.ravel(), T.ravel())[0, 1])}


def main():
    W = np.load(HERE / "results/reference_fit.npz")["pf_means_W"]            # proteins x genes
    scores = json.loads((HERE / "results/scores.json").read_text())["scores"]
    out = {"donors": {}, "summary": {}}
    for cohort, (analysis, key) in PARTS.items():
        adapt = load_recipient(cohort, "adaptation")
        score = load_recipient(cohort, "scoring")                             # checked against its seal
        rows = []
        for donor in sorted(set(score["donor"])):
            tag = f"{cohort}/{donor}/{analysis}"
            keep = scores[tag]["types"]
            ms, ma = score["donor"] == donor, adapt["donor"] == donor
            Cb, Cw, sx, sy, n = decompose(score["x"][ms], score["y"][ms], score["labels"][key][ms], keep)
            Tb, Tw = scaled(Cb, sx, sy), scaled(Cw, sx, sy)
            T = Tb + Tw
            xa = adapt["x"][ma]
            Sb, Sw, sa, _, na = decompose(xa, xa, adapt["labels"][key][ma], keep)
            Pb, Pw = scaled(Sb, sa, sa) @ W.T, scaled(Sw, sa, sa) @ W.T
            nT = np.linalg.norm(T)
            row = {"tag": tag, "types": keep, "scoring_cells": n, "adaptation_cells": na,
                   "truth": {"between_share": float(np.linalg.norm(Tb) / nT),
                             "within_share": float(np.linalg.norm(Tw) / nT),
                             "cosine_between_within": float(np.sum(Tb * Tw) / (np.linalg.norm(Tb) * np.linalg.norm(Tw)))},
                   "between": metrics(Pb, Tb), "within": metrics(Pw, Tw), "total_direct": metrics(Pb + Pw, T),
                   "predicted_norm_ratio": {"between": float(np.linalg.norm(Pb) / np.linalg.norm(Tb)),
                                            "within": float(np.linalg.norm(Pw) / np.linalg.norm(Tw))}}
            rows.append(row)
            out["donors"][tag] = row
            print(tag, n, {k: round(row[k]["E"], 3) for k in ("between", "within", "total_direct")},
                  round(row["truth"]["within_share"], 3), flush=True)
        get = lambda f: np.array([f(r) for r in rows])
        out["summary"][cohort] = {
            "donors": len(rows),
            "between_share": {"mean": float(get(lambda r: r["truth"]["between_share"]).mean()),
                              "min": float(get(lambda r: r["truth"]["between_share"]).min()),
                              "max": float(get(lambda r: r["truth"]["between_share"]).max())},
            "within_share": {"mean": float(get(lambda r: r["truth"]["within_share"]).mean()),
                             "min": float(get(lambda r: r["truth"]["within_share"]).min()),
                             "max": float(get(lambda r: r["truth"]["within_share"]).max())},
            **{f"E_{part}": {"mean": float(get(lambda r: r[part]["E"]).mean()),
                             "min": float(get(lambda r: r[part]["E"]).min()),
                             "max": float(get(lambda r: r[part]["E"]).max()),
                             "donors_below_1": int((get(lambda r: r[part]["E"]) < 1).sum())}
               for part in ("between", "within", "total_direct")},
            **{f"r_{part}": float(get(lambda r: r[part]["entry_correlation"]).mean())
               for part in ("between", "within", "total_direct")},
            "norm_ratio_within": float(get(lambda r: r["predicted_norm_ratio"]["within"]).mean()),
            "norm_ratio_between": float(get(lambda r: r["predicted_norm_ratio"]["between"]).mean())}
    (HERE / "results/posthoc_mixture.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out["summary"], indent=1))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
