"""Estimators, controls and endpoint of the perturbation test (PLAN.md).

Training populations are target x condition groups of training cells. Pairing-
free learning uses RNA moments from part-0 cells and protein moments from
part-1 cells of each population (never the same cell), with the frozen rules of
cross_study (estimators.fit_pf_means: pooled within-population SD units, ridge
penalty by three-fold cross-validation over targets, five-value grid with the
edge rule). Paired comparators use all cells of each training population with
their pairing. Predictions for a held-out group use only its adaptation cells,
each assay separately.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE.parent / "cross_study"))
sys.path.append(str(HERE.parent / "spectral_transfer"))
sys.path.append(str(HERE.parent / "recoverability"))

import estimators as est  # noqa: E402
from data import shrink, standardize  # noqa: E402
from gaussian_transfer import Marginal, saturate  # noqa: E402

from compat import opnorm, psd_power, supported_basis, svd  # noqa: E402
from noise import noise_fraction, signal_correlation, split_halves  # noqa: E402
from pdata import clr, lognorm  # noqa: E402

CONTROL_DRAWS = 10
CONTROL_SEED = 20261011


def features(d):
    return lognorm(d["counts"], d["library"]), d["y"]


def keys_of(d):
    return np.array([f"{t}|{c}" for t, c in zip(d["target"], d["condition"])])


def eligible_training(pop, part, minimum):
    keys = sorted(set(pop))
    return [k for k in keys if np.sum((pop == k) & (part == 0)) >= minimum and np.sum((pop == k) & (part == 1)) >= minimum]


def moments(x, y, pop, part, keys):
    """Pairing-free population moments: RNA from part 0, protein from part 1."""
    raw = []
    for k in keys:
        m = pop == k
        xr, yr = x[m & (part == 0)], y[m & (part == 1)]
        raw.append({"patient": k, "nx": len(xr), "mx": xr.mean(0), "Sx": np.cov(xr.T, bias=True),
                    "ny": len(yr), "my": yr.mean(0), "Sy": np.cov(yr.T, bias=True)})
    return raw


def centre_within_condition(raw):
    """Population means centred on their condition's cell-weighted mean (perturbation-induced variation only)."""
    out = [dict(r) for r in raw]
    for c in sorted({r["patient"].split("|")[1] for r in raw}):
        group = [r for r in out if r["patient"].split("|")[1] == c]
        wx = np.array([r["nx"] for r in group], float)
        wy = np.array([r["ny"] for r in group], float)
        cx = sum(w * r["mx"] for w, r in zip(wx, group)) / wx.sum()
        cy = sum(w * r["my"] for w, r in zip(wy, group)) / wy.sum()
        for r in group:
            r["mx"], r["my"] = r["mx"] - cx, r["my"] - cy
    return out


def target_folds(keys):
    folds = est.patient_folds(sorted({k.split("|")[0] for k in keys}))
    return {k: folds[k.split("|")[0]] for k in keys}


def fit_channel(raw):
    sd_x, sd_y = est.pooled_sd(raw)
    units = est.pf_units(raw, sd_x, sd_y)
    f = est.fit_pf_means(units, target_folds([r["patient"] for r in raw]))
    VS, _ = supported_basis(f["G"], float(f["ridge"]))
    return {"W": f["W"], "B": f["B"], "G": f["G"], "ridge": float(f["ridge"]), "scale": float(f["scale"]),
            "at_edge": bool(f["tuning"]["at_edge"]), "VS": VS, "n_train": len(raw)}


def derangement(n, rng):
    while True:
        perm = rng.permutation(n)
        if np.all(perm != np.arange(n)):
            return perm


def control_channels(x, y, pop, part, keys, cond, centre=False):
    """Matched regression controls, CONTROL_DRAWS each: pseudo-populations and within-condition derangements.

    Pseudo-populations: within each condition, the population labels of the cells of eligible
    populations are permuted, so the populations keep their number and approximate size but
    differ only by sampling and by condition. Derangements: every population's protein
    moments are replaced by those of another population of the same condition.
    """
    rng = np.random.default_rng(CONTROL_SEED)
    use = np.isin(pop, keys)
    out = {}
    for k in range(CONTROL_DRAWS):
        shuffled = pop.copy()
        for c in sorted(set(cond[use])):
            m = np.flatnonzero(use & (cond == c))
            shuffled[m] = pop[m][rng.permutation(len(m))]
        praw = moments(x, y, shuffled, part, keys)
        out[f"pseudo{k:02d}"] = fit_channel(centre_within_condition(praw) if centre else praw)
    raw = moments(x, y, pop, part, keys)
    for k in range(CONTROL_DRAWS):
        praw = [dict(r) for r in raw]
        for c in sorted({r["patient"].split("|")[1] for r in raw}):
            idx = [i for i, r in enumerate(raw) if r["patient"].split("|")[1] == c]
            perm = derangement(len(idx), rng)
            for i, j in zip(idx, perm):
                src = raw[idx[j]]
                praw[i] = dict(raw[i], ny=src["ny"], my=src["my"], Sy=src["Sy"])
        out[f"derange{k:02d}"] = fit_channel(centre_within_condition(praw) if centre else praw)
    return out


def paired_people(x, y, pop, keys):
    people = []
    for k in keys:
        m = pop == k
        xs, _ = standardize(x[m])
        ys, _ = standardize(y[m])
        n = int(m.sum())
        Rx_raw, Ry_raw, Rxy = xs.T @ xs / n, ys.T @ ys / n, xs.T @ ys / n
        people.append({"patient": k, "n": n, "marginal": Marginal(shrink(Rx_raw), shrink(Ry_raw)), "Rx_raw": Rx_raw,
                       "Ry_raw": Ry_raw, "Rxy": Rxy})
    return people


def paired_references(people, closed_form=True, log=print):
    fold_of = target_folds([q["patient"] for q in people])
    out = {"reference_regression": est.fit_reference_regression(people, fold_of)}
    conds = sorted({q["patient"].split("|")[1] for q in people})
    out["transferred_correlation"] = {c: est.pooled([q for q in people if q["patient"].split("|")[1] == c], "Rxy")
                                      for c in conds}
    if closed_form:
        out["paired_closed_form"] = est.fit_paired_closed_form(people, fold_of, log=log)
    return out


def condition_marginals(x, y, counts, library, pop, part, keys):
    """Pooled within-population correlation of training cells per condition: RNA (part 0, measured and latent), protein (part 1)."""
    out = {}
    for c in sorted({k.split("|")[1] for k in keys}):
        ks = [k for k in keys if k.split("|")[1] == c]
        rx = np.isin(pop, ks) & (part == 0)
        ry = np.isin(pop, ks) & (part == 1)
        xc = x[rx].copy()
        yc = y[ry].copy()
        for k in ks:
            a, b = pop[rx] == k, pop[ry] == k
            xc[a] -= xc[a].mean(0)
            yc[b] -= yc[b].mean(0)
        xs, _ = standardize(xc)
        ys, _ = standardize(yc)
        Rx, Ry = shrink(xs.T @ xs / len(xs)), shrink(ys.T @ ys / len(ys))
        h1, h2 = split_halves(counts[rx], library[rx])
        nu = noise_fraction(h1, h2, pop[rx], ks)
        out[c] = {"Rx": Rx, "Ry": Ry, "Rz": signal_correlation(Rx, nu), "nu": nu}
    return out


def group_inputs(counts, library, y):
    """Within-assay inputs of one held-out group (adaptation cells): no cross-assay statistic is formed."""
    x = lognorm(counts, library)
    xs, _ = standardize(x)
    ys, _ = standardize(y)
    n = len(x)
    Rx, Ry = shrink(xs.T @ xs / n), shrink(ys.T @ ys / n)
    h1, h2 = split_halves(counts, library)
    nu = noise_fraction(h1, h2)
    return {"x": x, "y": y, "Rx": Rx, "Ry": Ry, "nu": nu, "Rz": signal_correlation(Rx, nu), "n": n}


def transfer(Rx, Ry, B, m=None):
    """Closed-form transfer; pass a cached Marginal m of (Rx, Ry) to avoid recomputing square roots."""
    m = m or Marginal(Rx, Ry)
    U, beta, Vt = svd(m.Rx @ B @ m.Ry)
    return m.Rx @ (U * saturate(beta)) @ Vt @ m.Ry


def diagnostics(a, ch):
    VS = ch["VS"]
    if VS.shape[1] == 0:
        return {"dim_S": 0, "coverage": 0.0, "F_latent": 0.0, "F_measured": 0.0}
    P = VS @ VS.T
    row = {"dim_S": int(VS.shape[1]), "coverage": float(np.trace(P @ a["Rz"] @ P) / np.trace(a["Rz"]))}
    Ryih = psd_power(a["Ry"], -0.5)
    for k, R in (("latent", a["Rz"]), ("measured", a["Rx"])):
        QN, _ = np.linalg.qr(psd_power(R, -0.5) @ VS)
        row[f"F_{k}"] = opnorm(QN.T @ psd_power(R, 0.5) @ ch["W"].T @ Ryih)
    return row


def scoring_targets(counts, library, y, part):
    """Cross-correlation of the scoring cells, and of its two sub-halves (for the unbiased squared norm)."""
    x = lognorm(counts, library)

    def cc(m):
        xs, _ = standardize(x[m])
        ys, _ = standardize(y[m])
        return xs.T @ ys / m.sum()
    return {"T": cc(np.ones(len(x), bool)), "TA": cc(part == 0), "TB": cc(part == 1)}


def endpoint_terms(C, t):
    """Unbiased estimates of ||T||^2 - ||C - T||^2 (numerator) and ||T||^2 (denominator)."""
    return 2 * float(np.sum(C * t["T"])) - float(np.sum(C * C)), float(np.sum(t["TA"] * t["TB"]))
