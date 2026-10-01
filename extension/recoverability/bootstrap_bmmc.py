#!/usr/bin/env python3
"""Bone-marrow test, step 2: sampling uncertainty of the compatibility statistic (PLAN.md).

For the frozen, P1 and colon channels and every recipient analysis, 200
replicates each resample training populations (blood patients; P1: patients
with all their cell-type groups, re-centred within type; colon: donors with all
their biopsies) and refit with the channel's penalty scale fixed, and resample
the recipient analysis's adaptation cells. Reports ||F|| with measured and with
latent RNA covariance, its 5% and 95% bootstrap quantiles and the basic
bootstrap lower bound 2 F_hat - q_0.95. Reads the adaptation file only.
Writes results/bmmc_bootstrap.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
CS = HERE.parent / "cross_study"
sys.path.insert(0, str(CS))

import estimators as est  # noqa: E402
from data import shrink, standardize, type_centre  # noqa: E402

from bmmc import load  # noqa: E402
from compat import opnorm, psd_power, supported_basis  # noqa: E402
from freeze import check_freeze  # noqa: E402
from noise import lognorm, noise_fraction, signal_correlation, split_halves  # noqa: E402
from predict_bmmc import ANALYSES, load_channels  # noqa: E402
from training import colon_units, load_training_panel  # noqa: E402

REPLICATES = 200
SEED = 20261003
MIN_UNIT = 50


def raw_type_units(train):
    keys = sorted({(p, t) for p, t in zip(train["patient"], train["initial"])})
    raw = []
    for p, t in keys:
        m = (train["patient"] == p) & (train["initial"] == t)
        rna, prot = m & (train["half"] == 0), m & (train["half"] == 1)
        if rna.sum() < MIN_UNIT or prot.sum() < MIN_UNIT:
            continue
        xr, yr = train["x"][rna], train["y"][prot]
        raw.append({"patient": f"{p}|{t}", "donor": p, "type": t, "nx": len(xr), "mx": xr.mean(0),
                    "Sx": np.cov(xr.T, bias=True), "ny": len(yr), "my": yr.mean(0), "Sy": np.cov(yr.T, bias=True)})
    return raw


def centre_types(raw):
    out = [dict(r) for r in raw]
    for t in sorted({r["type"] for r in out}):
        group = [r for r in out if r["type"] == t]
        wx = np.array([r["nx"] for r in group], float)
        wy = np.array([r["ny"] for r in group], float)
        cx = sum(w * r["mx"] for w, r in zip(wx, group)) / wx.sum()
        cy = sum(w * r["my"] for w, r in zip(wy, group)) / wy.sum()
        for r in group:
            r["mx"], r["my"] = r["mx"] - cx, r["my"] - cy
    return out


def refit(raw, scale):
    sd_x, sd_y = est.pooled_sd(raw)
    W, _, _, ridge, G = est.ecological(est.pf_units(raw, sd_x, sd_y), scale)
    VS, _ = supported_basis(G, ridge)
    return W, VS


def main():
    check_freeze()
    info = json.loads((HERE / "results/bmmc_channels.json").read_text())
    chans = load_channels()
    units = json.loads((HERE / "results/bmmc_units.json").read_text())
    train = load_training_panel()
    patients = sorted(set(train["patient"]))
    raw_all = {r["patient"]: r for r in est.pf_raw_moments(train, patients)}
    by_donor = {}
    for r in raw_type_units(train):
        by_donor.setdefault(r["donor"], []).append(r)
    craw, donor_of = colon_units()
    colon_donors = sorted(set(donor_of.values()))
    rec = load("adaptation")
    analyses = []
    for batch, unit in units.items():
        m = rec["batch"] == batch
        counts, library = rec["counts"][m], rec["library"][m]
        x0, y0 = lognorm(counts, library), rec["y"][m]
        h1, h2 = split_halves(counts, library)
        for analysis, key in ANALYSES:
            if key is None:
                analyses.append({"tag": f"{batch}/{analysis}", "x": x0, "y": y0, "h1": h1, "h2": h2, "labels": None})
            else:
                labels = rec["labels"][key][m]
                types = unit["types"][key]
                cells = np.isin(labels, types)
                analyses.append({"tag": f"{batch}/{analysis}", "x": x0[cells], "y": y0[cells], "h1": h1[cells],
                                 "h2": h2[cells], "labels": labels[cells], "types": types})

    def moments(a, idx=None):
        x, y, h1, h2, labels = a["x"], a["y"], a["h1"], a["h2"], a["labels"]
        if idx is not None:
            x, y, h1, h2 = x[idx], y[idx], h1[idx], h2[idx]
            labels = None if labels is None else labels[idx]
        types = None if labels is None else [t for t in a["types"] if np.any(labels == t)]
        if labels is not None:
            x, y = type_centre(x, labels, types), type_centre(y, labels, types)
        xs, _ = standardize(x)
        ys, _ = standardize(y)
        n = len(x)
        Rx, Ry = shrink(xs.T @ xs / n), shrink(ys.T @ ys / n)
        return Rx, Ry, signal_correlation(Rx, noise_fraction(h1, h2, labels, types))

    def statistics(fits, a, idx=None):
        Rx, Ry, Rz = moments(a, idx)
        Ryih = psd_power(Ry, -0.5)
        roots = {k: (psd_power(R, 0.5), psd_power(R, -0.5)) for k, R in (("F_measured", Rx), ("F_signal", Rz))}
        out = {}
        for name, (W, VS) in fits.items():
            for k, (Rh, Rih) in roots.items():
                if VS.shape[1] == 0:
                    out[(name, k)] = 0.0
                    continue
                QN, _ = np.linalg.qr(Rih @ VS)
                out[(name, k)] = opnorm(QN.T @ Rh @ W.T @ Ryih)
        return out

    base = {name: (chans[name]["W"], chans[name]["VS"]) for name in ("frozen", "P1", "colon")}
    point = {a["tag"]: statistics(base, a) for a in analyses}
    scale = {name: info[name]["scale"] for name in base}
    rng = np.random.default_rng(SEED)
    boot = {a["tag"]: {k: [] for k in point[a["tag"]]} for a in analyses}
    for rep in range(REPLICATES):
        chosen = [patients[i] for i in rng.choice(len(patients), len(patients), replace=True)]
        fits = {"frozen": refit([dict(raw_all[p], patient=f"{p}#{j}") for j, p in enumerate(chosen)], scale["frozen"])}
        praw = [dict(r, patient=f"{r['patient']}#{j}") for j, p in enumerate(chosen) for r in by_donor.get(p, [])]
        fits["P1"] = refit(centre_types(praw), scale["P1"])
        cd = [colon_donors[i] for i in rng.choice(len(colon_donors), len(colon_donors), replace=True)]
        craw_b = [dict(r, patient=f"{r['patient']}#{j}") for j, d in enumerate(cd) for r in craw if donor_of[r["patient"]] == d]
        fits["colon"] = refit(craw_b, scale["colon"])
        for a in analyses:
            idx = rng.integers(0, len(a["x"]), len(a["x"]))
            for k, v in statistics(fits, a, idx).items():
                boot[a["tag"]][k].append(v)
        if rep % 10 == 0:
            print("replicate", rep, flush=True)
    out = []
    for a in analyses:
        for (name, k), v in point[a["tag"]].items():
            b = np.array(boot[a["tag"]][(name, k)])
            out.append({"tag": a["tag"], "channel": name, "statistic": k, "point": v,
                        "q05": float(np.quantile(b, 0.05)), "q95": float(np.quantile(b, 0.95)),
                        "basic_lower": float(2 * v - np.quantile(b, 0.95))})
    (HERE / "results/bmmc_bootstrap.json").write_text(json.dumps({"replicates": REPLICATES, "rows": out}, indent=1) + "\n")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
