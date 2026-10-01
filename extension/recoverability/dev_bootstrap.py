#!/usr/bin/env python3
"""Development analysis on data already scored (exploratory; not a test).

Does the incompatibility of the between-population channel with recipient
covariances exceed sampling uncertainty? For the frozen means channel and the
within-type channel P1, and for every recipient analysis of the untouched
blood cohort and colon, the compatibility statistic ||F|| (compat.py) is
computed with the measured RNA correlation matrix (as originally reported)
and with its latent part (noise.py). Uncertainty: 200 replicates, each
resampling training patients (P1: patients with all their cell-type units)
with the penalty scale fixed, and the recipient's cells. The closed-form
transfer's implied channel is compared with W on S (channel_departure).
Only within-assay moments of recipients are used. Writes results/dev_bootstrap.json.
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
from data import load_training, shrink, standardize, type_centre  # noqa: E402
from predict import ANALYSES  # noqa: E402

from compat import channel_departure, compatibility_statistic, psd_power, supported_basis  # noqa: E402
from dev_cache import channels, load_counts  # noqa: E402
from noise import lognorm, noise_fraction, signal_correlation, split_halves  # noqa: E402

REPLICATES = 200
SEED = 20260927
MIN_UNIT = 50


def raw_type_units(train):
    """posthoc_within.type_units without the centring, which is redone per replicate."""
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
    W, b, psi, ridge, G = est.ecological(est.pf_units(raw, sd_x, sd_y), scale)
    VS, _ = supported_basis(G, ridge)
    return W, VS


def recipient_units():
    scores = json.loads((CS / "results/scores.json").read_text())["scores"]
    units = []
    for cohort, analyses in ANALYSES.items():
        rec = load_counts(cohort, "adaptation")
        for donor in sorted(set(rec["donor"])):
            m = rec["donor"] == donor
            counts, library = rec["counts"][m], rec["library"][m]
            x0, y0 = lognorm(counts, library), rec["y"][m]
            h1, h2 = split_halves(counts, library)
            for analysis, key in analyses:
                tag = f"{cohort}/{donor}/{analysis}"
                if key is None:
                    units.append({"tag": tag, "x": x0, "y": y0, "h1": h1, "h2": h2, "labels": None, "types": None})
                else:
                    types = scores[tag]["types"]
                    labels = rec["labels"][key][m]
                    cells = np.isin(labels, types)
                    units.append({"tag": tag, "x": x0[cells], "y": y0[cells], "h1": h1[cells], "h2": h2[cells],
                                  "labels": labels[cells], "types": types})
    return units


def unit_moments(u, idx=None):
    x, y, h1, h2, labels = u["x"], u["y"], u["h1"], u["h2"], u["labels"]
    if idx is not None:
        x, y, h1, h2 = x[idx], y[idx], h1[idx], h2[idx]
        labels = None if labels is None else labels[idx]
    types = None if labels is None else [t for t in u["types"] if np.any(labels == t)]
    if labels is not None:
        x, y = type_centre(x, labels, types), type_centre(y, labels, types)
    xs, _ = standardize(x)
    ys, _ = standardize(y)
    n = len(x)
    Rx, Ry = shrink(xs.T @ xs / n), shrink(ys.T @ ys / n)
    nu = noise_fraction(h1, h2, labels, types)
    return Rx, Ry, signal_correlation(Rx, nu)


def main():
    chans = channels()
    train = load_training()
    patients = sorted(set(train["patient"]))
    raw_all = {r["patient"]: r for r in est.pf_raw_moments(train, patients)}
    traw = raw_type_units(train)
    ref = np.load(CS / "results/reference_fit.npz")
    frozen_scale = float(ref["pf_means_ridge"]) / float(np.trace(ref["pf_means_G"]))
    p1_scale = float(chans["P1"]["ridge"]) / float(np.trace(chans["P1"]["G"]))
    units = recipient_units()
    print(len(units), "units", flush=True)
    point = {}
    for name in ("frozen", "P1"):
        W, B = chans[name]["W"], chans[name]["B"]
        VS, _ = supported_basis(chans[name]["G"], float(chans[name]["ridge"]))
        for u in units:
            Rx, Ry, Rz = unit_moments(u)
            C_measured = est.closed_form(Rx, Ry, B)
            C_signal = est.closed_form(Rz, Ry, B)
            point[(name, u["tag"])] = {
                "F_measured": compatibility_statistic(W, VS, Rx, Ry),
                "F_signal": compatibility_statistic(W, VS, Rz, Ry),
                "departure_closed_measured": channel_departure(C_measured, Rx, W, VS),
                "departure_closed_signal": channel_departure(C_signal, Rz, W, VS),
                "dim_S": int(VS.shape[1])}
    rng = np.random.default_rng(SEED)
    boot = {k: {"F_measured": [], "F_signal": []} for k in point}
    for rep in range(REPLICATES):
        sample = rng.choice(len(patients), len(patients), replace=True)
        chosen = [patients[i] for i in sample]
        raw = [dict(raw_all[p], patient=f"{p}#{j}") for j, p in enumerate(chosen)]
        fits = {"frozen": refit(raw, frozen_scale)}
        by_donor = {}
        for r in traw:
            by_donor.setdefault(r["donor"], []).append(r)
        praw = [dict(r, patient=f"{r['patient']}#{j}") for j, p in enumerate(chosen) for r in by_donor.get(p, [])]
        fits["P1"] = refit(centre_types(praw), p1_scale)
        for u in units:
            idx = rng.integers(0, len(u["x"]), len(u["x"]))
            Rx, Ry, Rz = unit_moments(u, idx)
            Ryih = psd_power(Ry, -0.5)
            roots = {k: (psd_power(R, 0.5), psd_power(R, -0.5)) for k, R in (("F_measured", Rx), ("F_signal", Rz))}
            for name, (W, VS) in fits.items():
                for k, (Rh, Rih) in roots.items():
                    QN, _ = np.linalg.qr(Rih @ VS)
                    boot[(name, u["tag"])][k].append(float(np.linalg.norm(QN.T @ Rh @ W.T @ Ryih, 2)))
        if rep % 10 == 0:
            print("replicate", rep, flush=True)
    out = []
    for (name, tag), v in point.items():
        row = {"channel": name, "tag": tag, **v}
        for stat in ("F_measured", "F_signal"):
            b = np.array(boot[(name, tag)][stat])
            row[stat + "_q05"], row[stat + "_q95"] = float(np.quantile(b, 0.05)), float(np.quantile(b, 0.95))
            row[stat + "_basic_lower"] = float(2 * v[stat] - np.quantile(b, 0.95))
        out.append(row)
    (HERE / "results/dev_bootstrap.json").write_text(json.dumps({"replicates": REPLICATES, "rows": out}, indent=1) + "\n")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
