#!/usr/bin/env python3
"""Development analysis on data already scored (exploratory; not a test).

Hypothesis: population means average out RNA sampling noise, so the channel
they identify maps latent RNA state to protein, whereas a recipient's measured
RNA covariance also contains sampling noise. Applying the channel to the
measured covariance then implies more RNA-protein dependence than exists, most
of all within cell types, where latent variation is small. Replacing R_x by its
signal part (count splitting; noise.py) should make the between-population
channel compatible with within-cell covariances and reduce transfer error.

For every scored recipient analysis (untouched blood cohort and colon) and the
frozen and P1 channels, this compares prediction errors against the stored
targets and recomputes the compatibility statistics with the signal part.
Also reports the label-based between-type estimator (type means per assay).
Writes results/dev_noise.json.
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
from data import standardize, type_centre  # noqa: E402
from predict import ANALYSES, within_assay  # noqa: E402

from dev_cache import channels, load_counts  # noqa: E402
from diagnostics import psd_power, supported_projection  # noqa: E402
from noise import lognorm, noise_fraction, signal_correlation, split_halves  # noqa: E402

LABEL_FOR_TOTAL = {"hao": ("l1", "within_l1"), "colon": ("coarse", "within_coarse")}


def rel(C, T):
    return float(np.linalg.norm(C - T) / np.linalg.norm(T))


def sharp_F(W, VM, Rx, Ry):
    Rxh, Rxih, Ryih = psd_power(Rx, 0.5), psd_power(Rx, -0.5), psd_power(Ry, -0.5)
    QN, _ = np.linalg.qr(Rxih @ VM)
    return float(np.linalg.svd(QN.T @ Rxh @ W.T @ Ryih, compute_uv=False)[0])


def between_label(x, y, labels, keep):
    """Between-type cross-correlation from type means computed separately in each assay."""
    cells = np.isin(labels, keep)
    x, y, labels = x[cells], y[cells], labels[cells]
    xs, _ = standardize(x)
    ys, _ = standardize(y)
    C = np.zeros((x.shape[1], y.shape[1]))
    for t in keep:
        m = labels == t
        C += m.mean() * np.outer(xs[m].mean(0), ys[m].mean(0))
    return C


def main():
    chans = channels()
    for ch in chans.values():
        g, V = np.linalg.eigh((ch["G"] + ch["G"].T) / 2)
        h = np.clip(g, 0, None) / (np.clip(g, 0, None) + float(ch["ridge"]))
        ch["VM"] = V[:, h >= 0.5]
        ch["P"] = ch["VM"] @ ch["VM"].T
    scores = json.loads((CS / "results/scores.json").read_text())["scores"]
    truth = np.load(CS / "results/truth.npz")
    paired_B = np.load(CS / "results/reference_fit.npz")["paired_B"]
    rows = []
    for cohort, analyses in ANALYSES.items():
        rec = load_counts(cohort, "adaptation")
        for donor in sorted(set(rec["donor"])):
            m = rec["donor"] == donor
            counts, library, y0 = rec["counts"][m], rec["library"][m], rec["y"][m]
            x0 = lognorm(counts, library)
            h1, h2 = split_halves(counts, library)
            for analysis, key in analyses:
                tag = f"{cohort}/{donor}/{analysis}"
                T = truth[f"{tag}/T"]
                row = {"tag": tag, "cohort": cohort, "donor": donor, "analysis": analysis}
                if key is not None:
                    types = scores[tag]["types"]
                    labels = rec["labels"][key][m]
                    cells = np.isin(labels, types)
                    x = type_centre(x0[cells], labels[cells], types)
                    y = type_centre(y0[cells], labels[cells], types)
                    nu = noise_fraction(h1[cells], h2[cells], labels[cells], types)
                else:
                    x, y = x0, y0
                    nu = noise_fraction(h1, h2)
                    lkey, ltag = LABEL_FOR_TOTAL[cohort]
                    ltypes = scores[f"{cohort}/{donor}/{ltag}"]["types"]
                    labels = rec["labels"][lkey][m]
                    cells = np.isin(labels, ltypes)
                    row["E_label_between"] = rel(between_label(x, y, labels, ltypes), T)
                Rx, Ry, _, _ = within_assay(x, y)
                Rz = signal_correlation(Rx, nu)
                row.update({"cells": int(len(x)), "nu_mean": float(nu.mean()), "nu_median": float(np.median(nu)),
                            "E_reported": scores[tag]["arms"]["pf_means"]["E"],
                            "E_paired_closed_form": scores[tag]["arms"]["paired_closed_form"]["E"],
                            "E_paired_closed_form_signal": rel(est.closed_form(Rz, Ry, paired_B), T)})
                for name, ch in chans.items():
                    W, B, P, VM = ch["W"], ch["B"], ch["P"], ch["VM"]
                    WS = W @ P
                    r_meas = np.diag(WS @ Rx @ WS.T)
                    r_sig = np.diag(WS @ Rz @ WS.T)
                    row[name] = {
                        "E_closed": rel(est.closed_form(Rx, Ry, B), T),
                        "E_closed_signal": rel(est.closed_form(Rz, Ry, B), T),
                        "E_linear_S": rel(Rx @ WS.T, T),
                        "E_linear_S_signal": rel(Rz @ WS.T, T),
                        "E_linear": rel(Rx @ W.T, T),
                        "E_linear_signal": rel(Rz @ W.T, T),
                        "F_measured": sharp_F(W, VM, Rx, Ry),
                        "F_signal": sharp_F(W, VM, Rz, Ry),
                        "rmax_measured": float(r_meas.max()), "rmax_signal": float(r_sig.max()),
                        "frac_over_measured": float(np.mean(r_meas > 1)),
                        "frac_over_signal": float(np.mean(r_sig > 1)),
                        "coverage_measured": float(np.trace(P @ Rx @ P) / np.trace(Rx)),
                        "coverage_signal": float(np.trace(P @ Rz @ P) / np.trace(Rz))}
                rows.append(row)
                print(tag, round(row["nu_mean"], 2), "frozen", {k: round(v, 2) for k, v in row["frozen"].items()},
                      "| P1 E", round(row["P1"]["E_closed"], 2), round(row["P1"]["E_closed_signal"], 2),
                      round(row["P1"]["E_linear_S_signal"], 2), "| label", round(row.get("E_label_between", -1), 2),
                      flush=True)
    (HERE / "results/dev_noise.json").write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
