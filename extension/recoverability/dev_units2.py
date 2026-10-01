#!/usr/bin/env python3
"""Development table for the failure model (exploratory; data already scored).

Every recipient analysis of the untouched blood cohort and colon (48) crossed
with 62 channels on the original panel (206 genes, 122 proteins): the frozen
means channel, P1, the 40 learning-curve subsets and 20 donor-aware protein
permutations (training.py defines the same families for the bone-marrow panel).
For each pair: errors of closed-form transfer with measured and with latent RNA
covariance, the two unpaired diagnostics (coverage of latent RNA variance by S;
compatibility statistic ||F|| with latent covariance) and the controls
(between-type share, cells, training populations, covariance shift).
Writes results/dev_units2.json.
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
sys.path.insert(0, str(HERE.parent / "spectral_transfer"))

import estimators as est  # noqa: E402
from data import load_training, shrink, standardize, type_centre  # noqa: E402
from gaussian_transfer import Marginal, transfer  # noqa: E402
from predict import ANALYSES  # noqa: E402

from compat import psd_power, supported_basis  # noqa: E402
from dev_cache import channels, load_counts  # noqa: E402
from diagnostics import between_share, covariance_shift  # noqa: E402
from noise import lognorm, noise_fraction, signal_correlation, split_halves  # noqa: E402
from dev_noise import between_label  # noqa: E402
from training import LC_REPEATS, LC_SIZES, PERM_SEED, PERMUTATIONS, derangement  # noqa: E402

LABEL_FOR_TOTAL = {"hao": ("l1", "within_l1"), "colon": ("coarse", "within_coarse")}


def mean_corr(units):
    S = sum(u[4] for u in units) / len(units)
    d = np.sqrt(np.diag(S))
    return S / np.outer(d, d)


def fit(raw, folds):
    sd_x, sd_y = est.pooled_sd(raw)
    units = est.pf_units(raw, sd_x, sd_y)
    f = est.fit_pf_means(units, folds)
    VS, _ = supported_basis(f["G"], f["ridge"])
    return {"W": f["W"], "B": f["B"], "VS": VS, "n_train": len(raw), "R_train": mean_corr(units)}


def build_channels():
    cached = channels()
    train = load_training()
    patients = sorted(set(train["patient"]))
    raw_all = {r["patient"]: r for r in est.pf_raw_moments(train, patients)}
    raw = [raw_all[p] for p in patients]
    folds = est.patient_folds(patients)
    sd_x, sd_y = est.pooled_sd(raw)
    R_frozen = mean_corr(est.pf_units(raw, sd_x, sd_y))
    out = {}
    for name in ("frozen", "P1"):
        c = cached[name]
        VS, _ = supported_basis(c["G"], float(c["ridge"]))
        out[name] = {"W": c["W"], "B": c["B"], "VS": VS, "n_train": 118 if name == "frozen" else None,
                     "R_train": R_frozen}
    out["P1"]["n_train"] = int(json.loads((CS / "results/posthoc_within.json").read_text())["units"])
    lc = json.loads((CS / "results/learning_curve.json").read_text())
    rng = np.random.default_rng(lc["seed"])
    plan = [(s, rng.choice(len(patients), s, replace=False)) for s in LC_SIZES for _ in range(LC_REPEATS)]
    for i, (size, idx) in enumerate(plan):
        subset = [patients[j] for j in sorted(idx)]
        out[f"lc{i:02d}"] = fit([raw_all[p] for p in subset], est.patient_folds(subset))
    prng = np.random.default_rng(PERM_SEED)
    for k in range(PERMUTATIONS):
        perm = derangement(len(raw), prng)
        praw = [dict(r, ny=raw[j]["ny"], my=raw[j]["my"], Sy=raw[j]["Sy"]) for r, j in zip(raw, perm)]
        out[f"perm{k:02d}"] = fit(praw, folds)
    return out


def main():
    chans = build_channels()
    print(len(chans), "channels", flush=True)
    scores = json.loads((CS / "results/scores.json").read_text())["scores"]
    truth = np.load(CS / "results/truth.npz")
    paired_B = np.load(CS / "results/reference_fit.npz")["paired_B"]
    rows, unit_rows = [], []
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
                if key is not None:
                    types = scores[tag]["types"]
                    labels = rec["labels"][key][m]
                    cells = np.isin(labels, types)
                    x = type_centre(x0[cells], labels[cells], types)
                    y = type_centre(y0[cells], labels[cells], types)
                    nu = noise_fraction(h1[cells], h2[cells], labels[cells], types)
                    fb = 0.0
                else:
                    x, y = x0, y0
                    nu = noise_fraction(h1, h2)
                    lkey, ltag = LABEL_FOR_TOTAL[cohort]
                    ltypes = scores[f"{cohort}/{donor}/{ltag}"]["types"]
                    labels = rec["labels"][lkey][m]
                    fb = 0.5 * (between_share(x, labels, ltypes) + between_share(y, labels, ltypes))
                    C_label = between_label(x, y, labels, ltypes)
                xs, _ = standardize(x)
                ys, _ = standardize(y)
                n = len(x)
                Rx, Ry = shrink(xs.T @ xs / n), shrink(ys.T @ ys / n)
                Rz = signal_correlation(Rx, nu)
                marg = {"measured": Marginal(Rx, Ry), "signal": Marginal(Rz, Ry)}
                roots = {k: (psd_power(R, 0.5), psd_power(R, -0.5)) for k, R in (("measured", Rx), ("signal", Rz))}
                Ryih = psd_power(Ry, -0.5)
                normT = np.linalg.norm(T)
                unit_rows.append({"tag": tag, "E_label_between": float(np.linalg.norm(C_label - T) / normT)
                                  if key is None else 1.0,
                                  "E_paired_measured": float(np.linalg.norm(transfer(marg["measured"], paired_B)[0] - T)
                                                             / normT),
                                  "E_paired_signal": float(np.linalg.norm(transfer(marg["signal"], paired_B)[0] - T)
                                                           / normT),
                                  "nu_mean": float(nu.mean()), "nu_median": float(np.median(nu))})
                for name, ch in chans.items():
                    W, B, VS = ch["W"], ch["B"], ch["VS"]
                    P = VS @ VS.T
                    row = {"tag": tag, "cohort": cohort, "donor": donor, "analysis": analysis, "channel": name,
                           "family": name.rstrip("0123456789") if name not in ("frozen", "P1") else name,
                           "dim_S": int(VS.shape[1]), "cells": int(n), "n_train": int(ch["n_train"]),
                           "between_share": float(fb), "shift": covariance_shift(Rx, ch["R_train"]),
                           "nu_mean": float(nu.mean())}
                    for k, R in (("measured", Rx), ("signal", Rz)):
                        C = transfer(marg[k], B)[0]
                        row[f"E_{k}"] = float(np.linalg.norm(C - T) / normT)
                        row[f"coverage_{k}"] = float(np.trace(P @ R @ P) / np.trace(R))
                        Rh, Rih = roots[k]
                        if VS.shape[1]:
                            QN, _ = np.linalg.qr(Rih @ VS)
                            row[f"F_{k}"] = float(np.linalg.norm(QN.T @ Rh @ W.T @ Ryih, 2))
                        else:
                            row[f"F_{k}"] = 0.0
                        WS = W @ P
                        r = np.diag(WS @ R @ WS.T)
                        row[f"excess_{k}"] = float(np.sum(np.clip(r - 1, 0, None)) / max(np.sum(r), 1e-12))
                        row[f"frac_over_{k}"] = float(np.mean(r > 1))
                    rows.append(row)
                print(tag, flush=True)
    (HERE / "results/dev_units2.json").write_text(json.dumps({"pairs": rows, "units": unit_rows}, indent=1) + "\n")
    print(len(rows), "rows")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
