#!/usr/bin/env python3
"""Development analysis on data already scored (exploratory; not a test).

Why is the exact identified set empty? For the frozen means channel and the
post hoc within-type channel (P1), this computes the largest singular value of
the whitened identified block F (Supplementary Note 6) in four versions:

  reported   shrunk recipient correlation matrices, as in identifiability.py;
  clr        the same after removing the protein log-ratio null direction.
             Protein values are centred log ratios, so every cell's protein
             vector sums to zero and, after standardization by the recipient's
             SDs, R_y has an exact null vector n proportional to those SDs. A
             channel fitted in training SD units need not be orthogonal to n,
             and any component along n implies variance where the recipient
             has none. The component is removed and R_y restricted to n-perp;
  raw        as clr, with unshrunk sample correlation matrices;
  diag       per-protein implied explained variance r_j (no whitening).

Only within-assay moments of the recipients' adaptation cells are used.
Writes results/dev_compatibility.json.
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
from data import load_recipient, load_training, shrink, standardize, type_centre  # noqa: E402
from posthoc_within import type_units  # noqa: E402
from predict import ANALYSES  # noqa: E402

from diagnostics import psd_power, supported_projection  # noqa: E402


def null_basis(v):
    """Orthonormal basis of the complement of v."""
    v = v / np.linalg.norm(v)
    Q, _ = np.linalg.qr(np.column_stack([v, np.eye(len(v))]))
    H = Q[:, 1:len(v)]
    return H - np.outer(v, v @ H)


def identified_block(W, VM, Rx, Ry):
    """Largest singular value of F = Q_N^T Rx^1/2 W^T Ry^-1/2 (Supplementary Note 6)."""
    Rxh, Rxih, Ryih = psd_power(Rx, 0.5), psd_power(Rx, -0.5), psd_power(Ry, -0.5)
    QN, _ = np.linalg.qr(Rxih @ VM)
    F = QN.T @ Rxh @ W.T @ Ryih
    U, s, Vt = np.linalg.svd(F, full_matrices=False)
    return s, Vt[0] @ Ryih          # top protein direction in (restricted) protein coordinates


def main():
    ref = np.load(CS / "results/reference_fit.npz")
    scores = json.loads((CS / "results/scores.json").read_text())["scores"]
    train = load_training()
    traw = type_units(train)
    tsd_x, tsd_y = est.pooled_sd(traw)
    tunits = est.pf_units(traw, tsd_x, tsd_y)
    donor_fold = est.patient_folds(sorted({r["donor"] for r in traw}))
    fit = est.fit_pf_means(tunits, {r["patient"]: donor_fold[r["donor"]] for r in traw})
    chans = {}
    for name, W, G, ridge in (("frozen", ref["pf_means_W"], ref["pf_means_G"], float(ref["pf_means_ridge"])),
                              ("P1", fit["W"], fit["G"], fit["ridge"])):
        P, dim, _ = supported_projection(G, ridge)
        g, V = np.linalg.eigh((G + G.T) / 2)
        h = np.clip(g, 0, None) / (np.clip(g, 0, None) + ridge)
        chans[name] = {"W": W, "VM": V[:, h >= 0.5], "dim": dim}
    out = []
    for cohort, analyses in ANALYSES.items():
        rec = load_recipient(cohort, "adaptation")
        for donor in sorted(set(rec["donor"])):
            m = rec["donor"] == donor
            for analysis, key in analyses:
                tag = f"{cohort}/{donor}/{analysis}"
                x, y = rec["x"][m], rec["y"][m]
                if key is not None:
                    types = scores[tag]["types"]
                    labels = rec["labels"][key][m]
                    cells = np.isin(labels, types)
                    x, y = type_centre(x[cells], labels[cells], types), type_centre(y[cells], labels[cells], types)
                xs, sdx = standardize(x)
                ys, sdy = standardize(y)
                n = len(x)
                Rx_raw, Ry_raw = xs.T @ xs / n, ys.T @ ys / n
                Rx, Ry = shrink(Rx_raw), shrink(Ry_raw)
                nvec = sdy / np.linalg.norm(sdy)
                H = null_basis(nvec)
                row = {"tag": tag, "cohort": cohort, "analysis": analysis, "cells": n,
                       "Ry_raw_min_eig": float(np.linalg.eigvalsh(Ry_raw).min()),
                       "Ry_raw_null_residual": float(np.linalg.norm(Ry_raw @ nvec))}
                for name, ch in chans.items():
                    W, VM = ch["W"], ch["VM"]
                    s_rep, top = identified_block(W, VM, Rx, Ry)
                    s_clr, _ = identified_block(H.T @ W, VM, Rx, H.T @ Ry @ H)
                    s_raw, _ = identified_block(H.T @ W, VM, Rx_raw + 1e-9 * np.eye(len(Rx)), H.T @ Ry_raw @ H)
                    WS = W @ VM @ VM.T
                    r = np.diag(WS @ Rx @ WS.T)          # diag(Ry) = 1 on the correlation scale
                    row[name] = {"F_reported": float(s_rep[0]), "F_clr": float(s_clr[0]), "F_raw": float(s_raw[0]),
                                 "n_sv_over_1_reported": int(np.sum(s_rep > 1)),
                                 "n_sv_over_1_clr": int(np.sum(s_clr > 1)),
                                 "top_alignment_with_null": float(abs(top @ nvec) / np.linalg.norm(top)),
                                 "W_null_share": float(np.linalg.norm(nvec @ W) ** 2 / np.linalg.norm(W) ** 2),
                                 "r_max": float(r.max()), "r_mean": float(r.mean()),
                                 "frac_r_over_1": float(np.mean(r > 1))}
                out.append(row)
                print(tag, {k: (round(v["F_reported"], 2), round(v["F_clr"], 2), round(v["F_raw"], 2),
                                round(v["r_max"], 2)) for k, v in row.items() if isinstance(v, dict)},
                      round(row["Ry_raw_min_eig"], 4), flush=True)
    (HERE / "results/dev_compatibility.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
