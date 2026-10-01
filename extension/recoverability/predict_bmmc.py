#!/usr/bin/env python3
"""Bone-marrow test, step 1: predictions, unpaired diagnostics and reference selections (PLAN.md).

Reads the adaptation file only. Refuses to run unless PLAN.md and the code
match results/freeze.json. For every recipient analysis (bmmc_units.json) and
channel (bmmc_channels.npz) it computes the closed-form predictions with
measured and with latent RNA covariance and the diagnostics; the label-based
between-type estimator and the recipient benchmark (own adaptation pairs) are
added per analysis; entrywise compatibility bounds for the frozen channel;
paired-reference selections for the within-coarse target. SHA-256 digests of
every prediction matrix are recorded, so that the evaluation can recompute
predictions and prove they are the ones fixed here.

Writes results/bmmc_predictions.npz (frozen, P1 and colon channels, label and
benchmark arms), results/bmmc_bounds.npz, results/bmmc_selections.npz,
results/bmmc_diagnostics.json, results/bmmc_digests.json and
results/bmmc_manifest.json.
"""

from __future__ import annotations

import hashlib
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
from data import shrink, standardize, type_centre  # noqa: E402
from gaussian_transfer import Marginal, saturate  # noqa: E402

import reference as ref  # noqa: E402
from bmmc import load  # noqa: E402
from compat import completion_bounds, opnorm, psd_power, supported_basis, svd  # noqa: E402
from diagnostics import between_share, covariance_shift  # noqa: E402
from freeze import check_freeze  # noqa: E402
from noise import lognorm, noise_fraction, signal_correlation, split_halves  # noqa: E402

ANALYSES = (("total", None), ("within_coarse", "coarse"), ("within_fine", "fine"))
STORED = ("frozen", "P1", "colon")
BUDGETS = (50, 100, 200, 400, 800)
DRAWS = 50
STRATEGIES = ("random", "balanced", "guided")
SELECTION_SEED = 20261002
RESULT_FILES = ["bmmc_predictions.npz", "bmmc_bounds.npz", "bmmc_selections.npz", "bmmc_diagnostics.json",
                "bmmc_digests.json", "bmmc_channels.npz", "bmmc_channels.json", "bmmc_units.json", "bmmc_seal.json",
                "failure_model.json", "freeze.json"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(M):
    return hashlib.sha256(np.ascontiguousarray(M, dtype=np.float64).tobytes()).hexdigest()


def load_channels():
    z = np.load(HERE / "results/bmmc_channels.npz")
    info = json.loads((HERE / "results/bmmc_channels.json").read_text())
    out = {}
    for name in info:
        c = {k: z[f"{name}/{k}"] for k in ("W", "B", "G", "ridge", "R_train")}
        VS, _ = supported_basis(c["G"], float(c["ridge"]))
        out[name] = {"W": c["W"], "B": c["B"], "VS": VS, "R_train": c["R_train"], "n_train": info[name]["n_train"]}
    return out


def family(name):
    return name if name in ("frozen", "P1", "colon") else name.rstrip("0123456789")


def between_parts(x, y, labels, types):
    """Recipient between-type RNA covariance and label-based between-type cross-correlation (total SDs)."""
    cells = np.isin(labels, types)
    x, y, labels = x[cells], y[cells], labels[cells]
    xs, _ = standardize(x)
    ys, _ = standardize(y)
    Sb = np.zeros((x.shape[1], x.shape[1]))
    Cb = np.zeros((x.shape[1], y.shape[1]))
    for t in types:
        m = labels == t
        a, b = xs[m].mean(0), ys[m].mean(0)
        Sb += m.mean() * np.outer(a, a)
        Cb += m.mean() * np.outer(a, b)
    return Sb, Cb


def sample_inputs(rec, batch, unit):
    m = rec["batch"] == batch
    counts, library = rec["counts"][m], rec["library"][m]
    s = {"x0": lognorm(counts, library), "y0": rec["y"][m], "barcodes": rec["barcode"][m],
         "coarse": rec["labels"]["coarse"][m], "fine": rec["labels"]["fine"][m]}
    s["h1"], s["h2"] = split_halves(counts, library)
    s["Sb"], s["Cb"] = between_parts(s["x0"], s["y0"], s["coarse"], unit["types"]["coarse"])
    return s


def analysis_inputs(s, key, types):
    if key is None:
        x, y, nu = s["x0"], s["y0"], noise_fraction(s["h1"], s["h2"])
        cells = np.ones(len(x), bool)
    else:
        labels = s[key]
        cells = np.isin(labels, types)
        x = type_centre(s["x0"][cells], labels[cells], types)
        y = type_centre(s["y0"][cells], labels[cells], types)
        nu = noise_fraction(s["h1"][cells], s["h2"][cells], labels[cells], types)
    xs, _ = standardize(x)
    ys, _ = standardize(y)
    n = len(x)
    Rx, Ry = shrink(xs.T @ xs / n), shrink(ys.T @ ys / n)
    return {"x": x, "y": y, "cells": cells, "nu": nu, "Rx": Rx, "Ry": Ry, "Rz": signal_correlation(Rx, nu),
            "barcodes": s["barcodes"][cells]}


def arm_predictions(a, s, key):
    """Channel-free arms: recipient benchmark (own adaptation pairs), independence, label-based between."""
    W_bench, bench = est.benchmark(a["x"], a["y"], a["barcodes"])
    arms = {"benchmark": a["Rx"] @ W_bench, "independence": np.zeros((a["x"].shape[1], a["y"].shape[1]))}
    if key is None:
        arms["label_between"] = s["Cb"]
    return arms, bench


def transfer(marginal, B):
    """gaussian_transfer.transfer with the SVD fallback of amendment 1 (identical when gesdd converges)."""
    U, beta, Vt = svd(marginal.Rx @ B @ marginal.Ry)
    gamma = saturate(beta)
    return marginal.Rx @ (U * gamma) @ Vt @ marginal.Ry, gamma


def channel_predictions(a, ch):
    """Closed-form transfer with measured and with latent RNA covariance."""
    return {k: transfer(Marginal(R, a["Ry"]), ch["B"])[0] for k, R in (("measured", a["Rx"]), ("signal", a["Rz"]))}


def channel_diagnostics(a, ch, roots, Ryih):
    W, VS = ch["W"], ch["VS"]
    P = VS @ VS.T
    row = {"dim_S": int(VS.shape[1]), "n_train": int(ch["n_train"]), "shift": covariance_shift(a["Rx"], ch["R_train"])}
    for k, R in (("measured", a["Rx"]), ("signal", a["Rz"])):
        row[f"coverage_{k}"] = float(np.trace(P @ R @ P) / np.trace(R))
        Rh, Rih = roots[k]
        if VS.shape[1]:
            QN, _ = np.linalg.qr(Rih @ VS)
            row[f"F_{k}"] = opnorm(QN.T @ Rh @ W.T @ Ryih)
        else:
            row[f"F_{k}"] = 0.0
        WS = W @ P
        r = np.diag(WS @ R @ WS.T)
        row[f"excess_{k}"] = float(np.sum(np.clip(r - 1, 0, None)) / max(np.sum(r), 1e-12))
        row[f"frac_over_{k}"] = float(np.mean(r > 1))
    return row


def type_agreement(s, ch):
    if ch["VS"].shape[1] == 0:
        return None
    Cc = s["Sb"] @ (ch["W"] @ ch["VS"] @ ch["VS"].T).T
    return float(np.sum(Cc * s["Cb"]) / (np.linalg.norm(Cc) * np.linalg.norm(s["Cb"])))


def main():
    freeze = check_freeze()
    chans = load_channels()
    units = json.loads((HERE / "results/bmmc_units.json").read_text())
    rec = load("adaptation")
    stored, bounds, digests, diag = {}, {}, {}, []
    sel_index, sel_offsets, sel_meta = [], [0], []
    rng = np.random.default_rng(SELECTION_SEED)
    for batch, unit in units.items():
        s = sample_inputs(rec, batch, unit)
        ctypes = unit["types"]["coarse"]
        agreement = {name: type_agreement(s, ch) for name, ch in chans.items()}
        composition = 0.5 * (between_share(s["x0"], s["coarse"], ctypes) + between_share(s["y0"], s["coarse"], ctypes))
        for analysis, key in ANALYSES:
            a = analysis_inputs(s, key, None if key is None else unit["types"][key])
            tag = f"{batch}/{analysis}"
            roots = {k: (psd_power(a[R], 0.5), psd_power(a[R], -0.5)) for k, R in (("measured", "Rx"), ("signal", "Rz"))}
            Ryih = psd_power(a["Ry"], -0.5)
            arms, bench = arm_predictions(a, s, key)
            for arm, C in arms.items():
                digests[f"{tag}/{arm}"] = digest(C)
                stored[f"{tag}/{arm}"] = C
            common = {"tag": tag, "batch": batch, "donor": unit["donor"], "analysis": analysis, "cells": int(len(a["x"])),
                      "between_share": float(composition) if key is None else 0.0, "nu_mean": float(a["nu"].mean()),
                      "benchmark_lambda": bench["lambda"]}
            for name, ch in chans.items():
                for k, C in channel_predictions(a, ch).items():
                    digests[f"{tag}/{name}/{k}"] = digest(C)
                    if name in STORED:
                        stored[f"{tag}/{name}/{k}"] = C
                row = dict(common, channel=name, family=family(name), type_agreement=agreement[name])
                row.update(channel_diagnostics(a, ch, roots, Ryih))
                diag.append(row)
            b = completion_bounds(chans["frozen"]["W"], chans["frozen"]["VS"], a["Rz"], a["Ry"])
            if b is not None:
                bounds[f"{tag}/lower"], bounds[f"{tag}/upper"] = b[0], b[1]
                bounds[f"{tag}/centre"] = b[2]
            frozen = [r for r in diag if r["tag"] == tag and r["channel"] == "frozen"][0]
            print(tag, len(a["x"]), "frozen F", round(frozen["F_signal"], 3), "coverage",
                  round(frozen["coverage_signal"], 3), flush=True)
        cells = np.isin(s["coarse"], ctypes)
        xk, yk, lk = s["x0"][cells], s["y0"][cells], s["coarse"][cells]
        summ = ref.summaries(xk, yk, lk, ctypes)
        for N in BUDGETS:
            for strategy in STRATEGIES:
                for draw in range(DRAWS):
                    idx = ref.select(ref.allocate(strategy, N, summ, rng), lk, ctypes, rng)
                    sel_index.append(idx.astype(np.int32))
                    sel_offsets.append(sel_offsets[-1] + len(idx))
                    sel_meta.append((batch, N, strategy, draw))
            for draw in range(DRAWS):
                idx = np.sort(rng.choice(len(xk), size=min(N, len(xk)), replace=False))
                sel_index.append(idx.astype(np.int32))
                sel_offsets.append(sel_offsets[-1] + len(idx))
                sel_meta.append((batch, N, "paired_only", draw))
    np.savez_compressed(HERE / "results/bmmc_predictions.npz", **stored)
    np.savez_compressed(HERE / "results/bmmc_bounds.npz", **bounds)
    np.savez_compressed(HERE / "results/bmmc_selections.npz", index=np.concatenate(sel_index),
                        offsets=np.array(sel_offsets, np.int64),
                        batch=np.array([t[0] for t in sel_meta]), budget=np.array([t[1] for t in sel_meta]),
                        strategy=np.array([t[2] for t in sel_meta]), draw=np.array([t[3] for t in sel_meta]))
    (HERE / "results/bmmc_diagnostics.json").write_text(json.dumps(diag, indent=1) + "\n")
    (HERE / "results/bmmc_digests.json").write_text(json.dumps(digests, indent=0) + "\n")
    manifest = {n: sha(HERE / "results" / n) for n in RESULT_FILES}
    manifest["frozen_at"] = freeze["frozen_at"]
    manifest["note"] = "written before the scoring file was opened"
    (HERE / "results/bmmc_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
