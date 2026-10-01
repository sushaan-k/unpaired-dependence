#!/usr/bin/env python3
"""Predict every recipient donor's cross-correlation matrix for every arm (PLAN.md).

Reads the frozen reference fit and the recipients' ADAPTATION files only. Writes
results/predictions.npz and results/predictions_manifest.json (SHA-256 of the
predictions, the reference fit, the plan and the seals) before any scoring
file is opened.

Within-type analyses keep cell types with at least 10 adaptation cells in the
donor; evaluate.py applies the scoring-half part of the rule (at least 10
scoring cells) and recomputes a donor's within-type predictions if that drops
a type (reported).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import estimators as est
from data import load_recipient, shrink, standardize, type_centre

HERE = Path(__file__).resolve().parent
ARMS = ("pf_means", "pf_moment", "paired_closed_form", "reference_regression", "transferred_correlation",
        "transferred_covariance", "variances_only", "recipient_benchmark", "independence")
ANALYSES = {"hao": (("total", None), ("within_l1", "l1"), ("within_l2", "l2")),
            "colon": (("total", None), ("within_coarse", "coarse"))}
MIN_CELLS = 10


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference():
    z = np.load(HERE / "results/reference_fit.npz")
    return {"pf_means": z["pf_means_B"], "pf_moment": z["pf_moment_B"], "paired": z["paired_B"],
            "W_ref": z["paired_W_ref"], "corr": z["paired_corr"], "cov": z["paired_cov"]}


def within_assay(x, y):
    """Shrunk within-assay correlation matrices and raw SDs; no cross-moment is formed."""
    xs, sdx = standardize(x)
    ys, sdy = standardize(y)
    n = len(x)
    return shrink(xs.T @ xs / n), shrink(ys.T @ ys / n), sdx, sdy


def arm_predictions(x, y, barcodes, ref):
    Rx, Ry, sdx, sdy = within_assay(x, y)
    W_bench, bench = est.benchmark(x, y, barcodes)        # the only arm that uses recipient pairs
    inv = lambda s: np.where(s > 0, 1.0 / np.where(s > 0, s, 1.0), 0.0)
    p, q = Rx.shape[0], Ry.shape[0]
    preds = {
        "pf_means": est.closed_form(Rx, Ry, ref["pf_means"]),
        "pf_moment": est.closed_form(Rx, Ry, ref["pf_moment"]),
        "paired_closed_form": est.closed_form(Rx, Ry, ref["paired"]),
        "reference_regression": Rx @ ref["W_ref"],
        "transferred_correlation": ref["corr"],
        "transferred_covariance": ref["cov"] * np.outer(inv(sdx), inv(sdy)),
        "variances_only": est.closed_form(np.eye(p), np.eye(q), ref["paired"]),
        "recipient_benchmark": Rx @ W_bench,
        "independence": np.zeros((p, q)),
    }
    return preds, bench


def kept_types(labels, minimum=MIN_CELLS):
    types, counts = np.unique(labels, return_counts=True)
    return sorted(t for t, c in zip(types, counts) if c >= minimum)


def donor_predictions(rec, donor, analysis, label_key, ref, keep=None):
    m = rec["donor"] == donor
    x, y, barcodes = rec["x"][m], rec["y"][m], rec["barcode"][m]
    info = {"cells": int(m.sum())}
    if label_key is not None:
        labels = rec["labels"][label_key][m]
        keep = kept_types(labels) if keep is None else keep
        cells = np.isin(labels, keep)
        x = type_centre(x[cells], labels[cells], keep)
        y = type_centre(y[cells], labels[cells], keep)
        barcodes = barcodes[cells]
        info.update({"types": keep, "cells": int(cells.sum())})
    preds, bench = arm_predictions(x, y, barcodes, ref)
    info["benchmark"] = bench
    return preds, info


def main():
    ref = reference()
    out, meta = {}, {}
    for cohort, analyses in ANALYSES.items():
        rec = load_recipient(cohort, "adaptation")
        for donor in sorted(set(rec["donor"])):
            for analysis, label_key in analyses:
                preds, info = donor_predictions(rec, donor, analysis, label_key, ref)
                for arm in ARMS:
                    out[f"{cohort}/{donor}/{analysis}/{arm}"] = preds[arm]
                meta[f"{cohort}/{donor}/{analysis}"] = info
                print(cohort, donor, analysis, info["cells"], "benchmark lambda", info["benchmark"]["lambda"],
                      flush=True)
    path = HERE / "results/predictions.npz"
    np.savez_compressed(path, **out)
    (HERE / "results/prediction_info.json").write_text(json.dumps(meta, indent=1, default=float) + "\n")
    manifest = {"predictions.npz": sha256(path),
                "prediction_info.json": sha256(HERE / "results/prediction_info.json"),
                "reference_fit.npz": sha256(HERE / "results/reference_fit.npz"),
                "PLAN.md": sha256(HERE / "PLAN.md"),
                "seal.json": sha256(HERE / "seal.json"), "hao_seal.json": sha256(HERE / "hao_seal.json"),
                "note": "written before any scoring file was opened"}
    (HERE / "results/predictions_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
