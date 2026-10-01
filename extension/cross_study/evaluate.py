#!/usr/bin/env python3
"""Open the sealed scoring files and score the frozen predictions (PLAN.md, "Endpoints").

Refuses to run unless the prediction files match predictions_manifest.json.
Writes results/scores.json and results/truth.npz (scoring-cell cross-correlation
and RNA correlation matrices per donor and analysis; derived matrices only).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from data import cross_correlation, load_recipient, shrink, standardize, type_centre
from predict import ANALYSES, ARMS, MIN_CELLS, donor_predictions, reference

HERE = Path(__file__).resolve().parent


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_manifest():
    manifest = json.loads((HERE / "results/predictions_manifest.json").read_text())
    for name in ("predictions.npz", "prediction_info.json", "reference_fit.npz"):
        assert sha256(HERE / "results" / name) == manifest[name], name
    text = (HERE / "PLAN.md").read_text()
    prefix = text[:text.index("## Post hoc analysis P1") - 1] if "## Post hoc analysis P1" in text else text
    assert hashlib.sha256(prefix.encode()).hexdigest() == manifest["PLAN.md"], "PLAN.md"
    for name in ("seal.json", "hao_seal.json"):
        assert sha256(HERE / name) == manifest[name], name
    return manifest


def metrics(C, T):
    return {"E": float(np.linalg.norm(C - T) / np.linalg.norm(T)),
            "entry_correlation": float(np.corrcoef(C.ravel(), T.ravel())[0, 1]) if np.any(C) else 0.0}


def scoring_target(score, donor, key, keep):
    m = score["donor"] == donor
    x, y = score["x"][m], score["y"][m]
    if key is not None:
        labels = score["labels"][key][m]
        cells = np.isin(labels, keep)
        x = type_centre(x[cells], labels[cells], keep)
        y = type_centre(y[cells], labels[cells], keep)
    xs, _ = standardize(x)
    return cross_correlation(x, y), shrink(xs.T @ xs / len(x)), len(x)


def main():
    manifest = check_manifest()
    preds = np.load(HERE / "results/predictions.npz")
    info = json.loads((HERE / "results/prediction_info.json").read_text())
    ref = reference()
    scores, truth, recomputed = {}, {}, []
    for cohort, analyses in ANALYSES.items():
        adapt = load_recipient(cohort, "adaptation")
        score = load_recipient(cohort, "scoring")          # SHA-256 checked against the seal
        for donor in sorted(set(score["donor"])):
            for analysis, key in analyses:
                tag = f"{cohort}/{donor}/{analysis}"
                P = {arm: preds[f"{tag}/{arm}"] for arm in ARMS}
                keep = None
                if key is not None:
                    keep_a = info[tag]["types"]
                    labels = score["labels"][key][score["donor"] == donor]
                    keep = [t for t in keep_a if np.sum(labels == t) >= MIN_CELLS]
                    if keep != keep_a:                       # scoring half drops a type: same rule, recomputed
                        P, _ = donor_predictions(adapt, donor, analysis, key, ref, keep=keep)
                        recomputed.append({"tag": tag, "dropped": sorted(set(keep_a) - set(keep))})
                T, Rx_score, n = scoring_target(score, donor, key, keep)
                scores[tag] = {"scoring_cells": n, "types": keep,
                               "arms": {arm: metrics(P[arm], T) for arm in ARMS}}
                truth[f"{tag}/T"], truth[f"{tag}/Rx"] = T, Rx_score
                print(tag, n, {a: round(v["E"], 3) for a, v in scores[tag]["arms"].items()}, flush=True)
    np.savez_compressed(HERE / "results/truth.npz", **truth)
    record = {"manifest": manifest, "recomputed_within_type": recomputed, "scores": scores}
    (HERE / "results/scores.json").write_text(json.dumps(record, indent=1) + "\n")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
