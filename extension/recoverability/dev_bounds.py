#!/usr/bin/env python3
"""Development check of the association bounds (exploratory; data already scored).

For the frozen channel and every recipient analysis of the untouched blood
cohort and colon: with the latent RNA covariance, is the compatible set
nonempty, what share of target entries lies inside the entrywise bounds, how
wide are the bounds, and how often do they determine the sign of an entry
(and correctly)? Writes results/dev_bounds.json.
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

from data import shrink, standardize, type_centre  # noqa: E402
from predict import ANALYSES  # noqa: E402

from compat import completion_bounds, supported_basis  # noqa: E402
from dev_cache import channels, load_counts  # noqa: E402
from noise import lognorm, noise_fraction, signal_correlation, split_halves  # noqa: E402


def summary(bounds, T):
    lo, hi, centre = bounds
    inside = (T >= lo) & (T <= hi)
    signed = (lo > 0) | (hi < 0)
    correct = np.where(lo > 0, T > 0, T < 0)
    return {"inside": float(inside.mean()), "mean_halfwidth": float(np.mean(hi - lo) / 2),
            "mean_abs_target": float(np.mean(np.abs(T))), "sign_determined": float(signed.mean()),
            "sign_correct": float(correct[signed].mean()) if signed.any() else None,
            "centre_error": float(np.linalg.norm(centre - T) / np.linalg.norm(T))}


def main():
    ch = channels()["frozen"]
    VS, _ = supported_basis(ch["G"], float(ch["ridge"]))
    W = ch["W"]
    scores = json.loads((CS / "results/scores.json").read_text())["scores"]
    truth = np.load(CS / "results/truth.npz")
    out = {}
    for cohort, analyses in ANALYSES.items():
        rec = load_counts(cohort, "adaptation")
        for donor in sorted(set(rec["donor"])):
            m = rec["donor"] == donor
            counts, library, y0 = rec["counts"][m], rec["library"][m], rec["y"][m]
            x0 = lognorm(counts, library)
            h1, h2 = split_halves(counts, library)
            for analysis, key in analyses:
                tag = f"{cohort}/{donor}/{analysis}"
                if key is not None:
                    types = scores[tag]["types"]
                    labels = rec["labels"][key][m]
                    cells = np.isin(labels, types)
                    x = type_centre(x0[cells], labels[cells], types)
                    y = type_centre(y0[cells], labels[cells], types)
                    nu = noise_fraction(h1[cells], h2[cells], labels[cells], types)
                else:
                    x, y, nu = x0, y0, noise_fraction(h1, h2)
                xs, _ = standardize(x)
                ys, _ = standardize(y)
                Rx, Ry = shrink(xs.T @ xs / len(x)), shrink(ys.T @ ys / len(x))
                Rz = signal_correlation(Rx, nu)
                T = truth[f"{tag}/T"]
                b = completion_bounds(W, VS, Rz, Ry)
                out[tag] = None if b is None else summary(b, T)
                print(tag, out[tag], flush=True)
    (HERE / "results/dev_bounds.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
