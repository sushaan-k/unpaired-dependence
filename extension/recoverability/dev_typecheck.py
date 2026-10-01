#!/usr/bin/env python3
"""Development check of a channel against the recipient's own cell-type means (exploratory).

A recipient's cell types are populations: their RNA means (from RNA-only cells)
and protein means (from protein-only cells) give the recipient's between-type
cross-covariance directly, and a channel predicts it from the recipient's
between-type RNA covariance alone. Their agreement (cosine similarity) tests
the channel on the recipient's own between-population variation without any
paired cell; it cannot test the within-type channel. Adds 'type_agreement' to
every channel x donor of the development table. Writes results/dev_typecheck.json.
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

from data import standardize  # noqa: E402
from predict import ANALYSES  # noqa: E402

from dev_cache import load_counts  # noqa: E402
from dev_units2 import build_channels  # noqa: E402
from noise import lognorm  # noqa: E402

LABEL = {"hao": ("l1", "within_l1"), "colon": ("coarse", "within_coarse")}


def between_parts(x, y, labels, types):
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


def main():
    chans = build_channels()
    scores = json.loads((CS / "results/scores.json").read_text())["scores"]
    out = {}
    for cohort in ANALYSES:
        rec = load_counts(cohort, "adaptation")
        key, tag = LABEL[cohort]
        for donor in sorted(set(rec["donor"])):
            m = rec["donor"] == donor
            x, y = lognorm(rec["counts"][m], rec["library"][m]), rec["y"][m]
            types = scores[f"{cohort}/{donor}/{tag}"]["types"]
            Sb, Cb = between_parts(x, y, rec["labels"][key][m], types)
            for name, ch in chans.items():
                if ch["VS"].shape[1] == 0:
                    continue
                Cc = Sb @ (ch["W"] @ ch["VS"] @ ch["VS"].T).T
                cos = float(np.sum(Cc * Cb) / (np.linalg.norm(Cc) * np.linalg.norm(Cb)))
                out[f"{cohort}/{donor}|{name}"] = {"type_agreement": cos,
                                                   "type_scale": float(np.linalg.norm(Cc) / np.linalg.norm(Cb))}
            print(cohort, donor, round(out[f"{cohort}/{donor}|frozen"]["type_agreement"], 3),
                  round(out[f"{cohort}/{donor}|P1"]["type_agreement"], 3), flush=True)
    (HERE / "results/dev_typecheck.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
