#!/usr/bin/env python3
"""Development analysis on data already scored (exploratory; not a test).

Computes the unpaired coverage, compatibility and control variables for every
recipient analysis and channel already evaluated in extension/cross_study:
the frozen means channel, the rank-32 moment channel, the within-type channel
of post hoc analysis P1, and the 41 learning-curve refits (all-cell analyses).
Outcomes (relative error E, independence = 1) are read from the stored score
files; they are used only to relate the diagnostics to failure, never to build
them. Writes results/dev_units.json.
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
from data import load_recipient, load_training, type_centre  # noqa: E402
from posthoc_within import type_units  # noqa: E402
from predict import ANALYSES, within_assay  # noqa: E402

from diagnostics import (between_share, compatibility, coverage, covariance_shift,  # noqa: E402
                         rowspace_projection, supported_projection)

LABEL_FOR_TOTAL = {"hao": ("l1", "within_l1"), "colon": ("coarse", "within_coarse")}


def recipient_units():
    scores = json.loads((CS / "results/scores.json").read_text())["scores"]
    units = []
    for cohort, analyses in ANALYSES.items():
        rec = load_recipient(cohort, "adaptation")
        key_total, tag_total = LABEL_FOR_TOTAL[cohort]
        for donor in sorted(set(rec["donor"])):
            m = rec["donor"] == donor
            for analysis, key in analyses:
                tag = f"{cohort}/{donor}/{analysis}"
                x, y = rec["x"][m], rec["y"][m]
                if key is not None:
                    types = scores[tag]["types"]
                    labels = rec["labels"][key][m]
                    cells = np.isin(labels, types)
                    x = type_centre(x[cells], labels[cells], types)
                    y = type_centre(y[cells], labels[cells], types)
                    fb = 0.0
                else:
                    types = scores[f"{cohort}/{donor}/{tag_total}"]["types"]
                    labels = rec["labels"][key_total][m]
                    fb = 0.5 * (between_share(x, labels, types) + between_share(y, labels, types))
                Rx, Ry, _, _ = within_assay(x, y)
                units.append({"tag": tag, "cohort": cohort, "donor": donor, "analysis": analysis,
                              "Rx": Rx, "Ry": Ry, "cells": int(len(x)), "between_share": fb})
            print(cohort, donor, flush=True)
    return units, scores


def mean_training_correlation(units_pf):
    S = sum(u[4] for u in units_pf) / len(units_pf)
    d = np.sqrt(np.diag(S))
    return S / np.outer(d, d)


def channels(train):
    """Every channel with its W, supported projection P, interaction B and training size."""
    ref = np.load(CS / "results/reference_fit.npz")
    patients = sorted(set(train["patient"]))
    raw_all = {r["patient"]: r for r in est.pf_raw_moments(train, patients)}
    raw = [raw_all[p] for p in patients]
    sd_x, sd_y = est.pooled_sd(raw)
    R_frozen = mean_training_correlation(est.pf_units(raw, sd_x, sd_y))
    out = {}
    P, dim, eff = supported_projection(ref["pf_means_G"], float(ref["pf_means_ridge"]))
    out["frozen"] = {"W": ref["pf_means_W"], "P": P, "dim": dim, "B": ref["pf_means_B"],
                     "n_train": len(patients), "R_train": R_frozen}
    Pm, dim_m = rowspace_projection(ref["pf_moment_W"])
    out["moment"] = {"W": ref["pf_moment_W"], "P": Pm, "dim": dim_m, "B": ref["pf_moment_B"],
                     "n_train": len(patients), "R_train": R_frozen}
    # post hoc P1: donor-by-type units centred within type (refitted by the frozen rules)
    traw = type_units(train)
    tsd_x, tsd_y = est.pooled_sd(traw)
    tunits = est.pf_units(traw, tsd_x, tsd_y)
    donor_fold = est.patient_folds(sorted({r["donor"] for r in traw}))
    fit = est.fit_pf_means(tunits, {r["patient"]: donor_fold[r["donor"]] for r in traw})
    P1, dim1, _ = supported_projection(fit["G"], fit["ridge"])
    out["P1"] = {"W": fit["W"], "P": P1, "dim": dim1, "B": fit["B"], "n_train": len(traw),
                 "R_train": mean_training_correlation(tunits)}
    # learning-curve refits (same subsets as cross_study/learning_curve.py)
    lc = json.loads((CS / "results/learning_curve.json").read_text())
    rng = np.random.default_rng(lc["seed"])
    plan = [(s, rng.choice(len(patients), s, replace=False)) for s in (8, 16, 32, 64) for _ in range(10)]
    plan.append((len(patients), np.arange(len(patients))))
    for i, ((size, idx), row) in enumerate(zip(plan, lc["fits"])):
        subset = [patients[j] for j in sorted(idx)]
        sraw = [raw_all[p] for p in subset]
        ssd_x, ssd_y = est.pooled_sd(sraw)
        sunits = est.pf_units(sraw, ssd_x, ssd_y)
        sfit = est.fit_pf_means(sunits, est.patient_folds(subset))
        Ps, dims, _ = supported_projection(sfit["G"], sfit["ridge"])
        assert dims == row["dimension_M"] and size == row["patients"], (i, dims, row["dimension_M"])
        out[f"lc{i:02d}"] = {"W": sfit["W"], "P": Ps, "dim": dims, "B": sfit["B"], "n_train": size,
                             "R_train": mean_training_correlation(sunits), "E": row["E"]}
    return out


def outcome(channel, name, tag, scores, p1):
    if name == "frozen":
        return scores[tag]["arms"]["pf_means"]["E"]
    if name == "moment":
        return scores[tag]["arms"]["pf_moment"]["E"]
    if name == "P1":
        return p1["E"][tag]
    key = "/".join(tag.split("/")[:2])
    return channel["E"].get(key) if tag.endswith("/total") else None


def main():
    train = load_training()
    units, scores = recipient_units()
    chans = channels(train)
    p1 = json.loads((CS / "results/posthoc_within.json").read_text())
    rows = []
    for name, ch in chans.items():
        for u in units:
            E = outcome(ch, name, u["tag"], scores, p1)
            if E is None:
                continue
            row = {"channel": name, "family": name if not name.startswith("lc") else "learning_curve",
                   "tag": u["tag"], "cohort": u["cohort"], "donor": u["donor"], "analysis": u["analysis"],
                   "E": float(E), "coverage": coverage(ch["P"], u["Rx"]), "dim": ch["dim"],
                   "cells": u["cells"], "n_train": ch["n_train"], "between_share": u["between_share"],
                   "shift": covariance_shift(u["Rx"], ch["R_train"])}
            row.update(compatibility(ch["W"], ch["P"], u["Rx"], u["Ry"]))
            rows.append(row)
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results/dev_units.json").write_text(json.dumps(rows, indent=1) + "\n")
    print(len(rows), "units")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
