#!/usr/bin/env python3
"""Post hoc analysis P1 (PLAN.md; exploratory): populations that vary within cell types.

Training units are Stephenson patient x initial_clustering groups with at least 50
cells in each hash half. RNA moments come from the RNA half and protein moments
from the protein half of each group. Unit means are centred within type, so the
channel is learned from within-type, between-patient variation only. The frozen
pf_means rules then give B, which is scored against the frozen targets.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import estimators as est
from data import load_recipient, load_training, type_centre
from identifiability import floor, identified_subspace
from predict import ANALYSES, within_assay

HERE = Path(__file__).resolve().parent
MIN_CELLS = 50


def type_units(train):
    keys = sorted({(p, t) for p, t in zip(train["patient"], train["initial"])})
    raw = []
    for p, t in keys:
        m = (train["patient"] == p) & (train["initial"] == t)
        rna, prot = m & (train["half"] == 0), m & (train["half"] == 1)
        if rna.sum() < MIN_CELLS or prot.sum() < MIN_CELLS:
            continue
        xr, yr = train["x"][rna], train["y"][prot]
        raw.append({"patient": f"{p}|{t}", "donor": p, "type": t, "nx": len(xr), "mx": xr.mean(0),
                    "Sx": np.cov(xr.T, bias=True), "ny": len(yr), "my": yr.mean(0), "Sy": np.cov(yr.T, bias=True)})
    for t in sorted({r["type"] for r in raw}):
        group = [r for r in raw if r["type"] == t]
        wx = np.array([r["nx"] for r in group], float)
        wy = np.array([r["ny"] for r in group], float)
        cx = sum(w * r["mx"] for w, r in zip(wx, group)) / wx.sum()
        cy = sum(w * r["my"] for w, r in zip(wy, group)) / wy.sum()
        for r in group:
            r["mx"], r["my"] = r["mx"] - cx, r["my"] - cy
    return raw


def main():
    train = load_training()
    raw = type_units(train)
    sd_x, sd_y = est.pooled_sd(raw)
    units = est.pf_units(raw, sd_x, sd_y)
    donor_fold = est.patient_folds(sorted({r["donor"] for r in raw}))
    fold_of = {r["patient"]: donor_fold[r["donor"]] for r in raw}
    fit = est.fit_pf_means(units, fold_of)
    VM, _, h = identified_subspace(fit["G"], fit["ridge"])
    truth = np.load(HERE / "results/truth.npz")
    scores = json.loads((HERE / "results/scores.json").read_text())["scores"]
    out = {"units": len(raw), "patients": len({r["donor"] for r in raw}),
           "types": {t: sum(r["type"] == t for r in raw) for t in sorted({r["type"] for r in raw})},
           "scale": fit["scale"], "tuning": fit["tuning"], "dimension_M": int(VM.shape[1]),
           "effective_dimension": float(h.sum()), "E": {}, "rho": {}, "floor": {}}
    P = VM @ VM.T
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
                    x = type_centre(x[cells], labels[cells], types)
                    y = type_centre(y[cells], labels[cells], types)
                Rx, Ry, _, _ = within_assay(x, y)
                T = truth[f"{tag}/T"]
                out["E"][tag] = float(np.linalg.norm(est.closed_form(Rx, Ry, fit["B"]) - T) / np.linalg.norm(T))
                out["rho"][tag] = float(np.trace(P @ Rx @ P) / np.trace(Rx))
                out["floor"][tag] = floor(T, truth[f"{tag}/Rx"], VM)
    summary = {}
    for cohort, analyses in ANALYSES.items():
        for analysis, _ in analyses:
            tags = [k for k in out["E"] if k.startswith(cohort + "/") and k.endswith("/" + analysis)]
            summary[f"{cohort}/{analysis}"] = {
                "mean_E": float(np.mean([out["E"][k] for k in tags])),
                "donors_below_1": int(sum(out["E"][k] < 1 for k in tags)),
                "mean_E_frozen_pf_means": float(np.mean([scores[k]["arms"]["pf_means"]["E"] for k in tags])),
                "mean_E_paired": float(np.mean([scores[k]["arms"]["paired_closed_form"]["E"] for k in tags])),
                "mean_rho": float(np.mean([out["rho"][k] for k in tags])),
                "mean_floor": float(np.mean([out["floor"][k] for k in tags]))}
    out["summary"] = summary
    (HERE / "results/posthoc_within.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("units", "patients", "types", "scale", "dimension_M",
                                          "effective_dimension")}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
