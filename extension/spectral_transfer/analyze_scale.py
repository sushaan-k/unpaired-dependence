#!/usr/bin/env python3
"""Prespecified summaries of PLAN.md for designs D1, D2 and D3.

Primary endpoint: relative Frobenius error of the predicted cross-covariance
against the scoring-half paired cross-covariance, averaged over recipients.
Primary contrasts (closed_form minus comparator; negative favours the closed
form): variances_only, regression, reference_corr, reference_cov. Paired donor
bootstrap, 20,000 draws, seed 20260927; Bonferroni across the four contrasts.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PRIMARY = ("variances_only", "regression", "reference_corr", "reference_cov")
METRICS = ("cov_rel_error", "cov_corr", "r2")


def table(result, arms=None):
    donors = list(result["donors"])
    arms = arms or list(result["donors"][donors[0]]["arms"])
    return donors, arms, {m: np.array([[result["donors"][d]["arms"][a][m] for a in arms] for d in donors])
                          for m in METRICS}


def contrasts(values, arms, target, comparators, family):
    n = len(values)
    draws = np.random.default_rng(20260927).integers(0, n, (20000, n))
    out = {}
    for comp in comparators:
        diff = values[:, arms.index(target)] - values[:, arms.index(comp)]
        boot = diff[draws].mean(axis=1)
        out[comp] = {"mean_difference": float(diff.mean()),
                     "bootstrap_95": np.quantile(boot, (0.025, 0.975)).tolist(),
                     "bootstrap_familywise": np.quantile(boot, (0.025 / family, 1 - 0.025 / family)).tolist(),
                     "donors_favouring_target": int(np.sum(diff < 0))}
    return out


def summarize(path, extra=None):
    result = json.loads(path.read_text())
    donors, arms, values = table(result)
    if extra is not None:
        extra_result = json.loads(extra.read_text())
        _, extra_arms, extra_values = table(extra_result)
        assert list(extra_result["donors"]) == donors
        arms = arms + extra_arms
        values = {m: np.concatenate([values[m], extra_values[m]], axis=1) for m in METRICS}
    summary = {"donors": donors, "genes": result["genes"], "proteins": result["proteins"],
               "mean": {m: dict(zip(arms, values[m].mean(0).tolist())) for m in METRICS},
               "primary_contrasts": contrasts(values["cov_rel_error"], arms, "closed_form", PRIMARY, 4)}
    if extra is not None:
        summary["pairing_free_vs_paired"] = {
            target: contrasts(values["cov_rel_error"], arms, target,
                              ("closed_form", "regression", "variances_only", "independence"), 4)
            for target in ("pf_lowrank_closed_form", "pf_means_closed_form")}
    if "cell_coupling" in arms:
        summary["cells_vs_closed_form"] = contrasts(values["cov_rel_error"], arms, "cell_coupling",
                                                    ("closed_form",), 1)
    return summary


def main():
    report = {}
    if (RESULTS / "d1_lodo.json").exists():
        extra = RESULTS / "d3_pairing_free.json"
        report["D1_leave_one_donor_out"] = summarize(RESULTS / "d1_lodo.json",
                                                     extra if extra.exists() else None)
    if (RESULTS / "d2_hc_to_uc.json").exists():
        report["D2_healthy_to_colitis"] = summarize(RESULTS / "d2_hc_to_uc.json")
    (RESULTS / "scale_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
