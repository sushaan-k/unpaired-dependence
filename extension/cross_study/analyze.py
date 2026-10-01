#!/usr/bin/env python3
"""Prespecified summaries of PLAN.md ("Hypotheses and success criteria").

Bootstrap: 10,000 resamples of donors (seed 20260926), shared by every statistic
of a cohort. "95% lower/upper bound" = 2.5th/97.5th percentile. H4 uses
Bonferroni over 8 comparisons (percentiles 0.3125 and 99.6875).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from predict import ANALYSES, ARMS

HERE = Path(__file__).resolve().parent
BOOT, SEED = 10000, 20260926
BASELINES = ("reference_regression", "transferred_correlation", "transferred_covariance", "variances_only")
WITHIN = {"hao": "within_l1", "colon": "within_coarse"}


def interval(samples, level=0.95):
    a = 100 * (1 - level) / 2
    return [float(np.percentile(samples, a)), float(np.percentile(samples, 100 - a))]


def summarize(scores, perm, cohort, idx):
    donors = sorted({k.split("/")[1] for k in scores if k.startswith(cohort + "/")})
    E = {an: {arm: np.array([scores[f"{cohort}/{d}/{an}"]["arms"][arm]["E"] for d in donors]) for arm in ARMS}
         for an, _ in ANALYSES[cohort]}
    R = {an: {arm: np.array([scores[f"{cohort}/{d}/{an}"]["arms"][arm]["entry_correlation"] for d in donors])
              for arm in ARMS} for an, _ in ANALYSES[cohort]}
    boot = lambda v: v[idx].mean(axis=1)
    out = {"donors": donors,
           "per_donor_E": {an: {arm: E[an][arm].tolist() for arm in ARMS} for an in E},
           "mean_E": {an: {arm: float(E[an][arm].mean()) for arm in ARMS} for an in E},
           "mean_entry_correlation": {an: {arm: float(R[an][arm].mean()) for arm in ARMS} for an in R}}
    tot = E["total"]
    # H1
    observed = float(tot["pf_means"].mean())
    null = np.mean(np.array([perm[f"{cohort}/{d}"] for d in donors]), axis=0)
    out["H1"] = {"observed_mean_E": observed, "null_mean_E": float(null.mean()),
                 "null_min": float(null.min()), "null_quantiles_5_50_95": np.percentile(null, [5, 50, 95]).tolist(),
                 "p": float((1 + np.sum(null <= observed)) / (len(null) + 1)),
                 "per_donor_p": [float((1 + np.sum(np.array(perm[f"{cohort}/{d}"]) <= tot["pf_means"][i]))
                                       / (len(perm[f"{cohort}/{d}"]) + 1)) for i, d in enumerate(donors)]}
    out["H1"]["success"] = out["H1"]["p"] < 0.05
    # H2
    share = (1 - tot["pf_means"]) / (1 - tot["paired_closed_form"])
    out["H2"] = {"per_donor_share": share.tolist(), "mean_share": float(share.mean()),
                 "bootstrap_95": interval(boot(share))}
    out["H2"]["success"] = out["H2"]["bootstrap_95"][0] > 0.5
    share_moment = (1 - tot["pf_moment"]) / (1 - tot["paired_closed_form"])
    out["share_pf_moment"] = {"per_donor": share_moment.tolist(), "mean": float(share_moment.mean()),
                              "bootstrap_95": interval(boot(share_moment))}
    # H3
    w = E[WITHIN[cohort]]
    out["H3"] = {"analysis": WITHIN[cohort]}
    for arm in ("pf_means", "pf_moment", "paired_closed_form", "recipient_benchmark"):
        out["H3"][arm] = {"mean_E": float(w[arm].mean()), "bootstrap_95": interval(boot(w[arm])),
                          "donors_below_1": int(np.sum(w[arm] < 1))}
    out["H3"]["success"] = out["H3"]["pf_means"]["bootstrap_95"][1] < 1
    wshare = (1 - w["pf_means"]) / (1 - w["paired_closed_form"])
    out["H3"]["share_of_paired_within"] = {"mean": float(wshare.mean()), "bootstrap_95": interval(boot(wshare))}
    # H4
    out["H4"] = {}
    for base in BASELINES:
        diff = tot[base] - tot["paired_closed_form"]
        ci = interval(boot(diff), 1 - 0.05 / 8)
        out["H4"][base] = {"mean_difference": float(diff.mean()), "bonferroni_interval": ci,
                           "donors_favouring_closed_form": int(np.sum(diff > 0)), "success": ci[0] > 0}
    # descriptive
    diff = tot["pf_moment"] - tot["pf_means"]
    out["pf_moment_minus_pf_means"] = {"mean": float(diff.mean()), "bootstrap_95": interval(boot(diff)),
                                       "donors_moment_better": int(np.sum(diff < 0))}
    for an in E:
        out.setdefault("vs_benchmark", {})[an] = {
            arm: {"mean_difference": float((E[an][arm] - E[an]["recipient_benchmark"]).mean()),
                  "donors_better_than_benchmark": int(np.sum(E[an][arm] < E[an]["recipient_benchmark"]))}
            for arm in ARMS if arm != "recipient_benchmark"}
        out.setdefault("pf_vs_paired", {})[an] = {
            "pf_means_minus_paired": float((E[an]["pf_means"] - E[an]["paired_closed_form"]).mean()),
            "donors_pf_means_better": int(np.sum(E[an]["pf_means"] < E[an]["paired_closed_form"])),
            "pf_moment_minus_paired": float((E[an]["pf_moment"] - E[an]["paired_closed_form"]).mean()),
            "donors_pf_moment_better": int(np.sum(E[an]["pf_moment"] < E[an]["paired_closed_form"]))}
    return out


def main():
    scores = json.loads((HERE / "results/scores.json").read_text())
    perm = json.loads((HERE / "results/permutation_scores.json").read_text())["E"]
    summary = {"plan_sha256": scores["manifest"]["PLAN.md"],
               "recomputed_within_type": scores["recomputed_within_type"]}
    rng = np.random.default_rng(SEED)
    for cohort in ("hao", "colon"):
        n = len({k.split("/")[1] for k in scores["scores"] if k.startswith(cohort + "/")})
        idx = rng.integers(0, n, size=(BOOT, n))
        summary[cohort] = summarize(scores["scores"], perm, cohort, idx)
        s = summary[cohort]
        print(cohort, "H1", s["H1"]["success"], round(s["H1"]["p"], 4), "H2", s["H2"]["success"],
              [round(v, 3) for v in [s["H2"]["mean_share"]] + s["H2"]["bootstrap_95"]],
              "H3", s["H3"]["success"], round(s["H3"]["pf_means"]["mean_E"], 3),
              "H4", {b: s["H4"][b]["success"] for b in BASELINES})
    (HERE / "results/summary.json").write_text(json.dumps(summary, indent=1) + "\n")


if __name__ == "__main__":
    main()
