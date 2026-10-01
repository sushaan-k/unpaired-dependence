#!/usr/bin/env python3
"""Post hoc (added after unsealing, in response to review): controls for the paired-reference analysis.

Reads results/bmmc_evaluation.json only (no raw data). Two questions:

1. Total cross-correlation. Unpaired type means alone (zero paired cells) are
   compared with type means plus randomly, evenly or Neyman-selected paired
   cells (the same stratified James-Stein estimator for the within-type part)
   and with random paired cells used alone. Differences from type means alone,
   with 95% intervals from 2,000 bootstrap resamples of donors.
2. Selection efficiency for the within-type part. For each guided budget N,
   the number of randomly selected cells with the same mean within-type error,
   interpolated linearly in log budget between the evaluated budgets (50 to
   800), and the saving 1 - N / N_random; intervals from resampling donors (the
   mean curves are recomputed in every replicate). Also per sample.

Writes results/posthoc_reference.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from stats import cluster_bootstrap

HERE = Path(__file__).resolve().parent
BUDGETS = (50, 100, 200, 400, 800)
BOOT, SEED = 2000, 20261010


def equivalent(curve, target):
    """Budget at which a decreasing error curve (over BUDGETS) equals target; nan outside the evaluated range."""
    logb = np.log(BUDGETS)
    for k in range(len(BUDGETS) - 1):
        a, b = curve[k], curve[k + 1]
        if (a - target) * (b - target) <= 0 and a != b:
            return float(np.exp(logb[k] + (a - target) / (a - b) * (logb[k + 1] - logb[k])))
    return float("nan")


def main():
    ev = json.loads((HERE / "results/bmmc_evaluation.json").read_text())["reference"]
    units = json.loads((HERE / "results/bmmc_units.json").read_text())
    samples = list(ev["per_sample"])
    donors = np.array([units[b]["donor"] for b in samples])
    ps = ev["per_sample"]
    within = {st: np.array([[ps[b]["budgets"][f"{N}/{st}"]["within"] for N in BUDGETS] for b in samples])
              for st in ("random", "balanced", "guided")}
    total = {st: np.array([[ps[b]["budgets"][f"{N}/{st}"]["total"] for N in BUDGETS] for b in samples])
             for st in ("random", "balanced", "guided", "paired_only")}
    means_only = np.array([ps[b]["between_only_total"] for b in samples])

    def ci(values, stat, seed):
        boot = cluster_bootstrap(donors, lambda i: stat(values[i]), BOOT, seed)
        boot = boot[np.isfinite(boot)]
        return [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]

    out = {"samples": len(samples), "donors": len(set(donors)), "type_means_only": {
        "mean": float(means_only.mean()), "ci": ci(means_only, np.mean, SEED)}, "total": {}}
    for k, N in enumerate(BUDGETS):
        row = {}
        for st in ("random", "balanced", "guided", "paired_only"):
            v = total[st][:, k]
            row[st] = {"mean": float(v.mean())}
            if st != "paired_only":
                d = v - means_only
                row[st]["minus_type_means_only"] = {"mean": float(d.mean()), "ci": ci(d, np.mean, SEED + 10 * k + 1),
                                                    "samples_lower": int(np.sum(d < 0))}
        d = total["guided"][:, k] - total["random"][:, k]
        row["guided_minus_random"] = {"mean": float(d.mean()), "ci": ci(d, np.mean, SEED + 10 * k + 2),
                                      "samples_lower": int(np.sum(d < 0))}
        out["total"][str(N)] = row
    d = total["paired_only"][:, -1] - means_only
    out["paired_only_800_minus_type_means_only"] = {"mean": float(d.mean()), "ci": ci(d, np.mean, SEED + 99)}

    out["within_equivalent"] = {}
    for k, N in enumerate(BUDGETS[:-1]):
        def pooled(i, k=k, N=N):
            n_star = equivalent(within["random"][i].mean(0), within["guided"][i, k].mean())
            return 1 - N / n_star
        n_star = equivalent(within["random"].mean(0), within["guided"][:, k].mean())
        per = np.array([equivalent(within["random"][j], within["guided"][j, k]) for j in range(len(samples))])
        boot = cluster_bootstrap(donors, pooled, BOOT, SEED + 200 + k)
        boot = boot[np.isfinite(boot)]
        saving_per = 1 - N / per
        out["within_equivalent"][str(N)] = {
            "random_cells_needed": n_star, "saving": 1 - N / n_star,
            "saving_ci": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
            "random_cells_needed_ci": [float(N / (1 - np.quantile(boot, 0.025))),
                                       float(N / (1 - np.quantile(boot, 0.975)))],
            "per_sample_random_cells": [float(v) for v in per],
            "per_sample_saving_mean": float(np.nanmean(saving_per)),
            "per_sample_saving_range": [float(np.nanmin(saving_per)), float(np.nanmax(saving_per))],
            "per_sample_outside_range": int(np.sum(~np.isfinite(per)))}
    (HERE / "results/posthoc_reference.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
