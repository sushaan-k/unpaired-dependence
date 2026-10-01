#!/usr/bin/env python3
"""Summaries S1-S4 of PLAN.md from certified endpoints and released arrays.

    python analyze_identified.py            # requires results/star_endpoints.npz
    python analyze_identified.py --s3-only  # held-out reversal summary only

Writes results/summary.json, results/recipient_queries.csv and the LaTeX
macro and figure files used by the revised manuscript.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.special import rel_entr
from scipy.stats import spearmanr

from identified import log_odds, minimax
from run_identified import RESULTS, load_recipients

COHORTS = ("Cambridge", "Newcastle", "T-ALL", "Colon")
MARKERS = ("CD4", "CD7", "CD14", "CD19", "CD33", "CD38", "CD44", "CD47", "CD52")
SEED = 20260926
DRAWS = 20000
FAMILY_TAIL = 0.025 / len(COHORTS)


def predicted_cells(data):
    return data["predictions"][:, 0, :, 1, 1], data["predictions"][:, 3, :, 1, 1]


def query_frequencies(data):
    m = np.repeat(data["rna_means"], 9, axis=1)          # q = 9 i + j
    n = np.tile(data["protein_means"], (1, 9))
    return m, n


def reversal_summary(data):
    """S3: held-out direction when full patterns reverse the frequency-only sign."""
    m, n = query_frequencies(data)
    t_freq, t_full = predicted_cells(data)
    s_freq = np.sign(log_odds(t_freq, m, n))
    s_full = np.sign(log_odds(t_full, m, n))
    truth = data["truth"]
    s_emp = np.sign(truth[..., 1, 1] * truth[..., 0, 0] - truth[..., 0, 1] * truth[..., 1, 0])
    reversal = s_freq != s_full
    usable = reversal & (s_emp != 0)
    agree = usable & (s_emp == s_full)
    out = {}
    for cohort in COHORTS:
        mask = data["cohorts"] == cohort
        a, u = agree[mask].sum(axis=1), usable[mask].sum(axis=1)
        draws = np.random.default_rng(SEED).integers(0, mask.sum(), (DRAWS, mask.sum()))
        totals = u[draws].sum(axis=1)
        boot = a[draws].sum(axis=1) / np.where(totals > 0, totals, np.nan)
        out[cohort] = {
            "recipients": int(mask.sum()),
            "recipient_queries": int(mask.sum() * 81),
            "reversals": int(reversal[mask].sum()),
            "reversals_with_tied_heldout_direction": int((reversal & (s_emp == 0))[mask].sum()),
            "recipients_with_a_reversal": int((reversal[mask].sum(axis=1) > 0).sum()),
            "agreement_with_full_patterns": float(a.sum() / u.sum()),
            "agreeing": int(a.sum()),
            "usable_reversals": int(u.sum()),
            "bootstrap_95": np.nanquantile(boot, (0.025, 0.975)).tolist(),
            "bootstrap_familywise": np.nanquantile(boot, (FAMILY_TAIL, 1 - FAMILY_TAIL)).tolist(),
            "bootstrap_draws_without_reversals": int(np.isnan(boot).sum()),
        }
    return out, {"s_freq": s_freq, "s_full": s_full, "s_emp": s_emp}


def pair_gain(data):
    """Held-out pair deviance of frequency-only minus full-pattern prediction."""
    truth, predictions = data["truth"], data["predictions"]
    deviance = 2 * rel_entr(truth[:, None], predictions).sum(axis=(3, 4))
    return deviance[:, 0] - deviance[:, 3], deviance


def identified_summary(data, star):
    m, n = query_frequencies(data)
    t_freq, t_full = predicted_cells(data)
    low = np.minimum.reduce([star["star_low"], t_freq, t_full])
    high = np.maximum.reduce([star["star_high"], t_freq, t_full])
    theta_low, theta_high = log_odds(low, m, n), log_odds(high, m, n)
    risk = np.array([[2 * minimax(low[p, q], high[p, q], m[p, q], n[p, q])[1]
                      for q in range(81)] for p in range(len(low))])
    ambiguous = (theta_low < 0) & (theta_high > 0)
    cognate = np.array([i == j for i in range(9) for j in range(9)])
    gain, deviance = pair_gain(data)
    product = m * (1 - m) * n * (1 - n)
    star_low_share = float(np.mean(star["star_low"] <= np.minimum(t_freq, t_full)))
    star_high_share = float(np.mean(star["star_high"] >= np.maximum(t_freq, t_full)))
    out = {}
    for cohort in COHORTS:
        mask = data["cohorts"] == cohort
        width = (theta_high - theta_low)[mask]
        within_risk = [spearmanr(r, g).statistic for r, g in zip(risk[mask], gain[mask])]
        within_product = [spearmanr(r, g).statistic for r, g in zip(product[mask], gain[mask])]
        out[cohort] = {
            "recipients": int(mask.sum()),
            "S1_direction_unidentified_fraction": float(ambiguous[mask].mean()),
            "S1_direction_unidentified_count": int(ambiguous[mask].sum()),
            "S1_cognate_fraction": float(ambiguous[mask][:, cognate].mean()),
            "S1_recipients_all_queries_unidentified": int(ambiguous[mask].all(axis=1).sum()),
            "S2_log_odds_width_median": float(np.median(width)),
            "S2_log_odds_width_quartiles": np.quantile(width, (0.25, 0.75)).tolist(),
            "S2_log_odds_width_min": float(width.min()),
            "S2_minimax_deviance_lower_bound_mean": float(risk[mask].mean()),
            "S2_minimax_deviance_lower_bound_quartiles": np.quantile(risk[mask], (0.25, 0.75)).tolist(),
            "heldout_mean_deviance_frequency_only": float(deviance[mask, 0].mean()),
            "heldout_mean_deviance_full_patterns": float(deviance[mask, 3].mean()),
            "S4_mean_within_spearman_bound_vs_gain": float(np.mean(within_risk)),
            "S4_mean_within_spearman_frequency_product_vs_gain": float(np.mean(within_product)),
        }
    arrays = {
        "t_low": low, "t_high": high, "theta_low": theta_low, "theta_high": theta_high,
        "risk": risk, "ambiguous": ambiguous, "gain": gain, "m": m, "n": n,
        "theta_freq": log_odds(t_freq, m, n), "theta_full": log_odds(t_full, m, n),
    }
    extent = {"star_attains_lower_endpoint_fraction": star_low_share,
              "star_attains_upper_endpoint_fraction": star_high_share}
    return out, arrays, extent


def empirical_log_odds(truth, cells=256):
    counts = truth * cells + 0.5
    return np.log(counts[..., 1, 1] * counts[..., 0, 0] / (counts[..., 0, 1] * counts[..., 1, 0]))


def write_csv(data, arrays, signs):
    empirical = empirical_log_odds(data["truth"])
    with (RESULTS / "recipient_queries.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cohort", "recipient", "rna", "protein", "rna_frequency",
                         "protein_frequency", "t_low", "t_high", "log_odds_low",
                         "log_odds_high", "log_odds_frequency_only", "log_odds_full_patterns",
                         "heldout_direction", "heldout_log_odds_haldane",
                         "minimax_deviance_lower_bound", "heldout_pair_gain"])
        for p in range(len(data["cohorts"])):
            for q in range(81):
                writer.writerow([
                    data["cohorts"][p], p, MARKERS[q // 9], MARKERS[q % 9],
                    *(f"{v:.10g}" for v in (arrays["m"][p, q], arrays["n"][p, q],
                      arrays["t_low"][p, q], arrays["t_high"][p, q],
                      arrays["theta_low"][p, q], arrays["theta_high"][p, q],
                      arrays["theta_freq"][p, q], arrays["theta_full"][p, q])),
                    int(signs["s_emp"][p, q]), f"{empirical[p, q]:.10g}",
                    f"{arrays['risk'][p, q]:.10g}", f"{arrays['gain'][p, q]:.10g}"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--s3-only", action="store_true")
    args = parser.parse_args()
    data = load_recipients()
    s3, signs = reversal_summary(data)
    summary = {"plan_sha256": __import__("hashlib").sha256(
        (Path(__file__).resolve().parent / "PLAN.md").read_bytes()).hexdigest(),
        "S3_reversal_direction": s3}
    if not args.s3_only:
        star = np.load(RESULTS / "star_endpoints.npz")
        assert np.array_equal(star["cohorts"], data["cohorts"])
        s124, arrays, extent = identified_summary(data, star)
        summary["S1_S2_S4_identified_set"] = s124
        summary["endpoint_sources"] = extent
        summary["certification"] = json.loads((RESULTS / "certification.json").read_text())
        write_csv(data, arrays, signs)
        np.savez_compressed(RESULTS / "identified_summary_arrays.npz", **arrays,
                            cohorts=data["cohorts"])
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
