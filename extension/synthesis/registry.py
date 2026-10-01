#!/usr/bin/env python3
"""Registry of every prespecified hypothesis in the manuscript and of every analysis done after scoring.

    python registry.py --out ../../revision/source      # registry_table.tex (Supplementary Table 1)
                                                        # and ../HYPOTHESES.md, ../hypotheses.json

Every estimate and verdict is read from the result records of the tests (JSON files, or the macro files their
asset scripts generate from them); only the wording of the hypotheses is written here, taken from the frozen plans.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT = HERE.parent
SRC = EXT.parent / "revision" / "source"


def load(rel):
    return json.loads((EXT / rel).read_text())


def macros(name):
    """\\newcommand values of a generated macro file, with balanced braces."""
    t = (SRC / name).read_text()
    out, i = {}, 0
    while (j := t.find("\\newcommand{\\", i)) >= 0:
        k = t.index("}", j + 13)
        depth, m = 0, k + 1
        while True:
            depth += {"{": 1, "}": -1}.get(t[m], 0)
            if depth == 0:
                break
            m += 1
        out[t[j + 13:k]] = t[k + 2:m].replace("\\ensuremath{-}", "-").replace("{,}", ",")
        i = m
    return out


def stamp(value):
    """A freeze time, recorded as '2026-09-28 19:35:44 EDT' or '2026-09-27T17:26:25-0400', as 'YYYY-MM-DD HH:MM'
    (EDT, as recorded). Read by pattern: strptime's %Z accepts only the host's own time-zone names."""
    m = re.fullmatch(r"(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}):\d{2}(?: EDT|[+-]\d{2}:?\d{2})", value)
    if not m:
        raise ValueError(value)
    return f"{m.group(1)} {m.group(2)}"


def f(v, d=2):
    return f"{v:.{d}f}".replace("-", "−")


def ci(lo, hi, d=2):
    return f"{f(lo, d)} to {f(hi, d)}"


def pct(v, d=0):
    return f"{100 * v:.{d}f}%".replace("-", "−")


def verdict(b):
    return "met" if b else "not met"


def rows():
    R = []

    def add(test, frozen, hid, text, criterion, estimate, result, record):
        R.append({"test": test, "frozen": frozen, "id": hid, "hypothesis": text, "criterion": criterion,
                  "estimate": estimate, "result": result, "record": record})

    # external test
    rec = "semipaired/external/results/overcite_plan_estimator.json"
    o = load(rec)
    fz = stamp(load("semipaired/external/results/freeze.json")["frozen_at"])
    t = "External test (OverCITE-seq ORF screen)"
    for hid, key, txt, crit in (
            ("H1 (primary)", "paired_only", "Fewer paired cells than paired-only estimation at 30% recovery",
             "95% lower bound of the ratio > 1"),
            ("H2", "semicca", "Fewer paired cells than SemiCCA", "95% lower bound > 1"),
            ("H3", "champollion", "No more than 1.25 times the paired cells of Champollion",
             "95% lower bound > 0.8")):
        s = o["savings"][key]["0.3"]
        vk = {"paired_only": "H1_vs_paired_only", "semicca": "H2_vs_semicca",
              "champollion": "H3_noninferior_to_champollion"}[key]
        add(t, fz, hid, txt, crit, f"{f(s['factor'], 1)} ({ci(*s['ci'], 1)})", verdict(o["verdicts"][vk]), rec)
    # benchmark
    t = "Benchmark (nine data sets)"
    fz = stamp(load("generality/results/freeze.json")["frozen_at"])
    g = load("generality/results/summary.json")
    law = load("generality/results/law.json")
    for hid, key, txt, crit, met in (
            ("G1 (primary)", "G1_paired_only/0.5", "Geometric-mean saving over paired-only estimation at the plan's "
             "target (half of the unshrunk estimator's recovery with every training cell paired)",
             "95% lower bound > 1", g["verdicts"]["G1_met"]),
            ("G2", "G2_semicca/0.5", "The same against SemiCCA", "95% lower bound > 1", g["verdicts"]["G2_met"])):
        s = g[key]
        add(t, fz, hid, txt, crit, f"{f(s['geometric_mean'])} ({ci(*s['ci'])})", verdict(met),
            "generality/results/summary.json")
    s = g["G3_selftune_vs_fixed/0.5"]
    add(t, fz, "G3", "Self-tuning against the fixed estimator (secondary)", "two-sided; no threshold",
        f"{f(s['geometric_mean'])} ({ci(*s['ci'])})", "reported", "generality/results/summary.json")
    add(t, fz, "G4", "Predicted and observed savings rank the data sets alike (structural law)",
        "Spearman > 0, one-sided P < 0.05", f"rho {f(law['spearman'])}, P = {f(law['p_one_sided'])}",
        verdict(law["G4_met"]), "generality/results/law.json")
    dev = (EXT / "generality" / "DEVIATIONS.md").read_text()
    am = re.search(r"## Amendment 1 \(recorded (\d{2}:\d{2}):\d{2} EDT, (\d+) September 2026, "
                   r"after ([a-z, ]+?) had been scored", dev)
    when = f"amendment recorded 2026-09-{int(am.group(2)):02d} {am.group(1)}, after {am.group(3)} had been scored"
    ga = load("generality/results/summary_amend1.json")
    la = load("generality/results/law_amend1.json")
    n_inf = len(ga["informative"])
    for hid, key, txt, met in (
            ("G1 (amended)", "informative/G1_paired_only/0.5",
             f"As G1, at half of the best recovery of any method, over the {n_inf} data sets with recovered "
             f"dependence ({when})", ga["verdicts"]["G1_amended_met"]),
            ("G2 (amended)", "informative/G2_semicca/0.5", "As G2 under the amended rule",
             ga["verdicts"]["G2_amended_met"])):
        s = ga[key]
        add(t, fz, hid, txt, "95% lower bound > 1", f"{f(s['geometric_mean'])} ({ci(*s['ci'])})", verdict(met),
            "generality/results/summary_amend1.json")
    add(t, fz, "G4 (amended)", "As G4 under the amended rule", "Spearman > 0, one-sided P < 0.05",
        f"rho {f(la['spearman'])}, P = {f(la['p_one_sided'])}", verdict(la["G4_amended_met"]),
        "generality/results/law_amend1.json")
    # law of the saving
    t = "Prospective test of the law (ten data sets, 29 problems)"
    fz = stamp(load("predictability/results/freeze.json")["frozen_at"])
    p = load("predictability/results/summary.json")
    tp, v = p["targets"]["0.5"], p["verdicts"]

    def err(d):
        """A median error on the log2 scale as a percentage, with its interval."""
        return f"{pct(2 ** d['median_abs_log2'] - 1)} ({pct(2 ** d['ci'][0] - 1)} to {pct(2 ** d['ci'][1] - 1)})"

    add(t, fz, "H1 (primary)", f"The law is calibrated ({tp['estimable']} of {tp['problems']} problems with an "
        "estimable saving)", "median error < 25%", err(tp["law"]), verdict(v["H1_law_calibrated"]),
        "predictability/results/summary.json")
    add(t, fz, "H2", "The law ranks the problems", "Spearman > 0, one-sided P < 0.05",
        f"rho {f(tp['spearman_law']['rho'])}, P < 0.001" if tp["spearman_law"]["p_one_sided"] < 0.001
        else f"rho {f(tp['spearman_law']['rho'])}, P = {f(tp['spearman_law']['p_one_sided'], 3)}",
        verdict(v["H2_law_ranks"]), "predictability/results/summary.json")
    add(t, fz, "H3", "No saving in random bases", "median saving in [0.8, 1.25]",
        f"{f(tp['random_control']['median_observed'])}", verdict(v["H3_random_control_no_saving"]),
        "predictability/results/summary.json")
    add(t, fz, "H4", "A pilot of up to 200 paired cells predicts the saving", "median error < 50%",
        err(tp["pilot"]), verdict(v["H4_pilot_calibrated"]), "predictability/results/summary.json")
    add(t, fz, "H5", "Unpaired noise with the development dependence profile predicts it", "median error < 50%",
        err(tp["transfer"]), verdict(v["H5_transfer_calibrated"]), "predictability/results/summary.json")
    add(t, fz, "H6", "Every saving lies within the unpaired bounds", "all in [0.8, 1.25 / min pi]",
        f"{len(tp['bound_violations'])} outside", verdict(v["H6_within_unpaired_bounds"]),
        "predictability/results/summary.json")
    # bone-marrow test
    t = "Bone-marrow test (four sites)"
    d = load("deployment/results/summary.json")
    fz = stamp(load("deployment/results/freeze.json")["frozen"])
    q = d["targets"]["0.5"]
    for hid, key, txt, crit in (
            ("D1 (primary)", "D1_other_vs_paired_only", "Saving over paired-only estimation with unpaired cells "
             "of other sites", "95% lower bound > 1"),
            ("D2", "D2_assay_vs_paired_only", "The same with nuclear RNA of other sites", "95% lower bound > 1"),
            ("D3", "D3_other_over_same", "Other sites need no more paired cells than the own site",
             "95% upper bound < 1.25")):
        r = q[key]
        lb = "at least " if r["numerator_relation"] == ">" and r["denominator_relation"] == "=" else ""
        est = ("neither curve reached the target" if r["numerator_relation"] == r["denominator_relation"] == ">"
               else f"{lb}{f(r['ratio'])} ({ci(*r['ci'])})")
        add(t, fz, hid, txt, crit, est, verdict(d["hypotheses"][hid[:2]]), "deployment/results/summary.json")
    # validation
    t = "Validation test (blood study not analysed before; atlas of another study)"
    s = load("validation/results/summary.json")
    fz = stamp(load("validation/results/freeze.json")["frozen"])
    q, b = s["targets"]["0.5"], s["budgets"]
    add(t, fz, "V1 (primary)", "Saving over paired-only estimation with the atlas", "95% lower bound > 1",
        f"at least {f(q['V1_atlas_vs_paired_only']['ratio'])} ({ci(*q['V1_atlas_vs_paired_only']['ci'])})",
        verdict(s["hypotheses"]["V1"]), "validation/results/summary.json")
    rf, rc = s["rf"]["atlas"], s["rf_ci"]["atlas"]
    add(t, fz, "V2", "The atlas arm recovers dependence with 50 and with 100 paired cells", "95% lower bounds > 0",
        f"{pct(rf[b.index(50)])} ({pct(rc[b.index(50)][0], 1)} to {pct(rc[b.index(50)][1])}); "
        f"{pct(rf[b.index(100)])} ({pct(rc[b.index(100)][0])} to {pct(rc[b.index(100)][1])})",
        verdict(s["hypotheses"]["V2"]), "validation/results/summary.json")
    add(t, fz, "V3", "Saving with the study's other pools and donors", "95% lower bound > 1",
        f"at least {f(q['V3_other_vs_paired_only']['ratio'], 1)} ({ci(*q['V3_other_vs_paired_only']['ci'], 1)})",
        verdict(s["hypotheses"]["V3"]), "validation/results/summary.json")
    add(t, fz, "V4", "The atlas needs no more paired cells than the paired samples' own unpaired cells",
        "95% upper bound < 1.25", f"{f(q['V4_atlas_over_own']['ratio'])} ({ci(*q['V4_atlas_over_own']['ci'])})",
        verdict(s["hypotheses"]["V4"]), "validation/results/summary.json")
    # biological readout of the validation
    t = "Readout of the validation test (specified after V1-V4 were scored)"
    rd = load("validation/readout/results/readout.json")
    fz = stamp(load("validation/readout/results/freeze.json")["frozen"])
    for hid, txt in (("B1", "Direction of reproducible within-cell-type associations: atlas minus paired-only"),
                     ("B2", "Top-100 nominations that are reproducible associations: atlas minus paired-only"),
                     ("B3", "Recovered fraction of cognate RNA-protein correlations: atlas minus paired-only")):
        dd = rd["differences"][hid]
        est = "; ".join(f"{B}: {f(100 * r['difference'], 1)} points "
                        f"({f(100 * r['ci'][0], 1)} to {f(100 * r['ci'][1], 1)})" for B, r in dd.items())
        add(t, fz, hid, txt, "95% lower bound > 0 at " + ", ".join(dd) + " paired cells", est,
            verdict(rd["hypotheses"][hid]), "validation/readout/results/readout.json")
    # other methods given the same paired cells and atlas
    t = "Other methods given the validation's atlas (specified after V1-V4 and B1-B3 were scored)"
    rv = load("validation/posthoc_review/results/review.json")
    fz = stamp(load("validation/posthoc_review/results/freeze.json")["frozen"])
    alt = rv["alternatives"]
    for hid, key, txt in (("K1", "K1_semicca_over_atlas", "SemiCCA (best of three forms), given the estimator's paired "
                           "cells, atlas and tuning information, needs more paired cells than the estimator"),
                          ("K2", "K2_regression_over_atlas",
                           "Reference regression (ridge; best of six forms) likewise")):
        r = alt["ratios"][key]
        pre = "at least " if r["numerator_relation"] == ">" and r["denominator_relation"] != ">" else ""
        add(t, fz, hid, txt, "95% lower bound of the ratio of paired cells > 1",
            f"{pre}{f(r['ratio'])} ({ci(*r['ci'])})", verdict(alt[hid]),
            "validation/posthoc_review/results/review.json")
    # learning without pairs in blood
    cs = load("cross_study/results/summary.json")
    m = re.search(r"\((\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}) EDT, before any recipient prediction\)",
                  (EXT / "cross_study" / "results" / "plan_sha256.txt").read_text())
    fz = f"{m.group(1)} {m.group(2)}"
    for coh, name in (("hao", "Learning without pairs, blood: confirmatory cohort"),
                      ("colon", "Learning without pairs, colon cohort (analysed before)")):
        c = cs[coh]
        t = name
        add(t, fz, "H1 (primary)", "Beats a donor-aware permutation null", "P < 0.05",
            f"error {f(c['H1']['observed_mean_E'])} against {f(c['H1']['null_mean_E'])}; P = {f(c['H1']['p'], 3)}",
            verdict(c["H1"]["success"]), "cross_study/results/summary.json")
        add(t, fz, "H2 (primary)", "Recovers most of the paired operator's improvement over independence",
            "95% lower bound of the share > 0.5",
            f"{pct(c['H2']['mean_share'])} ({pct(c['H2']['bootstrap_95'][0])} to {pct(c['H2']['bootstrap_95'][1])})",
            verdict(c["H2"]["success"]), "cross_study/results/summary.json")
        h3 = c["H3"]["pf_means"]
        add(t, fz, "H3", "Recovers dependence within cell types", "95% upper bound of the error < 1",
            f"{f(h3['mean_E'])} ({ci(*h3['bootstrap_95'])})",
            verdict(c["H3"].get("success", h3["bootstrap_95"][1] < 1)),
            "cross_study/results/summary.json")
        h4 = c["H4"]
        add(t, fz, "H4", "The paired closed form beats four matched baselines", "Bonferroni intervals exclude zero",
            f"{sum(v['success'] for v in h4.values())} of {len(h4)} baselines",
            verdict(all(v["success"] for v in h4.values())), "cross_study/results/summary.json")
    # perturbation and replication
    pm, rm = macros("pert_macros.tex"), macros("rep_macros.tex")
    e = load("perturbation/results/evaluation.json")["verdict"]
    t = "Learning without pairs, melanoma screen"
    fz = stamp(load("perturbation/results/freeze.json")["frozen_at"])
    for hid, key, txt, crit, mk in (
            ("H1", "H1_recovery", "Recovery of within-condition dependence in held-out targets",
             "95% lower bound > 0", "PtRfPrimary"),
            ("H2a", "H2_beats_pseudo_populations", "Above the pseudo-population control",
             "difference, lower bound > 0", "PtRfDiffPseudo"),
            ("H2b", "H2_beats_derangements", "Above the derangement control", "difference, lower bound > 0",
             "PtRfDiffDerange"),
            ("H3", "H3_nontargeting", "Recovery in non-targeting cells", "95% lower bound > 0", "PtNtPrimary"),
            ("H4", "H4_most_of_paired", "Most of what paired cells recover", "share, lower bound > 50%", "PtRfShare")):
        add(t, fz, hid, txt, crit, f"{pm[mk]}% ({pm[mk + 'Lo']}% to {pm[mk + 'Hi']}%)".replace("-", "−"),
            verdict(e[key]), "perturbation/results/evaluation.json")
    r = load("perturbation_replication/results/evaluation.json")["verdict"]
    t = "Learning without pairs, monocyte screen (replication)"
    fz = stamp(load("perturbation_replication/results/freeze.json")["frozen_at"])
    for hid, key, txt, crit, mk in (
            ("R1", "R1_recovery", "Recovery over all test groups", "95% lower bound > 0", "RpPrimary"),
            ("R2a", "R2_beats_pseudo_populations", "Above the pseudo-population control", "lower bound > 0",
             "RpDiffPseudo"),
            ("R2b", "R2_beats_derangements", "Above the derangement control", "lower bound > 0", "RpDiffDerange"),
            ("R3", "R3_ifn_share_above_half", "Interferon block: at least half of the paired closed form",
             "lower bound > 50%", "RpIfnShare"),
            ("R4", "R4_nontargeting", "Recovery in non-targeting cells", "lower bound > 0", "RpNtPrimary"),
            ("R5", "R5_exposure_targets_matter", "Removing the most exposed targets costs more than random ones",
             "lower bound > 0", "RpRemDiff")):
        add(t, fz, hid, txt, crit, f"{rm[mk]}% ({rm[mk + 'Lo']}% to {rm[mk + 'Hi']}%)".replace("-", "−"),
            verdict(r[key]), "perturbation_replication/results/evaluation.json")
    return R


POST_HOC = [
    ("Development comparisons and ablations (melanoma and monocyte screens)", "Supplementary Note on the estimator",
     "semipaired/"),
    ("Benchmark amendments: primary target half of the best recovery of any method; data sets without recovered "
     "dependence set aside (recorded after three and four of nine data sets were scored)", "Methods; savings table",
     "generality/DEVIATIONS.md, results/*_amend1.json"),
    ("Summary across test data sets (geometric mean 3.1)", "Results; savings table", "synthesis/savings_summary.py"),
    ("Development of the law on the benchmark data sets", "Supplementary Note on the law", "predictability/"),
    ("External test: the frozen scoring script named the one-sided estimator (deviation; both arms met H1-H3)",
     "Methods; Supplementary Note on the estimator", "semipaired/external/DEVIATIONS.md"),
    ("Bone-marrow test: centring by contrasts; RNA from nuclei (maps, axes, the law along each pool's axes, "
     "principal angles); population sizes; calibration of the noise estimate; paired-only control under both centrings",
     "Results; Supplementary Note on the bone-marrow test", "deployment/posthoc_*.py"),
    ("Fly connectome: unpaired neurons of other animals with contrasts", "Results; Supplementary Note on the "
     "bone-marrow test", "deployment/posthoc_fly_contrasts.py"),
    ("Validation test: budget axis counted in contrasts; results per donor and pool; leaving out each pool; atlas "
     "size; skewed or missing cell types in the atlas, and the skewed atlases scored without condition maps",
     "Results; Supplementary Note on the validation test",
     "synthesis/make_assets.py; validation/posthoc_robustness.py; validation/posthoc_composition_map.py"),
    ("Validation test: sample, donor and pool allocation of every fold, checked from sample labels (every fold "
     "donor-disjoint)", "Methods; Supplementary Note on the validation test", "validation/posthoc_allocation.py"),
    ("Validation test: Champollion given the same paired cells and atlas, on the first draw of every fold and budget "
     "(specified with K1-K2, descriptive)", "Results; Supplementary Note on the validation test",
     "validation/posthoc_review/"),
    ("Readout of the validation test, calibrated: significance thresholds, false-discovery-rate truth sets, "
     "majority-sign baselines, weighting of samples and cell types, bootstrap P-values, simultaneous intervals and "
     "Holm adjustment (specified with K1-K2)", "Results; Supplementary Note on the validation test",
     "validation/posthoc_review/"),
    ("Blood test of learning without pairs: dependence split into parts between and within cell types; "
     "identification analyses", "Supplementary Note on learning without pairs in blood", "cross_study/"),
    ("Melanoma screen: analyses of the interferon response and of removing perturbations (specified before "
     "computation, after the scoring cells were opened)", "Results; Supplementary Note on the perturbation screen",
     "perturbation/PLAN_programs.md"),
    ("Monocyte screen: a penalty fixed after unsealing", "Supplementary Note on the replication",
     "perturbation_replication/results/posthoc_penalty.json"),
    ("Check that every recovery curve averages the scores of single draws (every test rerun with its frozen code)",
     "Methods; Supplementary Note on revision", "checks/draw_scoring.py"),
    ("Analyses added in revision, in one plan hashed before any was computed: calibration of the noise estimate "
     "after contrasts; the atlas's axes and condition maps at every budget; maps reweighted to the paired cells' mix "
     "of cell types; one draw of paired cells and resampled atlas patients; named within-type relationships; "
     "calibration of the association test; finer cell states and technical covariates; the law with data sets as "
     "units; a pilot that sets the total budget; savings at fixed accuracy",
     "Results; Supplementary Note on revision", "second_review/PLAN.md"),
    ("Analyses added in revision, running: the blood COVID-19 data set read through the benchmark's reader for "
     "string arrays (its deviation 1); the atlas bootstrap restarted unchanged after the memory limit stopped its "
     "first start (deviations; no script, setting or value changed)", "Supplementary Note on revision",
     "second_review/DEVIATIONS.md"),
]


def tex(s):
    s = s.replace("−", "\\ensuremath{-}").replace("%", "\\%").replace("_", "\\_").replace("<", "$<$") \
        .replace(">", "$>$").replace("rho ", "$\\rho$ ").replace("pi]", "$\\pi$]")
    return s


# The manuscript's table states what was registered and when in words; the time stamps, file names and working
# details stay in HYPOTHESES.md and hypotheses.json, which keep the records' own wording.
PLAIN = {
    "External test: the frozen scoring script named the one-sided estimator (deviation; both arms met H1-H3)":
        "External test: the registered scoring code scored the one-sided variant of the estimator, not the two-sided "
        "variant named in the plan (deviation; both variants met H1-H3)",
    "Check that every recovery curve averages the scores of single draws (every test rerun with its frozen code)":
        "Check that every recovery curve averages the scores of single draws (every test rerun with its registered "
        "code)",
    "Monocyte screen: a penalty fixed after unsealing":
        "Monocyte screen: a penalty fixed after the scoring files had been opened",
    "Analyses added in revision, running: the blood COVID-19 data set read through the benchmark's reader for "
    "string arrays (its deviation 1); the atlas bootstrap restarted unchanged after the memory limit stopped its "
    "first start (deviations; no script, setting or value changed)":
        "Follow-up analyses: two runs departed from the planned order of execution (deviations; no code, setting or "
        "value changed)",
}


def plain(s):
    s = PLAIN.get(s, s)
    s = re.sub(r"\(amendment recorded [^)]*\)", "(amended after three data sets had been scored)", s)
    s = s.replace("Analyses added in revision, in one plan hashed before any was computed:",
                  "Follow-up analyses, registered together before any was computed:")
    return s.replace("Supplementary Note on revision", "Supplementary Note on follow-up analyses")


COLUMN = ">{\\raggedright\\arraybackslash}p{%g\\textwidth}"


def write_tex(R, out):
    lines = ["% Generated by extension/synthesis/registry.py. Do not edit.",
             "\\begin{longtable}{@{}" + "".join(COLUMN % w for w in (0.12, 0.32, 0.16, 0.22, 0.07)) + "@{}}",
             "\\caption{\\RegistryCaption}\\label{tab:registry}\\\\",
             "\\toprule", "Hypothesis & Statement & Criterion & Estimate (95\\% CI) & Result \\\\", "\\midrule",
             "\\endfirsthead",
             "\\multicolumn{5}{@{}l}{\\textit{\\tablename~\\thetable, continued}} \\\\",
             "\\toprule", "Hypothesis & Statement & Criterion & Estimate (95\\% CI) & Result \\\\", "\\midrule",
             "\\endhead", "\\bottomrule", "\\endfoot"]
    last = None
    for r in R:
        if r["test"] != last:
            lines.append("\\multicolumn{5}{@{}l}{\\textit{" + tex(r["test"]) + "}} \\\\")
            last = r["test"]
        lines.append(" & ".join([tex(r["id"]), tex(plain(r["hypothesis"])), tex(r["criterion"]), tex(r["estimate"]),
                                 tex(r["result"])]) + " \\\\")
    lines += ["\\midrule", "\\multicolumn{5}{@{}l}{\\textit{Analyses after scoring, amendments and deviations}} \\\\"]
    for what, where, code in POST_HOC:
        lines.append("\\multicolumn{3}{@{}" + COLUMN % 0.63 + "}{" + tex(plain(what)) + "} & "
                     "\\multicolumn{2}{" + COLUMN % 0.30 + "@{}}{" + tex(plain(where)) + "} \\\\")
    lines.append("\\end{longtable}")
    (out / "registry_table.tex").write_text("\n".join(lines) + "\n")


def write_md(R):
    lines = ["# Registry of hypotheses and analyses after scoring", "",
             "Generated by `synthesis/registry.py` from the result records; do not edit. Every prespecified hypothesis "
             "of the manuscript, its criterion, estimate and result, and every analysis done after scoring.", ""]
    last = None
    for r in R:
        if r["test"] != last:
            lines += ["", f"## {r['test']}", "", f"Frozen {r['frozen']} (EDT). Record: `{r['record']}`.", "",
                      "| Hypothesis | Statement | Criterion | Estimate (95% CI) | Result |", "|---|---|---|---|---|"]
            last = r["test"]
        lines.append(f"| {r['id']} | {r['hypothesis']} | {r['criterion']} | {r['estimate']} | {r['result']} |")
    lines += ["", "## Analyses after scoring, amendments and deviations", "",
              "| Analysis | Reported in | Code or record |", "|---|---|---|"]
    lines += [f"| {a} | {b} | `{c}` |" for a, b, c in POST_HOC]
    (EXT / "HYPOTHESES.md").write_text("\n".join(lines) + "\n")
    (EXT / "hypotheses.json").write_text(json.dumps({"prespecified": R, "after_scoring": [
        {"analysis": a, "reported_in": b, "code": c} for a, b, c in POST_HOC]}, indent=1, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(SRC))
    a = ap.parse_args()
    R = rows()
    write_tex(R, Path(a.out))
    write_md(R)
    print(len(R), "hypotheses;", sum(r["result"] == "met" for r in R), "met,",
          sum(r["result"] == "not met" for r in R), "not met,", sum(r["result"] == "reported" for r in R), "reported")


if __name__ == "__main__":
    main()
