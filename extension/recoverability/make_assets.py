#!/usr/bin/env python3
"""Manuscript assets of the recoverability study: macros, Fig. 5, a supplementary table and RESULTS.md.

Reads the development results (dev_*.json; exploratory) and the sealed bone-marrow
evaluation (bmmc_*.json). Per-score AUC intervals in Fig. 5c,d are computed here
with the donor bootstrap of evaluate_bmmc.py and are descriptive; the prespecified
comparisons are the differences in bmmc_evaluation.json.

    python make_assets.py --out ../../revision/source
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from stats import auc, cluster_bootstrap, logistic_score

HERE = Path(__file__).resolve().parent
R = HERE / "results"
SANS = "font=\\sffamily"
COMMON = (" axis x line*=bottom,axis y line*=left,tick align=outside,y tick style={draw=none},"
          "xmajorgrids=true,grid style={gray!18},axis line style={gray!55},"
          "tick label style={font=\\sffamily\\scriptsize},label style={font=\\sffamily\\small},clip=false,"
          "title style={font=\\sffamily\\small,at={(0,1)},anchor=base west,yshift=5pt}")
ANALYSES = ("total", "within_coarse", "within_fine")
BUDGETS = (50, 100, 200, 400, 800)
WORD = {50: "Fifty", 100: "Hundred", 200: "TwoHundred", 400: "FourHundred", 800: "EightHundred"}


def load(name):
    return json.loads((R / name).read_text())


def signed(text):
    return "\\ensuremath{-}" + text[1:] if text.startswith("-") else text


def f2(v):
    return signed(f"{v:.2f}")


def f3(v):
    return signed(f"{v:.3f}")


# ----------------------------------------------------------------------------- development numbers
def dev_macros():
    m = {}
    comp = load("dev_compatibility.json")
    tot = [r for r in comp if r["analysis"] == "total"]
    m["RcDevFReported"] = f2(np.median([r["frozen"]["F_reported"] for r in tot]))
    m["RcDevFClr"] = f2(np.median([r["frozen"]["F_clr"] for r in tot]))
    m["RcDevFRaw"] = f2(np.median([r["frozen"]["F_raw"] for r in tot]))
    boot = load("dev_bootstrap.json")["rows"]
    fz = [r for r in boot if r["channel"] == "frozen"]
    p1 = [r for r in boot if r["channel"] == "P1"]
    ftot = [r for r in fz if r["tag"].endswith("/total")]
    hao_tot = [r for r in ftot if r["tag"].startswith("hao")]
    col_tot = [r for r in ftot if r["tag"].startswith("colon")]
    m["RcDevTotalAnalyses"] = str(len(ftot))
    m["RcDevFMeasTotal"] = f2(np.median([r["F_measured"] for r in ftot]))
    m["RcDevFLatTotal"] = f2(np.median([r["F_signal"] for r in ftot]))
    m["RcDevFMeasAbove"] = str(sum(r["F_measured"] > 1 for r in ftot))
    m["RcDevFMeasLowerAbove"] = str(sum(r["F_measured_basic_lower"] > 1 for r in ftot))
    m["RcDevFLatBloodMin"] = f2(min(r["F_signal"] for r in hao_tot))
    m["RcDevFLatBloodMax"] = f2(max(r["F_signal"] for r in hao_tot))
    m["RcDevFLatBloodAbove"] = str(sum(r["F_signal"] > 1 for r in hao_tot))
    m["RcDevFLatBloodLowerAbove"] = str(sum(r["F_signal_basic_lower"] > 1 for r in hao_tot))
    m["RcDevFLatColonMax"] = f2(max(r["F_signal"] for r in col_tot))
    within = [r for r in fz if "/within" in r["tag"]]
    m["RcDevWithinAnalyses"] = str(len(within))
    m["RcDevFMeasWithin"] = f2(np.median([r["F_measured"] for r in within]))
    m["RcDevFLatWithin"] = f2(np.median([r["F_signal"] for r in within]))
    m["RcDevFLatWithinLowerAbove"] = str(sum(r["F_signal_basic_lower"] > 1 for r in within))
    m["RcDevPOneAnalyses"] = str(len(p1))
    m["RcDevPOneFLat"] = f2(np.median([r["F_signal"] for r in p1]))
    m["RcDevPOneLowerAbove"] = str(sum(r["F_signal_basic_lower"] > 1 for r in p1))
    m["RcDevDeparture"] = f2(np.median([r["departure_closed_signal"] for r in fz]))
    m["RcDevDepartureMin"] = f2(min(r["departure_closed_signal"] for r in fz))
    m["RcDevDepartureMax"] = f2(max(r["departure_closed_signal"] for r in fz))
    bounds = load("dev_bounds.json")
    nonempty = [v for v in bounds.values() if v is not None]
    m["RcDevBoundsNonempty"] = str(len(nonempty))
    m["RcDevBoundsAnalyses"] = str(len(bounds))
    m["RcDevBoundsSigned"] = str(sum(v["sign_determined"] > 0 for v in nonempty))
    m["RcDevBoundsInsideMin"] = f3(min(v["inside"] for v in nonempty))
    units = {u["tag"]: u for u in load("dev_units2.json")["units"]}
    pairs = [r for r in load("dev_units2.json")["pairs"] if r["channel"] == "frozen"]
    fr = {r["tag"]: r for r in pairs}
    for cohort, key in (("hao", "Blood"), ("colon", "Colon")):
        tags = [t for t in units if t.startswith(cohort) and t.endswith("/total")]
        lab = [units[t]["E_label_between"] for t in tags]
        m[f"RcDevLabel{key}"] = f2(np.mean(lab))
        m[f"RcDevLabel{key}Min"] = f2(min(lab))
        m[f"RcDevLabel{key}Max"] = f2(max(lab))
        m[f"RcDevPf{key}"] = f2(np.mean([fr[t]["E_signal"] for t in tags]))
        m[f"RcDevPaired{key}"] = f2(np.mean([units[t]["E_paired_measured"] for t in tags]))
    tags = [t for t in units if t.endswith("/total")]
    m["RcDevLabelBeatsPaired"] = str(sum(units[t]["E_label_between"] < units[t]["E_paired_measured"] for t in tags))
    m["RcDevLabelUnits"] = str(len(tags))
    dev_pairs = [r for r in load("dev_units2.json")["pairs"] if r["dim_S"] > 0]
    m["RcDevPairs"] = f"{len(dev_pairs):,}"
    return m


# ----------------------------------------------------------------------------- bone marrow
def scored_rows():
    """Non-abstaining analysis x channel pairs with errors and model scores (as evaluate_bmmc.py)."""
    diag = load("bmmc_diagnostics.json")
    errors = load("bmmc_errors.json")
    model = load("failure_model.json")["models"]
    rows = []
    for r in diag:
        r = dict(r, E_signal=errors[f'{r["tag"]}/{r["channel"]}/signal'])
        if r["dim_S"] > 0 and r["type_agreement"] is not None:
            rows.append(r)

    def score(mdl):
        X = np.array([[np.log(r["F_signal"] + 1e-3) if c == "log_F_signal" else r[c] for c in mdl["predictors"]]
                      for r in rows], float)
        return logistic_score((mdl["intercept"], np.array(mdl["slopes"]), np.array(mdl["mean"]),
                               np.array(mdl["sd"])), X)

    scores = {"Two failure modes": score(model["primary"]), "Label-free": score(model["label_free"]),
              "Coverage": -np.array([r["coverage_signal"] for r in rows]),
              "Composition": -np.array([r["between_share"] for r in rows]),
              "Covariance shift": np.array([r["shift"] for r in rows]),
              "Training size": -np.log([r["n_train"] for r in rows]),
              "Cells": -np.log([r["cells"] for r in rows]),
              "Compatibility": np.array([r["F_signal"] for r in rows]),
              "Type disagreement": -np.array([r["type_agreement"] for r in rows])}
    return rows, scores


def auc_intervals(rows, scores, subset=None, names=None, seed=7):
    y = np.array([r["E_signal"] >= 1 for r in rows])
    clusters = np.array([r["donor"] for r in rows])
    idx = np.arange(len(rows)) if subset is None else np.flatnonzero(subset)
    out = {}
    for i, name in enumerate(names or scores):
        v = scores[name][idx]
        yy, cc = y[idx], clusters[idx]
        b = cluster_bootstrap(cc, lambda j: auc(yy[j], v[j]), 2000, seed + i)
        b = b[np.isfinite(b)]
        out[name] = (auc(yy, v), float(np.quantile(b, .025)), float(np.quantile(b, .975)))
    return out


def equivalent_budget(curve_random, target):
    """Budget at which random selection reaches `target`, by log-linear interpolation of its mean curve."""
    for (n0, e0), (n1, e1) in zip(zip(BUDGETS, curve_random), zip(BUDGETS[1:], curve_random[1:])):
        if e0 >= target >= e1:
            return float(n0 * 2 ** (np.log2(n1 / n0) * (e0 - target) / (e0 - e1)))
    return None


def bmmc_macros(ev, rows, scores):
    m = {}
    seal = load("bmmc_seal.json")
    units = load("bmmc_units.json")
    m["RcBmSamples"] = str(len(units))
    m["RcBmDonors"] = str(len({u["donor"] for u in units.values()}))
    m["RcBmCells"] = f"{sum(sum(u['cells']) for u in units.values()):,}"
    m["RcBmGenes"] = str(len(seal["genes"]))
    m["RcBmProteins"] = str(len(seal["proteins"]))
    m["RcBmFineMin"] = str(min(len(u["types"]["fine"]) for u in units.values()))
    m["RcBmFineMax"] = str(max(len(u["types"]["fine"]) for u in units.values()))
    est = ev["estimators"]
    key = {"total": "Tot", "within_coarse": "Coarse", "within_fine": "Fine"}
    for an in ANALYSES:
        k = key[an]
        m[f"RcBmFrozen{k}"] = f2(est[an]["frozen/signal"]["mean"])
        m[f"RcBmFrozenMeas{k}"] = f2(est[an]["frozen/measured"]["mean"])
        m[f"RcBmFrozenBelow{k}"] = str(est[an]["frozen/signal"]["below_1"])
        m[f"RcBmPOne{k}"] = f2(est[an]["P1/signal"]["mean"])
        m[f"RcBmColon{k}"] = f2(est[an]["colon/signal"]["mean"])
        m[f"RcBmColonBelow{k}"] = str(est[an]["colon/signal"]["below_1"])
        m[f"RcBmBench{k}"] = f2(est[an]["benchmark"]["mean"])
    m["RcBmLabelTot"] = f2(est["total"]["label_between"]["mean"])
    comp = ev["compatibility"]
    m["RcBmFMeasTot"] = f2(comp["frozen/total/F_measured"]["median"])
    m["RcBmFLatTot"] = f2(comp["frozen/total/F_signal"]["median"])
    m["RcBmFLatMax"] = f2(max(comp[f"frozen/{an}/F_signal"]["range"][1] for an in ANALYSES))
    m["RcBmFMeasMin"] = f2(min(comp[f"frozen/{an}/F_measured"]["range"][0] for an in ANALYSES))
    m["RcBmFMeasLowerAbove"] = str(sum(comp[f"frozen/{an}/F_measured"]["lower_bound_above_1"] for an in ANALYSES))
    m["RcBmColonFLatMax"] = f2(max(comp[f"colon/{an}/F_signal"]["range"][1] for an in ANALYSES))
    m["RcBmPOneFLatMin"] = f2(min(comp[f"P1/{an}/F_signal"]["range"][0] for an in ANALYSES))
    m["RcBmPOneFLatMax"] = f2(max(comp[f"P1/{an}/F_signal"]["range"][1] for an in ANALYSES))
    m["RcBmPOneLowerAbove"] = str(sum(comp[f"P1/{an}/F_signal"]["lower_bound_above_1"] for an in ANALYSES))
    m["RcBmAnalyses"] = str(3 * len(units))
    f = ev["failure"]
    m["RcBmPairs"] = f"{f['pairs']:,}"
    m["RcBmFailures"] = str(f["failures"])
    m["RcBmPermAbstain"] = f"{100 * f['abstention']['perm']:.0f}"
    m["RcBmLcAbstain"] = f"{100 * f['abstention']['lc']:.1f}"
    names = {"primary": "Primary", "label_free": "LabelFree", "coverage": "Coverage", "composition": "Composition",
             "cells": "Cells", "n_train": "NTrain", "shift": "Shift", "compatibility": "Compat",
             "type_disagreement": "TypeDis"}
    for k, v in f["auc"].items():
        m[f"RcBmAuc{names[k]}"] = f2(v)
    for k, v in f["comparisons"].items():
        tag = "".join(names.get(x, x.capitalize()) for x in k.replace("_vs_", "|").split("|"))
        m[f"RcBmDiff{tag}"] = f3(v["difference"])
        m[f"RcBmDiff{tag}Lo"] = f3(v["ci"][0])
        m[f"RcBmDiff{tag}Hi"] = f3(v["ci"][1])
    hc = f["high_coverage"]
    m["RcBmHiPairs"] = str(hc["pairs"])
    m["RcBmHiFailures"] = str(hc["failures"])
    m["RcBmHiPOne"] = str(hc["families"]["P1"])
    m["RcBmHiFailuresPOne"] = str(sum(1 for r in rows if r["coverage_signal"] >= 0.4 and r["E_signal"] >= 1
                                      and r["family"] == "P1"))
    for k, tag in (("primary", "Primary"), ("type_disagreement", "TypeDis"), ("compatibility", "Compat"),
                   ("label_free", "LabelFree")):
        m[f"RcBmHiAuc{tag}"] = f2(hc[k]["auc"])
        m[f"RcBmHiAuc{tag}Lo"] = f2(hc[k]["ci"][0])
        m[f"RcBmHiAuc{tag}Hi"] = f2(hc[k]["ci"][1])
    for k, tag in (("primary", "Primary"), ("coverage", "Coverage"), ("shift", "Shift")):
        m[f"RcBmRank{tag}"] = f2(f["channel_ranking"][k]["concordance"])
    ref = ev["reference"]
    s = ref["summary"]
    for N in BUDGETS:
        for st in ("random", "balanced", "guided"):
            m[f"RcBmRef{st.capitalize()}{WORD[N]}"] = f2(s[str(N)][st]["within"])
        m[f"RcBmRefHybrid{WORD[N]}"] = f2(s[str(N)]["guided"]["hybrid_total"])
        m[f"RcBmRefPaired{WORD[N]}"] = f2(s[str(N)]["paired_only_total"])
    m["RcBmRefBetweenOnly"] = f2(ref["between_only_total"])
    t = ref["tests"]
    m["RcBmRefGainRandom"] = f3(-t["guided_vs_random"]["mean_difference"])
    m["RcBmRefGainRandomLo"] = f3(-t["guided_vs_random"]["ci"][1])
    m["RcBmRefGainRandomHi"] = f3(-t["guided_vs_random"]["ci"][0])
    m["RcBmRefGainBalanced"] = f3(-t["guided_vs_balanced"]["mean_difference"])
    m["RcBmRefGainBalancedLo"] = f3(-t["guided_vs_balanced"]["ci"][1])
    m["RcBmRefGainBalancedHi"] = f3(-t["guided_vs_balanced"]["ci"][0])
    lower_all = all(s[str(N)]["guided_minus_random"]["samples_lower"] == len(units) for N in BUDGETS)
    m["RcBmRefAllSamples"] = "all" if lower_all else "not all"
    rand = [s[str(N)]["random"]["within"] for N in BUDGETS]
    for N in (200, 800):
        eq = equivalent_budget(rand, s[str(N)]["guided"]["within"])
        m[f"RcBmRefEquiv{WORD[N]}"] = f"{eq:.0f}" if eq is not None else "above 800"
    b = ev["bounds"]
    m["RcBmBoundsSigned"] = str(sum(v["sign_determined"] > 0 for v in b.values()))
    m["RcBmBoundsAnalyses"] = str(len(b))
    m["RcBmBoundsHalfwidth"] = f2(np.mean([b[k]["mean_halfwidth"] for k in b if k.endswith("/total")]))
    m["RcBmBoundsTarget"] = f2(np.mean([b[k]["mean_abs_target"] for k in b if k.endswith("/total")]))
    v = ev["verdict"]
    m["RcBmHOne"] = "met" if v["H1_primary_beats_coverage"] else "not met"
    m["RcBmHThree"] = "met" if v["H3_high_coverage_failures_recognized"] else "not met"
    m["RcBmHSix"] = "met" if (v["H6_guided_beats_random"] and v["H6_guided_beats_balanced"]) else "not met"
    return m


# ----------------------------------------------------------------------------- post hoc analyses (after unsealing)
def posthoc_macros():
    m = {}
    ref = load("posthoc_reference.json")
    m["RcPhMeansOnly"] = f2(ref["type_means_only"]["mean"])
    m["RcPhMeansOnlyLo"] = f2(ref["type_means_only"]["ci"][0])
    m["RcPhMeansOnlyHi"] = f2(ref["type_means_only"]["ci"][1])
    for N in BUDGETS:
        t = ref["total"][str(N)]
        for st in ("random", "guided"):
            d = t[st]["minus_type_means_only"]
            tag = f"RcPhTot{WORD[N]}{st.capitalize()}"
            m[tag] = f2(t[st]["mean"])
            m[f"{tag}Diff"] = f3(d["mean"])
            m[f"{tag}DiffLo"] = f3(d["ci"][0])
            m[f"{tag}DiffHi"] = f3(d["ci"][1])
        m[f"RcPhTot{WORD[N]}Alone"] = f2(t["paired_only"]["mean"])
        g = t["guided_minus_random"]
        m[f"RcPhTot{WORD[N]}GuidedMinusRandom"] = f3(g["mean"])
        m[f"RcPhTot{WORD[N]}GuidedMinusRandomLo"] = f3(g["ci"][0])
        m[f"RcPhTot{WORD[N]}GuidedMinusRandomHi"] = f3(g["ci"][1])
    d = ref["paired_only_800_minus_type_means_only"]
    m["RcPhAloneEightHundredMinusMeans"] = f3(d["mean"])
    m["RcPhAloneEightHundredMinusMeansLo"] = f3(d["ci"][0])
    m["RcPhAloneEightHundredMinusMeansHi"] = f3(d["ci"][1])
    for N in BUDGETS[:-1]:
        w = ref["within_equivalent"][str(N)]
        m[f"RcPhSaving{WORD[N]}"] = f"{100 * w['saving']:.0f}"
        m[f"RcPhSaving{WORD[N]}Lo"] = f"{100 * w['saving_ci'][0]:.0f}"
        m[f"RcPhSaving{WORD[N]}Hi"] = f"{100 * w['saving_ci'][1]:.0f}"
        m[f"RcPhRandomCells{WORD[N]}"] = f"{w['random_cells_needed']:.0f}"
        m[f"RcPhRandomCells{WORD[N]}Lo"] = f"{w['random_cells_needed_ci'][0]:.0f}"
        m[f"RcPhRandomCells{WORD[N]}Hi"] = f"{w['random_cells_needed_ci'][1]:.0f}"
    dg = load("posthoc_diagnostic.json")
    for sec, tag in (("all_pairs", "All"), ("high_coverage", "Hi")):
        for k, t in (("identity_family_analysis", "Id"), ("identity_in_sample", "IdIn"), ("identity_family", "IdFam"),
                     ("primary_minus_identity_family_analysis", "IdDiff"), ("primary", "Primary")):
            v = dg[sec][k]
            fmt = f3 if "minus" in k else f2
            m[f"RcPh{t}{tag}"] = fmt(v["value"])
            m[f"RcPh{t}{tag}Lo"] = fmt(v["ci"][0])
            m[f"RcPh{t}{tag}Hi"] = fmt(v["ci"][1])
    lofo = dg["leave_family_out"]["P1"]
    for k, t in (("high_coverage", "Hi"), ("all_pairs", "All"), ("held_out_family", "Family")):
        m[f"RcPhLofoPOne{t}"] = f2(lofo[k]["value"])
        m[f"RcPhLofoPOne{t}Lo"] = f2(lofo[k]["ci"][0])
        m[f"RcPhLofoPOne{t}Hi"] = f2(lofo[k]["ci"][1])
    pw = dg["within_family"]["P1_within_high_coverage"]
    m["RcPhPOneWithinPairs"] = str(pw["pairs"])
    m["RcPhPOneWithinFailures"] = str(pw["failures"])
    for k, t in (("primary", "Primary"), ("coverage", "Coverage")):
        m[f"RcPhPOneWithin{t}"] = f2(pw[k]["value"])
        m[f"RcPhPOneWithin{t}Lo"] = f2(pw[k]["ci"][0])
        m[f"RcPhPOneWithin{t}Hi"] = f2(pw[k]["ci"][1])
    for fam, t in (("lc", "Lc"), ("frozen", "Frozen"), ("P1", "POne"), ("colon", "Colon")):
        for k, u in (("primary", "Primary"), ("coverage", "Coverage")):
            m[f"RcPhFam{t}{u}"] = f2(dg["within_family"][fam][k]["value"])
    lab = load("posthoc_labels.json")
    sm = lab["summary"]
    m["RcPhLabRna"] = f"{100 * sm['rna_accuracy']:.0f}"
    m["RcPhLabProtein"] = f"{100 * sm['protein_accuracy']:.0f}"
    m["RcPhLabAgree"] = f"{100 * sm['assays_agree']:.0f}"
    m["RcPhLabErr"] = f2(sm["type_means_error_assay_labels"])
    m["RcPhLabErrJoint"] = f2(sm["type_means_error_joint_labels"])
    m["RcPhLabDiff"] = f3(sm["assay_minus_joint"]["mean"])
    m["RcPhLabDiffLo"] = f3(sm["assay_minus_joint"]["ci"][0])
    m["RcPhLabDiffHi"] = f3(sm["assay_minus_joint"]["ci"][1])
    m["RcPhLabBetter"] = str(sm["frozen_minus_assay_label_type_means"]["samples_type_means_better"])
    fl = lab["failure"]
    m["RcPhLabAucAll"] = f2(fl["all_pairs"]["assay_labels"]["value"])
    m["RcPhLabAucHi"] = f2(fl["high_coverage"]["assay_labels"]["value"])
    m["RcPhLabAucDiff"] = f3(fl["all_pairs"]["difference"]["value"])
    sims = {"smoothed": load("noise_simulation.json")["rows"], "gene": load("noise_simulation_gene.json")["rows"]}
    rows = sims["smoothed"] + sims["gene"]
    settings = sorted({(r["depth"], r["analysis"]) for r in rows})

    def med(rs, f):
        return float(np.median([f(r) for r in rs]))

    per = [[r for r in sims[t] if (r["depth"], r["analysis"]) == st] for t in sims for st in settings]
    m["RcSimRows"] = str(len(rows))
    m["RcSimNuBiasMax"] = f2(max(abs(med(rs, lambda r: r["nu_bias_mean"])) for rs in per))
    m["RcSimNuMaeMax"] = f2(max(med(rs, lambda r: r["nu_abs_error_mean"]) for rs in per))
    m["RcSimNuMaeMin"] = f2(min(med(rs, lambda r: r["nu_abs_error_mean"]) for rs in per))
    m["RcSimCouplingMax"] = f"{100 * max(med(rs, lambda r: r['coupling_ratio']) for rs in per):.0f}"
    m["RcSimErrMeasMin"] = f2(min(med(rs, lambda r: r["error_measured"]) for rs in per))
    m["RcSimErrMeasMax"] = f2(max(med(rs, lambda r: r["error_measured"]) for rs in per))
    m["RcSimErrCorrMin"] = f2(min(med(rs, lambda r: r["error_corrected"]) for rs in per))
    m["RcSimErrCorrMax"] = f2(max(med(rs, lambda r: r["error_corrected"]) for rs in per))
    ratio_c = [med(rs, lambda r, c=c: r["F"][c]["corrected"] / r["F"][c]["true"]) for rs in per for c in ("frozen", "P1", "colon")]
    ratio_m = [med(rs, lambda r, c=c: r["F"][c]["measured"] / r["F"][c]["true"]) for rs in per for c in ("frozen", "P1", "colon")]
    m["RcSimCorrRatioMin"] = f2(min(ratio_c))
    m["RcSimCorrRatioMax"] = f2(max(ratio_c))
    m["RcSimMeasRatioMin"] = f2(min(ratio_m))
    m["RcSimMeasRatioMax"] = f2(max(ratio_m))
    fz = [med(rs, lambda r: r["F"]["frozen"]["corrected"] / r["F"]["frozen"]["true"]) for rs in per]
    m["RcSimFrozenRatioMin"] = f2(min(fz))
    m["RcSimFrozenRatioMax"] = f2(max(fz))
    trip = [(r["F"][c]["corrected"], r["F"][c]["true"], r["F"][c]["measured"]) for r in rows for c in ("frozen", "P1", "colon")]
    m["RcSimBracket"] = str(sum(a <= b <= c for a, b, c in trip))
    m["RcSimBracketTotal"] = str(len(trip))
    d1 = [r for r in rows if r["depth"] == 1.0 and r["analysis"] == "total"]
    m["RcSimFrozenTrue"] = f2(med(d1, lambda r: r["F"]["frozen"]["true"]))
    m["RcSimFrozenMeas"] = f2(med(d1, lambda r: r["F"]["frozen"]["measured"]))
    m["RcSimFrozenCorr"] = f2(med(d1, lambda r: r["F"]["frozen"]["corrected"]))
    m["RcSimFrozenTrueAbove"] = str(sum(r["F"]["frozen"]["true"] > 1 for r in d1))
    m["RcSimFrozenCorrAbove"] = str(sum(r["F"]["frozen"]["corrected"] > 1 for r in d1))
    m["RcSimDOne"] = str(len(d1))
    p1 = [r["F"]["P1"]["corrected"] > 1 for r in rows if r["F"]["P1"]["true"] > 1]
    oracle = [float(np.median([r["F"]["frozen"]["oracle"] / r["F"]["frozen"]["corrected"] for r in sims[t]])) for t in sims]
    m["RcSimOracleMin"] = f2(min(oracle))
    m["RcSimOracleMax"] = f2(max(oracle))
    m["RcSimPOneCorrAbove"] = str(sum(p1))
    m["RcSimPOneTrueAbove"] = str(len(p1))
    return m


# ----------------------------------------------------------------------------- figure
def figure(ev, rows, scores, overall, high):
    lines = ["% Generated by extension/recoverability/make_assets.py. Do not edit.",
             "\\definecolor{totcol}{HTML}{217D91}", "\\definecolor{coarsecol}{HTML}{C9803A}",
             "\\definecolor{finecol}{HTML}{8C5A2B}", "\\definecolor{failcol}{HTML}{B03A2E}",
             "\\definecolor{okcol}{HTML}{217D91}", "\\definecolor{donorcol}{HTML}{5D6670}",
             f"\\begin{{tikzpicture}}[{SANS}]"]
    est = ev["estimators"]
    arms = [("benchmark", "Own paired cells"), ("label_between", "Type means, unpaired"),
            ("colon/signal", "No pairs: colon"), ("P1/signal", "No pairs: blood types"),
            ("frozen/signal", "No pairs: blood")]           # y = 1..5
    colors = {"total": "totcol", "within_coarse": "coarsecol", "within_fine": "finecol"}
    offsets = {"total": .22, "within_coarse": 0.0, "within_fine": -.22}
    lines += ["\\begin{axis}[name=a,width=.40\\textwidth,height=5.0cm,xmin=0,xmax=1.85,ymin=.4,ymax=5.6,",
              " xtick={0,.5,1,1.5},ytick={1,2,3,4,5},yticklabels={" + ",".join("{" + a[1] + "}" for a in arms) + "},",
              " xlabel={Cross-correlation error (relative)},title={\\textbf{a}\\enspace Transfer to bone marrow},",
              COMMON + ",",
              " legend style={draw=none,font=\\sffamily\\scriptsize,at={(0.5,-0.30)},anchor=north,legend columns=3,"
              "column sep=5pt,cells={anchor=west}}]",
              "\\draw[dashed,black!45] (axis cs:1,.4) -- (axis cs:1,5.6);"]
    for an, label in (("total", "All cells"), ("within_coarse", "9 types"), ("within_fine", "45 types")):
        pts, means = [], []
        for y, (arm, _) in enumerate(arms, start=1):
            if arm not in est[an]:
                continue
            vals = est[an][arm]["values"]
            ys = np.linspace(y + offsets[an] - .07, y + offsets[an] + .07, len(vals))
            pts += [f"({min(v, 1.8):.4f},{yy:.3f})" for v, yy in zip(vals, ys)]
            means.append(f"({min(np.mean(vals), 1.8):.4f},{y + offsets[an]:.3f})")
        lines += [f"\\addplot[only marks,mark=*,mark size=.8pt,color={colors[an]}!75] coordinates {{ "
                  + " ".join(pts) + " };", f"\\addlegendentry{{{label}}}",
                  f"\\addplot[only marks,mark=|,mark size=3pt,line width=1.1pt,color=black,forget plot] coordinates {{ "
                  + " ".join(means) + " };"]
    lines.append("\\end{axis}")
    # b: diagnostic plane
    lines += ["\\begin{axis}[at={(a.east)},anchor=west,xshift=1.3cm,name=b,width=.40\\textwidth,height=5.0cm,",
              " xmin=0,xmax=.85,ymin=-.2,ymax=1,xtick={0,.2,.4,.6,.8},ytick={-.2,0,.2,.4,.6,.8,1},",
              " xlabel={Coverage of latent RNA variance by \\textit{S}},ylabel={Agreement with type means},",
              " title={\\textbf{b}\\enspace Two failure modes},",
              COMMON.replace("y tick style={draw=none},", "") + ",ymajorgrids=true,",
              " legend style={draw=none,font=\\sffamily\\scriptsize,at={(0.5,-0.30)},anchor=north,legend columns=3,"
              "column sep=5pt,cells={anchor=west}}]",
              "\\draw[dashed,black!45] (axis cs:.4,-.2) -- (axis cs:.4,1);"]
    fam_style = {"lc": ("*", ".55pt", "!35"), "perm": ("x", "1.2pt", "!70"), "frozen": ("*", "1.5pt", ""),
                 "colon": ("square*", "1.4pt", ""), "P1": ("triangle*", "1.7pt", "")}
    for fam in ("lc", "perm", "colon", "frozen", "P1"):
        for failed in (False, True):
            sel = [r for r in rows if r["family"] == fam and (r["E_signal"] >= 1) == failed]
            if not sel:
                continue
            mark, size, shade = fam_style[fam]
            col = ("failcol" if failed else "okcol") + shade
            pts = " ".join(f"({r['coverage_signal']:.3f},{max(r['type_agreement'], -.2):.3f})" for r in sel)
            forget = fam in ("lc", "perm") or failed
            lines.append(f"\\addplot[only marks,mark={mark},mark size={size},color={col}"
                         + (",forget plot" if forget else "") + f"] coordinates {{ {pts} }};")
            if not forget:
                label = {"frozen": "Blood", "colon": "Colon", "P1": "Blood types"}[fam]
                lines.append(f"\\addlegendentry{{{label}}}")
    lines += ["\\node[font=\\sffamily\\scriptsize,anchor=north east,color=okcol] at (axis cs:.84,.99) {success};",
              "\\node[font=\\sffamily\\scriptsize,anchor=north east,color=failcol] at (axis cs:.84,.90) {failure};",
              "\\end{axis}"]
    # c: AUC overall; d: AUC among high-coverage pairs
    order_c = ["Two failure modes", "Coverage", "Label-free", "Composition", "Covariance shift", "Training size",
               "Cells"][::-1]
    order_d = ["Two failure modes", "Type disagreement", "Compatibility", "Label-free"][::-1]
    for panel, (name, order, data, title, pos) in enumerate((
            ("c", order_c, overall, "\\textbf{c}\\enspace Predicting failure, all pairs",
             "at={(a.south west)},anchor=north west,yshift=-2.9cm"),
            ("d", order_d, high, "\\textbf{d}\\enspace Pairs with coverage of 0.4 or more",
             "at={(b.south west)},anchor=north west,yshift=-2.9cm"))):
        n = len(order)
        lines += [f"\\begin{{axis}}[{pos},name={name},width=.30\\textwidth,height={1.8 + .45 * n:.2f}cm,",
                  f" xmin=0,xmax=1,ymin=.4,ymax={n + .6},xtick={{0,.25,.5,.75,1}},ytick={{{','.join(str(i + 1) for i in range(n))}}},",
                  " yticklabels={" + ",".join("{" + o + "}" for o in order) + "},xlabel={AUC for failure},",
                  f" title={{{title}}},", COMMON + "]",
                  f"\\draw[dashed,black!45] (axis cs:.5,.4) -- (axis cs:.5,{n + .6});"]
        for y, o in enumerate(order, start=1):
            a, lo, hi = data[o]
            col = "black" if o == "Two failure modes" else "donorcol"
            lines += [f"\\draw[{col},line width=.8pt] (axis cs:{lo:.4f},{y}) -- (axis cs:{hi:.4f},{y});",
                      f"\\addplot[only marks,mark=*,mark size=1.6pt,color={col}] coordinates {{ ({a:.4f},{y}) }};"]
        lines.append("\\end{axis}")
    # e: within-type error against budget; f: total error
    s = ev["reference"]["summary"]
    lines += ["\\begin{axis}[at={(c.south west)},anchor=north west,yshift=-2.3cm,name=e,width=.34\\textwidth,height=4.4cm,",
              " xmode=log,log basis x=2,xtick={50,100,200,400,800},xticklabels={50,100,200,400,800},",
              " ymin=.5,ymax=1.1,xlabel={Paired cells per sample},ylabel={Within-type error},",
              " title={\\textbf{e}\\enspace Choosing a paired reference},",
              COMMON.replace("y tick style={draw=none},", "") + ",ymajorgrids=true,",
              " legend style={draw=none,fill=none,font=\\sffamily\\scriptsize,at={(0.03,0.03)},anchor=south west,"
              "legend columns=1,cells={anchor=west}}]",
              "\\draw[dashed,black!45] (axis cs:45,1) -- (axis cs:900,1);"]
    for st, col, mark, label in (("random", "donorcol", "*", "Random"), ("balanced", "coarsecol", "square*", "Balanced"),
                                 ("guided", "totcol", "triangle*", "Guided")):
        pts = " ".join(f"({N},{s[str(N)][st]['within']:.4f})" for N in BUDGETS)
        lines += [f"\\addplot[color={col},mark={mark},mark size=1.5pt,line width=.8pt] coordinates {{ {pts} }};",
                  f"\\addlegendentry{{{label}}}"]
    lines.append("\\end{axis}")
    lines += ["\\begin{axis}[at={(e.east)},anchor=west,xshift=2.9cm,name=f,width=.34\\textwidth,height=4.4cm,",
              " xmode=log,log basis x=2,xtick={50,100,200,400,800},xticklabels={50,100,200,400,800},",
              " ymin=0,ymax=1,ytick={0,.2,.4,.6,.8,1},xlabel={Paired cells per sample},ylabel={Total error},",
              " title={\\textbf{f}\\enspace Type means and paired cells},",
              COMMON.replace("y tick style={draw=none},", "") + ",ymajorgrids=true,",
              " legend style={draw=none,fill=none,font=\\sffamily\\scriptsize,at={(0.98,0.98)},anchor=north east,"
              "legend columns=1,cells={anchor=west}}]"]
    between = ev["reference"]["between_only_total"]
    for st, key, col, mark, label in (("guided", "hybrid_total", "totcol", "triangle*", "Type means + guided pairs"),
                                      ("random", "hybrid_total", "donorcol", "*", "Type means + random pairs"),
                                      (None, "paired_only_total", "coarsecol", "square*", "Random pairs alone")):
        pts = " ".join(f"({N},{(s[str(N)][st][key] if st else s[str(N)][key]):.4f})" for N in BUDGETS)
        lines += [f"\\addplot[color={col},mark={mark},mark size=1.5pt,line width=.8pt] coordinates {{ {pts} }};",
                  f"\\addlegendentry{{{label}}}"]
    lines += [f"\\addplot[black!60,dashed,line width=.8pt,domain=45:900] {{{between:.4f}}};",
              "\\addlegendentry{Type means alone}",
              "\\end{axis}", "\\end{tikzpicture}"]
    return "\n".join(lines) + "\n"


def table(ev):
    est, f = ev["estimators"], ev["failure"]
    names = [("frozen/signal", "No pairs: blood (latent RNA)"), ("frozen/measured", "No pairs: blood (measured RNA)"),
             ("P1/signal", "No pairs: blood cell types"), ("colon/signal", "No pairs: colon"),
             ("label_between", "Type means in each assay"), ("benchmark", "Own paired cells")]
    lines = ["% Generated by extension/recoverability/make_assets.py. Do not edit.",
             "\\begin{tabular}{lrrr}", "\\toprule",
             "Arm & All cells & 9 types & 45 types \\\\", "\\midrule"]
    for arm, label in names:
        cells = []
        for an in ANALYSES:
            cells.append(f"{est[an][arm]['mean']:.2f}" if arm in est[an] else "--")
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def auc_table(ev, overall, high):
    f = ev["failure"]
    comp = f["comparisons"]
    rows = [("Two failure modes", "primary", None), ("Coverage", "coverage", "primary_vs_coverage"),
            ("Label-free", "label_free", "label_free_vs_coverage"), ("Composition", "composition", "primary_vs_composition"),
            ("Covariance shift", "shift", "primary_vs_shift"), ("Training size", "n_train", "primary_vs_n_train"),
            ("Cells", "cells", "primary_vs_cells")]
    lines = ["% Generated by extension/recoverability/make_assets.py. Do not edit.",
             "\\begin{tabular}{lccc}", "\\toprule",
             "Score & AUC (95\\% CI) & Primary minus score (95\\% CI) & AUC, coverage $\\ge0.4$ \\\\", "\\midrule"]
    hi_names = {"Two failure modes": "Two failure modes", "Label-free": "Label-free"}
    for label, key, cmp in rows:
        a, lo, hi = overall[label]
        diff = "--"
        if cmp is not None:
            c = comp[cmp]
            diff = f"{f3(c['difference'])} ({f3(c['ci'][0])} to {f3(c['ci'][1])})"
            if cmp == "label_free_vs_coverage":
                diff += "$^{a}$"
        h = "--"
        if label in high:
            ha, hlo, hhi = high[label]
            h = f"{ha:.2f} ({hlo:.2f}--{hhi:.2f})"
        lines.append(f"{label} & {a:.2f} ({lo:.2f}--{hi:.2f}) & {diff} & {h} \\\\")
    for label in ("Type disagreement", "Compatibility"):
        ha, hlo, hhi = high[label]
        lines.append(f"{label} & -- & -- & {ha:.2f} ({hlo:.2f}--{hhi:.2f}) \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def reference_table(ev):
    s = ev["reference"]["summary"]
    lines = ["% Generated by extension/recoverability/make_assets.py. Do not edit.",
             "\\begin{tabular}{rcccccc}", "\\toprule",
             "& \\multicolumn{3}{c}{Within-type error} & \\multicolumn{3}{c}{Total error} \\\\",
             "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
             "Paired cells & Random & Balanced & Guided & Type means + random & Type means + guided & Random alone \\\\",
             "\\midrule",
             f"0 & -- & -- & -- & {ev['reference']['between_only_total']:.3f} & {ev['reference']['between_only_total']:.3f} & -- \\\\"]
    for N in BUDGETS:
        r = s[str(N)]
        lines.append(f"{N} & {r['random']['within']:.3f} & {r['balanced']['within']:.3f} & {r['guided']['within']:.3f} & "
                     f"{r['random']['hybrid_total']:.3f} & {r['guided']['hybrid_total']:.3f} & {r['paired_only_total']:.3f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def simulation_table():
    """Supplementary table: noise-correction validation by latent construction, depth and analysis."""
    lines = ["% Generated by extension/recoverability/make_assets.py. Do not edit.",
             "\\begin{tabular}{llrcccccc}", "\\toprule",
             "& & & & & \\multicolumn{2}{c}{Matrix error} & \\multicolumn{2}{c}{$\\|F\\|$ / true, blood channel} \\\\",
             "\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}",
             "Latent & Cells & Depth & Noise share & Share error & Measured & Corrected & Measured & Corrected \\\\",
             "\\midrule"]
    for name, label in (("noise_simulation.json", "Smoothed"), ("noise_simulation_gene.json", "Plus gene-specific")):
        rows = load(name)["rows"]
        for an, al in (("total", "All"), ("within_coarse", "Within types")):
            for depth in (0.5, 1.0, 2.0):
                rs = [r for r in rows if r["analysis"] == an and r["depth"] == depth]
                med = lambda f: float(np.median([f(r) for r in rs]))
                lines.append(f"{label} & {al} & {depth:g} & {med(lambda r: r['nu_true_median']):.2f} & "
                             f"{med(lambda r: r['nu_abs_error_mean']):.3f} & {med(lambda r: r['error_measured']):.2f} & "
                             f"{med(lambda r: r['error_corrected']):.2f} & "
                             f"{med(lambda r: r['F']['frozen']['measured'] / r['F']['frozen']['true']):.2f} & "
                             f"{med(lambda r: r['F']['frozen']['corrected'] / r['F']['frozen']['true']):.2f} \\\\")
                label = ""
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def identity_table():
    """Supplementary table: failure prediction against channel identity, leave-one-family-out and assay labels."""
    dg, lab = load("posthoc_diagnostic.json"), load("posthoc_labels.json")
    cell = lambda v: f"{v['value']:.2f} ({v['ci'][0]:.2f}--{v['ci'][1]:.2f})"
    rows = [("Frozen score (two failure modes)", dg["all_pairs"]["primary"], dg["high_coverage"]["primary"]),
            ("Label-free score", dg["all_pairs"]["label_free"], dg["high_coverage"]["label_free"]),
            ("Coverage", dg["all_pairs"]["coverage"], dg["high_coverage"]["coverage"]),
            ("Channel family, development rates", dg["all_pairs"]["identity_family"],
             dg["high_coverage"]["identity_family"]),
            ("Family and analysis kind, development rates", dg["all_pairs"]["identity_family_analysis"],
             dg["high_coverage"]["identity_family_analysis"]),
            ("Family and analysis kind, in-sample rates", dg["all_pairs"]["identity_in_sample"],
             dg["high_coverage"]["identity_in_sample"])]
    for fam, label in (("P1", "donor-by-type"), ("frozen", "blood"), ("lc", "donor-subset")):
        r = dg["leave_family_out"][fam]
        rows.append((f"Score refitted without the {label} family", r["all_pairs"], r["high_coverage"]))
    rows.append(("Frozen score, assay-specific labels", lab["failure"]["all_pairs"]["assay_labels"],
                 lab["failure"]["high_coverage"]["assay_labels"]))
    lines = ["% Generated by extension/recoverability/make_assets.py. Do not edit.",
             "\\begin{tabular}{lcc}", "\\toprule",
             f"Score & All pairs ($n={dg['pairs']:,}$) & Coverage $\\ge0.4$ ($n={dg['high_coverage_pairs']}$) \\\\",
             "\\midrule"]
    lines += [f"{label} & {cell(a)} & {cell(b)} \\\\" for label, a, b in rows]
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def model_macros():
    fm = load("failure_model.json")
    m = {"RcFmPairs": f"{fm['development_pairs']:,}", "RcFmFailures": f"{fm['failures']:,}"}
    names = {"coverage_signal": "Coverage", "log_F_signal": "LogF", "type_agreement": "Agreement",
             "between_share": "Composition"}
    for model, tag in (("primary", "Primary"), ("label_free", "LabelFree")):
        mdl = fm["models"][model]
        m[f"RcFm{tag}Auc"] = f2(mdl["development_auc"])
        for c, b in zip(mdl["predictors"], mdl["slopes"]):
            m[f"RcFm{tag}{names[c]}"] = f2(b)
    return m


def results_md(m, ev):
    v = ev["verdict"]
    lines = ["# Recoverability test on bone marrow: results", "",
             "Generated by `make_assets.py` from `results/bmmc_evaluation.json` (sealed test) and the development",
             "files `results/dev_*.json` (exploratory, already-scored cohorts).", "",
             "## Prespecified hypotheses", "",
             f"* H1, primary score beats coverage alone: {'met' if v['H1_primary_beats_coverage'] else 'not met'} "
             f"(AUC difference {m['RcBmDiffPrimaryCoverage']}, 95% CI {m['RcBmDiffPrimaryCoverageLo']} to "
             f"{m['RcBmDiffPrimaryCoverageHi']}).",
             "* H2, primary score beats each control: " + ", ".join(
                 f"{k} {'met' if ok else 'not met'}" for k, ok in v["H2_primary_beats_controls"].items()) + ".",
             f"* H3, failures recognized among high-coverage pairs: {'met' if v['H3_high_coverage_failures_recognized'] else 'not met'} "
             f"(AUC {m['RcBmHiAucPrimary']}, 95% CI {m['RcBmHiAucPrimaryLo']}-{m['RcBmHiAucPrimaryHi']}).",
             f"* H6, guided selection beats random and balanced: {'met' if v['H6_guided_beats_random'] and v['H6_guided_beats_balanced'] else 'not met'}.",
             "", "## All macros", "", "| Macro | Value |", "|---|---|"]
    lines += [f"| `\\{k}` | {val} |" for k, val in m.items()]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    ev = load("bmmc_evaluation.json")
    rows, scores = scored_rows()
    overall = auc_intervals(rows, scores, names=["Two failure modes", "Coverage", "Label-free", "Composition",
                                                  "Covariance shift", "Training size", "Cells"])
    hi = np.array([r["coverage_signal"] >= 0.4 for r in rows])
    high = auc_intervals(rows, scores, subset=hi, names=["Two failure modes", "Type disagreement", "Compatibility",
                                                          "Label-free"], seed=101)
    m = dev_macros()
    m.update(bmmc_macros(ev, rows, scores))
    m.update(model_macros())
    m.update(posthoc_macros())
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "recov_macros.tex").write_text(
        "% Generated by extension/recoverability/make_assets.py. Do not edit.\n"
        + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in m.items()))
    (args.out / "recov_figure.tex").write_text(figure(ev, rows, scores, overall, high))
    (args.out / "recov_table.tex").write_text(table(ev))
    (args.out / "recov_auc_table.tex").write_text(auc_table(ev, overall, high))
    (args.out / "recov_ref_table.tex").write_text(reference_table(ev))
    (args.out / "recov_sim_table.tex").write_text(simulation_table())
    (args.out / "recov_id_table.tex").write_text(identity_table())
    (R / "RESULTS.md").write_text(results_md(m, ev))
    print(json.dumps(m, indent=1))


if __name__ == "__main__":
    main()
