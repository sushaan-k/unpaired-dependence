#!/usr/bin/env python3
"""LaTeX macros, figure and table for the cross-study pairing-free results, and RESULTS.md.

Reads results/summary.json, identifiability.json, learning_curve.json,
colon_rigor.json, reference_fit.json, scores.json, prediction_info.json, the
two seals, and the exploratory post hoc results (posthoc_within.json,
posthoc_mixture.json, mixture_diagnostics.json). Every number in the
manuscript's cross-study text comes from here.

    python make_assets.py --out ../../revision/source --results-md results/RESULTS.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
R = HERE / "results"
COHORT_KEY = {"hao": "Hao", "colon": "Colon"}
ARM_KEY = {"pf_means": "PfMeans", "pf_moment": "PfMoment", "paired_closed_form": "Paired",
           "reference_regression": "Reg", "transferred_correlation": "Corr", "transferred_covariance": "Cov",
           "variances_only": "Var", "recipient_benchmark": "Bench", "independence": "Indep"}
FIG_ARMS = [("independence", "Independence"), ("variances_only", "Variances only"),
            ("transferred_covariance", "Transferred covariance"),
            ("transferred_correlation", "Transferred correlation"),
            ("reference_regression", "Reference regression"), ("paired_closed_form", "Paired closed form"),
            ("pf_moment", "No pairs: moments"), ("pf_means", "No pairs: means"),
            ("recipient_benchmark", "Own paired half")]
TABLE_ARMS = [("pf_means", "No pairs: means (primary)"), ("pf_moment", "No pairs: moments"),
              ("paired_closed_form", "Paired closed form"), ("reference_regression", "Reference regression"),
              ("transferred_correlation", "Transferred correlation"),
              ("transferred_covariance", "Transferred covariance"), ("variances_only", "Variances only"),
              ("recipient_benchmark", "Own paired half"), ("independence", "Independence")]
BASE_KEY = {"reference_regression": "Reg", "transferred_correlation": "Corr",
            "transferred_covariance": "Cov", "variances_only": "Var"}


def load(name):
    return json.loads((R / name).read_text())


def num(x, d=3):
    text = f"{abs(x):.{d}f}"
    return f"\\ensuremath{{-}}{text}" if x < 0 and float(text) != 0 else text


def pct(x, d=0):
    return num(100 * x, d)


def big(x):
    return f"{int(round(x)):,}".replace(",", "{,}")


def pval(p):
    return f"{p:.3f}" if p >= 0.001 else "<0.001"


def macros(summary, ident, curve, rigor, fit, scores, info, seals):
    m = {}
    pf, paired = fit["pf"], fit["paired"]
    m["XsPatients"] = str(pf["patients"])
    m["XsTrainCells"] = big(paired["cells"])
    m["XsHalfCells"] = big(pf["cells_rna_half"])
    feats = json.loads((HERE / "results/features.json").read_text())
    m["XsGenes"], m["XsProteins"] = str(len(feats["genes"])), str(len(feats["proteins"]))
    m["XsPfScale"] = f"{pf['pf_means']['scale']:g}"
    m["XsPfRank"] = str(pf["pf_moment"]["rank"])
    m["XsPfMomentIter"] = big(pf["pf_moment"]["iterations"])
    mant, expo = f"{pf['pf_moment']['grad_norm']:.1e}".split("e")
    m["XsPfMomentGrad"] = f"{mant}\\times10^{{{int(expo)}}}"
    m["XsPairedRidge"] = f"{paired['paired_closed_form']['ridge']:g}"
    m["XsRegLambda"] = f"{paired['reference_regression']['lambda']:g}"
    hao_seal = seals["hao"]
    m["XsHaoCells"] = big(sum(v["cells"] for v in hao_seal["files"].values()))
    m["XsHaoScoring"] = big(hao_seal["files"]["scoring"]["cells"])
    m["XsColonScoring"] = big(seals["colon"]["files"]["scoring"]["cells"])
    for cohort, s in ((c, summary[c]) for c in ("hao", "colon")):
        K = COHORT_KEY[cohort]
        m[f"Xs{K}Donors"] = str(len(s["donors"]))
        for arm, key in ARM_KEY.items():
            m[f"Xs{K}Err{key}"] = num(s["mean_E"]["total"][arm])
            m[f"Xs{K}Rho{key}"] = num(s["mean_entry_correlation"]["total"][arm], 3)
        h1 = s["H1"]
        m[f"Xs{K}NullErr"] = num(h1["null_mean_E"])
        m[f"Xs{K}NullMin"] = num(h1["null_min"])
        m[f"Xs{K}PermP"] = pval(h1["p"])
        m[f"Xs{K}PermMaxDonorP"] = pval(max(h1["per_donor_p"]))
        h2 = s["H2"]
        m[f"Xs{K}Share"] = pct(h2["mean_share"])
        m[f"Xs{K}ShareLo"], m[f"Xs{K}ShareHi"] = pct(h2["bootstrap_95"][0]), pct(h2["bootstrap_95"][1])
        sm = s["share_pf_moment"]
        m[f"Xs{K}ShareMoment"] = pct(sm["mean"])
        m[f"Xs{K}ShareMomentLo"], m[f"Xs{K}ShareMomentHi"] = pct(sm["bootstrap_95"][0]), pct(sm["bootstrap_95"][1])
        h3 = s["H3"]
        for arm in ("pf_means", "pf_moment", "paired_closed_form", "recipient_benchmark"):
            key = ARM_KEY[arm]
            m[f"Xs{K}WErr{key}"] = num(h3[arm]["mean_E"])
            m[f"Xs{K}WErr{key}Lo"] = num(h3[arm]["bootstrap_95"][0])
            m[f"Xs{K}WErr{key}Hi"] = num(h3[arm]["bootstrap_95"][1])
            m[f"Xs{K}WBelow{key}"] = str(h3[arm]["donors_below_1"])
        m[f"Xs{K}WShare"] = pct(h3["share_of_paired_within"]["mean"])
        m[f"Xs{K}WShareLo"] = pct(h3["share_of_paired_within"]["bootstrap_95"][0])
        m[f"Xs{K}WShareHi"] = pct(h3["share_of_paired_within"]["bootstrap_95"][1])
        for base, key in BASE_KEY.items():
            h4 = s["H4"][base]
            m[f"Xs{K}Diff{key}"] = num(h4["mean_difference"])
            m[f"Xs{K}Diff{key}Lo"] = num(h4["bonferroni_interval"][0])
            m[f"Xs{K}Diff{key}Hi"] = num(h4["bonferroni_interval"][1])
            m[f"Xs{K}Fav{key}"] = str(h4["donors_favouring_closed_form"])
        d = s["pf_moment_minus_pf_means"]
        m[f"Xs{K}MomentMinusMeans"] = num(d["mean"])
        m[f"Xs{K}MomentMinusMeansLo"] = num(d["bootstrap_95"][0])
        m[f"Xs{K}MomentMinusMeansHi"] = num(d["bootstrap_95"][1])
        m[f"Xs{K}MomentBetter"] = str(d["donors_moment_better"])
        pvp = s["pf_vs_paired"]["total"]
        m[f"Xs{K}PfMeansBetterThanPaired"] = str(pvp["donors_pf_means_better"])
        m[f"Xs{K}PfMomentBetterThanPaired"] = str(pvp["donors_pf_moment_better"])
        vb = s["vs_benchmark"]["total"]
        m[f"Xs{K}PfMeansBetterThanBench"] = str(vb["pf_means"]["donors_better_than_benchmark"])
        m[f"Xs{K}PairedBetterThanBench"] = str(vb["paired_closed_form"]["donors_better_than_benchmark"])
    for arm in ("pf_means", "paired_closed_form", "recipient_benchmark"):
        m[f"XsHaoWRho{ARM_KEY[arm]}"] = num(summary["hao"]["mean_entry_correlation"]["within_l1"][arm], 2)
    l2 = summary["hao"]["mean_E"]["within_l2"]
    for arm in ("pf_means", "pf_moment", "paired_closed_form", "recipient_benchmark"):
        m[f"XsHaoLTwoErr{ARM_KEY[arm]}"] = num(l2[arm])
    # identifiability
    m["XsDimM"] = str(ident["dimension_M"])
    m["XsEffDim"] = num(ident["effective_dimension"], 1)
    for cohort in ("hao", "colon"):
        K = COHORT_KEY[cohort]
        within = "within_l1" if cohort == "hao" else "within_coarse"
        for label, analysis in (("", "total"), ("W", within)):
            rows = [v for k, v in ident["donors"].items() if k.startswith(cohort + "/") and k.endswith("/" + analysis)]
            for field, key in (("rho", "Rho"), ("floor", "Floor"), ("radius", "Radius")):
                fmt = (lambda v: pct(v)) if field == "rho" else (lambda v: num(v, 2))
                m[f"Xs{K}{label}Id{key}"] = fmt(np.mean([r[field] for r in rows]))
                m[f"Xs{K}{label}Id{key}Min"] = fmt(np.min([r[field] for r in rows]))
                m[f"Xs{K}{label}Id{key}Max"] = fmt(np.max([r[field] for r in rows]))
    units = list(ident["donors"].values())            # every donor x analysis (48)
    m["XsIdUnits"] = str(len(units))
    m["XsFloorErrCorr"] = num(np.corrcoef([r["floor"] for r in units], [r["E_pf_means"] for r in units])[0, 1], 2)
    m["XsRhoErrCorr"] = num(np.corrcoef([r["rho"] for r in units], [r["E_pf_means"] for r in units])[0, 1], 2)
    maxF = [r["max_identified_singular_value"] for r in units]
    m["XsMaxFMin"], m["XsMaxFMax"] = num(min(maxF), 2), num(max(maxF), 2)
    m["XsMaxFAboveOne"] = str(sum(v > 1 for v in maxF))
    m["XsMaxFTotalMin"] = num(min(r["max_identified_singular_value"] for k, r in ident["donors"].items()
                                  if k.endswith("/total")), 2)
    for cohort in ("hao", "colon"):
        g = ident["per_gene"]["cohorts"][cohort]
        m[f"XsGeneSpearmanUnid{COHORT_KEY[cohort]}"] = num(g["spearman_unidentified_error"], 2)
        m[f"XsGeneSpearmanAxis{COHORT_KEY[cohort]}"] = num(g["spearman_h_error"], 2)
    sens = load("sensitivity_moment.json")
    m["XsRestartBChange"] = sci(sens["relative_change_B"])
    m["XsRestartEChange"] = sci(sens["max_abs_E_change"])
    # learning curve
    sizes = sorted({f["patients"] for f in curve["fits"]})
    for size in sizes:
        fits = [f for f in curve["fits"] if f["patients"] == size]
        for cohort in ("hao", "colon"):
            K = COHORT_KEY[cohort]
            E = [np.mean([v for k, v in f["E"].items() if k.startswith(cohort + "/")]) for f in fits]
            m[f"XsLc{K}{words(size)}"] = num(np.mean(E))
        m[f"XsLcDim{words(size)}"] = num(np.mean([f["dimension_M"] for f in fits]), 1 if len(fits) > 1 else 0)
    # colon rigor re-analysis
    donors = sorted(rigor)
    Em = np.array([rigor[d]["pf_means"]["E"] for d in donors])
    Er = np.array([rigor[d]["pf_moment"]["E"] for d in donors])
    m["XsRigErrMeans"], m["XsRigErrMoment"] = num(Em.mean()), num(Er.mean())
    m["XsRigNullMeans"] = num(np.mean([np.mean(rigor[d]["pf_means"]["perm_E"]) for d in donors]))
    m["XsRigNullMoment"] = num(np.mean([np.mean(rigor[d]["pf_moment"]["perm_E"]) for d in donors]))
    m["XsRigMaxPMeans"] = pval(max(rigor[d]["pf_means"]["p"] for d in donors))
    m["XsRigMaxPMoment"] = pval(max(rigor[d]["pf_moment"]["p"] for d in donors))
    m["XsRigDonorsSigMeans"] = str(sum(rigor[d]["pf_means"]["p"] < 0.05 for d in donors))
    m["XsRigDonorsSigMoment"] = str(sum(rigor[d]["pf_moment"]["p"] < 0.05 for d in donors))
    m["XsRigEdge"] = str(sum(rigor[d]["pf_means"]["tuning"]["at_edge"] for d in donors))
    m["XsRigExtended"] = str(sum(rigor[d]["pf_means"]["tuning"]["extensions"] > 0 for d in donors))
    ranks = sorted(rigor[d]["pf_moment"]["rank"] for d in donors)
    m["XsRigRankMin"], m["XsRigRankMax"] = str(ranks[0]), str(ranks[-1])
    m["XsRigConverged"] = str(sum(rigor[d]["pf_moment"]["converged"] for d in donors))
    m["XsRigDimMin"] = str(min(rigor[d]["identifiability"]["dimension_M"] for d in donors))
    m["XsRigDimMax"] = str(max(rigor[d]["identifiability"]["dimension_M"] for d in donors))
    m["XsRigBiopsiesMin"] = str(min(rigor[d]["training_biopsies"] for d in donors))
    m["XsRigBiopsiesMax"] = str(max(rigor[d]["training_biopsies"] for d in donors))
    for field, key in (("rho", "Rho"), ("floor", "Floor"), ("radius", "Radius")):
        value = np.mean([rigor[d]["identifiability"][field] for d in donors])
        m[f"XsRigId{key}"] = pct(value) if field == "rho" else num(value, 2)
    m["XsRigMaxF"] = num(max(rigor[d]["identifiability"]["max_identified_singular_value"] for d in donors), 2)
    d3 = json.loads((HERE.parent / "spectral_transfer/results/d1_lodo.json").read_text())["donors"]
    paired = np.array([d3[d]["arms"]["closed_form"]["cov_rel_error"] for d in donors])
    m["XsRigShareMeans"] = pct(np.mean((1 - Em) / (1 - paired)))
    m["XsRigShareMoment"] = pct(np.mean((1 - Er) / (1 - paired)))
    return m


def mixture_macros(p1, p2, mdiag):
    """Exploratory post hoc analyses P1 and P2 and the descriptive addition to P2."""
    m = {}
    s = p1["summary"]
    m["XsPoneUnits"] = str(p1["units"])
    m["XsPoneDim"] = str(p1["dimension_M"])
    m["XsPoneErr"] = num(s["hao/within_l1"]["mean_E"])
    m["XsPoneErrLTwo"] = num(s["hao/within_l2"]["mean_E"])
    m["XsPoneErrTotal"] = num(s["hao/total"]["mean_E"])
    m["XsPoneRho"] = pct(s["hao/within_l1"]["mean_rho"])
    m["XsPoneFloor"] = num(s["hao/within_l1"]["mean_floor"], 2)
    for cohort in ("hao", "colon"):
        K = COHORT_KEY[cohort]
        q = p2["summary"][cohort]
        for part, key in (("between_share", "BetweenShare"), ("within_share", "WithinShare")):
            m[f"XsMix{K}{key}"] = pct(q[part]["mean"])
            m[f"XsMix{K}{key}Min"], m[f"XsMix{K}{key}Max"] = pct(q[part]["min"]), pct(q[part]["max"])
        for part, key in (("E_between", "ErrBetween"), ("E_within", "ErrWithin"), ("E_total_direct", "ErrDirect")):
            m[f"XsMix{K}{key}"] = num(q[part]["mean"], 2)
            m[f"XsMix{K}{key}Min"], m[f"XsMix{K}{key}Max"] = num(q[part]["min"], 2), num(q[part]["max"], 2)
            m[f"XsMix{K}{key}Below"] = str(q[part]["donors_below_1"])
        m[f"XsMix{K}RBetween"], m[f"XsMix{K}RWithin"] = num(q["r_between"], 2), num(q["r_within"], 2)
        m[f"XsMix{K}NormWithin"] = num(q["norm_ratio_within"], 1)
        m[f"XsMix{K}NormBetween"] = num(q["norm_ratio_between"], 2)
        d = mdiag["summary"][cohort]
        for part, key in (("rho_between", "RhoBetween"), ("rho_within", "RhoWithin"),
                          ("between_variance_share", "VarBetween")):
            m[f"XsMix{K}{key}"] = pct(d[part]["mean"])
            m[f"XsMix{K}{key}Min"], m[f"XsMix{K}{key}Max"] = pct(d[part]["min"]), pct(d[part]["max"])
    return m


def sci(x):
    """Upper bound of the form a x 10^b with a single-digit a (for "below" statements)."""
    expo = int(np.floor(np.log10(x)))
    mant = int(np.ceil(x / 10 ** expo))
    if mant == 10:
        mant, expo = 1, expo + 1
    return f"{mant}\\times10^{{{expo}}}"


def words(n):
    return {8: "Eight", 16: "Sixteen", 32: "ThirtyTwo", 64: "SixtyFour"}.get(n, "All")


CLIP = 1.6
SANS = "font=\\sffamily"
COMMON = (" axis x line*=bottom,axis y line*=left,tick align=outside,y tick style={draw=none},"
          "xmajorgrids=true,grid style={gray!18},axis line style={gray!55},"
          "tick label style={font=\\sffamily\\scriptsize},label style={font=\\sffamily\\small},clip=false,"
          "title style={font=\\sffamily\\small,at={(0,1)},anchor=base west,yshift=5pt}")


def figure(summary, curve, entries=None, genes=None, prots=None):
    """Fig. 3. Panels a-b: per-donor error by arm (top row independence, bottom row own
    paired half); values above CLIP are drawn at CLIP with the mean printed. c: within-type
    donor means."""
    lines = ["% Generated by extension/cross_study/make_assets.py. Do not edit.",
             "\\definecolor{donorcol}{HTML}{5D6670}", "\\definecolor{haocol}{HTML}{217D91}",
             "\\definecolor{coloncol}{HTML}{A64442}", f"\\begin{{tikzpicture}}[{SANS}]"]
    order = list(reversed(FIG_ARMS))                   # y = 1 (bottom) is the last entry of FIG_ARMS
    n = len(order)
    ticks = ",".join(str(i + 1) for i in range(n))
    labels = ",".join("{" + label + "}" for _, label in order)
    panels = (("hao", "\\textbf{a}\\enspace Confirmatory blood cohort, 8 donors", "name=a"),
              ("colon", "\\textbf{b}\\enspace Colon, 12 donors", "at={(a.east)},anchor=west,xshift=0.9cm,name=b"))
    for cohort, title, pos in panels:
        s = summary[cohort]
        ylabels = f"yticklabels={{{labels}}}" if cohort == "hao" else "yticklabels={}"
        lines += [f"\\begin{{axis}}[{pos},width=.38\\textwidth,height=6.0cm,xmin=0,xmax={CLIP + .05},"
                  f"ymin=.4,ymax={n + .6},",
                  f" xtick={{0,.4,.8,1.2,1.6}},ytick={{{ticks}}},{ylabels},",
                  f" xlabel={{Cross-correlation error (relative)}},title={{{title}}},", COMMON + "]",
                  f"\\draw[dashed,black!45] (axis cs:1,.4) -- (axis cs:1,{n + .6});",
                  f"\\draw[black!25] (axis cs:0,1.5) -- (axis cs:{CLIP + .05},1.5);",
                  f"\\draw[black!25] (axis cs:0,3.5) -- (axis cs:{CLIP + .05},3.5);",
                  "\\addplot[only marks,mark=*,mark size=.9pt,color=donorcol!60] coordinates {"]
        k = len(s["donors"])
        for row, (arm, _) in enumerate(order):
            vals = s["per_donor_E"]["total"][arm]
            ys = np.linspace(row + 1 - .12, row + 1 + .12, k)
            lines.append(" " + " ".join(f"({min(v, CLIP):.4f},{y:.3f})" for v, y in zip(vals, ys)))
        lines[-1] += "};"
        means = " ".join(f"({min(s['mean_E']['total'][arm], CLIP):.4f},{row + 1})" for row, (arm, _) in enumerate(order))
        lines.append("\\addplot[only marks,mark=|,mark size=4pt,line width=1.2pt,color=black] coordinates { "
                     + means + " };")
        for row, (arm, _) in enumerate(order):
            mean = s["mean_E"]["total"][arm]
            if mean > CLIP:
                lines.append(f"\\node[font=\\sffamily\\tiny,anchor=west,black!60] at (axis cs:{CLIP + .06},{row + 1}) "
                             f"{{{mean:.1f}}};")
        lines.append("\\end{axis}")
    groups = [("hao", "within_l1", "Blood, 8 types"), ("hao", "within_l2", "Blood, 31 types"),
              ("colon", "within_coarse", "Colon, 14 types")]
    arms = [("recipient_benchmark", "Own paired half", "black!70", "diamond*"),
            ("paired_closed_form", "Paired closed form", "coloncol", "square*"),
            ("pf_moment", "No pairs: moments", "haocol!55", "triangle*"),
            ("pf_means", "No pairs: means", "haocol", "*")]
    lines += ["\\begin{axis}[at={(a.south west)},anchor=north west,yshift=-2.0cm,name=c,",
              f" width=.38\\textwidth,height=4.4cm,xmin=0,xmax={CLIP + .05},ymin=.4,ymax=3.6,",
              " xtick={0,.4,.8,1.2,1.6},ytick={1,2,3},yticklabels={"
              + ",".join("{" + g[2] + "}" for g in reversed(groups)) + "},",
              " xlabel={Within-type error (donor means)},",
              " title={\\textbf{c}\\enspace Within matched cell types},",
              COMMON + ",",
              " legend style={draw=none,font=\\sffamily\\scriptsize,at={(0.5,-0.36)},anchor=north,legend columns=2,"
              "cells={anchor=west},column sep=6pt}]",
              "\\draw[dashed,black!45] (axis cs:1,.4) -- (axis cs:1,3.6);"]
    for j, (arm, label, color, mark) in enumerate(arms):
        coords = []
        for g, (cohort, analysis, _) in enumerate(reversed(groups)):
            coords.append(f"({min(summary[cohort]['mean_E'][analysis][arm], CLIP):.4f},{g + 1 + (j - 1.5) * .14:.3f})")
        lines += [f"\\addplot[only marks,mark={mark},mark size=1.8pt,color={color}] coordinates {{ "
                  + " ".join(coords) + " };", f"\\addlegendentry{{{label}}}"]
    lines.append("\\end{axis}")
    if entries is not None:
        lines += figure_pairs(entries, genes, prots)
    lines.append("\\end{tikzpicture}")
    return "\n".join(lines) + "\n"


def figure_identification(ident, curve, p2, mdiag, compat):
    """Fig. 4. a-c: between-type and within-type parts per donor (size of each part of the
    truth, error of the channel's prediction of each part, share of the corresponding RNA
    covariance inside the effective supported subspace S). d: compatibility statistic ||F||
    estimated from measured and from noise-corrected RNA covariance against its true latent
    value, in simulations with known latent covariance at the observed depths
    (recoverability/simulate_noise.py; both truths, depth factor 1). e: learning curve
    of the primary pairing-free estimator."""
    lines = ["% Generated by extension/cross_study/make_assets.py. Do not edit.",
             "\\definecolor{donorcol}{HTML}{5D6670}", "\\definecolor{haocol}{HTML}{217D91}",
             "\\definecolor{coloncol}{HTML}{A64442}", "\\definecolor{betweencol}{HTML}{217D91}",
             "\\definecolor{withincol}{HTML}{C9803A}", f"\\begin{{tikzpicture}}[{SANS}]"]
    rows = [("colon", "within", "Colon, within"), ("colon", "between", "Colon, between"),
            ("hao", "within", "Blood, within"), ("hao", "between", "Blood, between")]   # y = 1..4
    donors = {c: sorted(k for k in p2["donors"] if k.startswith(c + "/")) for c in ("hao", "colon")}

    def values(cohort, part, which):
        out = []
        for tag in donors[cohort]:
            r, d = p2["donors"][tag], mdiag["donors"][tag]
            out.append({"share": r["truth"][f"{part}_share"], "error": r[part]["E"],
                        "rho": d[f"rho_{part}"]}[which])
        return out

    specs = [("share", "\\textbf{a}\\enspace Size of each part", "Norm relative to total", 0, 1.05,
              "{0,.25,.5,.75,1}", None, "name=pa"),
             ("error", "\\textbf{b}\\enspace Prediction error", "Relative error", 0, 2.7,
              "{0,.5,1,1.5,2,2.5}", 1, "at={(pa.east)},anchor=west,xshift=1.0cm,name=pb"),
             ("rho", "\\textbf{c}\\enspace Share inside \\textit{S}", "Share of RNA covariance", 0, 1.05,
              "{0,.25,.5,.75,1}", None, "at={(pb.east)},anchor=west,xshift=1.0cm,name=pc")]
    for which, title, xlabel, lo, hi, xt, ref, pos in specs:
        ylabels = ("yticklabels={" + ",".join("{" + r[2] + "}" for r in rows) + "}") if which == "share" \
            else "yticklabels={}"
        lines += [f"\\begin{{axis}}[{pos},width=.28\\textwidth,height=4.6cm,xmin={lo},xmax={hi},ymin=.4,ymax=4.6,",
                  f" xtick={xt},ytick={{1,2,3,4}},{ylabels},xlabel={{{xlabel}}},title={{{title}}},", COMMON + "]",
                  "\\draw[black!25] (axis cs:" + f"{lo}" + ",2.5) -- (axis cs:" + f"{hi}" + ",2.5);"]
        if ref is not None:
            lines.append(f"\\draw[dashed,black!45] (axis cs:{ref},.4) -- (axis cs:{ref},4.6);")
        for y, (cohort, part, _) in enumerate(rows, start=1):
            vals = values(cohort, part, which)
            ys = np.linspace(y - .15, y + .15, len(vals))
            color = "betweencol" if part == "between" else "withincol"
            lines.append(f"\\addplot[only marks,mark=*,mark size=.9pt,color={color}!70] coordinates {{ "
                         + " ".join(f"({v:.4f},{yy:.3f})" for v, yy in zip(vals, ys)) + " };")
            lines.append(f"\\addplot[only marks,mark=|,mark size=4pt,line width=1.2pt,color=black] coordinates "
                         f"{{ ({np.mean(vals):.4f},{y}) }};")
        lines.append("\\end{axis}")
    # d: compatibility statistic in simulations with known latent covariance (recoverability/simulate_noise.py)
    lines += ["\\begin{axis}[at={(pa.south west)},anchor=north west,yshift=-2.45cm,name=pd,",
              " width=.40\\textwidth,height=5.0cm,xmode=log,ymode=log,xmin=.4,xmax=8,ymin=.1,ymax=8,",
              " xtick={.5,1,2,4},xticklabels={0.5,1,2,4},ytick={.1,.3,1,3},yticklabels={0.1,0.3,1,3},",
              " xlabel={$\\|F\\|$, true latent RNA covariance},ylabel={$\\|F\\|$, estimated},",
              " title={\\textbf{d}\\enspace Compatibility in simulations},",
              COMMON.replace("y tick style={draw=none},", "") + ",ymajorgrids=true,",
              " legend style={draw=none,font=\\sffamily\\scriptsize,at={(0.5,-0.36)},anchor=north,"
              "legend columns=2,column sep=5pt,cells={anchor=west}}]",
              "\\draw[dashed,black!45] (axis cs:1,.1) -- (axis cs:1,8);",
              "\\draw[dashed,black!45] (axis cs:.4,1) -- (axis cs:8,1);",
              "\\draw[black!30] (axis cs:.4,.4) -- (axis cs:8,8);"]
    channels = (("frozen", "haocol", "Channel from donors"), ("P1", "withincol", "Channel from donor-type groups"),
                ("colon", "coloncol", "Channel from colon"))
    for channel, color, label in channels:
        for kind, style in (("measured", ",mark options={fill=white},forget plot"), ("corrected", "")):
            pts = [(r["F"][channel]["true"], r["F"][channel][kind]) for r in compat if r["depth"] == 1.0]
            lines.append(f"\\addplot[only marks,mark=*,mark size=1.1pt,color={color}{style}] coordinates {{ "
                         + " ".join(f"({a:.4f},{max(b, .1):.4f})" for a, b in pts) + " };")
            if kind == "corrected":
                lines.append(f"\\addlegendentry{{{label}}}")
    lines += ["\\addlegendimage{only marks,mark=*,mark size=1.1pt,color=black!55,mark options={fill=white}}",
              "\\addlegendentry{Measured covariance}",
              "\\addlegendimage{only marks,mark=*,mark size=1.1pt,color=black!55}",
              "\\addlegendentry{Noise-corrected}",
              "\\end{axis}"]
    sizes = sorted({f["patients"] for f in curve["fits"]})
    lines += ["\\begin{axis}[at={(pd.east)},anchor=west,xshift=2.0cm,name=pe,",
              " width=.33\\textwidth,height=5.0cm,xmode=log,log basis x=2,",
              f" xtick={{{','.join(str(v) for v in sizes)}}},xticklabels={{{','.join(str(v) for v in sizes)}}},",
              " ymin=0.2,ymax=1.2,xlabel={Training donors},ylabel={No pairs: mean error},",
              " title={\\textbf{e}\\enspace More training populations},",
              " title style={font=\\sffamily\\small,at={(0,1)},anchor=base west,yshift=5pt},",
              " axis x line*=bottom,axis y line*=left,tick align=outside,",
              " ymajorgrids=true,grid style={gray!18},axis line style={gray!55},",
              " tick label style={font=\\sffamily\\scriptsize},label style={font=\\sffamily\\small},",
              " legend style={draw=none,font=\\sffamily\\scriptsize,at={(0.5,-0.36)},anchor=north,"
              "legend columns=1,cells={anchor=west}}]",
              "\\draw[dashed,black!45] (axis cs:6,1) -- (axis cs:150,1);"]
    for cohort, color, label in (("hao", "haocol", "Confirmatory blood"), ("colon", "coloncol", "Colon")):
        pts = []
        for size in sizes:
            fits = [f for f in curve["fits"] if f["patients"] == size]
            E = [np.mean([v for k, v in f["E"].items() if k.startswith(cohort + "/")]) for f in fits]
            lo, hi, mid = min(E), max(E), float(np.mean(E))
            pts.append(f"({size},{mid:.4f}) += (0,{hi - mid:.4f}) -= (0,{mid - lo:.4f})")
        lines += [f"\\addplot[color={color},mark=*,mark size=1.5pt,line width=.8pt,error bars/.cd,y dir=both,"
                  "y explicit] coordinates { " + " ".join(pts) + " };", f"\\addlegendentry{{{label}}}"]
    lines += ["\\end{axis}", "\\end{tikzpicture}"]
    return "\n".join(lines) + "\n"


COGNATE = [("FCGR3A", "CD16"), ("KLRB1", "CD161"), ("IGHD", "IgD"), ("IL7R", "CD127")]
LINEAGE = [("LYZ", "CD371"), ("CD79A", "CD19")]
ENTRY_ARMS = ("pf_means", "paired_closed_form", "recipient_benchmark")


def entry_means(scores):
    """Donor-mean held-out and predicted cross-correlation matrices of the confirmatory cohort
    (descriptive; computed after scoring)."""
    T = np.load(R / "truth.npz")
    P = np.load(R / "predictions.npz")
    donors = sorted({k.split("/")[1] for k in scores["scores"] if k.startswith("hao/")})
    out = {}
    for analysis in ("total", "within_l1"):
        out[analysis] = {"truth": np.mean([T[f"hao/{d}/{analysis}/T"] for d in donors], 0)}
        for arm in ENTRY_ARMS:
            out[analysis][arm] = np.mean([P[f"hao/{d}/{analysis}/{arm}"] for d in donors], 0)
    return out


def entry_macros(entries, genes, prots):
    m = {}
    for g, pr in COGNATE + LINEAGE:
        i, j = genes.index(g), prots.index(pr)
        key = "".join(ch for ch in g if ch.isalpha()).title()
        for analysis, tag in (("total", "Tot"), ("within_l1", "W")):
            e = entries[analysis]
            m[f"XsPair{key}{tag}Truth"] = num(e["truth"][i, j], 2)
            m[f"XsPair{key}{tag}PfMeans"] = num(e["pf_means"][i, j], 2)
            m[f"XsPair{key}{tag}Paired"] = num(e["paired_closed_form"][i, j], 2)
            m[f"XsPair{key}{tag}Bench"] = num(e["recipient_benchmark"][i, j], 2)
    return m


def figure_pairs(entries, genes, prots):
    """Fig. 3d: within-type correlation of the four largest gene-own-protein pairs."""
    rows = list(reversed(COGNATE))
    lines = ["\\begin{axis}[at={(c.east)},anchor=west,xshift=2.1cm,name=d,",
             " width=.38\\textwidth,height=4.4cm,xmin=-.02,xmax=.72,ymin=.4,ymax=4.6,",
             " xtick={0,.2,.4,.6},ytick={1,2,3,4},yticklabels={"
             + ",".join("{" + f"\\textit{{{g}}}--{pr}" + "}" for g, pr in rows) + "},",
             " xlabel={Within-type correlation (donor means)},",
             " title={\\textbf{d}\\enspace Genes and their own proteins, within types},",
             COMMON + ",",
             " legend style={draw=none,font=\\sffamily\\scriptsize,at={(0.5,-0.36)},anchor=north,legend columns=2,"
             "cells={anchor=west},column sep=6pt}]"]
    marks = [("truth", "Held-out cells", "black", "x", "2.4pt"),
             ("recipient_benchmark", "Own paired half", "black!70", "diamond*", "1.8pt"),
             ("paired_closed_form", "Paired closed form", "coloncol", "square*", "1.8pt"),
             ("pf_means", "No pairs: means", "haocol", "*", "1.8pt")]
    e = entries["within_l1"]
    for j, (arm, label, color, mark, size) in enumerate(marks):
        coords = []
        for y, (g, pr) in enumerate(rows, start=1):
            coords.append(f"({e[arm][genes.index(g), prots.index(pr)]:.4f},{y + (j - 1.5) * .13:.3f})")
        lines += [f"\\addplot[only marks,mark={mark},mark size={size},color={color},line width=.8pt] coordinates {{ "
                  + " ".join(coords) + " };", f"\\addlegendentry{{{label}}}"]
    lines.append("\\end{axis}")
    return lines


def figure_entries(entries, genes, prots):
    """Extended Data figure: held-out against predicted entries (every sixteenth entry of the
    donor-mean matrices, plus the labelled pairs)."""
    lines = ["% Generated by extension/cross_study/make_assets.py. Do not edit.",
             "\\definecolor{donorcol}{HTML}{5D6670}", "\\definecolor{haocol}{HTML}{217D91}",
             "\\definecolor{coloncol}{HTML}{A64442}", f"\\begin{{tikzpicture}}[{SANS}]"]
    panels = [("total", "pf_means", "\\textbf{a}\\enspace All cells, no pairs", "name=a", -0.7, 1.0),
              ("within_l1", "pf_means", "\\textbf{b}\\enspace Within types, no pairs",
               "at={(a.east)},anchor=west,xshift=1.3cm,name=b", -0.6, 0.8),
              ("within_l1", "paired_closed_form", "\\textbf{c}\\enspace Within types, paired",
               "at={(b.east)},anchor=west,xshift=1.3cm,name=c", -0.6, 0.8)]
    for analysis, arm, title, pos, lo, hi in panels:
        e = entries[analysis]
        truth, pred = e["truth"].ravel(), e[arm].ravel()
        sub = np.arange(0, truth.size, 16)
        lines += [f"\\begin{{axis}}[{pos},width=.34\\textwidth,height=.34\\textwidth,xmin={lo},xmax={hi},"
                  f"ymin={lo},ymax={hi},",
                  " xlabel={Held-out correlation},ylabel={Predicted correlation},title={" + title + "},",
                  COMMON.replace("y tick style={draw=none},", "").replace("clip=false", "clip=true") + "]",
                  f"\\draw[black!35] (axis cs:{lo},{lo}) -- (axis cs:{hi},{hi});",
                  "\\addplot[only marks,mark=*,mark size=.35pt,color=donorcol!45] coordinates {"]
        chunk = [f"({truth[k]:.3f},{pred[k]:.3f})" for k in sub]
        lines += [" " + " ".join(chunk[i:i + 12]) for i in range(0, len(chunk), 12)]
        lines[-1] += "};"
        pairs = COGNATE if analysis == "within_l1" else COGNATE + LINEAGE
        pts = []
        for g, pr in pairs:
            i, j = genes.index(g), prots.index(pr)
            pts.append((e[arm][i, j], e["truth"][i, j], g, pr))
        x_left = 0.30 if analysis == "within_l1" else 0.36
        for k, (y, x, g, pr) in enumerate(sorted(pts, reverse=True)):
            color = "coloncol" if (g, pr) in COGNATE else "haocol"
            yk = -0.08 - 0.085 * k
            lines += [f"\\draw[{color}!60,line width=.3pt] (axis cs:{x:.3f},{y:.3f}) -- (axis cs:{x_left - .01:.3f},{yk:.3f});",
                      f"\\addplot[only marks,mark=*,mark size=1.6pt,color={color}] coordinates {{ ({x:.3f},{y:.3f}) }};",
                      f"\\node[font=\\sffamily\\tiny,anchor=west,color={color},inner sep=1pt] at (axis cs:{x_left:.3f},{yk:.3f}) "
                      f"{{\\textit{{{g}}}--{pr}}};"]
        lines.append("\\end{axis}")
    lines += ["\\end{tikzpicture}"]
    return "\n".join(lines) + "\n"



def mixture_table(p1, p2, mdiag):
    """Supplementary table: between-type and within-type parts (P2) with the descriptive
    addition, and the within-type training analysis (P1)."""
    rows = ["% Generated by extension/cross_study/make_assets.py. Do not edit.",
            "\\begin{tabular}{@{}lcccccccc@{}}", "\\toprule",
            " & \\multicolumn{2}{c}{Norm share} & \\multicolumn{2}{c}{Relative error} & "
            "\\multicolumn{2}{c}{Entry correlation} & \\multicolumn{2}{c}{Share inside $S$}\\\\",
            "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(l){8-9}",
            "Cohort & between & within & between & within & between & within & between & within\\\\",
            "\\midrule"]
    for cohort, label in (("hao", "Confirmatory blood (8 donors)"), ("colon", "Colon (12 donors)")):
        q, d = p2["summary"][cohort], mdiag["summary"][cohort]
        rows.append(f"{label} & {num(q['between_share']['mean'], 2)} & {num(q['within_share']['mean'], 2)} & "
                    f"{num(q['E_between']['mean'], 2)} & {num(q['E_within']['mean'], 2)} & "
                    f"{num(q['r_between'], 2)} & {num(q['r_within'], 2)} & "
                    f"{num(d['rho_between']['mean'], 2)} & {num(d['rho_within']['mean'], 2)}\\\\")
    rows += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(rows) + "\n"


def table(summary):
    rows = ["% Generated by extension/cross_study/make_assets.py. Do not edit.",
            "\\begin{tabular}{@{}lcccccc@{}}", "\\toprule",
            " & \\multicolumn{3}{c}{Confirmatory blood cohort (8 donors)} & \\multicolumn{3}{c}{Colon (12 donors)}\\\\",
            "\\cmidrule(lr){2-4}\\cmidrule(l){5-7}",
            "Arm & All cells & 8 types & 31 types & All cells & 14 types & Entry corr.\\\\", "\\midrule"]
    for arm, label in TABLE_ARMS:
        h, c = summary["hao"]["mean_E"], summary["colon"]["mean_E"]
        corr = summary["colon"]["mean_entry_correlation"]["total"][arm]
        rows.append(f"{label} & {num(h['total'][arm])} & {num(h['within_l1'][arm])} & {num(h['within_l2'][arm])}"
                    f" & {num(c['total'][arm])} & {num(c['within_coarse'][arm])} & {num(corr, 2)}\\\\")
    rows += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(rows) + "\n"


def continuation_macros(rigor, path):
    """Post hoc continuation of rank-8 fits that stopped at the iteration limit
    (colon_rigor_continuation.py); empty when every fit converged."""
    stopped = [d for d in rigor if not rigor[d]["pf_moment"]["converged"]]
    if not stopped:
        return {"XsRigContinued": "0"}
    cont = json.loads(path.read_text())
    assert sorted(cont) == sorted(stopped), "continuation file does not match the unconverged fits"
    change = [abs(cont[d]["continued_E"] - cont[d]["reported_E"]) for d in stopped]
    rebuilt = [abs(cont[d]["rebuilt_E"] - cont[d]["reported_E"]) for d in stopped]
    assert max(rebuilt) < 1e-5, "rebuilt fits do not reproduce the reported errors"
    bound = np.ceil(max(change) * 1000) / 1000                           # "less than" bound, 3 decimals
    return {"XsRigContinued": str(len(stopped)),
            "XsRigContinueConverged": str(sum(cont[d]["continued_converged"] for d in stopped)),
            "XsRigContinueMaxChange": num(max(change), 4),
            "XsRigContinueBound": num(bound if bound > max(change) else bound + 0.001, 3),
            "XsRigContinueMaxWChange": num(max(cont[d]["relative_change_W"] for d in stopped), 3)}


def rigor_table(rigor):
    rows = ["% Generated by extension/cross_study/make_assets.py. Do not edit.",
            "\\begin{tabular}{@{}lrrrrrrr@{}}", "\\toprule",
            "Recipient & Biopsies & Penalty & Means $E$ & $p$ & Rank-8 $E$ & $p$ & $\\dim M$\\\\", "\\midrule"]
    for d in sorted(rigor, key=lambda k: int(k.split("HS")[1])):
        r = rigor[d]
        conv = "" if r["pf_moment"]["converged"] else "$^*$"
        rows.append(f"{d.replace('XAUT1-', '')} & {r['training_biopsies']} & {r['pf_means']['scale']:g} & "
                    f"{r['pf_means']['E']:.3f} & {pval(r['pf_means']['p'])} & {r['pf_moment']['E']:.3f}{conv} & "
                    f"{pval(r['pf_moment']['p'])} & {r['identifiability']['dimension_M']}\\\\")
    rows += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(rows) + "\n"


def results_md(m, summary):
    lines = ["# Cross-study pairing-free transfer: results", "",
             "Generated by `make_assets.py` from the result files. Plan: `PLAN.md` "
             f"(SHA-256 `{summary['plan_sha256']}`).", "",
             "## Success criteria (PLAN.md)", "",
             "| Cohort | H1 permutation p | H2 share of paired improvement (95% CI) | H3 within-type E, pf_means (95% CI) | H4 closed form beats |",
             "|---|---|---|---|---|"]
    for cohort in ("hao", "colon"):
        s = summary[cohort]
        beats = [b for b in ("reference_regression", "transferred_correlation", "transferred_covariance",
                             "variances_only") if s["H4"][b]["success"]]
        lines.append(f"| {cohort} | {pval(s['H1']['p'])} ({'met' if s['H1']['success'] else 'not met'}) | "
                     f"{100 * s['H2']['mean_share']:.0f}% ({100 * s['H2']['bootstrap_95'][0]:.0f}-"
                     f"{100 * s['H2']['bootstrap_95'][1]:.0f}%; {'met' if s['H2']['success'] else 'not met'}) | "
                     f"{s['H3']['pf_means']['mean_E']:.3f} ({s['H3']['pf_means']['bootstrap_95'][0]:.3f}-"
                     f"{s['H3']['pf_means']['bootstrap_95'][1]:.3f}; {'met' if s['H3']['success'] else 'not met'}) | "
                     f"{', '.join(beats) if beats else 'none'} |")
    lines += ["", "## Mean relative error by arm", "",
              "| Arm | Blood: all | Blood: l1 types | Blood: l2 types | Colon: all | Colon: types |",
              "|---|---|---|---|---|---|"]
    for arm, label in TABLE_ARMS:
        h, c = summary["hao"]["mean_E"], summary["colon"]["mean_E"]
        lines.append(f"| {label} | {h['total'][arm]:.3f} | {h['within_l1'][arm]:.3f} | {h['within_l2'][arm]:.3f} | "
                     f"{c['total'][arm]:.3f} | {c['within_coarse'][arm]:.3f} |")
    lines += ["", "## All macros", "", "| Macro | Value |", "|---|---|"]
    lines += [f"| `\\{k}` | {v} |" for k, v in m.items()]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--results-md", type=Path, default=R / "RESULTS.md")
    args = parser.parse_args()
    summary, ident, curve = load("summary.json"), load("identifiability.json"), load("learning_curve.json")
    rigor, fit, scores = load("colon_rigor.json"), load("reference_fit.json"), load("scores.json")
    info = load("prediction_info.json")
    seals = {"hao": json.loads((HERE / "hao_seal.json").read_text()),
             "colon": json.loads((HERE / "seal.json").read_text())}
    m = macros(summary, ident, curve, rigor, fit, scores, info, seals)
    p1, p2, mdiag = load("posthoc_within.json"), load("posthoc_mixture.json"), load("mixture_diagnostics.json")
    m.update(mixture_macros(p1, p2, mdiag))
    m.update(continuation_macros(rigor, R / "colon_rigor_continuation.json"))
    args.out.mkdir(parents=True, exist_ok=True)
    feats = json.loads((R / "features.json").read_text())
    entries = entry_means(scores)
    m.update(entry_macros(entries, feats["genes"], feats["proteins"]))
    (args.out / "cross_macros.tex").write_text(
        "% Generated by extension/cross_study/make_assets.py. Do not edit.\n"
        + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in m.items()))
    (args.out / "cross_figure.tex").write_text(figure(summary, curve, entries, feats["genes"], feats["proteins"]))
    (args.out / "entries_figure.tex").write_text(figure_entries(entries, feats["genes"], feats["proteins"]))
    compat = [r for name in ("noise_simulation.json", "noise_simulation_gene.json")
              for r in json.loads((HERE.parent / "recoverability/results" / name).read_text())["rows"]]
    (args.out / "identification_figure.tex").write_text(figure_identification(ident, curve, p2, mdiag, compat))
    (args.out / "cross_table.tex").write_text(table(summary))
    (args.out / "mixture_table.tex").write_text(mixture_table(p1, p2, mdiag))
    (args.out / "cross_rigor_table.tex").write_text(rigor_table(rigor))
    args.results_md.write_text(results_md(m, summary))


if __name__ == "__main__":
    main()
