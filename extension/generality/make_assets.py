#!/usr/bin/env python3
"""Manuscript assets of the generality benchmark: macros, table and figure (revision/source/gen_*.tex) and
RESULTS.md, from results/<dataset>.json (frozen rule), results/<dataset>_amend1.json (amendments 1-2),
results/summary*.json and results/law*.json.

    python make_assets.py [--out ../../revision/source] [--results-md RESULTS.md]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
PRIMARY = ("hao", "stephenson", "bmmc_cite", "colon", "bmmc_multiome", "scala_m1", "gouwens_visp", "banc", "malecns")
TAG = {"hao": "Hao", "stephenson": "Stephenson", "bmmc_cite": "BmCite", "colon": "Colon",
       "bmmc_multiome": "BmMulti", "scala_m1": "Scala", "gouwens_visp": "Gouwens", "banc": "Banc",
       "malecns": "MaleCns", "banc_crossanimal": "BancX"}
LABEL = {"hao": "Blood (8 donors)", "stephenson": "Blood, COVID-19 (118 patients)",
         "bmmc_cite": "Bone marrow (9 donors)", "colon": "Colon (12 patients)",
         "bmmc_multiome": "Bone marrow multiome (10 donors)",
         "scala_m1": "Motor cortex (263 mice)", "gouwens_visp": "Visual cortex (362 days)",
         "banc": "Fly CNS, female (913 types)", "malecns": "Fly CNS, male (1,010 types)"}
SYSTEM = {"hao": "RNA--protein", "stephenson": "RNA--protein", "bmmc_cite": "RNA--protein", "colon": "RNA--protein",
          "bmmc_multiome": "RNA--chromatin", "scala_m1": "RNA--electrophysiology (Patch-seq)",
          "gouwens_visp": "RNA--electrophysiology (Patch-seq)", "banc": "Brain--nerve-cord wiring (connectomes)",
          "malecns": "Brain--nerve-cord wiring (connectomes)"}
UNITS = {"hao": "donors", "stephenson": "patients", "bmmc_cite": "donors", "colon": "patients",
         "bmmc_multiome": "donors", "scala_m1": "mice", "gouwens_visp": "days", "banc": "types", "malecns": "types"}
ARMNAME = {"bjs2/mapped": "proposed", "selftune/mapped": "self-tuning", "ridge_p/mapped": "ref.\\ regr.\\ + map",
           "ridge_u/mapped": "ref.\\ regr.\\ + map", "ridge_p/percond": "ref.\\ regression",
           "ridge_u/percond": "ref.\\ regression", "ridge_p/pooled": "ref.\\ regression",
           "ridge_u/pooled": "ref.\\ regression", "semicca/percond": "SemiCCA", "semicca/pooled": "SemiCCA",
           "semicca/mapped": "SemiCCA + map", "lowrank/pooled": "none (low rank predicts zero)"}
CURVE_PANELS = ("bmmc_multiome", "gouwens_visp", "banc")       # fixed before any result: one per non-CITE system
PAIRED_ONLY = [f"{d}/{m}" for d in ("js", "scose", "fcose", "lowrank") for m in ("pooled", "percond")]
SEMICCA = ["semicca/pooled", "semicca/percond"]
PROPOSED, SELFTUNED = "bjs2/mapped", "selftune/mapped"
SANS = "font=\\sffamily"
COMMON = (" axis x line*=bottom,axis y line*=left,tick align=outside,"
          "ymajorgrids=true,grid style={gray!18},axis line style={gray!55},"
          "tick label style={font=\\sffamily\\scriptsize},label style={font=\\sffamily\\small},clip=false,"
          "title style={font=\\sffamily\\small,at={(0,1)},anchor=base west,yshift=5pt}")
STYLES = {"prop": "color=propcol,line width=1.1pt,mark=*,mark size=1.5pt",
          "self": "color=selfcol,line width=.8pt,dashed,mark=*,mark size=1.2pt",
          "paired": "color=pairedcol,line width=.8pt,mark=square*,mark size=1.3pt",
          "semi": "color=semicol,line width=.8pt,mark=triangle*,mark size=1.6pt"}


def load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def fac(v):
    if v < 1.15:
        return f"{v:.2f}"
    return f"{v:.1f}"


def pct(v):
    p = 100 * v
    s = f"{p:.1f}" if abs(p) < 10 else f"{p:.0f}"
    return s.replace("-", "\\ensuremath{-}")


def cells(v, rel):
    t = f"{v:,.0f}"
    return {"=": t, ">": f"$>${t}", "<=": f"$\\leq${t}"}[rel]


def facrel(v):
    """A saving with its censoring: 'at least' when the comparator never reached the target."""
    pre = "$\\geq$" if v["comparator_relation"] == ">" else ("$\\leq$" if v["proposed_relation"] == ">" else "")
    return pre + fac(v["factor"])


def envelope(rf, arms):
    return np.nanmax(np.array([rf[a] for a in arms if a in rf]), axis=0)


def macros(res, am, summ, summ_a, law, law_a, xa, xa_a):
    m = {}
    for n in PRIMARY:
        t = TAG[n]
        if n in res:
            r = res[n]
            m[f"GenUnits{t}"] = f"{r['units']:,}"
            m[f"GenCells{t}"] = f"{r['cells']:,}"
            m[f"GenRef{t}"] = pct(r["reference_rf_paired_all"])
            v = r["savings"]["paired_only"]["0.5"]
            m[f"GenFrozenSave{t}"] = fac(v["factor"])
            b = r["budgets"]
            m[f"GenMaxBudget{t}"] = f"{b[-1]:,}"
            if 100 in b:
                i = b.index(100)
                m[f"GenRfHundred{t}"] = pct(r["rf"][PROPOSED][i])
                m[f"GenPairedOnlyHundred{t}"] = pct(envelope(r["rf"], PAIRED_ONLY)[i])
                m[f"GenSemiHundred{t}"] = pct(envelope(r["rf"], SEMICCA)[i])
            m[f"GenRfMax{t}"] = pct(r["rf"][PROPOSED][-1])
            m[f"GenPairedOnlyMax{t}"] = pct(envelope(r["rf"], PAIRED_ONLY)[-1])
        if n in am:
            a = am[n]
            m[f"GenLevel{t}"] = pct(a["level"])
            for key, lab in (("paired_only", "PairedOnly"), ("semicca", "Semi")):
                v = a["savings"][key]["0.5"]
                m[f"GenSave{lab}{t}"] = facrel(v)
                m[f"GenSave{lab}{t}Lo"] = fac(v["ci"][0])
                m[f"GenSave{lab}{t}Hi"] = fac(v["ci"][1])
            v = a["savings"]["paired_only"]["0.5"]
            m[f"GenCellsProposed{t}"] = cells(v["proposed_cells"], v["proposed_relation"])
            m[f"GenCellsPairedOnly{t}"] = cells(v["comparator_cells"], v["comparator_relation"])
            m[f"GenSelf{t}"] = fac(a["selftune_vs_fixed"]["0.5"]["factor"])
    if summ:
        g = summ["G1_paired_only/0.5"]
        m["GenFrozenGOne"], m["GenFrozenGOneLo"], m["GenFrozenGOneHi"] = fac(g["geometric_mean"]), fac(g["ci"][0]), fac(g["ci"][1])
        m["GenFrozenGOneMet"] = "met" if summ["verdicts"]["G1_met"] else "not met"
        g = summ["G2_semicca/0.5"]
        m["GenFrozenGTwo"], m["GenFrozenGTwoLo"], m["GenFrozenGTwoHi"] = fac(g["geometric_mean"]), fac(g["ci"][0]), fac(g["ci"][1])
        m["GenFrozenGTwoMet"] = "met" if summ["verdicts"]["G2_met"] else "not met"
        m["GenFrozenNTrivial"] = str(sum(abs(v - 1) < 1e-9 for v in summ["G1_paired_only/0.5"]["per_dataset"].values()))
        m["GenRefMax"] = pct(max(res[n]["reference_rf_paired_all"] for n in res))
    if summ_a:
        for scope, sc in (("informative", ""), ("all", "All")):
            for key, lab in (("G1_paired_only/0.5", "GOne"), ("G2_semicca/0.5", "GTwo"),
                             ("G3_selftune_vs_fixed/0.5", "GThree"), ("original_forms/0.5", "GOriginal"),
                             ("G1_paired_only/0.25", "GOneQuarter"), ("G1_paired_only/0.75", "GOneThreeQuarters")):
                v = summ_a[f"{scope}/{key}"]
                m[f"Gen{lab}{sc}"] = fac(v["geometric_mean"])
                m[f"Gen{lab}{sc}Lo"] = fac(v["ci"][0])
                m[f"Gen{lab}{sc}Hi"] = fac(v["ci"][1])
        vd = summ_a["verdicts"]
        m["GenGOneMet"] = "met" if vd["G1_amended_met"] else "not met"
        m["GenGTwoMet"] = "met" if vd["G2_amended_met"] else "not met"
        m["GenGOneAllMet"] = "met" if vd["G1_amended_all_met"] else "not met"
        m["GenGTwoAllMet"] = "met" if vd["G2_amended_all_met"] else "not met"
        inf = summ_a["informative"]
        m["GenNInformative"] = str(len(inf))
        m["GenNotInformative"] = ", ".join(LABEL[n].split(" (")[0].lower() for n in summ_a["not_informative"]) or "none"
        g1 = summ_a["informative/G1_paired_only/0.5"]
        m["GenNLbAboveOne"] = str(len(g1["lower_bound_above_one"]))
        m["GenNLbAboveOneSemi"] = str(len(summ_a["informative/G2_semicca/0.5"]["lower_bound_above_one"]))
        vals = list(g1["per_dataset"].values())
        m["GenSaveMin"], m["GenSaveMax"] = fac(min(vals)), fac(max(vals))
        g3 = summ_a["informative/G3_selftune_vs_fixed/0.5"]["per_dataset"]
        m["GenSelfBetter"] = str(sum(v > 1 for v in g3.values()))
    for law_, sfx in ((law, "Frozen"), (law_a, "")):
        if law_:
            m[f"GenLaw{sfx}Rho"] = f"{law_['spearman']:.2f}"
            p = law_["p_one_sided"]
            m[f"GenLaw{sfx}P"] = f"{p:.3f}" if p >= 0.001 else "$<$0.001"
            m[f"GenLaw{sfx}N"] = str(len(law_["datasets"]))
            m[f"GenLaw{sfx}Met"] = "met" if law_.get("G4_met", law_.get("G4_amended_met")) else "not met"
    lc = load(RES / "law_concordance_posthoc.json")
    if lc:
        m["GenLawC"] = f"{lc['concordance']:.2f}"
        m["GenLawCP"] = f"{lc['p_one_sided']:.3f}" if lc["p_one_sided"] >= 0.001 else "$<$0.001"
        m["GenLawCPairs"] = str(lc["usable_pairs"])
        cen = [n for n, c in lc["censored"].items() if c]
        unc = [n for n, c in lc["censored"].items() if not c]
        m["GenLawNCensored"] = str(len(cen))
        m["GenLawNUncensored"] = str(len(unc))
        r = [lc["observed_over_predicted"][n] for n in unc]
        m["GenLawRatioMin"], m["GenLawRatioMax"] = f"{min(r):.1f}", f"{max(r):.1f}"
        m["GenLawUncensoredRho"] = f"{lc['uncensored_spearman']:.2f}"
        m["GenLawNConsistent"] = str(sum(lc["censored_consistent"].values()))
    if xa_a:
        v = xa_a["savings"]["paired_only"]["0.5"]
        m["GenSavePairedOnlyBancX"] = facrel(v)
        m["GenSavePairedOnlyBancXLo"] = fac(v["ci"][0])
        m["GenSavePairedOnlyBancXHi"] = fac(v["ci"][1])
        m["GenLevelBancX"] = pct(xa_a["level"])
    if xa:
        b = xa["budgets"]
        if 100 in b:
            i = b.index(100)
            m["GenRfHundredBancX"] = pct(xa["rf"][PROPOSED][i])
            m["GenPairedOnlyHundredBancX"] = pct(envelope(xa["rf"], PAIRED_ONLY)[i])
            # eighth session (correction): comparators that kept positive recovery under the cross-animal shift
            ridge = [a for a in xa["rf"] if a.startswith(("ridge_p/", "ridge_u/"))]
            m["GenRidgeHundredBancX"] = pct(envelope(xa["rf"], ridge)[i])
            m["GenSemiHundredBancX"] = pct(envelope(xa["rf"], SEMICCA + ["semicca/mapped"])[i])
            m["GenSelfPooledHundredBancX"] = pct(xa["rf"]["selftune/pooled"][i])
        m["GenPropMaxBancX"] = pct(max(xa["rf"][PROPOSED]))
        m["GenPropPooledMaxBancX"] = pct(max(xa["rf"]["bjs2/pooled"]))
        m["GenRidgeMinBancX"] = pct(min(envelope(xa["rf"], [a for a in xa["rf"] if a.startswith(("ridge_p/", "ridge_u/"))])))
    return m


WORDS = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
         10: "ten", 11: "eleven", 12: "twelve"}


def words(m):
    """Counts below 13 spelled out, as in the manuscript's text."""
    for k in ("GenFrozenNTrivial", "GenNInformative", "GenNLbAboveOne", "GenNLbAboveOneSemi", "GenLawNCensored",
              "GenLawNUncensored", "GenLawNConsistent", "GenSelfBetter", "GenNData"):
        if k in m and m[k].isdigit() and int(m[k]) in WORDS:
            m[k] = WORDS[int(m[k])]
    return m


def write_macros(path, m):
    m = words(dict(m))
    lines = ["% Generated by extension/generality/make_assets.py. Do not edit."]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(m.items())]
    Path(path).write_text("\n".join(lines) + "\n")


def table(res, am, summ_a):
    inf = set(summ_a["informative"]) if summ_a else set(am)
    rows = ["% Generated by extension/generality/make_assets.py. Do not edit.",
            "\\begin{tabular}{@{}lrrlrrlll@{}}", "\\toprule",
            "Data set & Cells & Best (\\%) & Reached by & \\multicolumn{2}{c}{Paired cells to half of best} & "
            "Saving vs & Saving vs & Prespecified\\\\", "\\cmidrule(lr){5-6}",
            " & & & & Proposed & Paired only & paired only & SemiCCA & rule\\\\", "\\midrule"]
    last = None
    for n in PRIMARY:
        if n not in am:
            continue
        r, a = res[n], am[n]
        if SYSTEM[n] != last:
            rows.append(f"\\multicolumn{{9}}{{@{{}}l}}{{\\textit{{{SYSTEM[n]}}}}}\\\\")
            last = SYSTEM[n]
        po, se = a["savings"]["paired_only"]["0.5"], a["savings"]["semicca"]["0.5"]
        frozen = fac(r["savings"]["paired_only"]["0.5"]["factor"])
        if n in inf:
            s1 = f"{facrel(po)} ({fac(po['ci'][0])}--{fac(po['ci'][1])})"
            s2 = f"{facrel(se)} ({fac(se['ci'][0])}--{fac(se['ci'][1])})"
            c1, c2 = cells(po["proposed_cells"], po["proposed_relation"]), cells(po["comparator_cells"], po["comparator_relation"])
        else:
            s1 = s2 = "n.e."
            c1 = c2 = "--"
        rows.append(f"{LABEL[n]} & {r['cells']:,} & {pct(a['level'])} & {ARMNAME.get(a['level_arm'], a['level_arm'].replace('_', ' '))} & "
                    f"{c1} & {c2} & {s1} & {s2} & {frozen}\\\\")
    rows += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(rows) + "\n"


def curve(xs, ys, style, lo=-20):
    pts = " ".join(f"({x},{max(100 * y, lo):.2f})" for x, y in zip(xs, ys) if y == y)
    return [f"\\addplot[{style}] coordinates {{ {pts} }};"]


def curve_panel(r, a, name, at, title, ylabel=True):
    b, rf = r["budgets"], r["rf"]
    xt = ",".join(str(x) for x in b)
    ymax = max(20, int(np.ceil(100 * max(max(rf[PROPOSED]), a["level"]) / 20.0)) * 20)
    t = 50 * a["level"]
    lines = [f"\\begin{{axis}}[{at}name={name},scale only axis,width=.2\\textwidth,height=3.2cm,xmode=log,log basis x=2,",
             f" xmin={b[0] * .8},xmax={b[-1] * 1.25},ymin=-20,ymax={ymax},xtick={{{xt}}},xticklabels={{{xt}}},",
             " xlabel={Paired cells}," + ("ylabel={Recovered (\\%)}," if ylabel else ""),
             " x tick label style={font=\\sffamily\\tiny},",
             f" title={{{title}}},", COMMON + "]",
             f"\\draw[dashed,black!45] (axis cs:{b[0] * .8},{t:.1f}) -- (axis cs:{b[-1] * 1.25},{t:.1f});"]
    lines += curve(b, rf[PROPOSED], STYLES["prop"])
    lines += curve(b, envelope(rf, PAIRED_ONLY), STYLES["paired"])
    lines += curve(b, envelope(rf, SEMICCA), STYLES["semi"])
    lines.append("\\end{axis}")
    return lines


def forest(am, summ_a):
    inf = summ_a["informative"] if summ_a else [n for n in PRIMARY if n in am]
    names = [n for n in PRIMARY if n in am]
    k = len(names)
    lo_x, hi_x = 0.4, 16
    lines = [f"\\begin{{axis}}[name=a,scale only axis,width=.55\\textwidth,height=4.6cm,xmode=log,log basis x=2,",
             f" xmin={lo_x},xmax={hi_x},ymin=-.6,ymax={k + .6},xtick={{0.5,1,2,4,8,16}},xticklabels={{0.5,1,2,4,8,16}},",
             " ytick={" + ",".join(str(i) for i in range(k + 1)) + "},",
             " yticklabels={{\\textbf{Geometric mean}}," + ",".join("{" + LABEL[n] + "}" for n in reversed(names)) + "},",
             " xlabel={Paired cells saved (ratio)},",
             " title={\\textbf{a}\\enspace Paired cells saved},",
             COMMON.replace("ymajorgrids=true", "xmajorgrids=true") + ",y tick style={draw=none},"
             "yticklabel style={font=\\sffamily\\scriptsize}]",
             f"\\draw[black!50] (axis cs:1,-.6) -- (axis cs:1,{k + .6});"]
    for key, style, dy in (("paired_only", "color=pairedcol,mark=*,mark size=1.6pt", .14),
                           ("semicca", "color=semicol,mark=triangle*,mark size=1.8pt", -.14)):
        col = style.split(",")[0]
        for y, n in enumerate(reversed(names), start=1):
            if n not in inf:
                if key == "paired_only":
                    lines.append(f"\\node[font=\\sffamily\\tiny,color=black!60,anchor=west] at (axis cs:1.05,{y:.2f}) {{not estimable}};")
                continue
            v = am[n]["savings"][key]["0.5"]
            lo, hi = max(v["ci"][0], lo_x), min(v["ci"][1], hi_x)
            lines.append(f"\\draw[{col},line width=.7pt] (axis cs:{lo:.3f},{y + dy:.2f}) -- (axis cs:{hi:.3f},{y + dy:.2f});")
            mark = style if v["comparator_relation"] != ">" else style.replace("mark=*", "mark=o").replace("triangle*", "triangle")
            lines.append(f"\\addplot[only marks,{mark}] coordinates {{ ({min(max(v['factor'], lo_x), hi_x):.3f},{y + dy:.2f}) }};")
        if summ_a:
            g = summ_a[{"paired_only": "informative/G1_paired_only/0.5", "semicca": "informative/G2_semicca/0.5"}[key]]
            lines.append(f"\\draw[{col},line width=1pt] (axis cs:{g['ci'][0]:.3f},{dy:.2f}) -- (axis cs:{g['ci'][1]:.3f},{dy:.2f});")
            lines.append(f"\\addplot[only marks,{col},mark=diamond*,mark size=2.6pt] coordinates {{ ({g['geometric_mean']:.3f},{dy:.2f}) }};")
    lines.append("\\end{axis}")
    return lines


def law_panel(law, at):
    names = law["datasets"]
    xs = [law["predicted"][n] for n in names]
    ys = [law["observed"][n] for n in names]
    lo = min(xs + ys) / 1.4
    hi = max(xs + ys) * 1.4
    short = {"hao": "Blood", "stephenson": "COVID", "bmmc_cite": "Marrow", "colon": "Colon", "bmmc_multiome": "Multiome",
             "scala_m1": "M1", "gouwens_visp": "VISp", "banc": "BANC", "malecns": "Male CNS", "frangieh": "Melanoma",
             "papalexi": "Monocyte", "overcite": "OverCITE"}
    ticks = [t for t in (0.25, 0.5, 1, 2, 4, 8, 16, 32, 64) if lo <= t <= hi]
    tt = ",".join(f"{t:g}" for t in ticks)
    lc = load(RES / "law_concordance_posthoc.json")
    cens = lc["censored"] if lc else {}
    lines = [f"\\begin{{axis}}[{at}name=b,scale only axis,width=.22\\textwidth,height=4.4cm,xmode=log,ymode=log,",
             f" xmin={lo:.3f},xmax={hi:.3f},ymin={lo:.3f},ymax={hi:.3f},log basis x=2,log basis y=2,",
             f" xtick={{{tt}}},xticklabels={{{tt}}},ytick={{{tt}}},yticklabels={{{tt}}},",
             " xlabel={Predicted saving},ylabel={Observed saving},",
             " title={\\textbf{b}\\enspace Predicted and observed savings},",
             COMMON.replace("ymajorgrids=true", "xmajorgrids=true,ymajorgrids=true") + "]",
             f"\\draw[black!35,dashed] (axis cs:{lo:.3f},{lo:.3f}) -- (axis cs:{hi:.3f},{hi:.3f});"]
    for n, x, y in zip(names, xs, ys):
        col = "propcol" if n in PRIMARY else "semicol"
        mk = "mark=o" if cens.get(n) else "mark=*"
        if cens.get(n):
            lines.append(f"\\draw[->,color={col},line width=.5pt] (axis cs:{x:.3f},{y:.3f}) -- (axis cs:{x:.3f},{y * 1.35:.3f});")
        lines.append(f"\\addplot[only marks,color={col},{mk},mark size=1.6pt] coordinates {{ ({x:.3f},{y:.3f}) }};")
        lines.append(f"\\node[font=\\sffamily\\tiny,anchor=west,color=black!70] at (axis cs:{x * 1.08:.3f},{y:.3f}) {{{short.get(n, n)}}};")
    lines.append("\\end{axis}")
    return lines


def figure(res, am, summ_a, law_a):
    lines = ["% Generated by extension/generality/make_assets.py. Do not edit.",
             "\\definecolor{propcol}{HTML}{217D91}", "\\definecolor{pairedcol}{HTML}{5D6670}",
             "\\definecolor{semicol}{HTML}{C9803A}",
             f"\\begin{{tikzpicture}}[{SANS}]"]
    lines += forest(am, summ_a)
    titles = {"bmmc_multiome": "\\textbf{b}\\enspace RNA--chromatin", "gouwens_visp": "\\textbf{c}\\enspace Patch-seq",
              "banc": "\\textbf{d}\\enspace Fly connectome"}
    prev = None
    for n in CURVE_PANELS:
        if n not in res or n not in am:
            continue
        name = {"bmmc_multiome": "b", "gouwens_visp": "c", "banc": "d"}[n]
        at = ("at={(a.outer south west)},anchor=outer north west,yshift=-.3cm," if prev is None
              else f"at={{({prev}.outer north east)}},anchor=outer north west,xshift=.1cm,")
        lines += curve_panel(res[n], am[n], name, at, titles[n], ylabel=prev is None)
        prev = name
    if prev:
        lines += ["\\begin{axis}[hide axis,scale only axis,width=1pt,height=1pt,xmin=0,xmax=1,ymin=0,ymax=1,"
                  "at={(c.outer south)},anchor=north,yshift=-2pt,legend style={draw=none,"
                  "font=\\sffamily\\scriptsize,at={(0.5,0.5)},anchor=north,legend columns=3,column sep=5pt,"
                  "cells={anchor=west}}]",
                  f"\\addlegendimage{{{STYLES['prop']}}}", "\\addlegendentry{Proposed}",
                  f"\\addlegendimage{{{STYLES['paired']}}}", "\\addlegendentry{Paired cells only}",
                  f"\\addlegendimage{{{STYLES['semi']}}}", "\\addlegendentry{SemiCCA}", "\\end{axis}"]
    lines.append("\\end{tikzpicture}")
    return "\n".join(lines) + "\n"


def results_md(res, am, summ, summ_a, law, law_a, xa, xa_a):
    L = ["# Generality benchmark: results", "",
         "Frozen plan: `PLAN.md` (hashes in `results/freeze.json`); amendments and deviations: `DEVIATIONS.md`. Savings "
         "are cells the comparator envelope needs divided by cells the proposed estimator needs, with 95% intervals "
         "from 2,000 bootstrap resamples of held-out units.", "",
         "## Prespecified rule (targets relative to the fully paired reference)", ""]
    if summ:
        for key, lab in (("G1_paired_only/0.5", "G1"), ("G2_semicca/0.5", "G2"), ("G3_selftune_vs_fixed/0.5", "G3")):
            v = summ[key]
            L.append(f"* {lab}: geometric mean {v['geometric_mean']:.2f} (95% CI {v['ci'][0]:.2f}-{v['ci'][1]:.2f}); "
                     f"per data set {json.dumps({k: round(x, 2) for k, x in v['per_dataset'].items()})}")
        L.append(f"* Verdicts: {json.dumps(summ['verdicts'])}")
    if law:
        L.append(f"* G4: Spearman {law['spearman']:.2f}, one-sided p = {law['p_one_sided']:.4f}")
    L += ["", "## Amendments 1-2 (targets relative to the best recovery any method reached)", ""]
    if summ_a:
        L.append(f"* Informative data sets: {', '.join(summ_a['informative'])}; not informative: "
                 f"{', '.join(summ_a['not_informative']) or 'none'}")
        for scope in ("informative", "all"):
            for key, lab in (("G1_paired_only/0.5", "G1'"), ("G2_semicca/0.5", "G2'"), ("G3_selftune_vs_fixed/0.5", "G3'")):
                v = summ_a[f"{scope}/{key}"]
                L.append(f"* {lab} ({scope}): geometric mean {v['geometric_mean']:.2f} (95% CI {v['ci'][0]:.2f}-{v['ci'][1]:.2f})")
        L.append(f"* Verdicts: {json.dumps(summ_a['verdicts'])}")
    if law_a:
        L.append(f"* G4': Spearman {law_a['spearman']:.2f} over {len(law_a['datasets'])} data sets, one-sided p = "
                 f"{law_a['p_one_sided']:.4f}")
    L += ["", "| Data set | Units | Cells | Reference (%) | Best (%) by | Proposed cells | Paired-only cells | Saving (95% CI) | "
              "vs SemiCCA (95% CI) | Self-tuned vs fixed | Prespecified-rule saving |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for n in list(PRIMARY) + ["banc_crossanimal"]:
        r = res.get(n) if n in PRIMARY else xa
        a = am.get(n) if n in PRIMARY else xa_a
        if not r or not a:
            continue
        po, se, stf = a["savings"]["paired_only"]["0.5"], a["savings"]["semicca"]["0.5"], a["selftune_vs_fixed"]["0.5"]
        L.append(f"| {n} | {r['units']} | {r['cells']} | {100 * r['reference_rf_paired_all']:.1f} | {100 * a['level']:.1f} "
                 f"{a['level_arm']} | {po['proposed_relation']}{po['proposed_cells']:.0f} | "
                 f"{po['comparator_relation']}{po['comparator_cells']:.0f} | {po['factor']:.2f} ({po['ci'][0]:.2f}-{po['ci'][1]:.2f}) | "
                 f"{se['factor']:.2f} ({se['ci'][0]:.2f}-{se['ci'][1]:.2f}) | {stf['factor']:.2f} ({stf['ci'][0]:.2f}-{stf['ci'][1]:.2f}) | "
                 f"{r['savings']['paired_only']['0.5']['factor']:.2f} |")
    L += ["", "Recovered fraction (%) by budget:", ""]
    for n in list(PRIMARY) + ["banc_crossanimal"]:
        r = res.get(n) if n in PRIMARY else xa
        if not r:
            continue
        L.append(f"* {n} (budgets {', '.join(map(str, r['budgets']))}): proposed "
                 + " / ".join(f"{100 * v:.1f}" for v in r["rf"][PROPOSED]) + "; self-tuning "
                 + " / ".join(f"{100 * v:.1f}" for v in r["rf"][SELFTUNED]) + "; paired only "
                 + " / ".join(f"{100 * v:.1f}" for v in envelope(r["rf"], PAIRED_ONLY)) + "; SemiCCA "
                 + " / ".join(f"{100 * v:.1f}" for v in envelope(r["rf"], SEMICCA)) + "; bases from paired cells "
                 + " / ".join(f"{100 * v:.1f}" for v in r["rf"]["abl_pairbasis/pooled"]))
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE.parent.parent / "revision" / "source"))
    ap.add_argument("--results-md", default=str(HERE / "RESULTS.md"))
    a = ap.parse_args()
    res = {n: load(RES / f"{n}.json") for n in PRIMARY if (RES / f"{n}.json").exists()}
    am = {n: load(RES / f"{n}_amend1.json") for n in PRIMARY if (RES / f"{n}_amend1.json").exists()}
    summ, summ_a = load(RES / "summary.json"), load(RES / "summary_amend1.json")
    law, law_a = load(RES / "law.json"), load(RES / "law_amend1.json")
    xa, xa_a = load(RES / "banc_crossanimal.json"), load(RES / "banc_crossanimal_amend1.json")
    out = Path(a.out)
    write_macros(out / "gen_macros.tex", macros(res, am, summ, summ_a, law, law_a, xa, xa_a))
    (out / "gen_table.tex").write_text(table(res, am, summ_a))
    (out / "gen_figure.tex").write_text(figure(res, am, summ_a, law_a))
    Path(a.results_md).write_text(results_md(res, am, summ, summ_a, law, law_a, xa, xa_a))
    print(len(res), "data sets;", len(am), "amended")


if __name__ == "__main__":
    main()
