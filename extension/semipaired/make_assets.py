#!/usr/bin/env python3
"""Manuscript assets of the semi-paired study: macros (semi_macros.tex), the main figure (semi_figure.tex), the
external-test table (semi_table.tex) and the development table (semi_dev_table.tex).

    python make_assets.py --out ../../revision/source

Reads results/summary_<dataset>.json (summarize.py), results/<dataset>.json, logs/effective_dimension_*.json,
external/results/overcite.json and external/results/predict_info.json, and ../hybrid/results/summary.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
R = HERE / "results"
EXT = HERE / "external" / "results"
HYB = HERE.parent / "hybrid" / "results"
SANS = "font=\\sffamily"
COMMON = (" axis x line*=bottom,axis y line*=left,tick align=outside,"
          "ymajorgrids=true,grid style={gray!18},axis line style={gray!55},"
          "tick label style={font=\\sffamily\\scriptsize},label style={font=\\sffamily\\small},clip=false,"
          "title style={font=\\sffamily\\small,at={(0,1)},anchor=base west,yshift=5pt}")
PAIRED_ONLY = [f"{d}/{m}" for d in ("js", "scose", "fcose", "lowrank") for m in ("pooled", "percond")]
PROPOSED = "bjs2/mapped"
PLACEHOLDERS = []          # every macro is computed from results
NUM = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six"}


def load(path):
    return json.loads(Path(path).read_text())


def signed(text):
    return "\\ensuremath{-}" + text[1:] if text.startswith("-") else text


def pct(v, digits=None):
    p = 100 * v
    if digits is None:
        digits = 1 if abs(p) < 10 else 0
    return signed(f"{p:.{digits}f}")


def fac(v):
    """Ratios: two decimals near one (0.95-1.15) and below one, one decimal otherwise."""
    if v < 1.15:
        return f"{v:.2f}"
    return f"{v:.1f}"


def envelope(rf, arms, n):
    vals = np.array([rf[a] for a in arms if a in rf], float)
    return np.nanmax(vals, axis=0) if len(vals) else np.full(n, np.nan)


def dev_macros(ds, tag, s):
    rf, b = s["rf"], s["budgets"]
    i100 = b.index(100)
    m = {}
    m[f"Sp{tag}Groups"] = str(s["groups"])
    m[f"Sp{tag}Targets"] = str(s["targets"])
    m[f"Sp{tag}RfHundred"] = pct(rf[PROPOSED][i100])
    m[f"Sp{tag}RfFifty"] = pct(rf[PROPOSED][b.index(50)])
    m[f"Sp{tag}RfFourHundred"] = pct(rf[PROPOSED][b.index(400)])
    env = envelope(rf, PAIRED_ONLY, len(b))
    m[f"Sp{tag}PairedOnlyHundred"] = pct(env[i100])
    m[f"Sp{tag}PairedOnlyFourHundred"] = pct(env[b.index(400)])
    semi = envelope(rf, ["semicca/pooled", "semicca/percond"], len(b))
    m[f"Sp{tag}SemiccaHundred"] = pct(semi[i100])
    ridge = envelope(rf, ["ridge_u/pooled", "ridge_u/percond", "ridge_p/pooled", "ridge_p/percond"], len(b))
    m[f"Sp{tag}RidgeHundred"] = pct(ridge[i100])
    m[f"Sp{tag}OneSidedHundred"] = pct(rf["bjs/mapped"][i100])
    m[f"Sp{tag}PooledHundred"] = pct(rf["bjs2/pooled"][i100])
    m[f"Sp{tag}PercondHundred"] = pct(rf["bjs2/percond"][i100])
    m[f"Sp{tag}PairBasisHundred"] = pct(rf["abl/bjs_pairbasis"][i100])
    m[f"Sp{tag}MarkerHundred"] = pct(rf["abl/bjs_marker"][i100])
    m[f"Sp{tag}MapPairedHundred"] = pct(max(rf["abl/map_paired"][i100], -9.99))
    if "abl/map_measured" in rf:
        m[f"Sp{tag}MapMeasuredHundred"] = pct(rf["abl/map_measured"][i100])
    for n, w in ((500, "FiveHundred"), (2000, "TwoThousand"), (8000, "EightThousand")):
        if f"dose/{n}" in rf:
            m[f"Sp{tag}Dose{w}Hundred"] = pct(rf[f"dose/{n}"][i100])
    m[f"Sp{tag}PairedAll"] = pct(rf["paired_all"][0])
    m[f"Sp{tag}MapGain"] = f"{100 * (rf['bjs2/mapped'][i100] - rf['bjs2/pooled'][i100]):.1f}"
    # is the proposed estimator's uncorrected loss the lowest of all arms at every budget?
    loss = s["loss_rel"]
    others = [a for a in loss if not a.startswith(("bjs", "abl/", "dose/")) and a not in ("paired_all", "independence")]
    m[f"Sp{tag}LossLowestEverywhere"] = "yes" if all(
        loss[PROPOSED][i] <= min(loss[a][i] for a in others) for i in range(len(b))) else "no"
    # savings against paired-only estimation and the comparators in original forms, over targets 0.3-0.6
    for key, lab in (("savings_vs_paired_only", "PairedOnly"), ("savings_vs_published", "Published"),
                     ("savings_vs_best", "Best")):
        sv = s[key]
        facs = [v["factor"] for v in sv.values()]
        m[f"Sp{tag}Save{lab}Min"] = fac(min(facs))
        m[f"Sp{tag}Save{lab}Max"] = fac(max(facs))
        lo = min(v["factor_ci"][0] for v in sv.values())
        m[f"Sp{tag}Save{lab}CiMin"] = fac(lo)
        t4 = sv["0.4"]
        m[f"Sp{tag}Save{lab}Four"] = fac(t4["factor"])
        m[f"Sp{tag}Save{lab}FourLo"] = fac(t4["factor_ci"][0])
        m[f"Sp{tag}Save{lab}FourHi"] = fac(t4["factor_ci"][1])
    sb = s["savings_vs_best"]["0.4"]
    m[f"Sp{tag}CellsFour"] = f"{sb['proposed_cells']:.0f}"
    m[f"Sp{tag}PairedOnlyCellsFour"] = f"{s['savings_vs_paired_only']['0.4']['cells']:.0f}"
    vb = {d["budget"]: d for d in s["versus_best"]}
    m[f"Sp{tag}BestDiffHundred"] = pct(vb[100]["diff"])
    m[f"Sp{tag}BestDiffHundredLo"] = pct(vb[100]["ci"][0])
    m[f"Sp{tag}BestDiffHundredHi"] = pct(vb[100]["ci"][1])
    wins = sum(d["ci"][0] > 0 for d in s["versus_best"])
    m[f"Sp{tag}BestWins"] = str(wins)
    m[f"Sp{tag}Budgets"] = str(len(b))
    ch = s.get("versus_champollion")
    if ch:
        cb = ch["budgets"]
        j = cb.index(100)
        m[f"Sp{tag}ChampHundred"] = pct(ch["champollion"][j])
        m[f"Sp{tag}ChampPropHundred"] = pct(ch["proposed"][j])
        sv = ch["savings"]
        m[f"Sp{tag}ChampSaveFour"] = fac(sv["0.4"]["factor"])
        m[f"Sp{tag}ChampSaveFourLo"] = fac(sv["0.4"]["factor_ci"][0])
        m[f"Sp{tag}ChampSaveFourHi"] = fac(sv["0.4"]["factor_ci"][1])
        facs = [v["factor"] for v in sv.values()]
        m[f"Sp{tag}ChampSaveMin"] = fac(min(facs))
        m[f"Sp{tag}ChampSaveMax"] = fac(max(facs))
        m[f"Sp{tag}ChampWins"] = str(sum(d["ci"][0] > 0 for d in ch["differences"]))
        m[f"Sp{tag}ChampBudgets"] = str(len(cb))
        m[f"Sp{tag}ChampFolds"] = str(ch["folds"])
        m[f"Sp{tag}ChampGroups"] = str(ch["groups"])
    return m


def champ_text(sf, sp):
    """Supplementary sentences on Champollion in development (both screens)."""
    out = []
    ch = sf.get("versus_champollion")
    if ch:
        j = ch["budgets"].index(100)
        sv = ch["savings"]
        facs = [v["factor"] for v in sv.values()]
        lo = min(v["factor_ci"][0] for v in sv.values())
        out.append(f"In Frangieh, Champollion recovered {pct(ch['champollion'][j])}\\% at 100 paired cells against "
                   f"{pct(ch['proposed'][j])}\\% for the proposed estimator, and the proposed estimator needed "
                   f"{fac(min(facs))} to {fac(max(facs))} times fewer paired cells to reach recovered fractions of 0.3 to "
                   f"0.6 (lowest 95\\% bound, {fac(lo)}).")
    ch = sp.get("versus_champollion")
    if ch:
        j = ch["budgets"].index(100)
        sv = ch["savings"]
        r = {t: sv[t]["factor"] for t in ("0.3", "0.4", "0.5", "0.6")}
        ci = {t: sv[t]["factor_ci"] for t in r}
        out.append(f"In Papalexi ({ch['folds']} held-out targets, {ch['groups']} groups), Champollion recovered "
                   f"{pct(ch['champollion'][j])}\\% at 100 paired cells against {pct(ch['proposed'][j])}\\% for the "
                   f"proposed estimator, so it reached recovered fractions of 0.3 and 0.4 with fewer paired cells (ratios "
                   f"{fac(r['0.3'])} and {fac(r['0.4'])}; 95\\% CI, {fac(ci['0.3'][0])}--{fac(ci['0.3'][1])} and "
                   f"{fac(ci['0.4'][0])}--{fac(ci['0.4'][1])}), and the two were similar at 0.5 and 0.6 ({fac(r['0.5'])} "
                   f"and {fac(r['0.6'])}).")
    if out:
        out.append("Champollion's lasso weight was chosen per budget on the evaluation groups, which favours it.")
    return " ".join(out)


def ext_macros(o, info):
    m = {}
    b = o["budgets"]
    rf = o["rf"]
    m["SpExCells"] = f"{o['cells']:,}"
    m["SpExOrfs"] = str(len(o["held_out_orfs"]))
    m["SpExReservoir"] = str(info["n_reservoir"])
    m["SpExPoolPops"] = str(info["n_pool_keys"])
    m["SpExTarget"] = f"{o['target']:.1f}"
    for i, B in enumerate(b):
        w = {25: "TwentyFive", 50: "Fifty", 100: "Hundred", 200: "TwoHundred", 400: "FourHundred"}[B]
        m[f"SpExRf{w}"] = pct(rf[PROPOSED][i])
        m[f"SpExRf{w}Lo"] = pct(o["rf_ci"][PROPOSED][i][0])
        m[f"SpExRf{w}Hi"] = pct(o["rf_ci"][PROPOSED][i][1])
        m[f"SpExPairedOnly{w}"] = pct(envelope(rf, PAIRED_ONLY, len(b))[i])
        m[f"SpExSemi{w}"] = pct(envelope(rf, ["semicca/pooled", "semicca/percond"], len(b))[i])
        m[f"SpExChamp{w}"] = pct(rf["champollion/transport"][i])
        m[f"SpExLoss{w}"] = f"{o['loss_rel'][PROPOSED][i]:.2f}"
    for key, lab in (("paired_only", "PairedOnly"), ("semicca", "Semi"), ("champollion", "Champ"),
                     ("original_forms", "Original"), ("all_comparators", "All")):
        for t, tl in (("0.3", ""), ("0.2", "Two"), ("0.4", "Four")):
            v = o["savings"][key][t]
            m[f"SpExSave{lab}{tl}"] = fac(v["factor"])
            m[f"SpExSave{lab}{tl}Lo"] = fac(v["ci"][0])
            m[f"SpExSave{lab}{tl}Hi"] = fac(v["ci"][1])
            m[f"SpExSave{lab}{tl}Rel"] = {"=": "", ">": "more than ", "<=": "at most "}[v["comparator_relation"]]
            m[f"SpExCells{lab}{tl}"] = f"{v['comparator_cells']:.0f}"
        v = o["savings"][key]["0.3"]
        m["SpExCellsProposed"] = f"{v['proposed_cells']:.0f}"
        m["SpExCellsProposedRel"] = {"=": "", ">": "more than ", "<=": "at most "}[v["proposed_relation"]]
    # the paired cells needed are interpolated: the budgets between which each running-maximum curve crosses the
    # target, with the recovered fractions there
    for arm_rf, lab in ((rf[PROPOSED], "Proposed"), (envelope(rf, PAIRED_ONLY, len(b)), "PairedOnly")):
        env = np.maximum.accumulate(np.asarray(arm_rf, float))
        i = int(np.argmax(env >= o["target"]))
        if i > 0 and env[i] >= o["target"]:
            m[f"SpExCross{lab}LoBudget"] = f"{b[i - 1]:,}"
            m[f"SpExCross{lab}HiBudget"] = f"{b[i]:,}"
            m[f"SpExCross{lab}LoRf"] = pct(env[i - 1])
            m[f"SpExCross{lab}HiRf"] = pct(env[i])
    vd = o["verdicts"]
    m["SpExHOne"] = "met" if vd["H1_vs_paired_only"] else "not met"
    m["SpExHTwo"] = "met" if vd["H2_vs_semicca"] else "not met"
    m["SpExHThree"] = "met" if vd["H3_noninferior_to_champollion"] else "not met"
    vb = {d["budget"]: d for d in o["versus_best"]}
    m["SpExBestHundred"] = vb[100]["best"].replace("_", "\\_")
    m["SpExBestDiffHundred"] = pct(vb[100]["diff"])
    m["SpExBestDiffHundredLo"] = pct(vb[100]["ci"][0])
    m["SpExBestDiffHundredHi"] = pct(vb[100]["ci"][1])
    m["SpExPairedAll"] = pct(o["reference_rf"]["paired_all/all"])
    i100 = b.index(100)
    loss = o["loss_rel"]
    m["SpExLossPairedOnlyHundred"] = f"{min(loss[a][i100] for a in PAIRED_ONLY if a in loss):.2f}"
    m["SpExLossSemiHundred"] = f"{min(loss[a][i100] for a in ('semicca/pooled', 'semicca/percond')):.2f}"
    m["SpExLossChampHundred"] = f"{loss['champollion/transport'][i100]:.2f}"
    m["SpExLossBestOtherHundred"] = f"{min(v[i100] for a, v in loss.items() if not a.startswith('bjs')):.2f}"
    m["SpExLossIndependence"] = f"{o['reference_loss_rel'].get('independence/all', 1.0):.2f}"
    fr = load(EXT / "freeze.json")["frozen_at"]          # e.g. 2026-09-27T14:40:12-0400
    hh, mm = fr[11:13], fr[14:16]
    m["SpExFrozenAt"] = f"{hh}:{mm} EDT on {int(fr[8:10])} September 2026"
    sec = info.get("seconds_per_fit", {})
    if sec:
        m["SpExSecondsProposed"] = f"{1000 * sec['bjs2']:.0f}"
        m["SpExSecondsChampFit"] = f"{sec['champollion_fit']:.1f}"
        m["SpExSecondsChamp"] = f"{sec['champollion']:.1f}"
    return m


def hybrid_macros():
    s = load(HYB / "summary.json")
    m = {}
    for ds, tag in (("frangieh", "Fr"), ("papalexi", "Pa")):
        rf = s[ds]["parts"]["test"]["table"]
        m[f"Hy{tag}HybridHundred"] = pct(rf["hybrid"]["100"][0])
        m[f"Hy{tag}BlockHundred"] = pct(rf["block_js"]["100"][0])
        m[f"Hy{tag}NoiseAware"] = pct(rf["noise_aware"]["100"][0])
        m[f"Hy{tag}Current"] = pct(rf["current"]["100"][0])
        m[f"Hy{tag}PairedJsHundred"] = pct(rf["paired_js"]["100"][0])
        m[f"Hy{tag}HybridPcHundred"] = pct(rf["hybrid_pc"]["100"][0])
    return m


def effdim_macros():
    """Effective noise dimension tr(S)/lambda_max(S) of every block at the largest budget checked, where the largest
    eigenvalue is estimated with least upward bias (medians over draws)."""
    m = {}
    vals, small = [], []
    for ds in ("frangieh", "papalexi"):
        p = HERE / "logs" / f"effective_dimension_{ds}.json"
        if not p.exists():
            continue
        e = load(p)
        top = max(e, key=int)
        vals += e[top]["effective_dimension_median"]
        small += e[min(e, key=int)]["effective_dimension_median"]
        tag = {"frangieh": "Fr", "papalexi": "Pa"}[ds]
        m[f"SpEffDimBudget{tag}"] = f"{int(top):,}"
    if vals:
        m["SpEffDimMin"] = f"{min(vals):.1f}"
        m["SpEffDimSmallMin"] = f"{min(small):.1f}"
    return m


def bound_macros():
    """Numerical check of the risk bound for the implemented estimator (bound_check.py): each reservoir as the
    population, 200 draws with replacement per budget."""
    m, ratios, effdims, ideal = {}, [], [], True
    for ds, tag in (("overcite", "Oc"), ("frangieh", "Fr"), ("papalexi", "Pa")):
        p = HERE / "logs" / f"bound_check_{ds}.json"
        if not p.exists():
            return {}
        r = load(p)
        ratios += [v["bound"] / v["risk_implemented"] for v in r["budgets"].values()]
        ideal &= all(v["risk_implemented"] <= v["risk_idealized"] for v in r["budgets"].values())
        effdims.append(r["min_effective_dimension"])
        m[f"SpBoundCells{tag}"] = f"{r['reservoir_cells']:,}"
        m[f"SpBoundBudgetMax{tag}"] = f"{max(int(B) for B in r['budgets']):,}"
        m[f"SpBoundCovered{tag}"] = "yes" if all(v["bound_covers_implemented"] for v in r["budgets"].values()) else "no"
        m[f"SpBoundCondition{tag}"] = "yes" if r["condition_tau_ge_4rho"] else "no"
        m["SpBoundDraws"] = str(r["draws"])
    m["SpBoundRatioMin"] = f"{min(ratios):.1f}"
    m["SpBoundRatioMax"] = f"{max(ratios):.1f}"
    m["SpBoundEffDimMin"] = f"{min(effdims):.1f}"
    m["SpBoundBelowIdealized"] = "yes" if ideal else "no"
    return m


def selftune_macros():
    """Development comparison of the self-tuning variants (dev_selftune.py, DEV_SELFTUNE.md)."""
    m, crit = {}, {}
    for ds, tag in (("frangieh", "Fr"), ("papalexi", "Pa")):
        p = R / f"selftune_{ds}.json"
        if not p.exists():
            return {}
        r = load(p)
        i50 = r["budgets"].index(50)
        m[f"SpSelf{tag}Fifty"] = pct(r["rf"]["V2"][i50])
        m[f"SpFixed{tag}Fifty"] = pct(r["rf"]["V0"][i50])
        for k, v in r["criterion"].items():
            crit.setdefault(k, []).append(v)
    mean = {k: sum(v) / len(v) for k, v in crit.items()}
    m["SpSelfCriterion"] = f"{100 * mean['V2']:.1f}"
    m["SpFixedCriterion"] = f"{100 * mean['V0']:.1f}"
    m["SpSelfPartitionCriterion"] = f"{100 * mean['V1']:.1f}"
    m["SpSelfFullCriterion"] = f"{100 * mean['V3']:.1f}"
    return m


def provenance_macros():
    """Provenance of the plan's estimator in the external test (external/provenance_check.py)."""
    p = EXT / "provenance.json"
    if not p.exists():
        return {}
    r = load(p)
    t = r["manifest_written_at"]
    return {"SpProvArrays": str(r["arrays_per_arm"]["bjs2/mapped"]),
            "SpProvMaxDiff": f"{max(r['max_abs_difference_recomputed_vs_stored'].values()):g}",
            "SpProvManifestAt": f"{t[11:13]}:{t[14:16]}:{t[17:19]} EDT"}


def curve_lines(xs, ys, style, legend=None, lo=-20):
    pts = " ".join(f"({x},{max(100 * y, lo):.2f})" for x, y in zip(xs, ys) if y == y)
    out = [f"\\addplot[{style}] coordinates {{ {pts} }};"]
    if legend:
        out.append(f"\\addlegendentry{{{legend}}}")
    return out


STYLES = {"prop": "color=propcol,line width=1.1pt,mark=*,mark size=1.5pt",
          "paired": "color=pairedcol,line width=.8pt,mark=square*,mark size=1.3pt",
          "semi": "color=semicol,line width=.8pt,mark=triangle*,mark size=1.6pt",
          "champ": "color=champcol,line width=.8pt,mark=diamond*,mark size=1.7pt",
          "ridge": "color=ridgecol,line width=.8pt,dashed,mark=o,mark size=1.3pt"}


def dev_panel(s, name, at, title, xmax, legend=False):
    rf, b = s["rf"], s["budgets"]
    xt = ",".join(str(x) for x in b)
    lines = [f"\\begin{{axis}}[{at}name={name},scale only axis,width=.3\\textwidth,height=3.9cm,xmode=log,log basis x=2,",
             f" xmin=40,xmax={xmax},ymin=-20,ymax=100,xtick={{{xt}}},xticklabels={{{xt}}},ytick={{-20,0,20,40,60,80,100}},",
             " xlabel={Paired cells},ylabel={Dependence recovered (\\%)},",
             f" title={{{title}}},", COMMON + "]",
             f"\\draw[dashed,black!45] (axis cs:40,30) -- (axis cs:{xmax},30);"]
    lines += curve_lines(b, rf[PROPOSED], STYLES["prop"])
    lines += curve_lines(b, envelope(rf, PAIRED_ONLY, len(b)), STYLES["paired"])
    lines += curve_lines(b, envelope(rf, ["semicca/pooled", "semicca/percond"], len(b)), STYLES["semi"])
    lines += curve_lines(b, envelope(rf, ["ridge_u/pooled", "ridge_u/percond", "ridge_p/pooled", "ridge_p/percond"],
                                     len(b)), STYLES["ridge"])
    ch = s.get("versus_champollion")
    if ch and ch["groups"] == s["groups"]:          # plotted only when it covers every evaluation group
        lines += curve_lines(ch["budgets"], ch["champollion"], STYLES["champ"])
    lines.append("\\end{axis}")
    return lines


def mech_panel(sf, sp, at):
    """Recovery at 100 paired cells: the proposed estimator and its ablations in both screens."""
    rows = [("bjs2/mapped", "Proposed"), ("bjs2/pooled", "Without the condition map"),
            ("bjs/mapped", "RNA eigenbasis only"), ("abl/bjs_pairbasis", "Bases from the paired cells"),
            ("abl/bjs_marker", "Marker coordinates, random blocks"), ("abl/map_measured", "Map from measured RNA"),
            ("dose/500", "500 unpaired cells per assay"), ("js/pooled", "Paired cells only (James--Stein)")]
    n = len(rows)
    lines = [f"\\begin{{axis}}[{at}name=c,scale only axis,width=.19\\textwidth,height=4.2cm,xmin=-5,xmax=60,",
             f" ymin=.4,ymax={n + .6},ytick={{{','.join(str(i) for i in range(1, n + 1))}}},",
             " yticklabels={" + ",".join("{" + r[1] + "}" for r in reversed(rows)) + "},",
             " xtick={0,20,40,60},xlabel={Recovered at 100 paired cells (\\%)},",
             " title={\\textbf{c}\\enspace Where the saving comes from},",
             COMMON.replace("ymajorgrids=true", "xmajorgrids=true") + ",y tick style={draw=none}]"]
    for s, style, dy in ((sf, "only marks,mark=*,mark size=1.7pt,color=propcol", .15),
                         (sp, "only marks,mark=o,mark size=1.7pt,color=pairedcol", -.15)):
        b = s["budgets"]
        i = b.index(100)
        pts = []
        for y, (a, _) in enumerate(reversed(rows), start=1):
            if a in s["rf"]:
                pts.append(f"({max(100 * s['rf'][a][i], -5):.2f},{y + dy:.2f})")
        lines.append(f"\\addplot[{style}] coordinates {{ {' '.join(pts)} }};")
    lines.append("\\end{axis}")
    return lines


def ext_panel(o, at):
    b = o["budgets"]
    rf = o["rf"]
    ci = o["rf_ci"]
    xt = ",".join(str(x) for x in b)
    lines = [f"\\begin{{axis}}[{at}name=d,scale only axis,width=.24\\textwidth,height=4.2cm,xmode=log,log basis x=2,",
             f" xmin=20,xmax=480,ymin=-20,ymax=80,xtick={{{xt}}},xticklabels={{{xt}}},ytick={{-20,0,20,40,60,80}},",
             " xlabel={Paired cells},ylabel={Dependence recovered (\\%)},",
             " title={\\textbf{d}\\enspace Prespecified external test},", COMMON + "]",
             "\\draw[dashed,black!45] (axis cs:20,30) -- (axis cs:480,30);"]
    lo = " ".join(f"({x},{max(100 * c[0], -20):.2f})" for x, c in zip(b, ci[PROPOSED]))
    hi = " ".join(f"({x},{min(100 * c[1], 80):.2f})" for x, c in zip(reversed(b), reversed(ci[PROPOSED])))
    lines.append(f"\\fill[propcol,opacity=.15] plot coordinates {{ {lo} {hi} }} -- cycle;")
    lines += curve_lines(b, rf[PROPOSED], STYLES["prop"])
    lines += curve_lines(b, envelope(rf, PAIRED_ONLY, len(b)), STYLES["paired"])
    lines += curve_lines(b, envelope(rf, ["semicca/pooled", "semicca/percond"], len(b)), STYLES["semi"])
    lines += curve_lines(b, rf["champollion/transport"], STYLES["champ"])
    lines += curve_lines(b, envelope(rf, ["ridge_u/pooled", "ridge_u/percond", "ridge_p/pooled", "ridge_p/percond"],
                                     len(b)), STYLES["ridge"])
    lines.append("\\end{axis}")
    return lines


def figure(sf, sp, o):
    lines = ["% Generated by extension/semipaired/make_assets.py. Do not edit.",
             "\\definecolor{propcol}{HTML}{217D91}", "\\definecolor{pairedcol}{HTML}{5D6670}",
             "\\definecolor{semicol}{HTML}{C9803A}", "\\definecolor{champcol}{HTML}{8E4B8C}",
             "\\definecolor{ridgecol}{HTML}{9AA3AB}", f"\\begin{{tikzpicture}}[{SANS}]"]
    lines += dev_panel(sf, "a", "", "\\textbf{a}\\enspace Melanoma screen (development)", 3600)
    lines += dev_panel(sp, "b", "at={(a.outer north east)},anchor=outer north west,xshift=.3cm,",
                       "\\textbf{b}\\enspace Monocyte screen (development)", 1800)
    lines += ["\\path (a.outer south west) -- (b.outer south east) coordinate[pos=.5] (legab);",
              "\\begin{axis}[hide axis,scale only axis,width=1pt,height=1pt,xmin=0,xmax=1,ymin=0,ymax=1,at={(legab)},"
              "anchor=north,yshift=-2pt,legend style={draw=none,font=\\sffamily\\scriptsize,at={(0.5,0.5)},anchor=north,"
              "legend columns=5,column sep=6pt,cells={anchor=west}}]",
              f"\\addlegendimage{{{STYLES['prop']}}}", "\\addlegendentry{Proposed}",
              f"\\addlegendimage{{{STYLES['paired']}}}", "\\addlegendentry{Paired cells only}",
              f"\\addlegendimage{{{STYLES['semi']}}}", "\\addlegendentry{SemiCCA}",
              f"\\addlegendimage{{{STYLES['champ']}}}", "\\addlegendentry{Champollion}",
              f"\\addlegendimage{{{STYLES['ridge']}}}", "\\addlegendentry{Reference regression}",
              "\\end{axis}"]
    lines += mech_panel(sf, sp, "at={(a.outer south west)},anchor=outer north west,yshift=-.8cm,")
    if o is not None:
        lines += ext_panel(o, "at={(c.outer north east)},anchor=outer north west,xshift=.4cm,")
    lines.append("\\end{tikzpicture}")
    return "\n".join(lines) + "\n"


def ext_table(o):
    """External test: recovered fraction by budget and uncorrected loss for each arm, then paired cells needed at the
    accuracy target and the prespecified comparisons."""
    b = o["budgets"]
    rf, loss = o["rf"], o["loss_rel"]
    arms = [("Proposed", [PROPOSED]),
            ("Paired cells only$^a$", PAIRED_ONLY),
            ("SemiCCA$^b$", ["semicca/pooled", "semicca/percond"]),
            ("Champollion", ["champollion/transport"]),
            ("Reference regression$^b$", ["ridge_u/pooled", "ridge_u/percond", "ridge_p/pooled", "ridge_p/percond"]),
            ("Any comparator with the map", [a for a in rf if a.endswith("/mapped") and not a.startswith("bjs")])]
    n = len(b)
    lines = ["% Generated by extension/semipaired/make_assets.py. Do not edit.",
             "\\begin{tabular}{@{}l" + "r" * n + "rr@{}}", "\\toprule",
             "& \\multicolumn{" + str(n) + "}{c}{Recovered (\\%) at paired cells} & \\multicolumn{2}{c}{Loss} \\\\",
             "\\cmidrule(lr){2-" + str(n + 1) + "}\\cmidrule(l){" + str(n + 2) + "-" + str(n + 3) + "}",
             "Estimator & " + " & ".join(str(x) for x in b) + " & 100 & 400 \\\\", "\\midrule"]
    i100, i400 = b.index(100), b.index(400)
    for label, members in arms:
        members = [a for a in members if a in rf]
        env = envelope(rf, members, n)
        l1 = min(loss[a][i100] for a in members)
        l4 = min(loss[a][i400] for a in members)
        lines.append(label + " & " + " & ".join(pct(v) for v in env) + f" & {l1:.3f} & {l4:.3f} \\\\")
    lines += ["\\midrule",
              "\\multicolumn{" + str(n + 3) + "}{@{}l}{Paired cells to reach a recovered fraction of "
              f"{o['target']:.1f} (proposed: " + rel(o["savings"]["paired_only"][str(o["target"])], "proposed") + ")} \\\\",
              "Comparison & \\multicolumn{2}{r}{Cells} & \\multicolumn{" + str(n - 2) + "}{r}{Ratio (95\\% CI)} & "
              "\\multicolumn{2}{r}{Verdict} \\\\"]
    t = str(o["target"])
    vd = o["verdicts"]
    for key, label, vk in (("paired_only", "H1: paired cells only", "H1_vs_paired_only"),
                           ("semicca", "H2: SemiCCA", "H2_vs_semicca"),
                           ("champollion", "H3: Champollion (non-inferiority)", "H3_noninferior_to_champollion"),
                           ("original_forms", "Every comparator, original form", None),
                           ("all_comparators", "Every comparator, with or without map", None)):
        v = o["savings"][key][t]
        verdict = ("met" if vd[vk] else "not met") if vk else "--"
        lines.append(f"{label} & \\multicolumn{{2}}{{r}}{{{rel(v, 'comparator')}}} & \\multicolumn{{{n - 2}}}{{r}}"
                     f"{{{fac(v['factor'])} ({fac(v['ci'][0])}--{fac(v['ci'][1])})}} & \\multicolumn{{2}}{{r}}{{{verdict}}} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def rel(v, who):
    r = v[f"{who}_relation"]
    c = v[f"{who}_cells"]
    return {"=": f"{c:.0f}", ">": f"$>${c:.0f}", "<=": f"$\\le${c:.0f}"}[r]


def dev_table(sf, sp):
    """Development recovered fractions (%) of the proposed estimator, its ablations and the comparators, both screens
    in one tabular (columns: the Frangieh budgets; Papalexi has no 3,200-cell budget)."""
    rows = [("Proposed (two-sided, mapped)", [PROPOSED]), ("Two-sided, pooled", ["bjs2/pooled"]),
            ("Two-sided, per condition", ["bjs2/percond"]), ("One-sided (RNA eigenbasis), mapped", ["bjs/mapped"]),
            ("Bases from the paired cells", ["abl/bjs_pairbasis"]), ("Marker coordinates, random blocks", ["abl/bjs_marker"]),
            ("Map from measured RNA covariance", ["abl/map_measured"]),
            ("500 unpaired cells per assay", ["dose/500"]), ("2,000 unpaired cells per assay", ["dose/2000"]),
            ("Paired only (best of four)", PAIRED_ONLY),
            ("SemiCCA", ["semicca/pooled", "semicca/percond"]),
            ("Reference regression", ["ridge_u/pooled", "ridge_u/percond", "ridge_p/pooled", "ridge_p/percond"]),
            ("Any comparator with the map", None),
            ("Champollion$^a$", "champ")]
    cols = sf["budgets"]
    n = len(cols)
    lines = ["% Generated by extension/semipaired/make_assets.py. Do not edit.",
             "\\begin{tabular}{@{}l" + "r" * n + "@{}}", "\\toprule",
             "Paired cells & " + " & ".join(f"{x:,}" for x in cols) + " \\\\"]
    for s, name in ((sf, "Frangieh (melanoma)"), (sp, "Papalexi (monocyte)")):
        b = s["budgets"]
        rf = s["rf"]
        ch = s.get("versus_champollion")
        lines += ["\\midrule", f"\\multicolumn{{{n + 1}}}{{@{{}}l}}{{\\textit{{{name}}}}} \\\\"]
        for label, arms in rows:
            if arms == "champ":
                if not ch:
                    continue
                vals = [ch["champollion"][ch["budgets"].index(x)] if x in ch["budgets"] else np.nan for x in b]
            elif arms is None:
                vals = envelope(rf, [a for a in rf if a.endswith("/mapped") and not a.startswith("bjs")], len(b))
            else:
                if not all(a in rf for a in arms):
                    continue
                vals = envelope(rf, arms, len(b)) if len(arms) > 1 else rf[arms[0]]
            by = dict(zip(b, vals))
            cells = [("--" if x not in by or by[x] != by[x] else pct(max(by[x], -9.99))) for x in cols]
            lines.append(label + " & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def external_results_md(o, of, info):
    """external/RESULTS.md: the external test in plain text (plan's estimator; frozen script's arm as deviation 1)."""
    b = o["budgets"]
    rf, loss, ci = o["rf"], o["loss_rel"], o["rf_ci"]
    t = str(o["target"])
    rows = [("Proposed (bjs2/mapped)", [PROPOSED]), ("One-sided (bjs/mapped)", ["bjs/mapped"]),
            ("Paired cells only (best of four)", PAIRED_ONLY), ("SemiCCA", ["semicca/pooled", "semicca/percond"]),
            ("Champollion", ["champollion/transport"]),
            ("Reference regression", ["ridge_u/pooled", "ridge_u/percond", "ridge_p/pooled", "ridge_p/percond"]),
            ("Any comparator with the map", [a for a in rf if a.endswith("/mapped") and not a.startswith("bjs")])]
    L = ["# External test (OverCITE-seq): results", "",
         f"Frozen {o['freeze']}; predictions hashed before the held-out files were opened (`results/manifest.json`).",
         f"{len(o['held_out_orfs'])} held-out ORFs, {o['cells']} cells; reservoir {info['n_reservoir']} paired cells; "
         f"20 draws per budget. Generated by `../make_assets.py` from `results/overcite_plan_estimator.json` (the "
         "plan's estimator) and `results/overcite.json` (the frozen script's arm; `DEVIATIONS.md`).", "",
         "Recovered fraction (%) of within-condition cross-correlation among held-out ORFs, and uncorrected relative "
         "loss:", "", "| Arm | " + " | ".join(str(x) for x in b) + " | loss at 100 | loss at 400 |",
         "|---|" + "---|" * (len(b) + 2)]
    i1, i4 = b.index(100), b.index(400)
    for label, arms in rows:
        arms = [a for a in arms if a in rf]
        env = envelope(rf, arms, len(b))
        L.append(f"| {label} | " + " | ".join(f"{100 * v:.1f}" for v in env) +
                 f" | {min(loss[a][i1] for a in arms):.3f} | {min(loss[a][i4] for a in arms):.3f} |")
    L += ["", f"Proposed estimator, 95% intervals: " + "; ".join(
        f"{x}: {100 * c[0]:.1f} to {100 * c[1]:.1f}" for x, c in zip(b, ci[PROPOSED])),
        f"All training cells used as paired cells: {100 * o['reference_rf']['paired_all/all']:.1f}%.", "",
        f"Paired cells to reach a recovered fraction of {t} (proposed: "
        f"{o['savings']['paired_only'][t]['proposed_cells']:.0f}); ratio = comparator cells / proposed cells:", ""]
    lab = {"paired_only": "H1, paired cells only", "semicca": "H2, SemiCCA", "champollion": "H3, Champollion "
           "(non-inferiority: lower bound > 0.8)", "original_forms": "every comparator in its original form",
           "all_comparators": "every comparator, with or without the map"}
    for k, name in lab.items():
        for tt in ("0.2", t, "0.4"):
            v = o["savings"][k][tt]
            L.append(f"* {name}, target {tt}: {v['comparator_relation']} {v['comparator_cells']:.0f} cells; ratio "
                     f"{v['factor']:.2f} ({v['ci'][0]:.2f} to {v['ci'][1]:.2f})")
    L += ["", "Verdicts (plan's estimator): " + ", ".join(f"{k} {v}" for k, v in o["verdicts"].items()),
          "Verdicts (frozen script's arm, bjs/mapped): " + ", ".join(f"{k} {v}" for k, v in of["verdicts"].items()),
          "", "Differences from the best comparator (including those with the map), percentage points:", ""]
    L += [f"* {d['budget']}: {100 * d['diff']:.1f} ({100 * d['ci'][0]:.1f} to {100 * d['ci'][1]:.1f}) against "
          f"{d['best']}" for d in o["versus_best"]]
    sec = info.get("seconds_per_fit", {})
    if sec:
        L += ["", f"Time per fit: proposed {1000 * sec['bjs2']:.0f} ms (with the map), Champollion "
              f"{sec['champollion_fit']:.1f} s (fit) and {sec['champollion']:.1f} s (with transport and start-up)."]
    return "\n".join(L) + "\n"


def write_macros(path, m):
    lines = ["% Generated by extension/semipaired/make_assets.py. Do not edit."]
    for k in sorted(m):
        lines.append(f"\\newcommand{{\\{k}}}{{{m[k]}}}")
    Path(path).write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE.parent.parent / "revision" / "source"))
    args = ap.parse_args()
    out = Path(args.out)
    sf, sp = load(R / "summary_frangieh.json"), load(R / "summary_papalexi.json")
    m = {}
    m.update(dev_macros("frangieh", "Fr", sf))
    m.update(dev_macros("papalexi", "Pa", sp))
    m.update(hybrid_macros())
    ct = champ_text(sf, sp)
    if ct:
        m["SpDevChampText"] = ct
    m.update(effdim_macros())
    m.update(bound_macros())
    m.update(provenance_macros())
    m.update(selftune_macros())
    o = None
    if (EXT / "overcite_plan_estimator.json").exists():
        # the plan's proposed estimator (bjs2/mapped); the frozen script's output for bjs/mapped is reported as
        # deviation 1 (external/DEVIATIONS.md)
        o = load(EXT / "overcite_plan_estimator.json")
        m.update(ext_macros(o, load(EXT / "predict_info.json")))
        of = load(EXT / "overcite.json")
        t = str(of["target"])
        m["SpExFrozenArmCells"] = f"{of['savings']['paired_only'][t]['proposed_cells']:.0f}"
        for key, lab in (("paired_only", "PairedOnly"), ("semicca", "Semi"), ("champollion", "Champ")):
            v = of["savings"][key][t]
            m[f"SpExFrozenArm{lab}"] = fac(v["factor"])
            m[f"SpExFrozenArm{lab}Lo"] = fac(v["ci"][0])
            m[f"SpExFrozenArm{lab}Hi"] = fac(v["ci"][1])
        m["SpExFrozenArmAllMet"] = "yes" if all(of["verdicts"][k] for k in ("H1_vs_paired_only", "H2_vs_semicca",
                                                                          "H3_noninferior_to_champollion")) else "no"
        (out / "semi_table.tex").write_text(ext_table(o))
        (EXT.parent / "RESULTS.md").write_text(external_results_md(o, of, load(EXT / "predict_info.json")))
    for k in PLACEHOLDERS:
        m.setdefault(k, "\\textbf{[pending]}")
    write_macros(out / "semi_macros.tex", m)
    (out / "semi_figure.tex").write_text(figure(sf, sp, o))
    (out / "semi_dev_table.tex").write_text(dev_table(sf, sp))
    print(len(m), "macros")


if __name__ == "__main__":
    main()
