#!/usr/bin/env python3
"""Manuscript assets of the perturbation test: macros, Fig. 6, a supplementary table and results/RESULTS.md.

    python make_assets.py --out ../../revision/source
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
R = HERE / "results"
SANS = "font=\\sffamily"
COMMON = (" axis x line*=bottom,axis y line*=left,tick align=outside,y tick style={draw=none},"
          "xmajorgrids=true,grid style={gray!18},axis line style={gray!55},"
          "tick label style={font=\\sffamily\\scriptsize},label style={font=\\sffamily\\small},clip=false,"
          "title style={font=\\sffamily\\small,at={(0,1)},anchor=base west,yshift=5pt}")
CONDITIONS = (("control", "Alone"), ("ifng", "IFN$\\gamma$"), ("coculture", "With T cells"))
ARMS = [("benchmark", "Own paired cells"), ("transferred_correlation", "Transferred correlation"),
        ("reference_regression", "Reference regression"), ("paired_closed_form", "Paired closed form"),
        ("pf_measured", "No pairs: conditions pooled"), ("control_within_derange", "No pairs: deranged"),
        ("control_within_pseudo", "No pairs: pseudo-populations"),
        ("pf_within_measured", "No pairs: perturbations")]          # y = 1..8


def load(name):
    return json.loads((R / name).read_text())


def signed(text):
    return "\\ensuremath{-}" + text[1:] if text.startswith("-") else text


def pct(v, ref=None):
    """Percentage: one decimal when the reference value (default v) is below 10%, integer otherwise."""
    p = 100 * v
    r = 100 * (v if ref is None else ref)
    return signed(f"{p:.1f}" if abs(r) < 10 else f"{p:.0f}")


def macros(ev, seal, chans, diag):
    m = {"PtCells": f"{seal['cells']:,}", "PtTargets": str(seal["targets"]),
         "PtHeldTargets": str(len(seal["heldout_targets"])), "PtTrainTargets": str(len(seal["training_targets"])),
         "PtSingleCells": f"{seal['classes']['single']:,}", "PtNTCells": f"{seal['classes']['NT']:,}",
         "PtGenes": str(len(seal["genes"])), "PtEncodingGenes": str(len(seal["encoding_genes"])),
         "PtProteins": str(len(seal["proteins"])), "PtTrainPops": str(len(chans["populations"])),
         "PtGroups": str(ev["heldout"]["groups"]), "PtHeldEligible": str(ev["heldout"]["targets"]),
         "PtDimS": str(chans["channels"]["pf_within"]["dim_S"]),
         "PtDimSPooled": str(chans["channels"]["pf"]["dim_S"])}
    e = ev["heldout"]["estimates"]
    names = {"pf_within_measured": "Primary", "pf_within_latent": "PrimaryLatent", "pf_within_regression": "Regression",
             "pf_within_condition": "CondMarg", "pf_measured": "Pooled", "paired_closed_form": "Paired",
             "paired_closed_form_latent": "PairedLatent", "reference_regression": "RefReg",
             "transferred_correlation": "TransCorr", "benchmark": "Bench", "control_within_pseudo": "Pseudo",
             "control_within_derange": "Derange", "control_pseudo": "PooledPseudo", "control_derange": "PooledDerange",
             "primary_minus_within_pseudo": "DiffPseudo", "primary_minus_within_derange": "DiffDerange",
             "share_of_paired": "Share", "primary_minus_unstratified": "DiffPooled"}
    for k, tag in names.items():
        m[f"PtRf{tag}"] = pct(e[k]["value"])
        m[f"PtRf{tag}Lo"] = pct(e[k]["ci"][0], e[k]["value"])
        m[f"PtRf{tag}Hi"] = pct(e[k]["ci"][1], e[k]["value"])
    for c, tag in (("control", "Alone"), ("ifng", "Ifng"), ("coculture", "Cocult")):
        for k, t in (("pf_within_measured", "Primary"), ("paired_closed_form", "Paired"),
                     ("transferred_correlation", "TransCorr")):
            m[f"PtCond{tag}{t}"] = pct(ev["heldout"]["by_condition"][c][k])
            m[f"PtNt{tag}{t}"] = pct(ev["nt"]["by_condition"][c][k])
    nt = ev["nt"]
    for k, tag in (("pf_within_measured", "Primary"), ("paired_closed_form", "Paired"), ("pf_measured", "Pooled"),
                   ("transferred_correlation", "TransCorr"), ("reference_regression", "RefReg"), ("benchmark", "Bench")):
        m[f"PtNt{tag}"] = pct(nt["pooled"][k])
        m[f"PtNt{tag}Lo"] = pct(nt["pooled_ci"][k][0], nt["pooled"][k])
        m[f"PtNt{tag}Hi"] = pct(nt["pooled_ci"][k][1], nt["pooled"][k])
    m["PtNtShare"] = pct(nt["share_of_paired"]["value"])
    m["PtNtShareLo"] = pct(nt["share_of_paired"]["ci"][0], nt["share_of_paired"]["value"])
    m["PtNtShareHi"] = pct(nt["share_of_paired"]["ci"][1], nt["share_of_paired"]["value"])
    held = [r for r in diag if r["part"] == "heldout"]
    ntd = [r for r in diag if r["part"] == "nt"]
    m["PtCoverage"] = pct(np.median([r["pf_within"]["coverage"] for r in held]))
    m["PtCoverageNt"] = pct(np.median([r["pf_within"]["coverage"] for r in ntd]))
    m["PtAdaptMedian"] = f"{np.median([r['n_adapt'] for r in held]):.0f}"
    m["PtAdaptNtMin"] = f"{min(r['n_adapt'] for r in ntd):,}"
    m["PtAdaptNtMax"] = f"{max(r['n_adapt'] for r in ntd):,}"
    dev = load("dev_cf.json")["summary"]
    m["PtDevPrimary"] = pct(dev["pf_within_measured"]["rf"])
    m["PtDevPooled"] = pct(dev["pf_measured"]["rf"])
    m["PtDevPaired"] = pct(dev["paired_closed_form"]["rf"])
    v = ev["verdict"]
    for k, tag in (("H1_recovery", "HOne"), ("H3_nontargeting", "HThree"), ("H4_most_of_paired", "HFour")):
        m[f"Pt{tag}"] = "met" if v[k] else "not met"
    m["PtHTwo"] = "met" if (v["H2_beats_pseudo_populations"] and v["H2_beats_derangements"]) else "not met"
    return m


def direction(seal):
    z = np.load(R / "channels.npz")
    v = z["pf_within/VS"][:, 0]
    W = z["pf_within/W"]
    sign = 1.0 if v[np.argmax(np.abs(v))] > 0 else -1.0
    v = sign * v
    genes = np.array(seal["genes"])
    top = np.argsort(-np.abs(v))[:10]
    resp = W @ v
    ptop = np.argsort(-np.abs(resp))[:5]
    names = {"HLA_A": "HLA-ABC", "HLA_E": "HLA-E", "CD274": "PD-L1"}
    return ([(genes[i], float(v[i])) for i in top],
            [(names.get(seal["proteins"][i], seal["proteins"][i]), float(resp[i])) for i in ptop])


def figure(ev, seal, prog, des):
    e = ev["heldout"]["estimates"]
    nt = ev["nt"]
    lines = ["% Generated by extension/perturbation/make_assets.py. Do not edit.",
             "\\definecolor{heldcol}{HTML}{217D91}", "\\definecolor{ntcol}{HTML}{C9803A}",
             "\\definecolor{donorcol}{HTML}{5D6670}", f"\\begin{{tikzpicture}}[{SANS}]",
             "\\begin{axis}[name=a,scale only axis,width=.21\\textwidth,height=4.3cm,xmin=-80,xmax=105,ymin=.4,ymax=8.6,",
             " xtick={-75,-50,-25,0,25,50,75,100},xticklabels={,$-$50,,0,,50,,100},ytick={1,...,8},",
             " yticklabels={" + ",".join("{" + a[1] + "}" for a in ARMS) + "},",
             " xlabel={Dependence recovered (\\%)},",
             " title={\\textbf{a}\\enspace All genes and proteins},", COMMON + "]",
             "\\draw[dashed,black!45] (axis cs:0,.4) -- (axis cs:0,8.6);"]
    held_pts, nt_pts = [], []
    for y, (k, _) in enumerate(ARMS, start=1):
        val, (lo, hi) = e[k]["value"], e[k]["ci"]
        lines.append(f"\\draw[heldcol,line width=.8pt] (axis cs:{100 * lo:.2f},{y + .14}) -- (axis cs:{100 * hi:.2f},{y + .14});")
        held_pts.append(f"({100 * val:.2f},{y + .14})")
        if k in nt["pooled"]:
            v2, (l2, h2) = nt["pooled"][k], nt["pooled_ci"][k]
            lines.append(f"\\draw[ntcol,line width=.8pt] (axis cs:{100 * l2:.2f},{y - .14}) -- (axis cs:{100 * h2:.2f},{y - .14});")
            nt_pts.append(f"({100 * v2:.2f},{y - .14})")
    lines += ["\\addplot[only marks,mark=*,mark size=1.6pt,color=heldcol] coordinates { " + " ".join(held_pts) + " };",
              "\\addplot[only marks,mark=diamond*,mark size=2pt,color=ntcol] coordinates { " + " ".join(nt_pts) + " };",
              "\\end{axis}"]
    lines += program_figure_lines(prog, des, ev['heldout']['targets'])
    lines.append("\\end{tikzpicture}")
    genes, prots = direction(seal)
    return "\n".join(lines) + "\n", genes, prots


REP = HERE.parent / "perturbation_replication" / "results"


def program_macros(prog):
    h, n = prog["heldout"], prog["nt"]
    iv = h["intervals"]
    m = {"PgIfnGenes": str(len(prog["blocks"]["ifn"]["genes"])),
         "PgIfnProteinCount": str(len(prog["blocks"]["ifn"]["proteins"]))}
    for lab, tag in (("ifn/rf/pf_within_measured", "IfnPf"), ("ifn/rf/paired_closed_form", "IfnPaired"),
                     ("ifn/rf/transferred_correlation", "IfnTrans"), ("ifn/rf/reference_regression", "IfnRefReg"),
                     ("ifn/rf/benchmark", "IfnBench"), ("ifn/rf/replicate", "IfnRepl"),
                     ("ifn/share_paired", "IfnShare"), ("ifn/share_transferred", "IfnShareTrans"),
                     ("along_S/rf/pf_within_measured", "AlongPf"), ("along_S/rf/paired_closed_form", "AlongPaired"),
                     ("along_S/share_paired", "AlongShare"), ("complement/rf/pf_within_measured", "CompPf"),
                     ("complement/rf/paired_closed_form", "CompPaired"),
                     ("complement/rf/transferred_correlation", "CompTrans"), ("ifn_minus_complement/pf", "Diff"),
                     ("ifn_genes/rf/pf_within_measured", "IfnGenesPf"),
                     ("ifn_genes/rf/paired_closed_form", "IfnGenesPaired")):
        v = iv[lab]
        m[f"Pg{tag}"] = pct(v["value"])
        m[f"Pg{tag}Lo"] = pct(v["ci"][0], v["value"])
        m[f"Pg{tag}Hi"] = pct(v["ci"][1], v["value"])
    m["PgIfnRel"] = f"{h['ifn']['reliability']:.2f}"
    m["PgCompRel"] = f"{h['complement']['reliability']:.2f}"
    for c, tag in (("control", "Alone"), ("ifng", "Ifng"), ("coculture", "Cocult")):
        m[f"PgIfn{tag}Pf"] = pct(h["ifn"]["by_condition"][c]["pf_within_measured"])
        m[f"PgIfn{tag}Paired"] = pct(h["ifn"]["by_condition"][c]["paired_closed_form"])
    for k, tag in (("pf_within_measured", "Pf"), ("paired_closed_form", "Paired"), ("transferred_correlation", "Trans"),
                   ("replicate", "Repl")):
        m[f"PgNtIfn{tag}"] = pct(n["ifn"]["rf"][k])
        m[f"PgNtComp{tag}"] = pct(n["complement"]["rf"][k])
    m["PgNtIfnRel"] = f"{n['ifn']['reliability']:.2f}"
    others = h["support_vs_recovery"]
    m["PgOtherMax"] = pct(max(others["recovery"]))
    m["PgOtherCount"] = str(len(others["programs"]))
    m["PgSupportIfn"] = f"{h['ifn']['support']:.3f}"
    m["PgSupportOtherMax"] = f"{max(others['support']):.3f}"
    v = h["verdict"]
    m["PgPOne"] = "met" if v["P1_ifn_share_above_half"] else "not met"
    m["PgPTwo"] = "met" if v["P2_along_S_share_above_half"] else "not met"
    m["PgPThree"] = "met" if v["P3_ifn_above_complement"] else "not met"
    return m


def design_macros(des):
    m = {"DsNoneAll": pct(des["none"]["all"]), "DsNoneIfn": pct(des["none"]["ifn"])}
    words = {5: "Five", 10: "Ten", 20: "Twenty"}
    for k in (5, 10, 20):
        a = des["ablation"][str(k)]
        for b, tag in (("all", "All"), ("ifn", "Ifn")):
            e = a[b]
            m[f"Ds{words[k]}{tag}Exp"] = pct(e["remove_exposure"])
            m[f"Ds{words[k]}{tag}Str"] = pct(e["remove_strength"])
            m[f"Ds{words[k]}{tag}Rand"] = pct(e["remove_random_mean"])
            m[f"Ds{words[k]}{tag}RandLow"] = pct(e["remove_random_q05"])
            d = e["random_minus_exposure"]
            m[f"Ds{words[k]}{tag}Diff"] = pct(d["value"])
            m[f"Ds{words[k]}{tag}DiffLo"] = pct(d["ci"][0], d["value"])
            m[f"Ds{words[k]}{tag}DiffHi"] = pct(d["ci"][1], d["value"])
            m[f"Ds{words[k]}{tag}Below"] = str(e["random_below_exposure"])
            d = e["random_minus_strength"]
            m[f"Ds{words[k]}{tag}DiffStr"] = pct(d["value"])
            m[f"Ds{words[k]}{tag}DiffStrLo"] = pct(d["ci"][0], d["value"])
            m[f"Ds{words[k]}{tag}DiffStrHi"] = pct(d["ci"][1], d["value"])
        m[f"Ds{words[k]}DimExp"] = str(a["dim_S"]["exposure"])
        m[f"Ds{words[k]}DimRand"] = f"{a['dim_S']['random_mean']:.1f}"
        m[f"Ds{words[k]}ZeroRand"] = str(a["dim_S"]["random_zero"])
    top = sorted(des["exposure"], key=lambda t: -des["exposure"][t])
    m["DsTopFour"] = ", ".join(top[:3]) + " and " + top[3]
    m["DsTopFive"] = ", ".join(top[:4]) + " and " + top[4]
    tops = sorted(des["strength"], key=lambda t: -des["strength"][t])
    m["DsTopStrengthFive"] = ", ".join(tops[:4]) + " and " + tops[4]
    gap = des["exposure"][top[3]] / des["exposure"][top[4]]
    m["DsExposureGap"] = f"{gap:.0f}"
    loo = des["leave_one_out"]
    m["DsLooSpearman"] = signed(f"{loo['spearman_exposure']:.2f}")
    m["DsLooP"] = f"{loo['p_exposure']:.2f}"
    m["DsLooTop"] = " and ".join(loo["top_drop"][:2])
    m["DsDraws"] = str(len(des["ablation"]["5"]["all"]["random_values"]))
    v = des["verdict"]
    m["DsDOne"] = "met" if all(v.values()) else ("partly met" if any(v.values()) else "not met")
    return m


def replication_panel_values():
    ev = json.loads((REP / "evaluation.json").read_text())
    ph = json.loads((REP / "posthoc_penalty.json").read_text())
    s = ev["test"]["summary"]
    rows = [("Pairing-free, frozen", s["all"]["pf_within_measured"], s["ifn"]["pf_within_measured"]),
            ("Fixed scale 0.1$^{\\ast}$", ph["test"]["scale0.1/all"], ph["test"]["scale0.1/ifn"]),
            ("Fixed scale 1$^{\\ast}$", ph["test"]["scale1/all"], ph["test"]["scale1/ifn"]),
            ("Paired closed form", s["all"]["paired_closed_form"], s["ifn"]["paired_closed_form"]),
            ("Transferred correlation", s["all"]["transferred_correlation"], s["ifn"]["transferred_correlation"])]
    return rows[::-1]


def program_figure_lines(prog, des, n_targets):
    h, n = prog["heldout"], prog["nt"]
    iv = h["intervals"]
    arms = [("replicate", "Replicate half"), ("benchmark", "Own paired cells"),
            ("transferred_correlation", "Transferred correlation"), ("paired_closed_form", "Paired closed form"),
            ("pf_within_measured", "No pairs: perturbations")]
    lo_clip = -50
    lines = ["\\begin{axis}[at={(a.outer north east)},anchor=outer north west,xshift=.5cm,name=b,scale only axis,",
             f" width=.21\\textwidth,height=4.3cm,xmin={lo_clip},xmax=105,ymin=.4,ymax={len(arms) + .6},",
             " xtick={-50,-25,0,25,50,75,100},xticklabels={$-$50,,0,,50,,100},",
             f" ytick={{{','.join(str(i) for i in range(1, len(arms) + 1))}}},",
             " yticklabels={" + ",".join("{" + a[1] + "}" for a in arms) + "},",
             " xlabel={Dependence recovered (\\%)},title={\\textbf{b}\\enspace Interferon program},", COMMON + "]",
             f"\\draw[dashed,black!45] (axis cs:0,.4) -- (axis cs:0,{len(arms) + .6});"]
    series = {"ifn": [], "comp": [], "nt": []}
    for y, (k, _) in enumerate(arms, start=1):
        v = h["ifn"]["rf"][k]
        key = f"ifn/rf/{k}"
        if key in iv:
            lo, hi = iv[key]["ci"]
            lines.append(f"\\draw[heldcol,line width=.8pt] (axis cs:{max(100 * lo, lo_clip):.2f},{y + .18}) -- "
                         f"(axis cs:{100 * hi:.2f},{y + .18});")
        series["ifn"].append(f"({max(100 * v, lo_clip):.2f},{y + .18})")
        series["comp"].append(f"({max(100 * h['complement']['rf'][k], lo_clip):.2f},{y})")
        series["nt"].append(f"({max(100 * n['ifn']['rf'][k], lo_clip):.2f},{y - .18})")
    lines += ["\\addplot[only marks,mark=*,mark size=1.6pt,color=heldcol] coordinates { " + " ".join(series["ifn"]) + " };",
              "\\addplot[only marks,mark=o,mark size=1.6pt,color=donorcol] coordinates { " + " ".join(series["comp"]) + " };",
              "\\addplot[only marks,mark=diamond*,mark size=2pt,color=ntcol] coordinates { " + " ".join(series["nt"]) + " };",
              "\\end{axis}"]
    # shared legend of the first row
    lines += ["\\path (a.outer south west) -- (b.outer south east) coordinate[pos=.5] (legab);",
              "\\begin{axis}[hide axis,scale only axis,width=1pt,height=1pt,xmin=0,xmax=1,ymin=0,ymax=1,at={(legab)},"
              "anchor=north,yshift=-2pt,legend style={draw=none,font=\\sffamily\\scriptsize,at={(0.5,0.5)},anchor=north,"
              "legend columns=3,column sep=8pt,cells={anchor=west}}]",
              "\\addlegendimage{only marks,mark=*,mark size=1.6pt,color=heldcol}",
              f"\\addlegendentry{{Held-out targets ($n={n_targets}$)}}",
              "\\addlegendimage{only marks,mark=diamond*,mark size=2pt,color=ntcol}",
              "\\addlegendentry{Non-targeting cells}",
              "\\addlegendimage{only marks,mark=o,mark size=1.6pt,color=donorcol}",
              "\\addlegendentry{Held-out targets, other genes and proteins (\\textbf{b})}",
              "\\end{axis}"]
    # c: removal of the targets of highest exposure versus random targets (interferon block)
    lines += ["\\begin{axis}[at={(a.outer south west)},anchor=outer north west,yshift=-.75cm,name=c,scale only axis,",
              " width=.25\\textwidth,height=3.7cm,",
              " xmin=.5,xmax=3.5,ymin=-5,ymax=65,xtick={1,2,3},xticklabels={5,10,20},ytick={0,20,40,60},",
              " xlabel={Training targets removed},ylabel={Interferon block (\\%)},",
              " title={\\textbf{c}\\enspace Removing training perturbations},",
              COMMON.replace("y tick style={draw=none},", "").replace("xmajorgrids=true", "ymajorgrids=true") + ",",
              " legend style={draw=none,font=\\sffamily\\scriptsize,at={(0.5,-0.3)},anchor=north,legend columns=3,"
              "column sep=5pt,cells={anchor=west}}]",
              f"\\draw[dashed,black!45] (axis cs:.5,{100 * des['none']['ifn']:.2f}) -- (axis cs:3.5,{100 * des['none']['ifn']:.2f});"]
    rnd, exp_, strg = [], [], []
    for i, k in enumerate((5, 10, 20), start=1):
        vals = des["ablation"][str(k)]["ifn"]["random_values"]
        xs = np.linspace(i - .22, i + .22, len(vals))
        rnd += [f"({x:.3f},{100 * v:.2f})" for x, v in zip(xs, vals)]
        exp_.append(f"({i},{100 * des['ablation'][str(k)]['ifn']['remove_exposure']:.2f})")
        strg.append(f"({i + .3:.2f},{100 * des['ablation'][str(k)]['ifn']['remove_strength']:.2f})")
    lines += ["\\addplot[only marks,mark=*,mark size=.8pt,color=donorcol!55] coordinates { " + " ".join(rnd) + " };",
              "\\addlegendentry{Random targets}",
              "\\addplot[only marks,mark=*,mark size=2.1pt,color=black] coordinates { " + " ".join(exp_) + " };",
              "\\addlegendentry{Highest exposure}",
              "\\addplot[only marks,mark=square*,mark size=1.6pt,color=ntcol] coordinates { " + " ".join(strg) + " };",
              "\\addlegendentry{Strongest effect}", "\\end{axis}"]
    # d: sealed replication
    rows = replication_panel_values()
    lines += ["\\begin{axis}[at={(c.outer north east)},anchor=outer north west,xshift=.5cm,name=d,scale only axis,",
              f" width=.21\\textwidth,height=3.7cm,xmin=-100,xmax=105,ymin=.4,ymax={len(rows) + .6},",
              " xtick={-100,-50,0,50,100},xticklabels={$-$100,,0,,100},",
              f" ytick={{{','.join(str(i) for i in range(1, len(rows) + 1))}}},",
              " yticklabels={" + ",".join("{" + r[0] + "}" for r in rows) + "},",
              " xlabel={Dependence recovered (\\%)},title={\\textbf{d}\\enspace Independent screen, sealed},", COMMON + ",",
              " legend style={draw=none,font=\\sffamily\\scriptsize,at={(1.0,-0.3)},anchor=north east,legend columns=2,"
              "column sep=5pt,cells={anchor=west}}]",
              f"\\draw[dashed,black!45] (axis cs:0,.4) -- (axis cs:0,{len(rows) + .6});"]
    allp, ifnp, notes = [], [], []
    for y, (_, a_, i_) in enumerate(rows, start=1):
        allp.append(f"({max(100 * a_, -100):.2f},{y + .14})")
        ifnp.append(f"({max(100 * i_, -100):.2f},{y - .14})")
        if 100 * a_ < -100:
            notes.append(f"\\node[font=\\sffamily\\tiny,anchor=south west,inner sep=1pt] at (axis cs:-100,{y + .22}) "
                         f"{{$-${abs(100 * a_):.0f}\\%}};")
    lines += ["\\addplot[only marks,mark=o,mark size=1.6pt,color=donorcol] coordinates { " + " ".join(allp) + " };",
              "\\addlegendentry{All genes and proteins}",
              "\\addplot[only marks,mark=*,mark size=1.6pt,color=heldcol] coordinates { " + " ".join(ifnp) + " };",
              "\\addlegendentry{Interferon block}"] + [x.replace("%", "\\%") if "\\%" not in x else x for x in notes] + [
              "\\end{axis}"]
    return lines


def program_table(prog):
    h, n = prog["heldout"], prog["nt"]
    arms = [("pf_within_measured", "No pairs"), ("paired_closed_form", "Paired CF"),
            ("reference_regression", "Ref.\\ regr."), ("transferred_correlation", "Transf.\\ corr."),
            ("benchmark", "Own pairs"), ("replicate", "Replicate")]
    names = {"E2F_TARGETS": "E2F targets", "EPITHELIAL_MESENCHYMAL_TRANSITION": "Epithelial--mesenchymal transition",
             "G2M_CHECKPOINT": "G2M checkpoint", "HYPOXIA": "Hypoxia", "MYC_TARGETS_V1": "MYC targets",
             "OXIDATIVE_PHOSPHORYLATION": "Oxidative phosphorylation",
             "TNFA_SIGNALING_VIA_NFKB": "TNF-$\\alpha$ signalling via NF-$\\kappa$B"}
    label_of = lambda b: names.get(b.upper(), b.replace("_", " ").capitalize())
    blocks = [("ifn", "Interferon block"), ("along_S", "Along $S$"), ("ifn_genes", "Interferon genes, all proteins"),
              ("complement", "Complement")] + [(b, label_of(b)) for b in h["support_vs_recovery"]["programs"]]
    num = lambda v: signed(f"{100 * v:.1f}")
    lines = ["% Generated by extension/perturbation/make_assets.py. Do not edit.",
             "\\begin{tabular}{l" + "c" * (len(arms) + 1) + "}", "\\toprule",
             "Block & " + " & ".join(a[1] for a in arms) + " & Rel.\\ \\\\", "\\midrule"]
    for b, label in blocks:
        lines.append(f"{label} & " + " & ".join(num(h[b]["rf"][k]) for k, _ in arms) + f" & {h[b]['reliability']:.2f} \\\\")
    lines.append("\\midrule")
    for b, label in (("ifn", "Non-targeting: interferon block"), ("complement", "Non-targeting: complement")):
        lines.append(f"{label} & " + " & ".join(num(n[b]["rf"][k]) for k, _ in arms) + f" & {n[b]['reliability']:.2f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def table(ev):
    e = ev["heldout"]["estimates"]
    nt = ev["nt"]
    num = lambda v: signed(f"{100 * v:.1f}")
    rows = [("pf_within_measured", "No pairs: perturbations, measured RNA (primary)"),
            ("pf_within_latent", "No pairs: perturbations, corrected RNA"),
            ("pf_within_regression", "No pairs: perturbations, regression form"),
            ("pf_within_condition", "No pairs: perturbations, training marginals"),
            ("control_within_pseudo", "No pairs: pseudo-populations (10 draws)"),
            ("control_within_derange", "No pairs: derangements (10 draws)"),
            ("pf_measured", "No pairs: conditions pooled"),
            ("control_pseudo", "No pairs: conditions pooled, pseudo-populations"),
            ("control_derange", "No pairs: conditions pooled, derangements"),
            ("paired_closed_form", "Paired closed form"), ("paired_closed_form_latent", "Paired closed form, corrected RNA"),
            ("reference_regression", "Reference regression"), ("transferred_correlation", "Transferred correlation"),
            ("benchmark", "Own paired adaptation cells")]
    lines = ["% Generated by extension/perturbation/make_assets.py. Do not edit.",
             "\\begin{tabular}{lcccccc}", "\\toprule",
             "& \\multicolumn{4}{c}{Held-out targets} & \\multicolumn{2}{c}{Non-targeting cells} \\\\",
             "\\cmidrule(lr){2-5}\\cmidrule(lr){6-7}",
             "Arm & All (95\\% CI) & Alone & IFN$\\gamma$ & T cells & All (95\\% CI) & Error \\\\", "\\midrule"]
    for k, label in rows:
        v = e[k]
        conds = [num(ev["heldout"]["by_condition"][c][k]) if k in ev["heldout"]["by_condition"][c] else "--"
                 for c, _ in CONDITIONS]
        ntv = (f"{num(nt['pooled'][k])} ({num(nt['pooled_ci'][k][0])} to {num(nt['pooled_ci'][k][1])})"
               if k in nt["pooled"] else "--")
        err = f"{ev['heldout']['mean_rel_error'][k]:.3f}" if k in ev["heldout"]["mean_rel_error"] else "--"
        lines.append(f"{label} & {num(v['value'])} ({num(v['ci'][0])} to {num(v['ci'][1])}) & "
                     + " & ".join(conds) + f" & {ntv} & {err} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def results_md(m, ev, genes, prots):
    v = ev["verdict"]
    lines = ["# Perturbation test: results", "",
             "Generated by `make_assets.py` from `results/evaluation.json` (sealed test) and `results/dev_cf.json`",
             "(development on training targets, exploratory). Recovered fraction RF: share of the true within-group",
             "cross-covariance norm recovered (noise-unbiased); 0 for independence.", "",
             "## Prespecified hypotheses", ""]
    for k, ok in v.items():
        lines.append(f"* {k}: {'met' if ok else 'not met'}")
    lines += ["", "## Supported direction (training fit)", "",
              "Genes: " + ", ".join(f"{g} ({w:+.2f})" for g, w in genes),
              "", "Protein response to a unit move along it: " + ", ".join(f"{p} ({w:+.2f})" for p, w in prots),
              "", "## All macros", "", "| Macro | Value |", "|---|---|"]
    lines += [f"| `\\{k}` | {val} |" for k, val in m.items()]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    ev, seal, chans, diag = load("evaluation.json"), load("seal.json"), load("channels.json"), load("diagnostics.json")
    prog, des = load("programs.json"), load("design.json")
    m = macros(ev, seal, chans, diag)
    m.update(program_macros(prog))
    m.update(design_macros(des))
    fig, genes, prots = figure(ev, seal, prog, des)
    m["PtTopGenes"] = ", ".join(g for g, _ in genes[:6])
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "pert_macros.tex").write_text("% Generated by extension/perturbation/make_assets.py. Do not edit.\n"
                                             + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in m.items()))
    (args.out / "pert_figure.tex").write_text(fig)
    (args.out / "pert_table.tex").write_text(table(ev))
    (args.out / "pert_program_table.tex").write_text(program_table(prog))
    (R / "RESULTS.md").write_text(results_md(m, ev, genes, prots))
    print(json.dumps(m, indent=1))


if __name__ == "__main__":
    main()
