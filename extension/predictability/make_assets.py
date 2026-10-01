#!/usr/bin/env python3
"""Manuscript assets of the predictability study: macros and table (revision/source/pred_*.tex) and RESULTS.md,
from results/summary.json, rows.json, secondary.json, the per-data-set results and predictions, the development
results and the theory checks. The figure of the law (law_figure.tex) is drawn by synthesis/make_assets.py, which
imports this module's data-set labels, colours and development rows.

    python make_assets.py [--out ../../revision/source] [--results-md RESULTS.md]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
sys.path.insert(0, str(HERE))
NEW = ("sln111", "sln206", "pbmc10k", "malt10k", "bmcite", "fetal_cortex", "snare_cortex", "fafb_vpn", "banc_vpn",
       "human_gaba")
LABEL = {"sln111": "Mouse spleen and lymph node, 111 proteins", "sln206": "Mouse spleen and lymph node, 206 proteins",
         "pbmc10k": "Human blood, 14 proteins", "malt10k": "Human MALT lymphoma, 14 proteins",
         "bmcite": "Human bone marrow, 25 proteins", "fetal_cortex": "Human fetal cortex, RNA--chromatin",
         "snare_cortex": "Mouse cortex, RNA--chromatin", "fafb_vpn": "Fly visual neurons, FlyWire",
         "banc_vpn": "Fly visual neurons, BANC", "human_gaba": "Human cortex Patch-seq"}
SYSTEM = {"sln111": "prot", "sln206": "prot", "pbmc10k": "prot", "malt10k": "prot", "bmcite": "prot",
          "fetal_cortex": "chrom", "snare_cortex": "chrom", "fafb_vpn": "wire", "banc_vpn": "wire", "human_gaba": "ephys"}
SHORT = {"sln111": "Mouse lymphoid organs, 111 proteins", "sln206": "Mouse lymphoid organs, 206 proteins",
         "fafb_vpn": "Fly visual neurons, FlyWire", "banc_vpn": "Fly visual neurons, BANC"}
MARK = {"sln111": "*", "sln206": "square*", "fafb_vpn": "triangle*", "banc_vpn": "diamond*"}
COL = {"prot": "protcol", "chrom": "chromcol", "wire": "wirecol", "ephys": "ephyscol"}
SANS = "font=\\sffamily"
COMMON = (" axis x line*=bottom,axis y line*=left,tick align=outside,"
          "ymajorgrids=true,xmajorgrids=true,grid style={gray!18},axis line style={gray!55},"
          "tick label style={font=\\sffamily\\scriptsize},label style={font=\\sffamily\\small},clip=false,"
          "title style={font=\\sffamily\\small,at={(0,1)},anchor=base west,yshift=5pt}")
WORDS = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight",
         9: "nine", 10: "ten", 11: "eleven", 12: "twelve"}


def load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def pct(log2err):
    return f"{100 * (2 ** log2err - 1):.0f}"


def fac(v):
    return f"{v:.2f}" if v < 1.15 else f"{v:.1f}"


def dev_rows():
    """Development problems (post hoc) with an estimable saving at 0.5: observed and law."""
    import ptools as pt
    from dev_summary import saving
    out = []
    for p in sorted((RES / "dev").glob("*.json")):
        if p.name == "summary.json" or p.name.endswith("_pilot.json"):
            continue
        r = json.loads(p.read_text())
        w, pi = pt.shares(r["pred"]["population"]["blocks"])
        s, cens, _ = saving(r["observed"]["js"], r["observed"]["bjs2"], 0.5, r["budgets"])
        if not cens and not r["dataset"].startswith("scala"):
            out.append((r["dataset"], s, pt.saving_closed(w, pi, 0.5)))
    return out


def macros(summ, rows, sec, dev, chk, drec, frz):
    m = {}
    m["PredFrozenAt"] = frz["frozen_at"][11:16] + " EDT"
    m["PredDataAt"] = drec["written_at"][11:16] + " EDT"
    m["PredNData"] = str(len(summ["datasets"]))
    m["PredNProblems"] = str(summ["problems"])
    t5 = summ["targets"]["0.5"]
    est = [r for r in rows if r["target"] == 0.5 and r["estimable"]]
    m["PredNEstimable"] = str(t5["estimable"])
    m["PredNEstimableData"] = str(len({r["dataset"] for r in est}))
    m["PredNCensored"] = str(t5["problems"] - t5["estimable"])
    law = t5["law"]
    m["PredLawErr"] = pct(law["median_abs_log2"])
    m["PredLawErrLo"], m["PredLawErrHi"] = pct(law["ci"][0]), pct(law["ci"][1])
    m["PredLawWithin"] = f"{100 * law['within_25pct']:.0f}"
    m["PredLawRatio"] = f"{2 ** law['median_log2_bias']:.2f}"
    m["PredRho"] = f"{t5['spearman_law']['rho']:.2f}"
    p = t5["spearman_law"]["p_one_sided"]
    m["PredRhoRel"] = "<0.001" if p < 0.001 else f"={p:.3f}"
    m["PredRandomObs"] = f"{t5['random_control']['median_observed']:.2f}"
    m["PredRandomLaw"] = f"{t5['random_control']['median_law']:.2f}"
    m["PredRandomN"] = str(t5["random_control"]["n"])
    for key, lab in (("pilot", "Pilot"), ("transfer", "Transfer")):
        v = t5[key]
        m[f"Pred{lab}Err"] = pct(v["median_abs_log2"])
        m[f"Pred{lab}ErrLo"], m[f"Pred{lab}ErrHi"] = pct(v["ci"][0]), pct(v["ci"][1])
        m[f"Pred{lab}Within"] = f"{100 * v['within_25pct']:.0f}"
    for tg, lab in (("0.7", "Seven"), ("0.3", "Three")):
        v = summ["targets"][tg]
        m[f"PredLawErr{lab}"] = pct(v["law"]["median_abs_log2"])
        m[f"PredN{lab}"] = str(v["estimable"])
        m[f"PredLawRatio{lab}"] = f"{2 ** v['law']['median_log2_bias']:.2f}"
        m[f"PredPilotErr{lab}"] = pct(v["pilot"]["median_abs_log2"])
    vd = summ["verdicts"]
    for k, lab in (("H1_law_calibrated", "One"), ("H2_law_ranks", "Two"), ("H3_random_control_no_saving", "Three"),
                   ("H4_pilot_calibrated", "Four"), ("H5_transfer_calibrated", "Five"),
                   ("H6_within_unpaired_bounds", "Six")):
        m[f"PredH{lab}"] = "met" if vd[k] else "not met"
    obs = [r["observed"] for r in est]
    m["PredSaveMin"], m["PredSaveMax"] = fac(min(obs)), fac(max(obs))
    s5 = sec["summary"]["0.5"]
    m["PredBootErr"] = pct(abs(s5["boot_median_abs_log2"]))
    m["PredBootRatio"] = f"{2 ** s5['boot_median_log2_bias']:.2f}"
    m["PredCellsBlocksRatio"] = f"{2 ** s5['law_cells_blocks_median_log2']:.2f}"
    m["PredCellsJsRatio"] = f"{2 ** s5['law_cells_js_median_log2']:.2f}"
    m["PredBootErrSeven"] = pct(sec["summary"]["0.7"]["boot_median_abs_log2"])
    m["PredBootErrThree"] = pct(sec["summary"]["0.3"]["boot_median_abs_log2"])
    m["PredCurveErrBlocks"] = f"{100 * sec['summary']['curve_mean_abs_err']['bjs2']:.1f}"
    m["PredCurveErrJs"] = f"{100 * sec['summary']['curve_mean_abs_err']['js']:.1f}"
    cc = s5["censoring_checks"]
    agree = sum(v[0] for v in cc.values())
    total = sum(v[1] for v in cc.values())
    m["PredCensAgree"], m["PredCensTotal"] = str(agree), str(total)
    # SNARE-seq: cells the proposed estimator needed against the law's
    snare = [r for r in rows if r["dataset"] == "snare_cortex" and r["target"] == 0.5 and r["rel_blocks"] == "="]
    if snare:
        m["PredSnareObsMin"] = f"{min(r['cells_blocks'] for r in snare):,.0f}"
        m["PredSnareObsMax"] = f"{max(r['cells_blocks'] for r in snare):,.0f}"
        m["PredSnareLawMin"] = f"{min(r['law_cells_blocks'] for r in snare):,.0f}"
        m["PredSnareLawMax"] = f"{max(r['law_cells_blocks'] for r in snare):,.0f}"
    easy = sorted({r["dataset"] for r in rows if r["target"] == 0.5 and r["rel_blocks"] == "<="})
    m["PredNEasy"] = str(len(easy))
    # development (post hoc)
    e = [abs(np.log2(s / l)) for _, s, l in dev]
    m["PredDevN"] = str(len(dev))
    m["PredDevLawErr"] = pct(float(np.median(e)))
    m["PredDevLawErrMax"] = pct(float(np.max(e)))
    m["PredDevSaveMin"], m["PredDevSaveMax"] = fac(min(s for _, s, _ in dev)), fac(max(s for _, s, _ in dev))
    s13 = chk["S13_summary"]
    m["PredCheckCases"] = str(s13["cases"])
    m["PredCheckMaxRatio"] = f"{s13['max_gap_over_bound']:.2f}"
    # development signal-share profile: share of the dependence in the leading RNA eigen-blocks
    prof = json.loads((RES / "transfer_profile.json").read_text())
    wb = np.array(prof["w_bar"])
    m["PredProfileTop"] = f"{100 * wb[0].sum():.0f}"
    m["PredProfileTopTwenty"] = f"{100 * wb[:2].sum():.0f}"
    m["PredProfileN"] = str(len(prof["datasets"]))
    for k in ("PredNData", "PredNEstimableData", "PredNSeven", "PredNThree", "PredNEasy", "PredProfileN"):
        if m[k].isdigit() and int(m[k]) in WORDS:
            m[k] = WORDS[int(m[k])]
    return m


def write_macros(path, m):
    lines = ["% Generated by extension/predictability/make_assets.py. Do not edit."]
    for k in sorted(m):
        lines.append(f"\\newcommand{{\\{k}}}{{{m[k]}}}")
    Path(path).write_text("\n".join(lines) + "\n")


def table(rows, preds):
    reason = {}
    for n in NEW:
        rs = [r for r in rows if r["dataset"] == n and r["target"] == 0.5]
        if any(r["estimable"] for r in rs):
            reason[n] = ""
        elif all(r["rel_blocks"] == "<=" for r in rs):
            reason[n] = "both above target at 25 cells"
        elif all(r["rel_js"] == ">" for r in rs) and any(r["rel_blocks"] == "=" for r in rs):
            reason[n] = "paired-only not reached"
        else:
            reason[n] = "target not reached"
    L = ["% Generated by extension/predictability/make_assets.py. Do not edit.",
         "\\begin{tabular}{@{}llrrlllll@{}}", "\\toprule",
         "Data set & Pair & Cells & Panel & Observed & Law & Pilot & No pairs & Bound \\\\", "\\midrule"]
    pair = {"prot": "RNA--protein", "chrom": "RNA--chromatin", "wire": "wiring", "ephys": "RNA--electrophysiology"}
    for n in NEW:
        first = True
        for r in [r for r in rows if r["dataset"] == n and r["target"] == 0.5]:
            p = preds[n][r["problem"]]["n"]["p"]
            if r["estimable"]:
                o = f"{r['observed']:.2f}"
            elif r["rel_js"] == ">" and r["rel_blocks"] == "=":
                o = f"$\\geq${r['observed']:.2f}"
            else:
                o = "n.e."
            cells = (sum(preds[n][r["problem"]]["n"][k] for k in ("pool_x", "pool_y", "reservoir", "truth"))
                     if first else "")
            L.append(f"{LABEL[n] if first else ''} & {pair[SYSTEM[n]] if first else ''} & "
                     f"{cells if cells == '' else f'{cells:,}'} & {p} & {o} & {r['law']:.2f} & "
                     f"{r['pilot']:.2f} & {r['transfer']:.2f} & {r['bound']:.0f} \\\\")
            first = False
        if reason[n]:
            L.append(f"\\multicolumn{{9}}{{@{{}}l}}{{\\quad\\textit{{n.e.: {reason[n]}}}}} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(L) + "\n"


def results_md(summ, rows, sec):
    L = ["# Predictability study: results", "",
         "Frozen plan: `PLAN.md` (`results/freeze.json`); data record `results/data_freeze.json`; deviations "
         "`DEVIATIONS.md`. Savings are cells needed by one-block James-Stein divided by cells needed by block "
         "James-Stein in the unpaired bases, at a recovered fraction of the population's dependence.", "",
         f"* Verdicts: {json.dumps(summ['verdicts'])}"]
    for tg, v in summ["targets"].items():
        L.append(f"* Target {tg}: {v['estimable']} of {v['problems']} problems estimable; law median |log2| "
                 f"{v['law']['median_abs_log2']:.3f} (CI {v['law']['ci'][0]:.3f}-{v['law']['ci'][1]:.3f}), bias "
                 f"{v['law']['median_log2_bias']:+.3f}; pilot {v['pilot']['median_abs_log2']:.3f}; transfer "
                 f"{v['transfer']['median_abs_log2']:.3f}; Spearman {v.get('spearman_law', {}).get('rho', float('nan')):.3f}; "
                 f"random control {v['random_control']['median_observed']:.3f}")
    L.append(f"* Secondary: {json.dumps(sec['summary'])}")
    L += ["", "| Data set | Problem | Target | Observed | Estimable | Law | Pilot | Transfer | Bound | Random observed |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['dataset']} | {r['problem']} | {r['target']} | {r['observed']:.2f} ({r['rel_js']}/{r['rel_blocks']}) | "
                 f"{r['estimable']} | {r['law']:.2f} | {r['pilot']:.2f} | {r['transfer']:.2f} | {r['bound']:.1f} | "
                 f"{r['random_observed']:.2f} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE.parent.parent / "revision" / "source"))
    ap.add_argument("--results-md", default=str(HERE / "RESULTS.md"))
    a = ap.parse_args()
    summ, rows, sec = load(RES / "summary.json"), load(RES / "rows.json"), load(RES / "secondary.json")
    chk, drec, frz = load(RES / "check_theory.json"), load(RES / "data_freeze.json"), load(RES / "freeze.json")
    preds = {n: load(RES / f"{n}_predictions.json") for n in NEW}
    dev = dev_rows()
    out = Path(a.out)
    write_macros(out / "pred_macros.tex", macros(summ, rows, sec, dev, chk, drec, frz))
    (out / "pred_table.tex").write_text(table(rows, preds))
    Path(a.results_md).write_text(results_md(summ, rows, sec))
    print("assets written")


if __name__ == "__main__":
    main()
