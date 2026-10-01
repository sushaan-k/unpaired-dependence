#!/usr/bin/env python3
"""Manuscript assets of the replication: macros (rep_macros.tex), a supplementary table and results/RESULTS.md.

    python make_assets.py --out ../../revision/source
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
R = HERE / "results"
PRIMARY, PAIRED = "pf_within_measured", "paired_closed_form"


def load(name):
    return json.loads((R / name).read_text())


def signed(text):
    return "\\ensuremath{-}" + text[1:] if text.startswith("-") else text


def pct(v, ref=None):
    p, r = 100 * v, 100 * (v if ref is None else ref)
    return signed(f"{p:.1f}" if abs(r) < 10 else f"{p:.0f}")


def macros(ev, seal, diag):
    t, n = ev["test"], ev["nt"]
    iv = t["intervals"]
    m = {"RpCells": f"{seal['used_cells']:,}", "RpTargets": str(len(seal["targets"])), "RpGroups": str(t["groups"]),
         "RpTestTargets": str(t["targets"]), "RpReplicates": str(len(seal["replicates"])),
         "RpGenes": str(len(seal["genes"])), "RpProteins": str(len(seal["proteins"])),
         "RpIfnGenes": str(len(ev["blocks"]["ifn_genes"])),
         "RpDimSMedian": f"{np.median([r['dim_S'] for r in diag if r['part'] == 'test']):.0f}",
         "RpDimSMin": str(min(r["dim_S"] for r in diag if r["part"] == "test")),
         "RpDimSMax": str(max(r["dim_S"] for r in diag if r["part"] == "test"))}
    for lab, tag in (("primary", "Primary"), ("primary_minus_pseudo", "DiffPseudo"),
                     ("primary_minus_derange", "DiffDerange"), ("ifn_share_paired", "IfnShare"),
                     ("random_minus_exposure_removal", "RemDiff"), ("paired", "Paired"), ("ifn_primary", "IfnPrimary"),
                     ("ifn_paired", "IfnPaired"), ("share_paired", "Share"),
                     ("ifn_random_minus_exposure_removal", "IfnRemDiff"), ("along_S_primary", "AlongPrimary"),
                     ("along_S_share_paired", "AlongShare"), ("ifn_transferred", "IfnTrans"),
                     ("ifn_replicate", "IfnReplicate")):
        v = iv[lab]
        m[f"Rp{tag}"] = pct(v["value"])
        m[f"Rp{tag}Lo"] = pct(v["ci"][0], v["value"]) if v["ci"] else "--"
        m[f"Rp{tag}Hi"] = pct(v["ci"][1], v["value"]) if v["ci"] else "--"
    s = t["summary"]
    for b, tag in (("all", "All"), ("ifn", "Ifn"), ("complement", "Comp"), ("along_S", "Along")):
        for k, u in ((PRIMARY, "Pf"), (PAIRED, "Paired"), ("transferred_correlation", "Trans"),
                     ("reference_regression", "RefReg"), ("benchmark", "Bench"), ("replicate", "Repl"),
                     ("pseudo", "Pseudo"), ("derange", "Derange"), ("remove_exposure", "RemExp"),
                     ("remove_strength", "RemStr"), ("remove_random_mean", "RemRand")):
            m[f"Rp{tag}{u}"] = pct(s[b][k])
        m[f"Rp{tag}Rel"] = f"{s[b]['reliability']:.2f}"
    for rep, sr in t["by_replicate"].items():
        tag = {"rep1": "RepOne", "rep2": "RepTwo", "rep3": "RepThree", "rep4": "RepFour"}[rep.split("-")[0]]
        m[f"Rp{tag}Pf"] = pct(sr["all"][PRIMARY])
        m[f"Rp{tag}IfnPf"] = pct(sr["ifn"][PRIMARY])
        m[f"Rp{tag}IfnPaired"] = pct(sr["ifn"][PAIRED])
    m["RpNtPrimary"] = pct(n["intervals"]["primary"]["value"])
    m["RpNtPrimaryLo"] = pct(n["intervals"]["primary"]["ci"][0], n["intervals"]["primary"]["value"])
    m["RpNtPrimaryHi"] = pct(n["intervals"]["primary"]["ci"][1], n["intervals"]["primary"]["value"])
    m["RpNtPaired"] = pct(n["intervals"]["paired"]["value"])
    ns = n["summary"]
    m["RpNtIfnPf"] = pct(ns["ifn"][PRIMARY])
    m["RpNtIfnPaired"] = pct(ns["ifn"][PAIRED])
    m["RpNtIfnRepl"] = pct(ns["ifn"]["replicate"])
    removed = [r["sets"]["exposure"] for r in diag if r["part"] == "test"]
    counts = {}
    for s_ in removed:
        for g in s_:
            counts[g] = counts.get(g, 0) + 1
    m["RpTopRemoved"] = ", ".join(sorted(counts, key=lambda g: (-counts[g], g))[:5])
    ph = load("posthoc_penalty.json")
    for sc, tag in (("0.1", "PointOne"), ("1", "One")):
        m[f"RpFixed{tag}All"] = pct(ph["test"][f"scale{sc}/all"])
        m[f"RpFixed{tag}Ifn"] = pct(ph["test"][f"scale{sc}/ifn"])
        m[f"RpFixed{tag}Dim"] = f"{ph['test'][f'scale{sc}/dim_S']:.0f}"
    v = ev["verdict"]
    for k, tag in (("R1_recovery", "ROne"), ("R3_ifn_share_above_half", "RThree"), ("R4_nontargeting", "RFour"),
                   ("R5_exposure_targets_matter", "RFive")):
        m[f"Rp{tag}"] = "met" if v[k] else "not met"
    m["RpRTwo"] = "met" if (v["R2_beats_pseudo_populations"] and v["R2_beats_derangements"]) else "not met"
    return m


def table(ev):
    s, n = ev["test"]["summary"], ev["nt"]["summary"]
    rows = [(PRIMARY, "No pairs: perturbations (primary)"), ("pf_within_latent", "No pairs: corrected RNA"),
            ("pseudo", "No pairs: pseudo-populations"), ("derange", "No pairs: derangements"),
            ("remove_exposure", "No pairs: without 5 highest-exposure targets"),
            ("remove_strength", "No pairs: without 5 strongest targets"),
            ("remove_random_mean", "No pairs: without 5 random targets"),
            (PAIRED, "Paired closed form"), ("reference_regression", "Reference regression"),
            ("transferred_correlation", "Transferred correlation"), ("benchmark", "Own paired adaptation cells"),
            ("replicate", "Adaptation half as replicate")]
    num = lambda v: signed(f"{100 * v:.1f}")
    lines = ["% Generated by extension/perturbation_replication/make_assets.py. Do not edit.",
             "\\begin{tabular}{lccccc}", "\\toprule",
             "& \\multicolumn{3}{c}{Test targets} & \\multicolumn{2}{c}{Non-targeting cells} \\\\",
             "\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}",
             "Arm & All & Interferon block & Complement & All & Interferon block \\\\", "\\midrule"]
    for k, label in rows:
        cells = [num(s[b][k]) for b in ("all", "ifn", "complement")]
        cells += [num(n[b][k]) if k in n[b] else "--" for b in ("all", "ifn")]
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    ev, seal, diag = load("evaluation.json"), load("seal.json"), load("diagnostics.json")
    m = macros(ev, seal, diag)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "rep_macros.tex").write_text("% Generated by extension/perturbation_replication/make_assets.py. Do not edit.\n"
                                            + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in m.items()))
    (args.out / "rep_table.tex").write_text(table(ev))
    lines = ["# Replication in the Papalexi screen: results", "", "Generated by `make_assets.py` from "
             "`results/evaluation.json` (sealed).", "", "## Prespecified hypotheses", ""]
    lines += [f"* {k}: {'met' if v else 'not met'}" for k, v in ev["verdict"].items()]
    lines += ["", "## All macros", "", "| Macro | Value |", "|---|---|"] + [f"| `\\{k}` | {v} |" for k, v in m.items()]
    (R / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(m, indent=1))


if __name__ == "__main__":
    main()
