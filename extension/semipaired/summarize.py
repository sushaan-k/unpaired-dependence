#!/usr/bin/env python3
"""Summaries of the semi-paired study: recovered fraction and uncorrected loss by arm and budget, paired-cell
savings at accuracy targets with bootstrap intervals over held-out targets, mechanism ablations and dose-response.

    python summarize.py frangieh papalexi [--champ]

Writes results/summary_<dataset>.json and RESULTS.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROPOSED = "bjs2/mapped"
TARGETS = (0.3, 0.4, 0.5, 0.6)
BOOT, SEED = 1000, 20261039


def load(ds):
    return json.loads((HERE / "results" / f"{ds}.json").read_text())


def arrays(res, extra=None):
    rows = res["rows"]
    budgets = res["budgets"]
    den = np.array([r["den"] for r in rows])
    tn = np.array([r["tn"] for r in rows])
    arms = sorted({a for r in rows for a in r["num"]})
    num, loss = {}, {}
    for a in arms:
        num[a], loss[a] = {}, {}
        keys = sorted({k for r in rows for k in r["num"][a]}, key=int)
        for k in keys:
            if all(k in r["num"][a] for r in rows):
                num[a][int(k)] = np.array([r["num"][a][k] for r in rows])
                loss[a][int(k)] = np.array([r["loss"][a][k] for r in rows])
    if extra:
        for a, d in extra.items():
            num[a], loss[a] = d["num"], d["loss"]
    targets = np.array([r["target"] for r in rows])
    return budgets, den, tn, num, loss, targets


def curve(num, den, arm, budgets, idx=None, block=0):
    sel = slice(None) if idx is None else idx
    out = []
    for b in budgets:
        v = num[arm].get(b, num[arm].get(0))
        out.append(np.nan if v is None else float(v[sel, block].sum() / den[sel, block].sum()))
    return np.array(out)


def needed(budgets, values, target):
    """Paired cells needed to reach target on the running maximum of a curve (log-linear interpolation)."""
    env = np.maximum.accumulate(np.nan_to_num(values, nan=-np.inf))
    b = np.asarray(budgets, float)
    if target <= env[0]:
        return float(b[0]), "<="
    if target > env[-1]:
        return float(b[-1]), ">"
    i = int(np.argmax(env >= target))
    lo, hi = env[i - 1], env[i]
    f = (target - lo) / (hi - lo) if hi > lo else 1.0
    return float(np.exp(np.log(b[i - 1]) + f * (np.log(b[i]) - np.log(b[i - 1])))), "="


def analyse(res, comparators, extra=None):
    budgets, den, tn, num, loss, targets = arrays(res, extra)
    arms = list(num)
    table = {a: curve(num, den, a, budgets).tolist() for a in arms}
    loss_rel = {a: [float(loss[a].get(b, loss[a].get(0))[:, 0].sum() / tn[:, 0].sum()) for b in budgets] for a in arms}
    ids = sorted(set(targets))
    members = [np.flatnonzero(targets == t) for t in ids]
    rng = np.random.default_rng(SEED)
    picks = [np.concatenate([members[i] for i in rng.choice(len(ids), len(ids), replace=True)]) for _ in range(BOOT)]
    comps = [a for a in comparators if a in num]
    out = {"budgets": budgets, "groups": int(len(den)), "targets": int(len(ids)), "rf": table, "loss_rel": loss_rel,
           "blocks": res["blocks"], "rf_blocks": {a: [curve(num, den, a, budgets, block=k).tolist()
                                                      for k in range(den.shape[1])] for a in arms}}
    # the proposed estimator against the upper envelope of all comparators and against each comparator
    prop = curve(num, den, PROPOSED, budgets)
    env = np.nanmax(np.array([curve(num, den, a, budgets) for a in comps]), axis=0)
    diffs = []
    for i, b in enumerate(budgets):
        best = max(comps, key=lambda a: curve(num, den, a, budgets)[i])
        boots = [curve(num, den, PROPOSED, [b], idx)[0] - curve(num, den, best, [b], idx)[0] for idx in picks]
        diffs.append({"budget": b, "best": best, "proposed": float(prop[i]), "best_rf": float(env[i]),
                      "diff": float(prop[i] - env[i]),
                      "ci": [float(np.quantile(boots, .025)), float(np.quantile(boots, .975))]})
    out["versus_best"] = diffs
    sav = {}
    for tgt in TARGETS:
        n_p, r_p = needed(budgets, prop, tgt)
        n_e, r_e = needed(budgets, env, tgt)
        boots = []
        for idx in picks[:300]:
            pc = curve(num, den, PROPOSED, budgets, idx)
            ec = np.nanmax(np.array([curve(num, den, a, budgets, idx) for a in comps]), axis=0)
            bp, _ = needed(budgets, pc, tgt)
            be, _ = needed(budgets, ec, tgt)
            boots.append(be / bp)
        sav[str(tgt)] = {"proposed_cells": n_p, "proposed_relation": r_p, "comparators_cells": n_e,
                         "comparators_relation": r_e, "factor": n_e / n_p,
                         "factor_ci": [float(np.quantile(boots, .025)), float(np.quantile(boots, .975))]}
    out["savings_vs_best"] = sav
    # against (a) paired-only estimation (no unpaired information: James-Stein, SCOSE, FCOSE, low rank, pooled or
    # per condition) and (b) the comparators in their published forms (pooled or per condition, without the map)
    groups_ = {"paired_only": [a for a in num if a.split("/")[0] in ("js", "scose", "fcose", "lowrank")
                               and a.split("/")[-1] in ("pooled", "percond")],
               "published": [a for a in comps if a.split("/")[-1] in ("pooled", "percond") or a == "champollion"]}
    for label, members_ in groups_.items():
        envp = np.nanmax(np.array([curve(num, den, a, budgets) for a in members_]), axis=0)
        sp = {}
        for tgt in TARGETS:
            n_p, _ = needed(budgets, prop, tgt)
            n_e, r_e = needed(budgets, envp, tgt)
            boots = []
            for idx in picks[:300]:
                pc = curve(num, den, PROPOSED, budgets, idx)
                ec = np.nanmax(np.array([curve(num, den, a, budgets, idx) for a in members_]), axis=0)
                boots.append(needed(budgets, ec, tgt)[0] / needed(budgets, pc, tgt)[0])
            sp[str(tgt)] = {"cells": n_e, "relation": r_e, "factor": n_e / n_p,
                            "factor_ci": [float(np.quantile(boots, .025)), float(np.quantile(boots, .975))]}
        out[f"savings_vs_{label}"] = sp
    return out


def champollion(res, ds):
    """The proposed estimator against Champollion on the groups and budgets of results/champ_<ds>.json (best lasso
    weight per budget, re-chosen in every bootstrap resample)."""
    path = HERE / "results" / f"champ_{ds}.json"
    if not path.exists():
        return None
    ch = json.loads(path.read_text())
    full = max(len(r["num"]) for r in ch["rows"])
    ch["rows"] = [r for r in ch["rows"] if len(r["num"]) == full]          # folds completed at every budget
    ch["budgets"] = [b for b in ch["budgets"] if str(b) in ch["rows"][0]["num"]]
    keys = {r["key"]: r for r in ch["rows"]}
    rows = [r for r in res["rows"] if r["key"] in keys]
    budgets = [b for b in ch["budgets"] if all(str(b) in keys[r["key"]]["num"] for r in rows)]
    den = np.array([r["den"][0] for r in rows])
    tn = np.array([r["tn"][0] for r in rows])
    prop = np.array([[r["num"][PROPOSED][str(b)][0] for b in budgets] for r in rows])
    prop_loss = np.array([[r["loss"][PROPOSED][str(b)][0] for b in budgets] for r in rows])
    cs = [str(c) for c in ch["cs"]]
    cham = np.array([[[keys[r["key"]]["num"][str(b)][c][0] for c in cs] for b in budgets] for r in rows])
    cham_loss = np.array([[[keys[r["key"]]["loss"][str(b)][c][0] for c in cs] for b in budgets] for r in rows])

    def curves(idx):
        d = den[idx].sum()
        pc = prop[idx].sum(0) / d
        allc = cham[idx].sum(0) / d                     # budgets x cs
        return pc, allc.max(1), allc.argmax(1)
    pc, cc, ci = curves(slice(None))
    targets = np.array([r["target"] for r in rows])
    ids = sorted(set(targets))
    members = [np.flatnonzero(targets == t) for t in ids]
    rng = np.random.default_rng(SEED + 1)
    picks = [np.concatenate([members[i] for i in rng.choice(len(ids), len(ids), replace=True)]) for _ in range(BOOT)]
    sav = {}
    for tgt in TARGETS:
        n_p, r_p = needed(budgets, pc, tgt)
        n_c, r_c = needed(budgets, cc, tgt)
        boots = []
        for idx in picks:
            bp, bc, _ = curves(idx)
            boots.append(needed(budgets, bc, tgt)[0] / needed(budgets, bp, tgt)[0])
        sav[str(tgt)] = {"proposed_cells": n_p, "proposed_relation": r_p, "champollion_cells": n_c,
                         "champollion_relation": r_c, "factor": n_c / n_p,
                         "factor_ci": [float(np.quantile(boots, .025)), float(np.quantile(boots, .975))]}
    diffs = []
    for i, b in enumerate(budgets):
        bs = [curves(idx)[0][i] - curves(idx)[1][i] for idx in picks[:500]]
        diffs.append({"budget": b, "diff": float(pc[i] - cc[i]),
                      "ci": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]})
    return {"budgets": budgets, "groups": len(rows), "folds": len({r["target"] for r in rows}) if ds != "frangieh" else 1,
            "draws": ch["draws"],
            "proposed": pc.tolist(), "champollion": cc.tolist(), "c": [ch["cs"][k] for k in ci],
            "proposed_loss": (prop_loss.sum(0) / tn.sum()).tolist(),
            "champollion_loss": [float(cham_loss[:, i, k].sum() / tn.sum()) for i, k in enumerate(ci)],
            "savings": sav, "differences": diffs}


def fmt(v):
    return "" if v is None or not np.isfinite(v) else f"{100 * v:.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="+")
    args = ap.parse_args()
    lines = ["# Semi-paired study (development): results", "",
             "Recovered fraction (%) of within-group gene-protein cross-correlation in held-out groups, pooled over",
             "groups and averaged over draws of paired calibration cells; uncorrected relative loss in the second",
             "table. Generated by `summarize.py`.", ""]
    for ds in args.datasets:
        res = load(ds)
        arms = sorted({a for r in res["rows"] for a in r["num"]})
        comparators = [a for a in arms if not a.startswith(("bjs", "abl/", "dose/")) and a not in (
            "paired_all", "independence")]
        s = analyse(res, comparators)
        champ = champollion(res, ds)
        if champ:
            s["versus_champollion"] = champ
        (HERE / "results" / f"summary_{ds}.json").write_text(json.dumps(s, indent=1) + "\n")
        lines += [f"## {ds} ({s['groups']} groups, {s['targets']} targets)", "",
                  "| Arm | " + " | ".join(str(b) for b in s["budgets"]) + " |", "|---|" + "---|" * len(s["budgets"])]
        first = ["bjs2/mapped", "bjs2/pooled", "bjs2/percond", "bjs/mapped", "bjs/pooled", "bjs/percond"]
        order = first + [a for a in arms if a not in first]
        for a in order:
            if a in s["rf"]:
                lines.append(f"| {a} | " + " | ".join(fmt(v) for v in s["rf"][a]) + " |")
        lines += ["", "Uncorrected relative loss (sum over groups of ||C - T||^2 / ||T||^2):", "",
                  "| Arm | " + " | ".join(str(b) for b in s["budgets"]) + " |", "|---|" + "---|" * len(s["budgets"])]
        for a in ("bjs2/mapped", "bjs2/pooled", "bjs/mapped", "js/pooled", "semicca/pooled", "semicca/mapped",
                  "ridge_u/pooled", "ridge_u/mapped", "paired_all", "independence"):
            if a in s["loss_rel"]:
                lines.append(f"| {a} | " + " | ".join(f"{v:.3f}" for v in s["loss_rel"][a]) + " |")
        lines += ["", "Proposed (bjs2/mapped) against the best comparator at each budget (difference, 95% CI over targets):", ""]
        lines += [f"* {d['budget']}: {fmt(d['proposed'])} vs {fmt(d['best_rf'])} ({d['best']}); "
                  f"{fmt(d['diff'])} ({fmt(d['ci'][0])} to {fmt(d['ci'][1])})" for d in s["versus_best"]]
        lines += ["", "Paired cells to reach an accuracy target (proposed vs upper envelope of all comparators):", ""]
        for t, v in s["savings_vs_best"].items():
            lines.append(f"* RF {t}: {v['proposed_relation']} {v['proposed_cells']:.0f} vs {v['comparators_relation']} "
                         f"{v['comparators_cells']:.0f}; factor {v['factor']:.2f} ({v['factor_ci'][0]:.2f} to "
                         f"{v['factor_ci'][1]:.2f})")
        for key, label in (("savings_vs_paired_only", "paired-only estimation (no unpaired information)"),
                           ("savings_vs_published", "the comparators in their published forms (without the map)")):
            lines += ["", f"Against {label}:", ""]
            for t, v in s[key].items():
                lines.append(f"* RF {t}: {v['relation']} {v['cells']:.0f}; factor {v['factor']:.2f} "
                             f"({v['factor_ci'][0]:.2f} to {v['factor_ci'][1]:.2f})")
        if champ:
            lines += ["", f"Against Champollion ({champ['groups']} groups of {champ['folds']} folds; lasso weight c/sqrt(B), "
                      f"c chosen per budget on these groups; {champ['draws']} draws):", "",
                      "| Budget | " + " | ".join(str(b) for b in champ["budgets"]) + " |",
                      "|---|" + "---|" * len(champ["budgets"]),
                      "| bjs2/mapped | " + " | ".join(fmt(v) for v in champ["proposed"]) + " |",
                      "| Champollion | " + " | ".join(fmt(v) for v in champ["champollion"]) + " |",
                      "| c chosen | " + " | ".join(str(v) for v in champ["c"]) + " |",
                      "| Uncorrected loss, bjs2/mapped | " + " | ".join(f"{v:.3f}" for v in champ["proposed_loss"]) + " |",
                      "| Uncorrected loss, Champollion | " + " | ".join(f"{v:.3f}" for v in champ["champollion_loss"]) + " |",
                      ""]
            lines += [f"* RF {t}: {v['proposed_relation']} {v['proposed_cells']:.0f} vs {v['champollion_relation']} "
                      f"{v['champollion_cells']:.0f}; factor {v['factor']:.2f} ({v['factor_ci'][0]:.2f} to "
                      f"{v['factor_ci'][1]:.2f})" for t, v in champ["savings"].items()]
            (HERE / "results" / f"summary_{ds}.json").write_text(json.dumps(s, indent=1) + "\n")
        lines.append("")
    (HERE / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
