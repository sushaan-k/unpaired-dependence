#!/usr/bin/env python3
"""Decision rule and summaries of the hybrid development comparison (PLAN.md).

    python summarize.py            # reads results/frangieh.json and results/papalexi.json

Writes results/summary.json and RESULTS.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BOOT, SEED = 1000, 20261027
UNPAIRED = ("current", "splitup", "noise_aware")
COMPARATORS = ("independence", "current", "splitup", "noise_aware", "paired_corr", "paired_js", "paired_lowrank",
               "paired_eot", "block_js", "combine_global", "hybrid_pc")
REPORTED = COMPARATORS + ("hybrid", "hybrid_current", "paired_all")
CRIT_BUDGETS, SAVING_BUDGETS = (100, 200, 400), (100, 200)
POSTHOC = ("ridge_paired", "ridge_unpaired", "hybrid_pooled", "hybrid_pc_pooled")


def load(dataset):
    return json.loads((HERE / "results" / f"{dataset}.json").read_text())


def arrays(res, part):
    rows = res["parts"][part]["rows"]
    budgets = [int(b) for b in res["settings"]["budgets"]]
    den = np.array([r["den"] for r in rows])
    num = {}
    for a in REPORTED:
        num[a] = {}
        for b in budgets:
            key = "0" if a in UNPAIRED + ("independence", "paired_all") else str(b)
            if all(key in r["num"][a] for r in rows):
                num[a][b] = np.array([r["num"][a][key] for r in rows])
    targets = np.array([r["target"] for r in rows])
    path = HERE / "results" / f"{res['dataset']}_posthoc.json"
    if path.exists():
        ph = json.loads(path.read_text())["parts"][part]["rows"]
        assert [r["key"] for r in rows] == [r["key"] for r in ph]
        for a in POSTHOC:
            num[a] = {b: np.array([r["num"][a][str(b)] for r in ph]) for b in budgets if str(b) in ph[0]["num"][a]}
    return budgets, den, num, targets


def rf(num, den, idx=None, block=0):
    if idx is None:
        return float(num[:, block].sum() / den[:, block].sum())
    return float(num[idx, block].sum() / den[idx, block].sum())


def needed(env_budgets, env_values, value):
    """Paired cells the comparator envelope needs to reach value (log-linear interpolation)."""
    env = np.maximum.accumulate(np.asarray(env_values, float))
    b = np.asarray(env_budgets, float)
    if value <= env[0]:
        return float(b[0]), "<="
    if value > env[-1]:
        return float(b[-1]), ">"
    i = int(np.argmax(env >= value))
    lo, hi = env[i - 1], env[i]
    frac = (value - lo) / (hi - lo) if hi > lo else 1.0
    return float(np.exp(np.log(b[i - 1]) + frac * (np.log(b[i]) - np.log(b[i - 1])))), "="


def posthoc_table(dataset, res, part):
    path = HERE / "results" / f"{dataset}_posthoc.json"
    if not path.exists():
        return {}
    ph = json.loads(path.read_text())["parts"][part]["rows"]
    rows = res["parts"][part]["rows"]
    assert [r["key"] for r in rows] == [r["key"] for r in ph]
    den = np.array([r["den"] for r in rows])
    out = {}
    for a in POSTHOC:
        budgets = sorted({int(b) for r in ph for b in r["num"].get(a, {})})
        out[a] = {b: [float(v) for v in (np.array([r["num"][a][str(b)] for r in ph]).sum(0) / den.sum(0))]
                  for b in budgets}
    return out


def analyse(res):
    out = {"blocks": res["blocks"], "parts": {}}
    for part in ("test", "nt"):
        budgets, den, num, targets = arrays(res, part)
        table = {a: {b: [rf(v[b], den, block=k) for k in range(den.shape[1])] for b in v} for a, v in num.items()}
        block = {"table": table, "budgets": budgets, "groups": int(len(den)), "targets": int(len(set(targets)))}
        if part == "test":
            ids = sorted(set(targets))
            members = [np.flatnonzero(targets == t) for t in ids]
            rng = np.random.default_rng(SEED)
            picks = [np.concatenate([members[i] for i in rng.choice(len(ids), len(ids), replace=True)])
                     for _ in range(BOOT)]
            crit = {}
            for b in budgets:
                comp = {a: table[a][b][0] for a in COMPARATORS if b in table[a]}
                best = max(comp, key=comp.get)
                diff = table["hybrid"][b][0] - comp[best]
                boots = [rf(num["hybrid"][b], den, idx) - rf(num[best][b], den, idx) for idx in picks]
                crit[b] = {"best": best, "best_rf": comp[best], "hybrid_rf": table["hybrid"][b][0], "diff": diff,
                           "ci": [float(np.quantile(boots, .025)), float(np.quantile(boots, .975))]}
            env = [max(table[a][b][0] for a in COMPARATORS if b in table[a]) for b in budgets]
            savings = {}
            for b0 in budgets:
                if b0 not in table["hybrid"]:
                    continue
                n, how = needed(budgets, env, table["hybrid"][b0][0])
                boots = []
                for idx in picks[:200]:
                    e = [max(rf(num[a][b], den, idx) for a in COMPARATORS if b in num[a]) for b in budgets]
                    nb, _ = needed(budgets, e, rf(num["hybrid"][b0], den, idx))
                    boots.append(nb / b0)
                savings[b0] = {"needed": n, "relation": how, "factor": n / b0,
                               "factor_ci": [float(np.quantile(boots, .025)), float(np.quantile(boots, .975))]}
            block.update(criteria=crit, savings=savings, envelope=env)
            # value of unpaired marginal information: block_js against paired-only arms (and post hoc ridge arms)
            for label, arms_ in (("paired_only", ("paired_corr", "paired_js", "paired_lowrank", "paired_eot")),
                                 ("paired_only_and_ridge", ("paired_corr", "paired_js", "paired_lowrank", "paired_eot",
                                                            "ridge_paired", "ridge_unpaired"))):
                if not all(a in num for a in arms_):
                    continue
                envp = [max(table[a][b][0] for a in arms_ if b in table[a]) for b in budgets]
                sv = {}
                for b0 in budgets:
                    n_, how = needed(budgets, envp, table["block_js"][b0][0])
                    boots = []
                    for idx in picks[:200]:
                        e = [max(rf(num[a][b], den, idx) for a in arms_ if b in num[a]) for b in budgets]
                        nb, _ = needed(budgets, e, rf(num["block_js"][b0], den, idx))
                        boots.append(nb / b0)
                    sv[b0] = {"needed": n_, "relation": how, "factor": n_ / b0,
                              "factor_ci": [float(np.quantile(boots, .025)), float(np.quantile(boots, .975))]}
                block[f"block_js_savings_vs_{label}"] = sv
            block["pass1"] = all(crit[b]["ci"][0] > 0 for b in CRIT_BUDGETS if b in crit)
            block["pass2"] = all(savings[b]["factor"] >= 2 and savings[b]["relation"] != "<=" for b in SAVING_BUDGETS)
        block["posthoc"] = {a: table[a] for a in POSTHOC if a in table}
        out["parts"][part] = block
    out["fold_info"] = res["parts"]["test"]["fold_info"]
    return out


def fmt(v):
    return f"{100 * v:.1f}"


def report(summary):
    lines = ["# Hybrid development comparison: results", "",
             "Generated by `summarize.py` from `results/frangieh.json` and `results/papalexi.json`. Development data;",
             "nothing here is confirmatory. Recovered fraction (%) of within-group gene-protein cross-correlation,",
             "pooled over evaluation groups and averaged over draws of paired calibration cells.", ""]
    decision = all(s["parts"]["test"]["pass1"] and s["parts"]["test"]["pass2"] for s in summary.values())
    lines += [f"**Decision rule: {'met, pursue the hybrid' if decision else 'not met, the hybrid is not pursued'}.**", ""]
    for ds, s in summary.items():
        t = s["parts"]["test"]
        lines += [f"## {ds}: held-out targets ({t['groups']} groups, {t['targets']} targets)", "",
                  f"Criterion 1 (beats the best comparator at 100, 200, 400 cells): {'met' if t['pass1'] else 'not met'}. "
                  f"Criterion 2 (saving factor of at least 2 at 100 and 200 cells): {'met' if t['pass2'] else 'not met'}.",
                  "", "| Arm | " + " | ".join(str(b) for b in t["budgets"]) + " |",
                  "|---|" + "---|" * len(t["budgets"])]
        for a in REPORTED:
            if a in t["table"]:
                lines.append(f"| {a} | " + " | ".join(fmt(t["table"][a][b][0]) if b in t["table"][a] else "" for b in
                                                     t["budgets"]) + " |")
        lines += ["", "| Budget | Best comparator | Its RF | Hybrid RF | Difference (95% CI) | Cells the comparators need | Factor (95% CI) |",
                  "|---|---|---|---|---|---|---|"]
        for b in t["budgets"]:
            c = t["criteria"][b]
            sv = t["savings"].get(b)
            lines.append(f"| {b} | {c['best']} | {fmt(c['best_rf'])} | {fmt(c['hybrid_rf'])} | {fmt(c['diff'])} "
                         f"({fmt(c['ci'][0])} to {fmt(c['ci'][1])}) | {sv['relation']} {sv['needed']:.0f} | "
                         f"{sv['factor']:.2f} ({sv['factor_ci'][0]:.2f} to {sv['factor_ci'][1]:.2f}) |")
        for key, label in (("block_js_savings_vs_paired_only", "paired-only arms"),
                           ("block_js_savings_vs_paired_only_and_ridge", "paired-only and ridge arms")):
            if key in t:
                lines += ["", f"Paired cells that the {label} need to match `block_js` (factor, 95% CI):", ""]
                lines += [f"* {b} cells: {v['relation']} {v['needed']:.0f} ({v['factor']:.2f}; {v['factor_ci'][0]:.2f} to "
                          f"{v['factor_ci'][1]:.2f})" for b, v in t[key].items()]
        if t.get("posthoc"):
            lines += ["", "Post hoc arms (added after scoring; not in the decision rule):", "",
                      "| Arm | " + " | ".join(str(b) for b in t["budgets"]) + " |", "|---|" + "---|" * len(t["budgets"])]
            for a, v in t["posthoc"].items():
                lines.append(f"| {a} | " + " | ".join(fmt(v[b][0]) if b in v else "" for b in t["budgets"]) + " |")
        blocks = s["blocks"]
        for b in (200, 800):
            if b not in t["budgets"]:
                continue
            lines += ["", f"Recovery by block at {b} cells:", "", "| Arm | " + " | ".join(blocks) + " |",
                      "|---|" + "---|" * len(blocks)]
            for a in ("noise_aware", "paired_js", "block_js", "combine_global", "hybrid_pc", "hybrid", "paired_all"):
                if b in t["table"].get(a, {}):
                    lines.append(f"| {a} | " + " | ".join(fmt(v) for v in t["table"][a][b]) + " |")
        n = s["parts"]["nt"]
        lines += ["", f"Non-targeting cells ({n['groups']} groups), all genes and proteins:", "",
                  "| Arm | " + " | ".join(str(b) for b in n["budgets"]) + " |", "|---|" + "---|" * len(n["budgets"])]
        for a in REPORTED:
            if a in n["table"]:
                lines.append(f"| {a} | " + " | ".join(fmt(n["table"][a][b][0]) if b in n["table"][a] else "" for b in
                                                     n["budgets"]) + " |")
        dims = [f.get("dim_noise_aware") for f in s["fold_info"]]
        lines += ["", f"Programs selected by the noise-aware estimator (per fold): {sorted(set(dims))}; "
                      f"current estimator's supported dimension: {sorted({f.get('dim_current') for f in s['fold_info']})}.", ""]
    (HERE / "RESULTS.md").write_text("\n".join(lines) + "\n")
    return decision


def main():
    summary = {}
    for ds in ("frangieh", "papalexi"):
        if (HERE / "results" / f"{ds}.json").exists():
            summary[ds] = analyse(load(ds))
    decision = report(summary)
    (HERE / "results/summary.json").write_text(json.dumps({"decision": decision, **summary}, indent=1, default=float) + "\n")
    print((HERE / "RESULTS.md").read_text())


if __name__ == "__main__":
    main()
