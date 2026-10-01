#!/usr/bin/env python3
"""Post hoc, exploratory (not part of the frozen analysis or its amendments): the structural law with censoring
taken into account.

Several observed savings are lower bounds: the paired-only envelope never reached the target within the largest
budget, or the proposed estimator reached it at the smallest budget. Spearman's correlation treats these bounds as
exact values. Harrell's concordance index uses only pairs whose ordering is known. A pair is usable if the data set
with the smaller observed saving is uncensored, or if both are uncensored. Ties in the prediction count one half. A
permutation test permutes the predictions over data sets, which keeps the censoring pattern, and so tests the
association.

    python posthoc/law_concordance.py      # results/law_concordance_posthoc.json
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import grun as gr  # noqa: E402

PERMUTATIONS, SEED = 10000, 20261063


def earlier(name):
    """Observed saving and censoring at half the best recovery any method reached (amendment 1's rule), from the
    stored curves of the three earlier data sets."""
    SPR = gr.SP / "results"
    if name == "overcite":
        o = json.loads((gr.SP / "external/results/overcite_plan_estimator.json").read_text())
        rf, budgets = o["rf"], o["budgets"]
    else:
        res = json.loads((SPR / f"{name}.json").read_text())
        budgets = res["budgets"]
        den = sum(r["den"][0] for r in res["rows"])
        rf = {}
        for a in sorted({a for r in res["rows"] for a in r["num"]}):
            if all(str(B) in res["rows"][0]["num"].get(a, {}) for B in budgets):
                rf[a] = [sum(r["num"][a][str(B)][0] for r in res["rows"]) / den for B in budgets]
    methods = [a for a in rf if not a.startswith(("abl", "dose", "paired_all", "independence"))]
    L = max(max(rf[a]) for a in methods)
    env = np.nanmax(np.array([rf[a] for a in gr.PAIRED_ONLY if a in rf]), axis=0)
    n_c, r_c = gr.needed(env, 0.5 * L, budgets)
    n_p, r_p = gr.needed(rf["bjs2/mapped"], 0.5 * L, budgets)
    return n_c / n_p, (r_c == ">" or r_p == "<=")


def concordance(pred, obs, cens):
    num, den = 0.0, 0
    n = len(obs)
    for i in range(n):
        for j in range(n):
            if i == j or not obs[i] < obs[j] or cens[i]:
                continue                        # i has the smaller observed saving and must be exact
            den += 1
            num += 1.0 if pred[i] < pred[j] else (0.5 if pred[i] == pred[j] else 0.0)
    return num / den, den


def main():
    pred = json.loads((gr.RES / "law_predicted.json").read_text())
    names, obs, cens = [], [], []
    for n in gr.PRIMARY:
        a = json.loads((gr.RES / f"{n}_amend1.json").read_text())
        if a["level_ci"][0] <= 0:
            continue                            # not informative (amendment 2)
        v = a["savings"]["paired_only"]["0.5"]
        names.append(n)
        obs.append(v["factor"])
        cens.append(v["comparator_relation"] == ">" or v["proposed_relation"] == "<=")
    for n in ("frangieh", "papalexi", "overcite"):
        s, c = earlier(n)
        names.append(n)
        obs.append(s)
        cens.append(c)
    x = np.array([pred[n]["predicted_saving"] for n in names])
    y, c = np.array(obs), np.array(cens)
    C, pairs = concordance(x, y, c)
    rng = np.random.default_rng(SEED)
    null = np.array([concordance(rng.permutation(x), y, c)[0] for _ in range(PERMUTATIONS)])
    p = float((1 + np.sum(null >= C)) / (1 + PERMUTATIONS))
    unc = ~c
    ratio = {n: float(y[i] / x[i]) for i, n in enumerate(names)}
    out = {"note": "post hoc, exploratory", "datasets": names, "predicted": dict(zip(names, x.tolist())),
           "observed": dict(zip(names, y.tolist())), "censored": dict(zip(names, c.tolist())),
           "concordance": C, "usable_pairs": pairs, "p_one_sided": p, "permutations": PERMUTATIONS,
           "uncensored_spearman": float(__import__("scipy.stats", fromlist=["spearmanr"]).spearmanr(x[unc], y[unc]).correlation)
           if unc.sum() > 2 else None,
           "observed_over_predicted": ratio,
           "censored_consistent": {n: bool(y[i] <= x[i]) for i, n in enumerate(names) if c[i]}}
    (gr.RES / "law_concordance_posthoc.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k not in ("predicted", "observed")}, indent=1))


if __name__ == "__main__":
    main()
