#!/usr/bin/env python3
"""Development run of the paired-reference selection experiment on the untouched blood
cohort, already scored (exploratory; not a test). See reference.py.

For each donor: adaptation cells of the kept level-1 types are the unpaired
study and the pool of cells that could be paired; targets are the scoring
half's within-type cross-correlation (stored) and its total cross-correlation
over kept-type cells. Writes results/dev_reference.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
CS = HERE.parent / "cross_study"
sys.path.insert(0, str(CS))

from data import cross_correlation, load_recipient  # noqa: E402

import reference as ref  # noqa: E402

BUDGETS = (50, 100, 200, 400, 800)
DRAWS = 50
STRATEGIES = ("random", "balanced", "guided")


def rel(C, T):
    return float(np.linalg.norm(C - T) / np.linalg.norm(T))


def main():
    scores = json.loads((CS / "results/scores.json").read_text())["scores"]
    truth = np.load(CS / "results/truth.npz")
    adapt = load_recipient("hao", "adaptation")
    score = load_recipient("hao", "scoring")
    rng = np.random.default_rng(20260930)
    out = {}
    for donor in sorted(set(adapt["donor"])):
        tag = f"hao/{donor}/within_l1"
        types = scores[tag]["types"]
        m = adapt["donor"] == donor
        labels = adapt["labels"]["l1"][m]
        keep = np.isin(labels, types)
        x, y, labels = adapt["x"][m][keep], adapt["y"][m][keep], labels[keep]
        s = ref.summaries(x, y, labels, types)
        T_within = truth[f"{tag}/T"]
        ms = score["donor"] == donor
        sl = score["labels"]["l1"][ms]
        sk = np.isin(sl, types)
        T_total = cross_correlation(score["x"][ms][sk], score["y"][ms][sk])
        C_between = ref.between_estimate(s)
        res = {"pi": s["pi"].tolist(), "V": s["V"].tolist(), "types": types,
               "E_between_only_total": rel(C_between, T_total), "budgets": {}}
        for N in BUDGETS:
            row = {}
            for strat in STRATEGIES:
                ew, et, alloc = [], [], []
                for _ in range(DRAWS):
                    n_t = ref.allocate(strat, N, s, rng)
                    idx = ref.select(n_t, labels, types, rng)
                    Cw = ref.within_estimate(x, y, labels, idx, s)
                    ew.append(rel(Cw, T_within))
                    et.append(rel(C_between + ref.within_to_total(Cw, s), T_total))
                    alloc.append(n_t)
                row[strat] = {"E_within": float(np.mean(ew)), "E_total_hybrid": float(np.mean(et)),
                              "mean_alloc": np.mean(alloc, 0).tolist()}
            po = []
            for _ in range(DRAWS):
                idx = rng.choice(len(x), size=min(N, len(x)), replace=False)
                po.append(rel(ref.paired_only_total(x, y, idx, s), T_total))
            row["paired_only_total_random"] = float(np.mean(po))
            res["budgets"][str(N)] = row
            print(donor, N, {k: (round(v["E_within"], 3), round(v["E_total_hybrid"], 3)) for k, v in row.items()
                             if isinstance(v, dict)}, round(row["paired_only_total_random"], 3), flush=True)
        out[donor] = res
    (HERE / "results/dev_reference.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
