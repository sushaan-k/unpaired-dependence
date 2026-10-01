#!/usr/bin/env python3
"""Deviation 1 (DEVIATIONS.md): score the estimator that PLAN.md names as proposed, bjs2/mapped.

The frozen run_external.py set PROPOSED = "bjs/mapped" (the one-sided arm of the draft), so results/overcite.json
reports savings and verdicts for that arm. The plan specifies bjs2/mapped, whose predictions were computed and
hashed with all others before any held-out file was opened. This script applies the frozen scoring function
unchanged, with the plan's arm, to the same hashed predictions and held-out statistics.

    python rescore.py      # writes results/overcite_plan_estimator.json
"""

from __future__ import annotations

import json

import numpy as np
from threadpoolctl import threadpool_limits

import run_external as rx


def main():
    freeze = rx.check_freeze()
    man = json.loads((rx.HERE / "results/manifest.json").read_text())
    for f in ("predictions.npz", "predict_info.json"):
        assert rx.sha(rx.HERE / "results" / f) == man[f], f
    info = json.loads((rx.HERE / "results/predict_info.json").read_text())
    parts = [rx.od.load(p, info["genes"]) for p in ("test_adaptation", "test_scoring")]
    cat = {k: np.concatenate([p[k] for p in parts]) for k in ("counts", "library", "y", "part", "target", "condition")}
    pop = rx.pm.keys_of(cat)
    stats = rx.tg.group_stats(cat["counts"], cat["library"], cat["y"], pop, cat["part"])
    z = np.load(rx.HERE / "results/predictions.npz")
    rx.PROPOSED = "bjs2/mapped"
    out = rx.score(stats, {k: z[k] for k in z.files}, freeze["frozen_at"])
    out["cells"] = int(len(pop))
    out["proposed"] = rx.PROPOSED
    out["note"] = ("Deviation 1: the frozen scoring function applied to the plan's proposed estimator (bjs2/mapped); "
                   "results/overcite.json holds the frozen script's output for bjs/mapped")
    (rx.HERE / "results/overcite_plan_estimator.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"verdicts": out["verdicts"],
                      "savings": {k: v[str(rx.TARGET)] for k, v in out["savings"].items()}}, indent=1))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
