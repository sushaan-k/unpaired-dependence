#!/usr/bin/env python3
"""Secondary and exploratory analyses of the prospective test (after the primary summary; nothing here changes a
verdict).

    python psecondary.py      # results/secondary.json

Prespecified secondary (PLAN.md, "reported without thresholds"): the bootstrap law's savings; the law's predicted
curves against the observed ones. Exploratory (post hoc, not in the plan): consistency of the predictions with
censored observations, and where the law's bias comes from (cells needed by each estimator, predicted against
observed).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prun  # noqa: E402
import ptools as pt  # noqa: E402

RES = HERE / "results"


def main():
    prun.check_freeze()
    out = {"note": "secondary (bootstrap law, curves) and exploratory (censoring consistency, bias decomposition)",
           "problems": {}, "summary": {}}
    per_target = {str(t): {"boot": [], "law_js_cells": [], "law_blocks_cells": [], "censored": []}
                  for t in prun.TARGETS}
    curve_err = {"bjs2": [], "js": []}
    for name in prun.NEW:
        ev = json.loads((RES / f"{name}.json").read_text())
        pr = json.loads((RES / f"{name}_predictions.json").read_text())
        for tag, rec in ev["problems"].items():
            p = pr[tag]
            b = rec["budgets"]
            obs = rec["curves"]
            row = {"dataset": name, "problem": tag}
            if p["law_curves"]:
                for k in ("bjs2", "js"):
                    e = float(np.mean(np.abs(np.array(p["law_curves"][k]) - np.array(obs[k]))))
                    row[f"curve_abs_err_{k}"] = e
                    curve_err[k].append(e)
            for t in prun.TARGETS:
                ts = str(t)
                o = rec["targets"][ts]["observed"]
                law = p["law"][ts]
                bc = p["boot_curves"]
                if bc:
                    s_boot = prun.saving_obs(bc["js"], bc["bjs2"], t, b)
                    row[f"boot_{ts}"] = s_boot["saving"]
                    if o["estimable"] and s_boot["estimable"]:
                        per_target[ts]["boot"].append(np.log2(o["saving"] / s_boot["saving"]))
                if o["estimable"] and law:
                    per_target[ts]["law_js_cells"].append(np.log2(o["cells_js"] / law["cells_js"]))
                    per_target[ts]["law_blocks_cells"].append(np.log2(o["cells_blocks"] / law["cells_blocks"]))
                if not o["estimable"] and law:
                    # exploratory: is the law consistent with the censoring? (a) one-block curve never reached the
                    # target: its predicted cells should exceed the largest budget; (b) block curve above the target
                    # at the smallest budget: its predicted cells should be at most the smallest budget; (c) block
                    # curve never reached it: predicted cells above the largest budget
                    checks = []
                    if o["rel_js"] == ">":
                        checks.append(("js_not_reached", law["cells_js"] > b[-1]))
                    if o["rel_js"] == "<=":
                        checks.append(("js_above_at_start", law["cells_js"] <= b[0]))
                    if o["rel_blocks"] == ">":
                        checks.append(("blocks_not_reached", law["cells_blocks"] > b[-1]))
                    if o["rel_blocks"] == "<=":
                        checks.append(("blocks_above_at_start", law["cells_blocks"] <= b[0]))
                    if o["rel_blocks"] == "=":
                        checks.append(("blocks_cells_within_2x", abs(np.log2(o["cells_blocks"] / law["cells_blocks"])) <= 1))
                    per_target[ts]["censored"].append({"dataset": name, "problem": tag, "checks": checks})
            out["problems"][f"{name}/{tag}"] = row
    for ts, v in per_target.items():
        cens = v["censored"]
        flat = [c for r in cens for c in r["checks"]]
        out["summary"][ts] = {
            "boot_median_abs_log2": float(np.median(np.abs(v["boot"]))) if v["boot"] else None,
            "boot_median_log2_bias": float(np.median(v["boot"])) if v["boot"] else None, "boot_n": len(v["boot"]),
            "law_cells_js_median_log2": float(np.median(v["law_js_cells"])) if v["law_js_cells"] else None,
            "law_cells_blocks_median_log2": float(np.median(v["law_blocks_cells"])) if v["law_blocks_cells"] else None,
            "censored_problems": len(cens),
            "censoring_checks": {k: [int(sum(ok for kk, ok in flat if kk == k)), int(sum(1 for kk, _ in flat if kk == k))]
                                 for k in sorted({kk for kk, _ in flat})},
            "censoring_failures": [f"{r['dataset']}/{r['problem']}:{k}" for r in cens for k, ok in r["checks"] if not ok]}
    out["summary"]["curve_mean_abs_err"] = {k: float(np.median(v)) for k, v in curve_err.items()}
    (RES / "secondary.json").write_text(pt.jdump(out) + "\n")
    print(json.dumps(out["summary"], indent=1))


if __name__ == "__main__":
    main()
