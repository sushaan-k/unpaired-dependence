#!/usr/bin/env python3
"""Provenance of the plan's proposed estimator in the external test (deviation 1, DEVIATIONS.md).

    python provenance_check.py      # training file only; no held-out file is read

1. The frozen PLAN.md (hash checked against results/freeze.json) names bjs2/mapped as the proposed estimator.
2. results/predictions.npz, whose SHA-256 was written to results/manifest.json before any held-out file was
   opened, holds bjs2/mapped predictions for every budget, draw and condition.
3. Recomputing those predictions from the training file with the frozen code (run_external.py, run_study.py,
   sp_estimators.py, all hash-checked) reproduces the stored arrays exactly; the same holds for bjs/mapped, the
   arm the frozen scoring script named.
4. Records the hashes of both scoring outputs and of DEVIATIONS.md, and the order of the files' modification
   times, in results/provenance.json.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import run_external as rx

ARMS = ("bjs2/mapped", "bjs/mapped")


def main():
    freeze = rx.check_freeze()                                   # PLAN.md and every frozen file unchanged
    plan = (rx.HERE / "PLAN.md").read_text()
    naming = [ln.strip() for ln in plan.splitlines() if "bjs2/mapped" in ln or "Proposed estimator" in ln]
    assert any("Proposed estimator (`bjs2/mapped`)" in ln for ln in naming), "plan does not name bjs2/mapped"
    assert "bjs/mapped" not in plan.replace("bjs2/mapped", ""), "plan also names the one-sided arm"
    man = json.loads((rx.HERE / "results/manifest.json").read_text())
    assert rx.sha(rx.HERE / "results/predictions.npz") == man["predictions.npz"]
    z = np.load(rx.HERE / "results/predictions.npz")
    info = json.loads((rx.HERE / "results/predict_info.json").read_text())
    conds = info["conditions"]
    expected = {f"{a}/{B}/{dr}/{c}" for a in ARMS for B in rx.BUDGETS for dr in range(rx.DRAWS) for c in conds}
    present = expected & set(z.files)
    assert present == expected, f"missing {len(expected - present)} arrays"
    # recompute from the training file with the frozen code, exactly as predict() does
    genes = info["genes"]
    train = rx.od.load("train", genes)
    x, y = rx.pm.features(train)
    pop = rx.pm.keys_of(train)
    keys = rx.pm.eligible_training(pop, train["part"], rx.od.MIN_TRAIN_HALF)
    pool, un, X_res, Y_res, cond_res = rx.cm.setup_fold(train, x, y, pop, keys)
    ctx = rx.rs.build_context(train, x, y, pop, keys, pool, un)
    diff = {a: 0.0 for a in ARMS}
    for B in rx.BUDGETS:
        for dr in range(rx.DRAWS):
            rng = np.random.default_rng([rx.SEED, B, dr])
            pick = rng.choice(len(X_res), size=B, replace=False)
            X, Y = X_res[pick], Y_res[pick]
            for name in ("bjs2", "bjs"):
                crng = np.random.default_rng([rx.SEED + 1, B, dr])
                P = rx.rs.denoise(name, X, Y, ctx.Rx, ctx.Ry, ctx.n_x, ctx.n_y, ctx.Uz, ctx.blocks, crng)
                for c in conds:
                    stored = z[f"{name}/mapped/{B}/{dr}/{c}"]
                    diff[f"{name}/mapped"] = max(diff[f"{name}/mapped"], float(np.max(np.abs(ctx.mapped(P, c) - stored))))
    files = ["PLAN.md", "results/freeze.json", "results/predictions.npz", "results/predict_info.json",
             "results/manifest.json", "results/overcite.json", "results/overcite_plan_estimator.json",
             "DEVIATIONS.md", "rescore.py"]
    mtimes = {f: time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime((rx.HERE / f).stat().st_mtime)) for f in files}
    rec = {"checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "frozen_at": freeze["frozen_at"], "plan_sha256": rx.sha(rx.HERE / "PLAN.md"),
           "plan_lines_naming_the_proposed_estimator": naming,
           "manifest_written_at": man["written_at"], "predictions_sha256": man["predictions.npz"],
           "arrays_per_arm": {a: sum(1 for k in z.files if k.startswith(a + "/")) for a in ARMS},
           "max_abs_difference_recomputed_vs_stored": diff,
           "scoring_outputs": {"frozen_script (proposed = bjs/mapped)": rx.sha(rx.HERE / "results/overcite.json"),
                               "plan_estimator (proposed = bjs2/mapped)": rx.sha(rx.HERE / "results/overcite_plan_estimator.json"),
                               "DEVIATIONS.md": rx.sha(rx.HERE / "DEVIATIONS.md")},
           "file_modification_times": mtimes}
    assert all(v == 0.0 for v in diff.values()), diff
    (rx.HERE / "results/provenance.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
