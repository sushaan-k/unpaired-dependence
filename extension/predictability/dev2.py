#!/usr/bin/env python3
"""Development, second pass (post hoc): pilot predictors and the bootstrap law, on the stored observed curves.

    python dev2.py <name>      # results/dev/<name>_pilot.json
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ptools as pt  # noqa: E402
import dev  # noqa: E402

PILOTS = (50, 100, 200)
RESAMPLES = 30


def closed_curves(blocks, budgets):
    c = pt.curves(blocks, budgets, "lin")
    return None if c is None else {"bjs2": c[0].tolist(), "js": c[1].tolist()}


def run(name):
    t0 = time.time()
    r = json.loads((dev.OUT / f"{name}.json").read_text())
    budgets = r["budgets"]
    d = pt.load(dev.DATA / f"{name}.npz", name, allow_object=(name == "stephenson"))
    P = pt.Problem(d)
    out = {"dataset": name, "budgets": budgets}
    c = pt.boot_curves(P, P.Xr, P.Yr, budgets, draws=20, seed=1)
    out["boot_population"] = {"bjs2": c[0].tolist(), "js": c[1].tolist()}
    un = P.unpaired_noise()
    for m in PILOTS:
        X, Y = P.Xr[:m], P.Yr[:m]
        rec = {}
        mom = P.moments(X, Y, ell=False)
        rec["lin"] = closed_curves(mom, budgets)
        # thresholded: a block's signal counts only if its unbiased estimate exceeds twice its bootstrap sd
        rng = np.random.default_rng([7, m])
        reps = np.array([[b["r2"] for b in P.moments(X[i], Y[i], ell=False)]
                         for i in [rng.integers(0, m, m) for _ in range(RESAMPLES)]])
        sd = reps.std(0)
        thr = [dict(b, r2=b["r2"] if b["r2"] > 2 * s else 0.0) for b, s in zip(mom, sd)]
        rec["lin_thr"] = closed_curves(thr, budgets)
        # shrunk: squared norms of the proposed estimate from the pilot
        A, _ = P.fit(X, Y)
        shr = []
        k = 0
        for rows in P.rblocks:
            for cols in P.pblocks:
                shr.append(dict(mom[k], r2=float(np.sum(A[np.ix_(rows, cols)] ** 2))))
                k += 1
        rec["lin_shrunk"] = closed_curves(shr, budgets)
        # bootstrap law from the pilot
        c = pt.boot_curves(P, X, Y, budgets, draws=20, seed=2)
        rec["boot"] = None if c is None else {"bjs2": c[0].tolist(), "js": c[1].tolist()}
        # resampled pilots: spread of the closed-form prediction (for intervals)
        rec["lin_resampled"] = []
        for i in range(RESAMPLES):
            idx = rng.integers(0, m, m)
            cc = closed_curves(P.moments(X[idx], Y[idx], ell=False), budgets)
            rec["lin_resampled"].append(cc)
        out[f"pilot{m}"] = rec
    (dev.OUT / f"{name}_pilot.json").write_text(pt.jdump(out) + "\n")
    print(name, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        run(sys.argv[1])
