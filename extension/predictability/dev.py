#!/usr/bin/env python3
"""Development of the saving predictor on the nine benchmark data sets (post hoc; these data sets have been scored
before, so nothing here is a test).

    python dev.py selfcheck <name>     # fast estimators equal sp_estimators.block_js2 / js_matrix
    python dev.py run <name>           # curves and predictions -> results/dev/<name>.json

Per data set: roles, unpaired summaries and centred reservoir and truth cells (ptools.Problem); observed
recovered-fraction curves of block (proposed, pooled) and one-block (paired-only) James-Stein shrinkage, and of block
shrinkage in a random orthonormal basis pair (control), from DRAWS draws of B reservoir cells after the pilot cells;
predicted curves from block moments of (i) all reservoir cells, (ii) pilots of m cells, (iii) unpaired noise
only (bounds).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import ortho_group
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ptools as pt  # noqa: E402

DATA = Path("/home/claude/cbio/rawdata/generality")
OUT = HERE / "results" / "dev"
DRAWS = 20
PILOTS = (25, 50, 100, 200)
PILOT_MAX = 200
SEED = 20261072


def selfcheck(name):
    d = pt.load(DATA / f"{name}.npz", name, allow_object=(name == "stephenson"))
    P = pt.Problem(d)
    rng = np.random.default_rng(1)
    for B in (25, 200):
        idx = rng.choice(len(P.Xr), B, replace=False)
        X, Y = P.Xr[idx], P.Yr[idx]
        A, A1 = P.fit(X, Y)
        ref = pt.es.block_js2(X, Y, P.U, P.V, P.rblocks, P.pblocks)[0]
        ref1 = pt.es.js_matrix(X, Y)[0]
        print(B, float(np.abs(P.U @ A @ P.V.T - ref).max()), float(np.abs(P.U @ A1 @ P.V.T - ref1).max()))


def run(name, nx=None):
    t0 = time.time()
    d = pt.load(DATA / f"{name}.npz", name, allow_object=(name == "stephenson"))
    nx = int(nx) if nx else pt.gr.PANEL_X
    P = pt.Problem(d, n_x=nx)
    P.score_setup()
    tag = name if nx == pt.gr.PANEL_X else f"{name}_p{nx}"
    N = len(P.Xr)
    avail = N - PILOT_MAX
    budgets = [B for B in pt.BUDGETS if B <= avail]
    rng = np.random.default_rng([SEED, sum(map(ord, name))])
    Ur = ortho_group.rvs(P.U.shape[0], random_state=rng.integers(2 ** 31))
    Vr = ortho_group.rvs(P.V.shape[0], random_state=rng.integers(2 ** 31)) if P.V.shape[0] > 1 else np.eye(1)
    obs = {"bjs2": [], "js": [], "random": []}
    for B in budgets:
        acc = {k: [] for k in obs}
        for dr in range(DRAWS):
            r = np.random.default_rng([SEED, sum(map(ord, name)), B, dr])
            idx = PILOT_MAX + r.choice(avail, B, replace=False)
            X, Y = P.Xr[idx], P.Yr[idx]
            A, A1 = P.fit(X, Y)
            Ar, _ = P.fit(X, Y, Ur, Vr)
            acc["bjs2"].append(P.rf(A))
            acc["js"].append(P.rf(A1))
            acc["random"].append(P.rf(Ar, Ur, Vr))
        for k in obs:
            obs[k].append(float(np.mean(acc[k])))
    t1 = time.time()
    pred = {}
    mom_pop = P.moments(P.Xr, P.Yr)
    pred["population"] = {"blocks": mom_pop}
    for law in ("lin", "sph"):
        c = pt.curves(mom_pop, budgets, law)
        pred["population"][law] = None if c is None else {"bjs2": c[0].tolist(), "js": c[1].tolist()}
    mom_rand = P.moments(P.Xr, P.Yr, Ur, Vr)
    pred["random_population"] = {"blocks": mom_rand}
    for law in ("lin", "sph"):
        c = pt.curves(mom_rand, budgets, law)
        pred["random_population"][law] = None if c is None else {"bjs2": c[0].tolist(), "js": c[1].tolist()}
    for m in PILOTS:
        mom = P.moments(P.Xr[:m], P.Yr[:m])
        pred[f"pilot{m}"] = {"blocks": mom}
        for law in ("lin", "sph"):
            c = pt.curves(mom, budgets, law)
            pred[f"pilot{m}"][law] = None if c is None else {"bjs2": c[0].tolist(), "js": c[1].tolist()}
    un = P.unpaired_noise()
    pi_u = np.array([b["tau"] for b in un])
    pi_u = pi_u / pi_u.sum()
    pred["unpaired"] = {"blocks": un, "bound": float(1 / pi_u.min())}
    out = {"dataset": tag, "n": P.n, "budgets": budgets, "draws": DRAWS, "observed": obs, "pred": pred,
           "seconds": {"curves": t1 - t0, "predictions": time.time() - t1}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{tag}.json").write_text(pt.jdump(out) + "\n")
    print(name, P.n, budgets, {k: [round(v, 3) for v in vv] for k, vv in obs.items()}, f"{time.time() - t0:.0f}s",
          flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        {"selfcheck": selfcheck, "run": run}[sys.argv[1]](*sys.argv[2:])
