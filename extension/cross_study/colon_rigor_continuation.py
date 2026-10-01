#!/usr/bin/env python3
"""Continuation check for rank-8 fits of the colon re-analysis that stopped at the
iteration limit (post hoc; not in PLAN.md).

For every recipient whose rank-8 fit in results/colon_rigor/ did not converge, the
fit is rebuilt with the saved rank and penalty (reproducing the reported error) and
then continued from its final state for up to a further 20,000 L-BFGS-B
iterations. The change in the recipient's error is reported. Nothing reported in
the main analysis is replaced.

    python colon_rigor_continuation.py
"""

from __future__ import annotations

import json
import time

import numpy as np
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits

import estimators as est
from colon_pf_rigor import HERE, OUT, RANK, biopsy_units, cs, jsonable
from gaussian_transfer import Marginal, transfer


def main():
    data = cs.load(200)
    barcodes = np.load(cs.DATA, allow_pickle=False)["barcodes"].astype(str)
    target = HERE / "results/colon_rigor_continuation.json"
    report = json.loads(target.read_text()) if target.exists() else {}     # resumable
    for path in sorted(OUT.glob("*.json")):
        saved = json.loads(path.read_text())
        moment = saved["pf_moment"]
        donor = path.stem
        if moment["converged"] or donor in report:
            continue
        started = time.time()
        samples = sorted(set(data["sample"][data["donor"] != donor]))
        units, _ = biopsy_units(data, barcodes, samples)
        W0, b0, psi0, *_ = est.ecological(units, saved["pf_means"]["scale"])
        W, b, psi, res = est.moment_fit(units, RANK, moment["ridge"], (W0, b0, psi0))
        p = W0.shape[1]
        q = W0.shape[0]
        more = minimize(est.moment_negloglik_fast, res.x, args=(units, p, q, RANK, moment["ridge"]), jac=True,
                        method="L-BFGS-B", options=est.MOMENT_OPTIONS)
        z = more.x
        U, V = z[: q * RANK].reshape(q, RANK), z[q * RANK: q * RANK + p * RANK].reshape(p, RANK)
        psi2 = np.exp(z[q * RANK + p * RANK + q:])
        W2 = U @ V.T

        m = data["donor"] == donor
        adapt, score = m & (data["half"] == 0), m & (data["half"] == 1)
        _, mx, my, Sx, Sy, _ = cs.moments(data["x"][adapt], data["y"][adapt])
        xs, ys = data["x"][score], data["y"][score]
        marginal = Marginal(Sx, Sy)

        def error(B):
            return cs.evaluate(transfer(marginal, B)[0], Sx, mx, my, xs, ys)["cov_rel_error"]

        E1, E2 = error(W.T / psi[None, :]), error(W2.T / psi2[None, :])
        report[donor] = {"reported_E": moment["E"], "rebuilt_E": E1, "rebuilt_iterations": int(res.nit),
                         "rebuilt_objective": float(res.fun), "continued_E": E2,
                         "continued_iterations": int(more.nit), "continued_converged": bool(more.success),
                         "continued_message": str(more.message), "continued_objective": float(more.fun),
                         "relative_change_W": float(np.linalg.norm(W2 - W) / np.linalg.norm(W)),
                         "seconds": time.time() - started}
        print(donor, json.dumps(jsonable(report[donor])), flush=True)
        target.write_text(json.dumps(jsonable(report), indent=1) + "\n")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
