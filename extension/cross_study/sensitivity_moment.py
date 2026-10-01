#!/usr/bin/env python3
"""Post hoc sensitivity (not in PLAN.md): was the pf_moment fit finished?

The frozen rank-32 fit stopped when its line search failed (not by the
convergence criterion). Restart L-BFGS-B from that point with the same options,
up to three times, and report the change in objective, interaction and
endpoint for every recipient donor (total analysis). The frozen predictions
are not changed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits

import estimators as est
from data import load_recipient, load_training
from predict import within_assay

HERE = Path(__file__).resolve().parent


def main():
    fit = json.loads((HERE / "results/reference_fit.json").read_text())["pf"]
    ref = np.load(HERE / "results/reference_fit.npz")
    train = load_training()
    patients = sorted(set(train["patient"]))
    raw = est.pf_raw_moments(train, patients)
    units = est.pf_units(raw, ref["pf_sd_x"], ref["pf_sd_y"])
    r, ridge = fit["pf_moment"]["rank"], fit["pf_moment"]["ridge"]
    W, b, psi = ref["pf_moment_W"], ref["pf_moment_b"], ref["pf_moment_psi"]
    q, p = W.shape
    Uw, s, Vt = np.linalg.svd(W, full_matrices=False)
    z = np.concatenate([(Uw[:, :r] * np.sqrt(s[:r])).ravel(), (Vt[:r].T * np.sqrt(s[:r])).ravel(), b, np.log(psi)])
    start = est.moment_negloglik_fast(z, units, p, q, r, ridge)[0]
    history = []
    for k in range(3):
        res = minimize(est.moment_negloglik_fast, z, args=(units, p, q, r, ridge), jac=True, method="L-BFGS-B",
                       options=est.MOMENT_OPTIONS)
        history.append({"iterations": int(res.nit), "message": str(res.message), "objective": float(res.fun),
                        "grad_norm": float(np.linalg.norm(res.jac))})
        print(history[-1], flush=True)
        z = res.x
        if res.success:
            break
    U, V = z[: q * r].reshape(q, r), z[q * r: q * r + p * r].reshape(p, r)
    W2, psi2 = U @ V.T, np.exp(z[q * r + p * r + q:])
    B1, B2 = ref["pf_moment_B"], W2.T / psi2[None, :]
    truth = np.load(HERE / "results/truth.npz")
    change = {}
    for cohort in ("hao", "colon"):
        rec = load_recipient(cohort, "adaptation")
        for donor in sorted(set(rec["donor"])):
            m = rec["donor"] == donor
            Rx, Ry, _, _ = within_assay(rec["x"][m], rec["y"][m])
            T = truth[f"{cohort}/{donor}/total/T"]
            E1 = np.linalg.norm(est.closed_form(Rx, Ry, B1) - T) / np.linalg.norm(T)
            E2 = np.linalg.norm(est.closed_form(Rx, Ry, B2) - T) / np.linalg.norm(T)
            change[f"{cohort}/{donor}"] = [float(E1), float(E2)]
    out = {"start_objective": float(start), "restarts": history,
           "relative_change_B": float(np.linalg.norm(B2 - B1) / np.linalg.norm(B1)),
           "E_frozen_vs_restarted": change,
           "max_abs_E_change": float(max(abs(a - b) for a, b in change.values()))}
    (HERE / "results/sensitivity_moment.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "E_frozen_vs_restarted"}, indent=1))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
