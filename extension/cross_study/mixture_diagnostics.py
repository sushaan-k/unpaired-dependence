#!/usr/bin/env python3
"""Descriptive addition to P2, computed after P2 and not part of its plan.

For each recipient donor's adaptation cells (kept level-1 or coarse types), the
share of the between-type and of the within-type RNA covariance (total-SD
units) that lies inside the subspace M identified by the frozen pf_means fit,
rho = tr(P_M R P_M) / tr(R). Uses RNA only. Writes
results/mixture_diagnostics.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from data import load_recipient
from identifiability import identified_subspace
from posthoc_mixture import PARTS, decompose, scaled

HERE = Path(__file__).resolve().parent


def main():
    z = np.load(HERE / "results/reference_fit.npz")
    VM, _, _ = identified_subspace(z["pf_means_G"], float(z["pf_means_ridge"]))
    P = VM @ VM.T
    scores = json.loads((HERE / "results/scores.json").read_text())["scores"]
    out = {"dimension_M": int(VM.shape[1]), "donors": {}, "summary": {}}
    for cohort, (analysis, key) in PARTS.items():
        adapt = load_recipient(cohort, "adaptation")
        rows = []
        for donor in sorted(set(adapt["donor"])):
            tag = f"{cohort}/{donor}/{analysis}"
            m = adapt["donor"] == donor
            x = adapt["x"][m]
            Sb, Sw, s, _, _ = decompose(x, x, adapt["labels"][key][m], scores[tag]["types"])
            Rb, Rw = scaled(Sb, s, s), scaled(Sw, s, s)
            row = {"rho_between": float(np.trace(P @ Rb @ P) / np.trace(Rb)),
                   "rho_within": float(np.trace(P @ Rw @ P) / np.trace(Rw)),
                   "between_variance_share": float(np.trace(Rb) / (np.trace(Rb) + np.trace(Rw)))}
            out["donors"][tag] = row
            rows.append(row)
            print(tag, {k: round(v, 3) for k, v in row.items()}, flush=True)
        out["summary"][cohort] = {k: {"mean": float(np.mean([r[k] for r in rows])),
                                      "min": float(np.min([r[k] for r in rows])),
                                      "max": float(np.max([r[k] for r in rows]))} for k in rows[0]}
    (HERE / "results/mixture_diagnostics.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out["summary"], indent=1))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
