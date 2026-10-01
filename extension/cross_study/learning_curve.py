#!/usr/bin/env python3
"""Learning curve over the number of training populations (PLAN.md; descriptive).

pf_means is refitted, with its unchanged tuning rule, on random subsets of 8,
16, 32 and 64 Stephenson patients (10 subsets each, seed 20260928) and on all
patients. For each fit: dimension of the identified subspace M, effective
dimension tr(H), and for every recipient donor (total analysis) the endpoint E
and the RNA-only diagnostic rho.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import estimators as est
from data import load_recipient, load_training
from identifiability import identified_subspace
from predict import within_assay

HERE = Path(__file__).resolve().parent
SIZES, REPEATS, SEED = (8, 16, 32, 64), 10, 20260928


def main():
    train = load_training()
    patients = sorted(set(train["patient"]))
    raw_all = {r["patient"]: r for r in est.pf_raw_moments(train, patients)}
    truth = np.load(HERE / "results/truth.npz")
    recipients = []
    for cohort in ("hao", "colon"):
        rec = load_recipient(cohort, "adaptation")
        for donor in sorted(set(rec["donor"])):
            m = rec["donor"] == donor
            Rx, Ry, _, _ = within_assay(rec["x"][m], rec["y"][m])
            recipients.append((f"{cohort}/{donor}", Rx, Ry, truth[f"{cohort}/{donor}/total/T"]))
    rng = np.random.default_rng(SEED)
    plan = [(s, rng.choice(len(patients), s, replace=False)) for s in SIZES for _ in range(REPEATS)]
    plan.append((len(patients), np.arange(len(patients))))
    rows = []
    for size, idx in plan:
        subset = [patients[i] for i in sorted(idx)]
        raw = [raw_all[p] for p in subset]
        sd_x, sd_y = est.pooled_sd(raw)
        fit = est.fit_pf_means(est.pf_units(raw, sd_x, sd_y), est.patient_folds(subset))
        VM, _, h = identified_subspace(fit["G"], fit["ridge"])
        P = VM @ VM.T
        row = {"patients": size, "scale": fit["scale"], "at_edge": fit["tuning"]["at_edge"],
               "dimension_M": int(VM.shape[1]), "effective_dimension": float(h.sum()), "E": {}, "rho": {}}
        for tag, Rx, Ry, T in recipients:
            row["E"][tag] = float(np.linalg.norm(est.closed_form(Rx, Ry, fit["B"]) - T) / np.linalg.norm(T))
            row["rho"][tag] = float(np.trace(P @ Rx @ P) / np.trace(Rx))
        rows.append(row)
        print(size, row["dimension_M"], round(np.mean(list(row["E"].values())), 3), flush=True)
    (HERE / "results/learning_curve.json").write_text(json.dumps({"seed": SEED, "fits": rows}, indent=1) + "\n")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
