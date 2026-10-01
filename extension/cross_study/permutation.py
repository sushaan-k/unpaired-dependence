#!/usr/bin/env python3
"""Donor-aware permutation null for pf_means (PLAN.md, H1).

Each training patient's protein moments are paired with the RNA moments of
another patient from the same site (within-site derangement, no fixed points),
and pf_means is refitted with the unchanged tuning rule. 200 permutations,
seed 20260927. Writes the permuted interactions; evaluate_permutation() scores
them against the sealed targets after evaluate.py has run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import estimators as est
from data import load_recipient, load_training
from predict import within_assay

HERE = Path(__file__).resolve().parent
SEED, N = 20260927, 200


def derangement(groups, rng):
    """Map each member to another member of the same group; no fixed points."""
    mapping = {}
    for members in groups:
        members = sorted(members)
        if len(members) < 2:
            raise ValueError("a group with one member cannot be deranged")
        while True:
            idx = rng.permutation(len(members))
            if np.all(idx != np.arange(len(members))):
                break
        mapping.update({members[i]: members[j] for i, j in enumerate(idx)})
    return mapping


def fit_permutations():
    train = load_training()
    patients = sorted(set(train["patient"]))
    fold_of = est.patient_folds(patients)
    raw = est.pf_raw_moments(train, patients)
    sd_x, sd_y = est.pooled_sd(raw)
    by_patient = {r["patient"]: r for r in raw}
    site_of = {}
    for p in patients:
        sites = set(train["site"][train["patient"] == p])
        assert len(sites) == 1
        site_of[p] = sites.pop()
    groups = [[p for p in patients if site_of[p] == s] for s in sorted(set(site_of.values()))]
    rng = np.random.default_rng(SEED)
    Bs, scales = [], []
    for k in range(N):
        mapping = derangement(groups, rng)
        permuted = [{**r, "ny": by_patient[mapping[r["patient"]]]["ny"],
                     "my": by_patient[mapping[r["patient"]]]["my"],
                     "Sy": by_patient[mapping[r["patient"]]]["Sy"]} for r in raw]
        fit = est.fit_pf_means(est.pf_units(permuted, sd_x, sd_y), fold_of)
        Bs.append(fit["B"])
        scales.append(fit["scale"])
        print(k, fit["scale"], flush=True)
    np.savez_compressed(HERE / "results/permutation_B.npz", B=np.array(Bs), scales=np.array(scales))


def evaluate_permutations():
    """E of every permuted interaction for every recipient donor (total analysis)."""
    z = np.load(HERE / "results/permutation_B.npz")
    truth = np.load(HERE / "results/truth.npz")
    out = {}
    for cohort in ("hao", "colon"):
        rec = load_recipient(cohort, "adaptation")
        for donor in sorted(set(rec["donor"])):
            m = rec["donor"] == donor
            Rx, Ry, _, _ = within_assay(rec["x"][m], rec["y"][m])
            T = truth[f"{cohort}/{donor}/total/T"]
            out[f"{cohort}/{donor}"] = [float(np.linalg.norm(est.closed_form(Rx, Ry, B) - T) / np.linalg.norm(T))
                                        for B in z["B"]]
            print(cohort, donor, np.mean(out[f"{cohort}/{donor}"]), flush=True)
    (HERE / "results/permutation_scores.json").write_text(
        json.dumps({"scales": z["scales"].tolist(), "E": out}, indent=1) + "\n")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        {"fit": fit_permutations, "evaluate": evaluate_permutations}[sys.argv[1]]()
