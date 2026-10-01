#!/usr/bin/env python3
"""Pairing-free rigor re-analysis of colon D3 (PLAN.md; post hoc, not confirmatory).

Same recipients, genes, units and adaptation/scoring halves as spectral_transfer
D3 (colon_scale.load, salt spectral-transfer-v1; original units with the
colon_scale shrinkage). Changes, each fixing a stated weakness of D3:
  - training biopsy moments from disjoint cell halves (pf-halves-v1|sample|barcode):
    RNA from one half, protein from the other, so no cell contributes both;
  - donor-grouped folds (folds-v1|donor) instead of biopsy-level folds;
  - the five-value grid with the edge rule (pf_means); pf_moment at the D3 rank (8),
    run for up to 20,000 L-BFGS-B iterations (estimators.MOMENT_OPTIONS; PLAN.md,
    amendment 1); fits that stop at the limit are continued post hoc by
    colon_rigor_continuation.py;
  - donor-aware permutation: biopsy-level derangements in which no biopsy keeps
    protein moments from its own donor (200 for pf_means, 20 for pf_moment with
    the recipient's rank and penalty); p = (1 + #{perm <= observed}) / (N + 1);
  - identifiability diagnostics (Proposition S7).

    python colon_pf_rigor.py --donors XAUT1-HS1 ...   # resumable; one JSON per donor
    python colon_pf_rigor.py --merge
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import estimators as est
from identifiability import floor, identified_subspace, radius

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "spectral_transfer"))
import colon_scale as cs  # noqa: E402
from gaussian_transfer import Marginal, transfer  # noqa: E402

OUT = HERE / "results/colon_rigor"
SEED, N_MEANS, N_MOMENT = 20260929, 200, 20
RANK = 8                     # the D3 rank, kept fixed (PLAN.md, amendment 1)


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def biopsy_units(data, barcodes, samples):
    units, donor_of = [], {}
    for s in samples:
        idx = np.flatnonzero(data["sample"] == s)
        order = idx[np.argsort([sha(f"pf-halves-v1|{s}|{barcodes[i]}") for i in idx])]
        rna, prot = order[: len(order) // 2], order[len(order) // 2:]
        _, mx, _, Sx, _, _ = cs.moments(data["x"][rna], data["y"][rna])
        n, _, my, _, Sy, _ = cs.moments(data["x"][prot], data["y"][prot])
        units.append((s, n, mx, my, Sx, Sy))
        donor_of[s] = str(data["donor"][idx[0]])
    return units, donor_of


def cross_donor_derangement(units, donor_of, rng):
    ids = [u[0] for u in units]
    donors = np.array([donor_of[i] for i in ids])
    for _ in range(100000):
        perm = rng.permutation(len(ids))
        if np.all(donors[perm] != donors):
            return perm
    raise RuntimeError("no cross-donor derangement found")


def permuted(units, perm):
    return [(u[0], units[j][1], u[2], units[j][3], u[4], units[j][5]) for u, j in zip(units, perm)]


def run_donor(data, barcodes, donor):
    started = time.time()
    samples = sorted(set(data["sample"][data["donor"] != donor]))
    units, donor_of = biopsy_units(data, barcodes, samples)
    donor_fold = est.patient_folds(sorted(set(donor_of.values())))
    fold_of = {s: donor_fold[donor_of[s]] for s in samples}
    means = est.fit_pf_means(units, fold_of)
    moment = est.fit_pf_moment(units, fold_of, means["scale"], log=lambda m: None, rank=RANK)
    m = data["donor"] == donor
    adapt, score = m & (data["half"] == 0), m & (data["half"] == 1)
    n, mx, my, Sx, Sy, _ = cs.moments(data["x"][adapt], data["y"][adapt])
    xs, ys = data["x"][score], data["y"][score]
    marginal = Marginal(Sx, Sy)

    def error(B):
        return cs.evaluate(transfer(marginal, B)[0], Sx, mx, my, xs, ys)["cov_rel_error"]

    E_means, E_moment = error(means["B"]), error(moment["B"])
    rng = np.random.default_rng(SEED + sum(map(ord, donor)))
    perm_means, perm_moment = [], []
    for k in range(N_MEANS):
        perm = cross_donor_derangement(units, donor_of, rng)
        pu = permuted(units, perm)
        fit = est.fit_pf_means(pu, fold_of)
        perm_means.append(error(fit["B"]))
        if k < N_MOMENT:
            W0, b0, psi0, *_ = est.ecological(pu, means["scale"])
            W, b, psi, res = est.moment_fit(pu, moment["rank"], moment["ridge"], (W0, b0, psi0))
            perm_moment.append(error(W.T / psi[None, :]))
    VM, _, h = identified_subspace(means["G"], means["ridge"])
    _, _, _, Sx_score, _, T = cs.moments(xs, ys)                # T: the D3 target (unshrunk cross-covariance)
    r, _, sF = radius(Sx, Sy, means["W"], VM)
    P = VM @ VM.T
    return {
        "training_biopsies": len(samples), "training_donors": len(set(donor_of.values())),
        "pf_means": {"E": E_means, "scale": means["scale"], "tuning": means["tuning"],
                     "perm_E": perm_means,
                     "p": (1 + sum(e <= E_means for e in perm_means)) / (len(perm_means) + 1)},
        "pf_moment": {"E": E_moment, "rank": moment["rank"], "ridge": moment["ridge"],
                      "tuning": moment["tuning"], "converged": moment["converged"],
                      "iterations": moment["iterations"], "message": moment["message"],
                      "grad_norm": moment["grad_norm"], "perm_E": perm_moment,
                      "p": (1 + sum(e <= E_moment for e in perm_moment)) / (len(perm_moment) + 1)},
        "identifiability": {"dimension_M": int(VM.shape[1]), "effective_dimension": float(h.sum()),
                            "rho": float(np.trace(P @ Sx @ P) / np.trace(Sx)),
                            "floor": floor(T, Sx_score, VM), "radius": r / float(np.linalg.norm(T)),
                            "max_identified_singular_value": sF},
        "seconds": time.time() - started}


def jsonable(v):
    if isinstance(v, dict):
        return {str(k): jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--donors", nargs="*")
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.merge:
        merged = {p.stem: json.loads(p.read_text()) for p in sorted(OUT.glob("*.json"))}
        (HERE / "results/colon_rigor.json").write_text(json.dumps(merged, indent=1) + "\n")
        return
    data = cs.load(200)
    barcodes = np.load(cs.DATA, allow_pickle=False)["barcodes"].astype(str)
    donors = args.donors or sorted(set(data["donor"]))
    for donor in donors:
        path = OUT / f"{donor}.json"
        if path.exists():
            continue
        result = run_donor(data, barcodes, str(donor))
        path.write_text(json.dumps(jsonable(result), indent=1) + "\n")
        print(donor, round(result["pf_means"]["E"], 4), round(result["pf_moment"]["E"], 4),
              "p", round(result["pf_means"]["p"], 4), round(result["pf_moment"]["p"], 4),
              f"{result['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
