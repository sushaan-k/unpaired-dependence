#!/usr/bin/env python3
"""Fit every reference-side arm on the Stephenson training populations (PLAN.md, "Arms").

    python fit_reference.py pf        # pf_means and pf_moment (pairing-free; disjoint halves)
    python fit_reference.py paired    # paired closed form, reference regression, pooled matrices
    python fit_reference.py merge     # -> results/reference_fit.npz and results/reference_fit.json

No recipient data are read here.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import estimators as est
from data import DATA, load_training

HERE = Path(__file__).resolve().parent
OUT = DATA / "fits"


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


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


def pf():
    train = load_training()
    patients = sorted(set(train["patient"]))
    fold_of = est.patient_folds(patients)
    raw = est.pf_raw_moments(train, patients)
    sd_x, sd_y = est.pooled_sd(raw)
    units = est.pf_units(raw, sd_x, sd_y)
    t = time.time()
    means = est.fit_pf_means(units, fold_of)
    log(f"pf_means scale {means['scale']:g} ({time.time() - t:.1f}s) {means['tuning']}")
    t = time.time()
    moment = est.fit_pf_moment(units, fold_of, means["scale"], log=log)
    log(f"pf_moment rank {moment['rank']} converged {moment['converged']} it {moment['iterations']} "
        f"({time.time() - t:.0f}s)")
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "pf.npz", sd_x=sd_x, sd_y=sd_y, means_W=means["W"], means_b=means["b"],
             means_psi=means["psi"], means_B=means["B"], means_G=means["G"], means_ridge=means["ridge"],
             moment_W=moment["W"], moment_b=moment["b"], moment_psi=moment["psi"], moment_B=moment["B"],
             patients=np.array(patients))
    report = {"patients": len(patients), "cells_rna_half": int(sum(r["nx"] for r in raw)),
              "cells_protein_half": int(sum(r["ny"] for r in raw)),
              "pf_means": {k: means[k] for k in ("scale", "ridge", "tuning")},
              "pf_moment": {k: moment[k] for k in ("rank", "ridge", "tuning", "converged", "iterations",
                                                   "message", "grad_norm")}}
    (OUT / "pf.json").write_text(json.dumps(jsonable(report), indent=1) + "\n")


def paired():
    train = load_training()
    patients = sorted(set(train["patient"]))
    fold_of = est.patient_folds(patients)
    people = est.paired_people(train, patients)
    t = time.time()
    closed = est.fit_paired_closed_form(people, fold_of, log=log)
    log(f"paired closed form ridge {closed['ridge']:g} converged {closed['converged']} ({time.time() - t:.0f}s)")
    regression = est.fit_reference_regression(people, fold_of)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "paired.npz", B=closed["B"], W_ref=regression["W"], corr=est.pooled(people, "Rxy"),
             cov=est.pooled(people, "Cxy_raw"), patients=np.array(patients))
    report = {"paired_closed_form": {k: closed[k] for k in ("ridge", "tuning", "converged", "iterations",
                                                            "message", "grad_norm")},
              "reference_regression": {k: regression[k] for k in ("lambda", "tuning")},
              "cells": int(sum(q["n"] for q in people))}
    (OUT / "paired.json").write_text(json.dumps(jsonable(report), indent=1) + "\n")


def merge():
    import hashlib
    pfz, pz = np.load(OUT / "pf.npz"), np.load(OUT / "paired.npz")
    path = HERE / "results/reference_fit.npz"
    np.savez_compressed(path, **{f"pf_{k}": pfz[k] for k in pfz.files},
                        **{f"paired_{k}": pz[k] for k in pz.files})
    report = {"pf": json.loads((OUT / "pf.json").read_text()),
              "paired": json.loads((OUT / "paired.json").read_text()),
              "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (HERE / "results/reference_fit.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1)[:4000])


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        {"pf": pf, "paired": paired, "merge": merge}[sys.argv[1]]()
