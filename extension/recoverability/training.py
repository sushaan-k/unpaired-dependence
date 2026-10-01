"""Training populations restricted to the bone-marrow panel, and every channel of the test.

All channels use the frozen pairing-free rules of cross_study (estimators.fit_pf_means:
RNA moments from one hash half of each population's cells and protein moments
from the other, pooled within-population SD units, ridge penalty chosen by
patient-grouped three-fold cross-validation over a five-value grid with the
edge rule). Protein values are centred log ratios over the kept targets, so
they are recomputed for the reduced protein set.

Channels
  frozen      all 118 blood patients (population = patient);
  P1          patient x cell-type groups centred within type (posthoc_within);
  lcNN        the 40 random patient subsets of learning_curve.py (8, 16, 32, 64);
  permNN      20 donor-aware derangements: every patient's protein moments are
              replaced by another patient's, RNA moments are kept. The supported
              subspace, which depends on RNA means only, is unchanged; the
              channel is wrong by construction;
  colon       colon biopsies (population = biopsy) from the colon adaptation cells.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
CS = HERE.parent / "cross_study"
sys.path.insert(0, str(CS))

import estimators as est  # noqa: E402
from adt_matching import COLON, names_for  # noqa: E402
from data import COLON_DIR, DATA, _sha256, features  # noqa: E402

from bmmc import clr, seal  # noqa: E402

PERMUTATIONS = 20
PERM_SEED = 20261001
LC_SIZES, LC_REPEATS = (8, 16, 32, 64), 10


def panel():
    record = seal()
    feats = features()
    return ([feats["genes"].index(g) for g in record["genes"]],
            [feats["proteins"].index(p) for p in record["proteins"]], record)


def load_training_panel():
    gi, pi, _ = panel()
    d = np.load(DATA / "stephenson_training.npz", allow_pickle=False)
    return {"x": d["x"][:, gi].astype(float), "y": clr(d["adt"][:, pi]), "patient": d["patient"],
            "half": d["half"], "initial": d["initial"]}


def type_units(train, minimum=50):
    """posthoc_within.type_units on the reduced panel."""
    keys = sorted({(p, t) for p, t in zip(train["patient"], train["initial"])})
    raw = []
    for p, t in keys:
        m = (train["patient"] == p) & (train["initial"] == t)
        rna, prot = m & (train["half"] == 0), m & (train["half"] == 1)
        if rna.sum() < minimum or prot.sum() < minimum:
            continue
        xr, yr = train["x"][rna], train["y"][prot]
        raw.append({"patient": f"{p}|{t}", "donor": p, "type": t, "nx": len(xr), "mx": xr.mean(0),
                    "Sx": np.cov(xr.T, bias=True), "ny": len(yr), "my": yr.mean(0), "Sy": np.cov(yr.T, bias=True)})
    for t in sorted({r["type"] for r in raw}):
        group = [r for r in raw if r["type"] == t]
        wx = np.array([r["nx"] for r in group], float)
        wy = np.array([r["ny"] for r in group], float)
        cx = sum(w * r["mx"] for w, r in zip(wx, group)) / wx.sum()
        cy = sum(w * r["my"] for w, r in zip(wy, group)) / wy.sum()
        for r in group:
            r["mx"], r["my"] = r["mx"] - cx, r["my"] - cy
    return raw


def mean_correlation(units):
    """Mean within-population RNA correlation matrix of the training units (covariance-shift control)."""
    S = sum(u[4] for u in units) / len(units)
    d = np.sqrt(np.diag(S))
    return S / np.outer(d, d)


def fit(raw, fold_of):
    sd_x, sd_y = est.pooled_sd(raw)
    units = est.pf_units(raw, sd_x, sd_y)
    f = est.fit_pf_means(units, fold_of)
    return {"W": f["W"], "B": f["B"], "G": f["G"], "ridge": f["ridge"], "psi": f["psi"], "b": f["b"],
            "scale": f["scale"], "at_edge": f["tuning"]["at_edge"], "n_train": len(raw),
            "R_train": mean_correlation(units)}


def derangement(n, rng):
    while True:
        perm = rng.permutation(n)
        if np.all(perm != np.arange(n)):
            return perm


def colon_units():
    gi, pi, record = panel()
    feats = features()
    path = COLON_DIR / "colon_adaptation.npz"
    assert _sha256(path) == json.loads((CS / "seal.json").read_text())["files"]["adaptation"]["sha256"]
    d = np.load(path, allow_pickle=False)
    rna = sp.csr_matrix((d["rna_data"], d["rna_indices"], d["rna_indptr"]))
    library = np.asarray(rna.sum(axis=1)).ravel()
    names = list(d["rna_names"])
    counts = rna[:, [names.index(g) for g in record["genes"]]].toarray().astype(float)
    x = np.log1p(1e4 * counts / np.maximum(library, 1)[:, None])
    adt_names = list(d["adt_names"])
    y = clr(d["adt"][:, [adt_names.index(n) for n in names_for(COLON, record["proteins"])]])
    sample, donor, barcodes = d["CoLabs_sample"].astype(str), d["CoLabs_patient"].astype(str), d["barcodes"].astype(str)
    raw, donor_of = [], {}
    for s in sorted(set(sample)):
        idx = np.flatnonzero(sample == s)
        order = idx[np.argsort([hashlib.sha256(f"pf-halves-v1|{s}|{barcodes[i]}".encode()).hexdigest() for i in idx])]
        rna_i, prot_i = order[: len(order) // 2], order[len(order) // 2:]
        raw.append({"patient": s, "nx": len(rna_i), "mx": x[rna_i].mean(0), "Sx": np.cov(x[rna_i].T, bias=True),
                    "ny": len(prot_i), "my": y[prot_i].mean(0), "Sy": np.cov(y[prot_i].T, bias=True)})
        donor_of[s] = donor[idx[0]]
    return raw, donor_of


def all_channels(log=print):
    train = load_training_panel()
    patients = sorted(set(train["patient"]))
    raw_all = {r["patient"]: r for r in est.pf_raw_moments(train, patients)}
    raw = [raw_all[p] for p in patients]
    out = {"frozen": fit(raw, est.patient_folds(patients))}
    log("frozen", out["frozen"]["scale"])
    traw = type_units(train)
    donor_fold = est.patient_folds(sorted({r["donor"] for r in traw}))
    out["P1"] = fit(traw, {r["patient"]: donor_fold[r["donor"]] for r in traw})
    log("P1", out["P1"]["scale"], len(traw))
    lc = json.loads((CS / "results/learning_curve.json").read_text())
    rng = np.random.default_rng(lc["seed"])
    plan = [(s, rng.choice(len(patients), s, replace=False)) for s in LC_SIZES for _ in range(LC_REPEATS)]
    for i, (size, idx) in enumerate(plan):
        subset = [patients[j] for j in sorted(idx)]
        out[f"lc{i:02d}"] = fit([raw_all[p] for p in subset], est.patient_folds(subset))
    log("learning curve done")
    prng = np.random.default_rng(PERM_SEED)
    folds = est.patient_folds(patients)
    for k in range(PERMUTATIONS):
        perm = derangement(len(raw), prng)
        praw = [dict(r, ny=raw[j]["ny"], my=raw[j]["my"], Sy=raw[j]["Sy"]) for r, j in zip(raw, perm)]
        out[f"perm{k:02d}"] = fit(praw, folds)
    log("permutations done")
    craw, donor_of = colon_units()
    cfold = est.patient_folds(sorted(set(donor_of.values())))
    out["colon"] = fit(craw, {s: cfold[donor_of[s]] for s in donor_of})
    log("colon", out["colon"]["scale"], len(craw))
    return out
