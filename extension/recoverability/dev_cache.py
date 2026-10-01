"""Cached development inputs (channels and recipient counts) shared by the dev_* scripts.

Channels: the frozen means channel (cross_study/results/reference_fit.npz) and the
post hoc within-type channel P1 refitted by the frozen rules (posthoc_within.py).
Recipient loaders return raw RNA counts of the 206 genes with library sizes, so
that RNA-only noise estimates can be formed; no cross-assay moment is computed here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
CS = HERE.parent / "cross_study"
sys.path.insert(0, str(CS))

import estimators as est  # noqa: E402
from adt_matching import COLON, HAO, names_for  # noqa: E402
from data import COLON_DIR, DATA, _sha256, clr, features, load_training  # noqa: E402
from posthoc_within import type_units  # noqa: E402

CACHE = HERE / "results/dev_channels.npz"


def channels():
    if CACHE.exists():
        z = np.load(CACHE)
        return {name: {k.split("/", 1)[1]: z[k] for k in z.files if k.startswith(name + "/")}
                for name in ("frozen", "P1")}
    ref = np.load(CS / "results/reference_fit.npz")
    out = {"frozen": {"W": ref["pf_means_W"], "G": ref["pf_means_G"], "ridge": np.array(float(ref["pf_means_ridge"])),
                      "B": ref["pf_means_B"], "psi": ref["pf_means_psi"], "b": ref["pf_means_b"],
                      "sd_x": ref["pf_sd_x"], "sd_y": ref["pf_sd_y"]}}
    train = load_training()
    traw = type_units(train)
    tsd_x, tsd_y = est.pooled_sd(traw)
    tunits = est.pf_units(traw, tsd_x, tsd_y)
    donor_fold = est.patient_folds(sorted({r["donor"] for r in traw}))
    fit = est.fit_pf_means(tunits, {r["patient"]: donor_fold[r["donor"]] for r in traw})
    out["P1"] = {"W": fit["W"], "G": fit["G"], "ridge": np.array(fit["ridge"]), "B": fit["B"], "psi": fit["psi"],
                 "b": fit["b"], "sd_x": tsd_x, "sd_y": tsd_y}
    np.savez(CACHE, **{f"{n}/{k}": v for n, d in out.items() for k, v in d.items()})
    return out


def load_counts(cohort, part):
    """Raw RNA counts of the 206 genes, library sizes, CLR protein values and labels (adaptation files)."""
    feats = features()
    genes, proteins = feats["genes"], feats["proteins"]
    if cohort == "colon":
        path = COLON_DIR / f"colon_{part}.npz"
        seal = json.loads((CS / "seal.json").read_text())
        d = np.load(path, allow_pickle=False)
        rna = sp.csr_matrix((d["rna_data"], d["rna_indices"], d["rna_indptr"]))
        library = np.asarray(rna.sum(axis=1)).ravel()
        names = list(d["rna_names"])
        adt_names = list(d["adt_names"])
        adt = d["adt"][:, [adt_names.index(n) for n in names_for(COLON, proteins)]]
        donor, labels = d["CoLabs_patient"].astype(str), {"coarse": d["coarse_annotations_MK"].astype(str)}
    else:
        path = DATA / f"hao_{part}.npz"
        seal = json.loads((CS / "hao_seal.json").read_text())
        d = np.load(path, allow_pickle=False)
        rna = sp.csr_matrix((d["rna_data"], d["rna_indices"], d["rna_indptr"]))
        library = d["library"].astype(float)
        names = list(d["rna_names"])
        adt_names = list(d["adt_names"])
        adt = d["adt"][:, [adt_names.index(n) for n in names_for(HAO, proteins)]]
        donor, labels = d["donor"].astype(str), {"l1": d["l1"].astype(str), "l2": d["l2"].astype(str)}
    assert _sha256(path) == seal["files"][part]["sha256"], f"{path} does not match its seal"
    cols = [names.index(g) for g in genes]
    counts = rna[:, cols].toarray().astype(float)
    return {"counts": counts, "library": np.asarray(library, float), "y": clr(adt), "donor": donor,
            "labels": labels}
