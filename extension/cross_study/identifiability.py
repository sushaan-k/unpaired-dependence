#!/usr/bin/env python3
"""Identifiability analyses of PLAN.md (descriptive; Additional file 1, Proposition S7).

Population means identify the shared channel W only on M, the span of the
between-population variation of RNA means. For the ridge estimator the
operator H = G (G + lambda I)^-1 says how much of each direction it keeps; M is
taken as the directions with H-eigenvalue >= 1/2.

For each recipient donor:
  rho       share of the recipient's standardized RNA variance inside M (RNA only;
            computable without any pairs);
  floor     ||R_x (I - P_M) R_x^-1 T|| / ||T||: error of the oracle that knows the
            recipient's own channel on M and nothing else, i.e. the part of the truth
            that population means cannot identify (uses scoring cells; after unsealing);
  radius    Chebyshev radius of the set of cross-correlations compatible with the
            identified part and a valid joint law (contraction completion), / ||T||.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from data import load_recipient, type_centre
from predict import ANALYSES, within_assay

HERE = Path(__file__).resolve().parent


def psd_power(S, power):
    w, V = np.linalg.eigh((S + S.T) / 2)
    w = np.clip(w, 1e-12, None)
    return (V * w ** power) @ V.T


def identified_subspace(G, ridge):
    g, V = np.linalg.eigh((G + G.T) / 2)
    h = np.clip(g, 0, None) / (np.clip(g, 0, None) + ridge)
    order = np.argsort(-h)
    h, V = h[order], V[:, order]
    return V[:, h >= 0.5], V[:, h < 0.5], h


def complement(Q):
    """Orthonormal basis of the orthogonal complement of the orthonormal columns Q."""
    full, _ = np.linalg.qr(np.hstack([Q, np.eye(Q.shape[0])]))
    basis = full[:, Q.shape[1]:Q.shape[0]]
    return basis - Q @ (Q.T @ basis)


def radius(Rx, Ry, W, VM):
    """Chebyshev radius and centre of {C : C identified on M through W, [[Rx, C], [C^T, Ry]] >= 0}.

    Whitened K = Rx^-1/2 C Ry^-1/2 must be a contraction. Its component along
    N = Rx^-1/2 M is fixed by W on M (F); the component along the complement is
    Gamma (I - F^T F)^1/2 for any contraction Gamma. The radius is
    max ||Rx^1/2 Q_perp Gamma D Ry^1/2||_F = sqrt(sum_i s_i(L)^2 s_i(R)^2).
    """
    Rxh, Rxih, Ryh, Ryih = psd_power(Rx, 0.5), psd_power(Rx, -0.5), psd_power(Ry, 0.5), psd_power(Ry, -0.5)
    if VM.shape[1] == 0:
        QN = np.zeros((Rx.shape[0], 0))
    else:
        QN, _ = np.linalg.qr(Rxih @ VM)
    Qp = complement(QN) if QN.shape[1] else np.eye(Rx.shape[0])
    K = Rxh @ W.T @ Ryih
    F = QN.T @ K
    sF = np.linalg.svd(F, compute_uv=False) if F.size else np.zeros(1)
    w, V = np.linalg.eigh(np.eye(Ry.shape[0]) - F.T @ F)
    D = (V * np.sqrt(np.clip(w, 0, None))) @ V.T
    sL = np.linalg.svd(Rxh @ Qp, compute_uv=False)
    sR = np.linalg.svd(D @ Ryh, compute_uv=False)
    k = min(len(sL), len(sR))
    centre = Rxh @ QN @ F @ Ryh
    return float(np.sqrt(np.sum(sL[:k] ** 2 * sR[:k] ** 2))), centre, float(sF.max())


def floor(T, Rx_score, VM):
    P = VM @ VM.T
    Wt = np.linalg.solve(Rx_score, T)
    return float(np.linalg.norm(Rx_score @ (np.eye(len(P)) - P) @ Wt) / np.linalg.norm(T))


def recipient_marginals(cohort, donor, rec, key, types):
    m = rec["donor"] == donor
    x, y = rec["x"][m], rec["y"][m]
    if key is not None:
        labels = rec["labels"][key][m]
        cells = np.isin(labels, types)
        x, y = type_centre(x[cells], labels[cells], types), type_centre(y[cells], labels[cells], types)
    Rx, Ry, _, _ = within_assay(x, y)
    return Rx, Ry


def main():
    ref = np.load(HERE / "results/reference_fit.npz")
    G, ridge, W = ref["pf_means_G"], float(ref["pf_means_ridge"]), ref["pf_means_W"]
    VM, VU, h = identified_subspace(G, ridge)
    truth = np.load(HERE / "results/truth.npz")
    scores = json.loads((HERE / "results/scores.json").read_text())["scores"]
    out = {"training_patients": int(len(ref["pf_patients"])), "dimension_M": int(VM.shape[1]),
           "effective_dimension": float(h.sum()), "H_eigenvalues": h.tolist(), "genes": int(G.shape[0]),
           "donors": {}}
    for cohort, analyses in ANALYSES.items():
        rec = load_recipient(cohort, "adaptation")
        for donor in sorted(set(rec["donor"])):
            for analysis, key in analyses:
                tag = f"{cohort}/{donor}/{analysis}"
                types = scores[tag]["types"]
                Rx, Ry = recipient_marginals(cohort, donor, rec, key, types)
                T, Rx_score = truth[f"{tag}/T"], truth[f"{tag}/Rx"]
                r, centre, sF = radius(Rx, Ry, W, VM)
                P = VM @ VM.T
                out["donors"][tag] = {
                    "rho": float(np.trace(P @ Rx @ P) / np.trace(Rx)),
                    "floor": floor(T, Rx_score, VM),
                    "radius": r / float(np.linalg.norm(T)),
                    "centre_error": float(np.linalg.norm(centre - T) / np.linalg.norm(T)),
                    "max_identified_singular_value": sF,
                    "E_pf_means": scores[tag]["arms"]["pf_means"]["E"],
                    "E_pf_moment": scores[tag]["arms"]["pf_moment"]["E"],
                    "E_paired_closed_form": scores[tag]["arms"]["paired_closed_form"]["E"]}
                print(tag, {k: round(v, 3) for k, v in out["donors"][tag].items()}, flush=True)
    out["per_gene"] = per_gene(VM, truth, scores)
    (HERE / "results/identifiability.json").write_text(json.dumps(out, indent=1) + "\n")


def per_gene(VM, truth, scores):
    """Post hoc, descriptive (not in PLAN.md): which gene-protein relationships are identified.

    For each gene g, h_g = ||P_M e_g||^2 is the share of its standardized axis inside M
    (needs no pairs). For each recipient donor, the realized error of pf_means on gene g's
    row of the cross-correlation matrix is ||C_hat[g] - T[g]|| / ||T[g]||, and the oracle's
    unidentified share is ||[R_x (I - P_M) R_x^-1 T][g]|| / ||T[g]||. Both are averaged over
    donors within each cohort (total analysis).
    """
    preds = np.load(HERE / "results/predictions.npz")
    P = VM @ VM.T
    h = np.diag(P)
    out = {"h": h.tolist(), "cohorts": {}}
    for cohort in ANALYSES:
        tags = [k for k in scores if k.startswith(cohort + "/") and k.endswith("/total")]
        err, unid, size = [], [], []
        for tag in tags:
            T, Rx = truth[f"{tag}/T"], truth[f"{tag}/Rx"]
            C = preds[f"{tag}/pf_means"]
            norm = np.linalg.norm(T, axis=1)
            err.append(np.linalg.norm(C - T, axis=1) / norm)
            unid.append(np.linalg.norm(Rx @ (np.eye(len(P)) - P) @ np.linalg.solve(Rx, T), axis=1) / norm)
            size.append(norm)
        err, unid, size = np.mean(err, 0), np.mean(unid, 0), np.mean(size, 0)
        out["cohorts"][cohort] = {"row_error": err.tolist(), "row_unidentified": unid.tolist(),
                                  "row_norm": size.tolist(),
                                  "spearman_h_error": spearman(h, err), "spearman_unidentified_error":
                                  spearman(unid, err)}
    return out


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
