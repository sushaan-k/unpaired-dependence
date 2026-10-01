#!/usr/bin/env python3
"""Post hoc (after the deployment test was scored): how similar are the unpaired RNA correlation structures that
the three pools supply? For every fold, the overlap of the leading latent-RNA eigenvectors (mean squared cosine of
the principal angles between the top-k subspaces, k = 5, 20, 60) of the other-site CITE-seq pool and of the
multiome pool with those of the same-site CITE-seq pool, and the correlation between the matrices' off-diagonal
entries. Pool cells only; no held-out cell is read.

    python posthoc_assay.py     # writes results_posthoc/assay_structure.json
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import drun  # noqa: E402


def overlap(U, V, k):
    s = np.linalg.svd(U[:, :k].T @ V[:, :k], compute_uv=False)
    return float(np.mean(s ** 2))


def offdiag_corr(A, B):
    m = ~np.eye(len(A), dtype=bool)
    return float(np.corrcoef(A[m], B[m])[0, 1])


def main():
    cite, mult = drun.load("bmmc_cite"), drun.load("bmmc_multiome")
    part, half, res = drun.roles(cite)
    out = {}
    for fd in drun.folds(cite, mult):
        rr = np.flatnonzero(np.isin(cite["pop"], fd["other"]) & (part == 0))
        rp = np.flatnonzero(np.isin(cite["pop"], fd["other"]) & (part == 1))
        genes, yp = drun.panels(cite, mult, rr, rp)
        P = drun.pools(cite, mult, fd, part, res, genes, yp)
        Us = {k: P[k][1].Uz for k in P}
        Rz = {k: P[k][1].un.R["__all__"]["Rz"] for k in P}
        rec = {}
        for k in ("other", "assay"):
            rec[k] = {f"top{n}": overlap(Us["same"], Us[k], n) for n in (5, 20, 60)}
            rec[k]["offdiag_r"] = offdiag_corr(Rz["same"], Rz[k])
        out[fd["heldout"]] = rec
        print(fd["heldout"], json.dumps(rec), flush=True)
    summ = {k: {m: float(np.mean([out[b][k][m] for b in out])) for m in ("top5", "top20", "top60", "offdiag_r")}
            for k in ("other", "assay")}
    (HERE / "results_posthoc").mkdir(exist_ok=True)
    (HERE / "results_posthoc" / "assay_structure.json").write_text(json.dumps({"note": "post hoc; pool cells only",
                                                                              "mean": summ, "folds": out}, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
