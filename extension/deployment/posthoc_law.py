#!/usr/bin/env python3
"""Post hoc (after the deployment test was scored): what the law of the saving says about each source of unpaired
cells.

For every fold of the deployment test and every unpaired pool (own site, other sites, other sites with nuclear
RNA), the paired reservoir's within-population cross-covariance is expressed in that pool's bases (eigenvectors of
its latent RNA correlation and of its protein correlation), and the block moments of the law (unbiased signal r_b^2
and per-cell noise trace tau_b of each block) are computed from all reservoir cells, centred within their
populations by Helmert contrasts (posthoc_contrasts.contrasts). The predicted saving of block over one-block
James-Stein shrinkage at half of the dependence recovered then follows from the prospectively tested law
(extension/predictability/ptools.py, imported unchanged). No held-out cell is read.

    python posthoc_law.py    # results_posthoc/law_by_source.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "predictability"))
import drun  # noqa: E402
import posthoc_contrasts as pc  # noqa: E402
import ptools as pt  # noqa: E402

es = drun.es
OUT = HERE / "results_posthoc"
LAW_BUDGETS = [25 * 2 ** k for k in range(10)]   # 25 .. 12,800, as in the prospective test of the law


def moments(X, Y, U, V, rblocks, pblocks):
    N = len(X)
    XU, YV = X @ U, Y @ V
    M = XU.T @ YV / N
    nx, ny = XU * XU, YV * YV
    out = []
    for rows in rblocks:
        sx = nx[:, rows].sum(1)
        for cols in pblocks:
            Mb = M[np.ix_(rows, cols)]
            n2 = float(np.sum(Mb * Mb))
            t = (float(np.sum(sx * ny[:, cols].sum(1))) - N * n2) / (N * (N - 1))
            out.append({"r2": n2 - t, "tau": N * t, "d": len(rows) * len(cols)})
    return out


def main():
    cite, mult = drun.load("bmmc_cite"), drun.load("bmmc_multiome")
    part, half, res = drun.roles(cite)
    rec = {}
    t0 = time.time()
    for fd in drun.folds(cite, mult):
        f = fd["fold"]
        rr = np.flatnonzero(np.isin(cite["pop"], fd["other"]) & (part == 0))
        rp = np.flatnonzero(np.isin(cite["pop"], fd["other"]) & (part == 1))
        genes, yp = drun.panels(cite, mult, rr, rp)
        P = drun.pools(cite, mult, fd, part, res, genes, yp)
        rres = np.flatnonzero(np.isin(cite["pop"], fd["paired"]) & res)
        xr, _, _ = drun.rna(cite, rres, genes)
        yr = drun.prot(cite, rres, yp)
        popr = np.array([f"{p}|{c}" for p, c in zip(cite["pop"][rres], cite["cond"][rres])])
        X, Y, _ = pc.contrasts(xr, yr, popr)
        out = {}
        for key in ("same", "other", "assay"):
            ctx = P[key][1]
            U = ctx.Uz
            V, _ = es.eigenbasis(ctx.Ry)
            pblocks = es.eigen_blocks(V.shape[1], es.PROTEIN_EDGES)
            blocks = moments(X, Y, U, V, ctx.blocks, pblocks)
            w, pi = pt.shares(blocks)
            cv = pt.curves(blocks, LAW_BUDGETS, "lin")
            s_pred = None
            if cv is not None:
                nk, rk = pt.needed(cv[0], 0.5, LAW_BUDGETS)
                n1, r1 = pt.needed(cv[1], 0.5, LAW_BUDGETS)
                s_pred = n1 / nk
            lead = [i for i, _ in enumerate(blocks)][:len(es.PROTEIN_EDGES) + 1]   # RNA 1-5 x protein blocks
            out[key] = {"w": w.tolist(), "pi": pi.tolist(), "saving_law": s_pred,
                        "saving_closed": float(pt.saving_closed(w, pi, 0.5)),
                        "chi2": float(np.sum(np.where(pi > 0, (w - pi) ** 2 / pi, 0.0))),
                        "w_leading_rna5": float(sum(w[i] for i in lead)),
                        "pi_leading_rna5": float(sum(pi[i] for i in lead)),
                        "r2_total": float(sum(max(b["r2"], 0.0) for b in blocks)),
                        "tau_total": float(sum(b["tau"] for b in blocks))}
        rec[str(f)] = {"heldout": fd["heldout"], "site": fd["site"], "contrasts": int(len(X)), **out}
        print(f"fold {f}: " + ", ".join(f"{k} S={out[k]['saving_closed']:.2f} w5={out[k]['w_leading_rna5']:.2f} "
                                        f"pi5={out[k]['pi_leading_rna5']:.2f}" for k in out)
              + f" [{time.time() - t0:.0f}s]", flush=True)
    summ = {}
    for key in ("same", "other", "assay"):
        s = np.array([rec[f][key]["saving_closed"] for f in rec])
        summ[key] = {"saving_closed_geomean": float(np.exp(np.mean(np.log(s)))),
                     "saving_closed_range": [float(s.min()), float(s.max())],
                     "w_leading_rna5_mean": float(np.mean([rec[f][key]["w_leading_rna5"] for f in rec])),
                     "pi_leading_rna5_mean": float(np.mean([rec[f][key]["pi_leading_rna5"] for f in rec]))}
    (OUT / "law_by_source.json").write_text(json.dumps({"folds": rec, "summary": summ,
                                                         "written": time.strftime("%Y-%m-%d %H:%M:%S %Z")},
                                                        indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        drun.check_freeze()
        main()
