#!/usr/bin/env python3
"""Champollion (Samaran et al. 2026; github.com/cantinilab/champollion) as a comparator: run in its own environment.

    <champollion-venv>/bin/python champ_worker.py in.npz out.npz

in.npz: Xb (B x p), Yb (B x q) paired bridge cells (pooled-SD units, centred on unpaired population means);
        for each recipient group g: Xg_<g> (n x p) RNA and Yg_<g> (n x q) protein of its adaptation cells, used
        as two unpaired sets (pooled-SD units, centred within the group); params: epsilon, gamma, max_iter, seed.
out.npz: C_<g> (p x q), the cross-covariance of the entropic transport plan between the group's RNA and protein
         cells under the cost learned on the bridge, in the group's correlation units; plus fit diagnostics.
"""

import sys
import time

import anndata as ad
import mudata as md
import numpy as np
import torch

from champollion import Champollion


def main():
    z = np.load(sys.argv[1])
    eps, gamma, max_iter, seed = float(z["epsilon"]), float(z["gamma"]), int(z["max_iter"]), int(z["seed"])
    torch.set_num_threads(2)
    Xb, Yb = z["Xb"], z["Yb"]
    p, q = Xb.shape[1], Yb.shape[1]
    names = [f"c{i}" for i in range(len(Xb))]
    rna = ad.AnnData(Xb.astype(np.float32))
    rna.obs_names = names
    rna.var_names = [f"g{i}" for i in range(p)]
    prot = ad.AnnData(Yb.astype(np.float32))
    prot.obs_names = names
    prot.var_names = [f"p{j}" for j in range(q)]
    t0 = time.time()
    model = Champollion(epsilon=eps, gamma=gamma, use_keops=False, device="cpu", random_state=seed,
                        max_iter=max_iter, verbose=False)
    model.fit(md.MuData({"rna": rna, "protein": prot}), modality_1="rna", modality_2="protein")
    fit_time = time.time() - t0
    out = {"fit_seconds": fit_time, "A_norm": float(torch.linalg.norm(model.A_).item())}
    keys = sorted({k[3:] for k in z.files if k.startswith("Xg_")})
    for g in keys:
        Xg, Yg = z[f"Xg_{g}"].astype(np.float32), z[f"Yg_{g}"].astype(np.float32)
        a1 = ad.AnnData(Xg)
        a1.var_names = rna.var_names
        a1.obs_names = [f"r{i}" for i in range(len(Xg))]
        a2 = ad.AnnData(Yg)
        a2.var_names = prot.var_names
        a2.obs_names = [f"s{i}" for i in range(len(Yg))]
        res = model.transport({"rna": a1, "protein": a2}, store_plan=True)
        P = res.plan
        P = (P.detach().cpu().numpy() if hasattr(P, "detach") else np.asarray(P)).astype(float)
        P = P / P.sum()
        r, c = P.sum(1), P.sum(0)
        xc = Xg - r @ Xg
        yc = Yg - c @ Yg
        C = xc.T @ P @ yc
        sx = np.sqrt(np.maximum(r @ (xc * xc), 1e-12))
        sy = np.sqrt(np.maximum(c @ (yc * yc), 1e-12))
        out[f"C_{g}"] = C / np.outer(sx, sy)
    np.savez(sys.argv[2], **out)


if __name__ == "__main__":
    main()
