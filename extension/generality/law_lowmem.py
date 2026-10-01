#!/usr/bin/env python3
"""Deviation 2 (DEVIATIONS.md): run the frozen glaw.py with a memory-light but algebraically identical
block_moments, because the frozen version materialises every per-cell product vector (for a reservoir of ~12,000
cells and a 140 x 197 block, more than 2.6 GB) and was stopped by the memory limit. The identity used:
sum_i ||z_i - m||^2 = sum_i ||x_i||^2 ||y_i||^2 - N ||m||^2 for z_i = x_i y_i' restricted to the block.

    python law_lowmem.py check      # equality with the frozen function on small reservoirs
    python law_lowmem.py predict    # glaw.predict() with the light function (run through with_object_arrays.py)
"""

import sys

import numpy as np

import glaw


def block_moments(X, Y, U, V, rblocks, pblocks):
    N = len(X)
    XU, YV = X @ U, Y @ V
    a2, tau = [], []
    for rows in rblocks:
        xr = XU[:, rows]
        nx = np.einsum("ij,ij->i", xr, xr)
        for cols in pblocks:
            yc = YV[:, cols]
            ny = np.einsum("ij,ij->i", yc, yc)
            M = xr.T @ yc / N
            m2 = float(np.sum(M * M))
            t = (float(np.sum(nx * ny)) - N * m2) / (N - 1)
            tau.append(t)
            a2.append(max(m2 - t / N, 0.0))
    return np.array(a2), np.array(tau)


if __name__ == "__main__":
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        if sys.argv[1] == "check":
            glaw.gr.check_freeze()
            for name in ("scala_m1", "gouwens_visp", "banc"):
                X, Y, U, V, rb, pb = glaw.benchmark_fold0(name)
                a1, t1 = glaw.block_moments(X, Y, U, V, rb, pb)
                a2, t2 = block_moments(X, Y, U, V, rb, pb)
                print(name, len(X), "max relative difference", float(np.max(np.abs(t1 - t2) / np.abs(t1))),
                      float(np.max(np.abs(a1 - a2) / np.maximum(np.abs(a1), 1e-300))),
                      "predicted", glaw.predicted_saving(a1, t1)[0], glaw.predicted_saving(a2, t2)[0], flush=True)
        else:
            glaw.gr.check_freeze()
            glaw.block_moments = block_moments
            glaw.predict()
