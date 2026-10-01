"""Shared data handling of the semi-paired study (development on the two perturbation screens).

Setting (as extension/hybrid): training cells of eligible populations are split by
a fixed barcode hash into a 25% paired calibration reservoir and a 75% unpaired
pool (RNA from assay half 0, protein from half 1, never the same cell); held-out
groups are scored with the noise-unbiased recovered fraction of their
within-group gene-protein cross-correlation and with the uncorrected relative
squared error. This module adds unpaired latent (noise-corrected) RNA
correlations, pooled and per condition, and the encoding-gene map.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
HYB = HERE.parent / "hybrid"
sys.path.append(str(HYB))

import hmethods as hm  # noqa: E402

pm = hm.pm
from noise import noise_fraction, signal_correlation, split_halves  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


hr = _load("hybrid_run", HYB / "run.py")


def cond_of(key):
    return key.split("|")[1]


def within_population_corr(xs, pop, keys):
    """Pooled within-population covariance of standardized features, returned as a correlation matrix."""
    acc, n = 0.0, 0
    for k in keys:
        m = pop == k
        if m.sum() < 2:
            continue
        c = xs[m] - xs[m].mean(0)
        acc = acc + c.T @ c
        n += int(m.sum())
    S = acc / n
    d = np.sqrt(np.clip(np.diag(S), 1e-12, None))
    return S / np.outer(d, d), d


class Unpaired:
    """Unpaired summaries used by the semi-paired estimators: measured and latent RNA correlation and protein
    correlation, pooled over and within conditions, from pool cells only (RNA: half 0, protein: half 1)."""

    def __init__(self, pool, x, y, counts, library, pop, part, keys, seed=20261030):
        self.pool = pool
        keys = [k for k in keys if k in set(pool.keys)]
        rna = np.isin(pop, keys) & (part == 0)
        prot = np.isin(pop, keys) & (part == 1)
        xs, ys = x / pool.sd_x, y / pool.sd_y
        h1, h2 = split_halves(counts[rna], library[rna], seed=seed)
        self.conds = sorted({cond_of(k) for k in keys})
        self.R = {}
        for c in ["__all__"] + self.conds:
            ks = keys if c == "__all__" else [k for k in keys if cond_of(k) == c]
            mr = rna & np.isin(pop, ks)
            mp = prot & np.isin(pop, ks)
            Rx, _ = within_population_corr(xs[mr], pop[mr], ks)
            Ry, _ = within_population_corr(ys[mp], pop[mp], ks)
            sub = np.isin(pop[rna], ks)
            nu = noise_fraction(h1[sub], h2[sub], pop[rna][sub], ks)
            Rx_s = pm.shrink(Rx)
            self.R[c] = {"Rx": Rx_s, "Ry": pm.shrink(Ry), "Rz": signal_correlation(Rx_s, nu), "nu": nu,
                         "n_rna": int(mr.sum()), "n_prot": int(mp.sum())}


def encoding_sets(genes, proteins, encoding):
    """Indices of each protein's encoding genes present on the panel."""
    gi = {g: i for i, g in enumerate(genes)}
    return [[gi[g] for g in encoding[p] if g in gi] for p in proteins]


def setup_fold(train, x, y, pop, keys):
    """Pool, reservoir (centred calibration cells with their conditions) and unpaired summaries of one fold."""
    use = np.isin(pop, keys)
    res = hm.reservoir(train["cell"]) & use
    pl = use & ~res
    pool = hm.Pool(x[pl], y[pl], pop[pl], train["part"][pl], train["guide"][pl], keys)
    un = Unpaired(pool, x[pl], y[pl], train["counts"][pl], train["library"][pl], pop[pl], train["part"][pl], keys)
    rmask = res & np.isin(pop, pool.keys)
    X_res, Y_res = pool.centre_cells(x[rmask], y[rmask], pop[rmask])
    cond_res = np.array([cond_of(k) for k in pop[rmask]])
    return pool, un, X_res, Y_res, cond_res


def scores(C, T, TA, TB, masks):
    """Per-group, per-block terms: recovered-fraction numerator and denominator, and uncorrected loss terms."""
    if C.ndim == 2:
        C = np.broadcast_to(C, T.shape)
    num = 2 * np.einsum("gpq,bpq->gb", C * T, masks) - np.einsum("gpq,bpq->gb", C * C, masks)
    loss = np.einsum("gpq,bpq->gb", (C - T) ** 2, masks)
    return num, loss
