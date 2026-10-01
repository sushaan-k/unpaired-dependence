#!/usr/bin/env python3
"""Closed-form transfer on the released nine-marker binary data (post hoc, E1).

For all 95 adaptation-only recipients and 11 colon donors, with the unchanged
blood-reference interaction and the released smoothed adaptation histograms:

  frequencies    released individual-frequency arm (exact, 2^9 states)
  first_order    P11 = m n + (Sx B Sy)_ij
  closed_form    P11 = m n + C_ij, C from the Gaussian closed form with the
                 recipients' within-assay covariance matrices (O(p^3))
  pairwise_exact pairwise maximum-entropy laws matching each assay's means and
                 covariances, then exact matrix scaling (2^9 states)
  patterns       released both-pattern arm (exact, 2^9 states)
  independence   released independence arm

Cells are clipped to 1e-9 inside their Frechet bounds when a closed-form cell
leaves them; the number of clipped cells is reported. Loss: twice KL from the
held-out table, averaged over 81 queries then people. Paired recipient
bootstrap (20,000 draws, seed 20260928) for the share of the
frequencies-to-patterns gain retained, per cohort.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, rel_entr
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "identified_set"))
from identified import SUPPORT, kernel, scale                     # noqa: E402
from run_identified import load_recipients                          # noqa: E402
from gaussian_transfer import Marginal, first_order, transfer       # noqa: E402

COHORTS = ("Cambridge", "Newcastle", "T-ALL", "Colon")
PAIRS = np.array([(i, j) for i in range(9) for j in range(i + 1, 9)])
FEATURES = np.concatenate([SUPPORT, SUPPORT[:, PAIRS[:, 0]] * SUPPORT[:, PAIRS[:, 1]]], axis=1)


def pairwise_maxent(law):
    """Maximum-entropy law on {0,1}^9 with the same first and second moments."""
    target = law @ FEATURES
    theta = np.zeros(FEATURES.shape[1])

    def dual(t):
        logits = FEATURES @ t
        return logsumexp(logits) - t @ target

    for _ in range(500):
        logits = FEATURES @ theta
        fitted = np.exp(logits - logsumexp(logits))
        gradient = fitted @ FEATURES - target
        if np.max(np.abs(gradient)) < 1e-11:
            break
        centred = FEATURES - fitted @ FEATURES
        hessian = (centred * fitted[:, None]).T @ centred
        step = np.linalg.solve(hessian + 1e-12 * np.eye(len(theta)), gradient)
        value, size = dual(theta), 1.0
        while dual(theta - size * step) > value - 1e-4 * size * (gradient @ step) and size > 1e-10:
            size /= 2
        theta = theta - size * step
    fitted = np.exp(FEATURES @ theta - logsumexp(FEATURES @ theta))
    assert np.max(np.abs(fitted @ FEATURES - target)) < 1e-9, "moment matching failed"
    return fitted


def tables_from_cross(m, n, C):
    lo = np.maximum(0, m[:, None] + n[None, :] - 1)
    hi = np.minimum(m[:, None], n[None, :])
    raw = m[:, None] * n[None, :] + C
    t = np.clip(raw, lo + 1e-9, hi - 1e-9)
    clipped = int(np.sum(t != raw))
    T = np.stack([1 - m[:, None] - n[None, :] + t, n[None, :] - t, m[:, None] - t, t], -1)
    return T.reshape(81, 2, 2), clipped


def tables_from_joint(Q):
    both = SUPPORT.T @ Q @ SUPPORT
    m, n = Q.sum(1) @ SUPPORT, Q.sum(0) @ SUPPORT
    T = np.stack([1 - m[:, None] - n[None, :] + both, n[None, :] - both, m[:, None] - both, both], -1)
    return T.reshape(81, 2, 2)


def main():
    data = load_recipients()
    release = HERE.parent.parent / "release" / "analysis"
    inputs = np.load(release / "assay_resolution/reconstruction_inputs.npz")
    colon = np.load(release / "colon_generalization/results/predictions.npz")
    marginals = np.concatenate([inputs["strict_marginals"], colon["marginals"]])
    B, K = data["interaction"], kernel(data["interaction"])
    arms = ("frequencies", "first_order", "closed_form", "pairwise_exact", "patterns", "independence")
    loss = np.zeros((len(marginals), len(arms)))
    clipped = {"first_order": 0, "closed_form": 0}
    for p, (r, c) in enumerate(marginals):
        m, n = r @ SUPPORT, c @ SUPPORT
        Sx = (SUPPORT - m).T @ ((SUPPORT - m) * r[:, None])
        Sy = (SUPPORT - n).T @ ((SUPPORT - n) * c[:, None])
        marginal = Marginal(Sx, Sy)
        T_first, k1 = tables_from_cross(m, n, first_order(marginal, B))
        T_closed, k2 = tables_from_cross(m, n, transfer(marginal, B)[0])
        clipped["first_order"] += k1
        clipped["closed_form"] += k2
        Q, _, _ = scale(K, pairwise_maxent(r), pairwise_maxent(c))
        predictions = {"frequencies": data["predictions"][p, 0], "first_order": T_first,
                       "closed_form": T_closed, "pairwise_exact": tables_from_joint(Q),
                       "patterns": data["predictions"][p, 3], "independence": data["predictions"][p, 4]}
        for a, arm in enumerate(arms):
            loss[p, a] = np.mean(2 * rel_entr(data["truth"][p], predictions[arm]).sum((1, 2)))
    report = {"arms": arms, "clipped_cells": clipped, "cohorts": {}}
    for cohort in COHORTS:
        mask = data["cohorts"] == cohort
        values = loss[mask]
        mean = values.mean(0)
        draws = np.random.default_rng(20260928).integers(0, mask.sum(), (20000, mask.sum()))
        boot = values[draws].mean(1)
        entry = {"n": int(mask.sum()), "mean_loss": dict(zip(arms, mean.tolist()))}
        for arm in ("first_order", "closed_form", "pairwise_exact"):
            a = arms.index(arm)
            share = (boot[:, 0] - boot[:, a]) / (boot[:, 0] - boot[:, 4])
            entry[f"{arm}_gain_retained"] = float((mean[0] - mean[a]) / (mean[0] - mean[4]))
            entry[f"{arm}_gain_retained_95"] = np.quantile(share, (0.025, 0.975)).tolist()
            entry[f"{arm}_beats_frequencies"] = int(np.sum(values[:, a] < values[:, 0]))
        report["cohorts"][cohort] = entry
    out = HERE / "results" / "binary_released.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(HERE / "results" / "binary_released_losses.npz", loss=loss, arms=np.array(arms),
                        cohorts=data["cohorts"])
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
