#!/usr/bin/env python3
"""Negative control for D3, added after the D3 results were seen (labelled post hoc).

For each recipient donor, the protein moments of the training biopsies are
reassigned to other biopsies' RNA moments by a random derangement, breaking
the between-sample correspondence while keeping every within-assay moment.
The primary pairing-free estimator (pf_means) is refitted with the ridge
chosen for the unpermuted data, and the recipient's cross-covariance error is
recomputed. 200 derangements per donor, seed 20260929.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import colon_scale as cs
from gaussian_transfer import Marginal, transfer
from pairing_free import ecological, sample_moments

HERE = Path(__file__).resolve().parent
PERMUTATIONS = 200


def derangement(rng, n):
    while True:
        perm = rng.permutation(n)
        if np.all(perm != np.arange(n)):
            return perm


def main():
    data = cs.load(200)
    observed = json.loads((HERE / "results/d3_pairing_free.json").read_text())["donors"]
    rng = np.random.default_rng(20260929)
    donors = sorted(set(data["donor"]))
    out = {}
    for donor in donors:
        samples = sorted(set(data["sample"][data["donor"] != donor]))
        units = sample_moments(data, samples)
        scale = observed[donor]["ridge_scale"]
        m = data["donor"] == donor
        adapt, score = m & (data["half"] == 0), m & (data["half"] == 1)
        n, mx, my, Sx, Sy, _ = cs.moments(data["x"][adapt], data["y"][adapt])
        marginal = Marginal(Sx, Sy)
        xs, ys = data["x"][score], data["y"][score]
        W, b, psi, _ = ecological(units, scale)
        base = {"regression": cs.evaluate(Sx @ W.T, Sx, mx, my, xs, ys)["cov_rel_error"],
                "closed_form": cs.evaluate(transfer(marginal, W.T / psi[None, :])[0], Sx, mx, my, xs, ys)["cov_rel_error"]}
        null = {"regression": [], "closed_form": []}
        for _ in range(PERMUTATIONS):
            perm = derangement(rng, len(units))
            shuffled = [(units[i][0], units[i][1], units[perm[i]][2], units[i][3], units[perm[i]][4])
                        for i in range(len(units))]
            Wp, bp, psip, _ = ecological(shuffled, scale)
            null["regression"].append(cs.evaluate(Sx @ Wp.T, Sx, mx, my, xs, ys)["cov_rel_error"])
            null["closed_form"].append(cs.evaluate(transfer(marginal, Wp.T / psip[None, :])[0],
                                                   Sx, mx, my, xs, ys)["cov_rel_error"])
        out[donor] = {arm: {"observed": base[arm], "null_mean": float(np.mean(null[arm])),
                            "null_quantiles": np.quantile(null[arm], (0.025, 0.5, 0.975)).tolist(),
                            "p_value": float((1 + np.sum(np.array(null[arm]) <= base[arm])) / (1 + PERMUTATIONS))}
                      for arm in base}
        print(donor, {a: (round(v["observed"], 3), round(v["null_mean"], 3), v["p_value"]) for a, v in out[donor].items()}, flush=True)
    (HERE / "results/d3_permutation_control.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
