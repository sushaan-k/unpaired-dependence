#!/usr/bin/env python3
"""Learning curve for pairing-free estimation (post hoc; added after D3 results).

For each recipient donor, the primary pairing-free estimator (pf_means,
closed-form arm) is refitted on random subsets of the other donors' biopsy
samples of size 5, 10 and 15 (20 subsets each, seed 20260930), and on all of
them, with the ridge chosen for the full set. The endpoint is the
cross-covariance relative error on the recipient's scoring half.
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
SIZES = (5, 10, 15)
REPEATS = 20


def main():
    data = cs.load(200)
    observed = json.loads((HERE / "results/d3_pairing_free.json").read_text())["donors"]
    rng = np.random.default_rng(20260930)
    out = {}
    for donor in sorted(set(data["donor"])):
        samples = sorted(set(data["sample"][data["donor"] != donor]))
        units = sample_moments(data, samples)
        scale = observed[donor]["ridge_scale"]
        m = data["donor"] == donor
        adapt, score = m & (data["half"] == 0), m & (data["half"] == 1)
        _, mx, my, Sx, Sy, _ = cs.moments(data["x"][adapt], data["y"][adapt])
        marginal = Marginal(Sx, Sy)
        xs, ys = data["x"][score], data["y"][score]

        def error(subset):
            W, b, psi, _ = ecological([units[i] for i in subset], scale)
            C = transfer(marginal, W.T / psi[None, :])[0]
            return cs.evaluate(C, Sx, mx, my, xs, ys)["cov_rel_error"]

        curve = {str(size): [error(rng.choice(len(units), size, replace=False)) for _ in range(REPEATS)]
                 for size in SIZES}
        curve[str(len(units))] = [error(np.arange(len(units)))]
        out[donor] = curve
        print(donor, {k: round(float(np.mean(v)), 3) for k, v in curve.items()}, flush=True)
    (HERE / "results/d3_learning_curve.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
