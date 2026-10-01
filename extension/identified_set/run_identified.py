#!/usr/bin/env python3
"""Compute certified inner bounds for every recipient and query.

Recipients: 95 retrospective adaptation-only recipients (24 Cambridge,
56 Newcastle, 15 T-ALL) and the 11 colon donors, all with the unchanged
blood-reference interaction. Inputs are released arrays only.

    python run_identified.py compute --shard 0 --of 2
    python run_identified.py compute --shard 1 --of 2
    python run_identified.py merge
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from identified import SUPPORT, certify, first_order_vertices, inner_interval, kernel

HERE = Path(__file__).resolve().parent
RELEASE = HERE.parent.parent / "release" / "analysis"
RESULTS = HERE / "results"
PARTS = RESULTS / "per_recipient"
MARGIN = 1e-6


def load_recipients():
    """Marker frequencies, arm predictions and held-out tables for 106 recipients."""
    source = np.load(RELEASE / "assay_resolution/results/adaptation_only_results/source_fit.npz")
    inputs = np.load(RELEASE / "assay_resolution/reconstruction_inputs.npz")
    retro = np.load(RELEASE / "assay_resolution/results/adaptation_only_results/predictions.npz")
    retro_truth = np.load(RELEASE / "assay_resolution/results/adaptation_only_results/losses.npz")["truth"]
    colon = np.load(RELEASE / "colon_generalization/results/predictions.npz")
    colon_truth = np.load(RELEASE / "colon_generalization/results/losses.npz")["truth"]
    interaction = source["b"]
    assert np.array_equal(interaction, inputs["strict_b"])
    assert np.array_equal(inputs["states"], SUPPORT)
    marginals = np.concatenate([inputs["strict_marginals"], colon["marginals"]])
    predictions = np.concatenate([retro["predictions"], colon["predictions"]])
    truth = np.concatenate([retro_truth, colon_truth])
    cohorts = np.concatenate([retro["cohorts"], np.full(len(colon_truth), "Colon")])
    assert list(retro["methods"]) == list(colon["methods"])
    return {
        "interaction": interaction,
        "rna_means": marginals[:, 0] @ SUPPORT,
        "protein_means": marginals[:, 1] @ SUPPORT,
        "predictions": predictions,
        "truth": truth,
        "cohorts": cohorts,
        "methods": np.asarray(retro["methods"]),
    }


def compute(shard: int, of: int, people=None) -> None:
    data = load_recipients()
    K = kernel(data["interaction"])
    PARTS.mkdir(parents=True, exist_ok=True)
    count = len(data["cohorts"])
    order = range(shard, count, of) if people is None else people
    for person in order:
        path = PARTS / f"{person:03d}.npz"
        if path.exists():
            continue
        started = time.time()
        mu, nu = data["rna_means"][person], data["protein_means"][person]
        rows = {key: [] for key in ("t_low", "t_high", "z_low", "z_high",
                                    "evaluations", "failures", "first_order")}
        for i in range(9):
            for j in range(9):
                result = inner_interval(K, data["interaction"], mu, nu, i, j, margin=MARGIN)
                for key in ("t_low", "t_high", "z_low", "z_high", "evaluations", "failures"):
                    rows[key].append(result[key])
                rows["first_order"].append(first_order_vertices(data["interaction"], mu, nu, i, j)[0])
        temporary = path.with_suffix(".partial.npz")
        np.savez_compressed(temporary, **{k: np.asarray(v) for k, v in rows.items()},
                            seconds=time.time() - started)
        temporary.rename(path)
        print(f"recipient {person + 1}/{count} ({data['cohorts'][person]}) "
              f"{time.time() - started:.0f}s", flush=True)


def merge() -> None:
    """Combine shards and re-certify every saved endpoint at tolerance 1e-12."""
    data = load_recipients()
    K = kernel(data["interaction"])
    count = len(data["cohorts"])
    parts = [np.load(PARTS / f"{person:03d}.npz") for person in range(count)]
    stacked = {key: np.stack([part[key] for part in parts])
               for key in ("t_low", "t_high", "z_low", "z_high", "evaluations",
                           "failures", "first_order")}
    certified = np.empty((count, 81, 2))
    for person in range(count):
        mu, nu = data["rna_means"][person], data["protein_means"][person]
        for q in range(81):
            i, j = divmod(q, 9)
            certified[person, q, 0] = certify(K, mu, nu, i, j, stacked["z_low"][person, q], MARGIN)
            certified[person, q, 1] = certify(K, mu, nu, i, j, stacked["z_high"][person, q], MARGIN)
    search = np.stack([stacked["t_low"], stacked["t_high"]], axis=-1)
    difference = float(np.max(np.abs(certified - search)))
    assert difference < 1e-9, difference
    np.savez_compressed(
        RESULTS / "star_endpoints.npz",
        star_low=certified[..., 0], star_high=certified[..., 1],
        z_low=stacked["z_low"], z_high=stacked["z_high"],
        first_order=stacked["first_order"], evaluations=stacked["evaluations"],
        failures=stacked["failures"], cohorts=data["cohorts"],
        rna_means=data["rna_means"], protein_means=data["protein_means"],
        seconds=np.array([part["seconds"] for part in parts]), margin=MARGIN,
    )
    report = {
        "recipients": count,
        "queries_per_recipient": 81,
        "star_endpoints_certified": int(certified.size),
        "max_search_vs_certified_difference": difference,
        "scaling_failures_during_search": int(stacked["failures"].sum()),
        "objective_evaluations": int(stacked["evaluations"].sum()),
        "compute_seconds": float(sum(part["seconds"] for part in parts)),
    }
    (RESULTS / "certification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("compute", "merge"))
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--of", type=int, default=1)
    parser.add_argument("--people", type=int, nargs="*",
                        help="explicit recipient indices (order of computation only)")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.stage == "compute":
            compute(args.shard, args.of, args.people)
        else:
            merge()
