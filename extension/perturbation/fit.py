#!/usr/bin/env python3
"""Perturbation test, step 1: fit every channel and paired comparator on training targets only (PLAN.md).

Reads perturb_training.npz (seal-checked). Writes results/channels.npz and
results/channels.json: the pairing-free channels (condition-stratified, primary;
unstratified, secondary), their matched controls (pseudo-populations and
within-condition derangements, ten draws each), the paired comparators (paired
closed form, reference regression, transferred within-condition correlation) and
the pooled within-population correlation matrices of each condition.
"""

from __future__ import annotations

import json
import time

import numpy as np
from threadpoolctl import threadpool_limits

import pmethods as pm
from pdata import MIN_TRAIN_HALF, load

ARRAYS = ("W", "B", "G", "VS")


def main():
    t0 = time.time()
    d = load("training")
    x, y = pm.features(d)
    pop = pm.keys_of(d)
    keys = pm.eligible_training(pop, d["part"], MIN_TRAIN_HALF)
    raw = pm.moments(x, y, pop, d["part"], keys)
    chans = {"pf_within": pm.fit_channel(pm.centre_within_condition(raw)), "pf": pm.fit_channel(raw)}
    for k, v in pm.control_channels(x, y, pop, d["part"], keys, d["condition"], centre=True).items():
        chans[f"within_{k}"] = v
    for k, v in pm.control_channels(x, y, pop, d["part"], keys, d["condition"]).items():
        chans[k] = v
    print(f"{len(keys)} populations; channels fitted ({time.time() - t0:.0f}s)", flush=True)
    people = pm.paired_people(x, y, pop, keys)
    refs = pm.paired_references(people, closed_form=True)
    print(f"paired comparators fitted ({time.time() - t0:.0f}s)", flush=True)
    cm = pm.condition_marginals(x, y, d["counts"], d["library"], pop, d["part"], keys)
    arrays, info = {}, {"populations": keys, "channels": {}}
    for name, ch in chans.items():
        for a in ARRAYS:
            arrays[f"{name}/{a}"] = ch[a]
        arrays[f"{name}/ridge"] = np.array(ch["ridge"])
        info["channels"][name] = {"scale": ch["scale"], "ridge": ch["ridge"], "at_edge": ch["at_edge"],
                                  "n_train": ch["n_train"], "dim_S": int(ch["VS"].shape[1])}
    arrays["paired/B"] = refs["paired_closed_form"]["B"]
    arrays["paired/W_ref"] = refs["reference_regression"]["W"]
    for c, C in refs["transferred_correlation"].items():
        arrays[f"paired/corr/{c}"] = C
    for c, m in cm.items():
        for k in ("Rx", "Ry", "Rz", "nu"):
            arrays[f"condition/{c}/{k}"] = m[k]
    info["paired_closed_form"] = {k: refs["paired_closed_form"][k] for k in ("ridge", "converged", "iterations",
                                                                             "message", "grad_norm")}
    info["paired_closed_form"]["tuning"] = refs["paired_closed_form"]["tuning"]
    info["reference_regression"] = {"lambda": refs["reference_regression"]["lambda"],
                                    "tuning": refs["reference_regression"]["tuning"]}
    info["cells"] = {"rna_half": int(sum(r["nx"] for r in raw)), "protein_half": int(sum(r["ny"] for r in raw))}
    np.savez_compressed(pm.HERE / "results/channels.npz", **arrays)
    (pm.HERE / "results/channels.json").write_text(json.dumps(info, indent=1, default=float) + "\n")
    print(json.dumps({k: v for k, v in info.items() if k != "populations"}, indent=1, default=float)[:3000])
    print(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
