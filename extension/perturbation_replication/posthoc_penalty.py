#!/usr/bin/env python3
"""Post hoc, exploratory (after unsealing): does the replication fail because the penalty admits too many directions?

For every leave-one-target-out fold, the condition-centred channel is refitted on
the same training populations with the ridge scale fixed at 0.1 (the value that
cross-validation chose in the Frangieh screen) and at 1, and also truncated to its
single leading supported direction, and the recovered fraction is recomputed
(all genes and proteins, and the interferon block). Writes results/posthoc_penalty.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE.parent / "perturbation"))

import pmethods as pm  # noqa: E402
from evaluate import ifn_block  # noqa: E402
from rdata import MIN_ADAPT, MIN_TRAIN_HALF, load, seal  # noqa: E402

SCALES = (0.01, 0.1, 1.0)


def fit_fixed(raw, scale):
    sd_x, sd_y = pm.est.pooled_sd(raw)
    units = pm.est.pf_units(raw, sd_x, sd_y)
    W, b, psi, ridge, G = pm.est.ecological(units, scale)
    VS, _ = pm.supported_basis(G, ridge)
    return {"W": W, "B": W.T / psi[None, :], "VS": VS}


def main():
    d = load("train")
    x, y = pm.features(d)
    pop = pm.keys_of(d)
    keys = pm.eligible_training(pop, d["part"], MIN_TRAIN_HALF)
    blocks, _ = ifn_block(seal())
    out = {}
    for part in ("test", "nt"):
        ad, sc = load(f"{part}_adaptation"), load(f"{part}_scoring")
        ak, sk = pm.keys_of(ad), pm.keys_of(sc)
        eligible = [k for k in sorted(set(ak)) if np.sum(ak == k) >= MIN_ADAPT]
        folds = sorted({k.split("|")[0] for k in eligible}) if part == "test" else [None]
        acc = {}
        for held in folds:
            ks = [k for k in keys if k.split("|")[0] != held]
            m = np.isin(pop, ks)
            raw = pm.centre_within_condition(pm.moments(x[m], y[m], pop[m], d["part"][m], ks))
            fits = {f"scale{s:g}": fit_fixed(raw, s) for s in SCALES}
            for key in [k for k in eligible if part == "nt" or k.split("|")[0] == held]:
                i = np.flatnonzero(ak == key)
                j = np.flatnonzero(sk == key)
                a = pm.group_inputs(ad["counts"][i], ad["library"][i], ad["y"][i])
                t = pm.scoring_targets(sc["counts"][j], sc["library"][j], sc["y"][j], sc["part"][j])
                mx = pm.Marginal(a["Rx"], a["Ry"])
                for name, ch in fits.items():
                    C = pm.transfer(a["Rx"], a["Ry"], ch["B"], mx)
                    for bname, blk in (("all", None), ("ifn", blocks["ifn"])):
                        cut = (lambda M: M) if blk is None else (lambda M, blk=blk: M[np.ix_(blk[0], blk[1])])
                        e = acc.setdefault(f"{name}/{bname}", [0.0, 0.0])
                        Cb, T, TA, TB = cut(C), cut(t["T"]), cut(t["TA"]), cut(t["TB"])
                        e[0] += 2 * float(np.sum(Cb * T)) - float(np.sum(Cb * Cb))
                        e[1] += float(np.sum(TA * TB))
                    acc.setdefault(f"{name}/dim_S", []).append(int(ch["VS"].shape[1]))
        out[part] = {k: (v[0] / v[1] if not k.endswith("dim_S") else float(np.median(v))) for k, v in acc.items()}
        print(part, json.dumps(out[part], indent=1), flush=True)
    (HERE / "results/posthoc_penalty.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
