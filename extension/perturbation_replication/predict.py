#!/usr/bin/env python3
"""Replication, step 1: leave-one-target-out fits and predictions from adaptation cells only (PLAN.md).

Refuses to run unless PLAN.md and the code match results/freeze.json. For every
target with an eligible test group (at least 40 adaptation cells), all channels
and comparators are fitted on the training halves of the other targets, and
every arm's prediction for the target's test groups is computed from their
adaptation cells, each assay separately (the own-pairs benchmark and the
replicate arm, which use the group's adaptation pairs, are reference points
outside the setting). Non-targeting groups use fits on all targets. Writes
results/predictions.npz, results/digests.json, results/diagnostics.json and
results/manifest.json before the scoring files are opened.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE.parent / "perturbation"))

import pmethods as pm  # noqa: E402
from freeze import check_freeze  # noqa: E402
from rdata import MIN_ADAPT, MIN_TRAIN_HALF, load, sha  # noqa: E402

REMOVE = 5
DRAWS = 50
DRAW_SEED = 20261019
RESULT_FILES = ["predictions.npz", "digests.json", "diagnostics.json", "seal.json", "freeze.json"]


def digest(M):
    return hashlib.sha256(np.ascontiguousarray(M, dtype=np.float64).tobytes()).hexdigest()


def exposures(raw):
    """RNA-only exposure and effect strength per training target (as extension/perturbation/design.py)."""
    sd_x, _ = pm.est.pooled_sd(raw)
    out, strength = {}, {}
    for c in sorted({r["patient"].split("|")[1] for r in raw}):
        rs = [r for r in raw if r["patient"].split("|")[1] == c]
        w = np.array([r["nx"] for r in rs], float)
        mean = sum(wi * r["mx"] for wi, r in zip(w, rs)) / w.sum()
        R = sum(wi * r["Sx"] for wi, r in zip(w, rs)) / w.sum() / np.outer(sd_x, sd_x)
        trR, trR2 = float(np.trace(R)), float(np.sum(R * R))
        for r in rs:
            t = r["patient"].split("|")[0]
            d = (r["mx"] - mean) / sd_x
            out[t] = out.get(t, 0.0) + float(d @ R @ d) - trR2 / r["nx"]
            strength[t] = strength.get(t, 0.0) + float(d @ d) - trR / r["nx"]
    return out, strength


def fold_fits(d, x, y, pop, keys, held, rng):
    """All fits of one fold: training populations of every target except `held` (None: all targets)."""
    ks = [k for k in keys if k.split("|")[0] != held]
    m = np.isin(pop, ks)
    raw = pm.moments(x[m], y[m], pop[m], d["part"][m], ks)
    fits = {"pf": pm.fit_channel(pm.centre_within_condition(raw))}
    for name, ch in pm.control_channels(x[m], y[m], pop[m], d["part"][m], ks, d["condition"][m], centre=True).items():
        fits[f"within_{name}"] = ch
    people = pm.paired_people(x[m], y[m], pop[m], ks)
    refs = pm.paired_references(people, closed_form=True, log=lambda s: None)
    expo, strength = exposures(raw)
    targets = sorted(expo)
    sets = {"exposure": sorted(targets, key=lambda t: (-expo[t], t))[:REMOVE],
            "strength": sorted(targets, key=lambda t: (-strength[t], t))[:REMOVE]}
    for r in range(DRAWS):
        sets[f"random{r:02d}"] = sorted(rng.choice(targets, size=REMOVE, replace=False))
    for name, s in sets.items():
        sub = [r for r in raw if r["patient"].split("|")[0] not in set(s)]
        fits[f"abl_{name}"] = pm.fit_channel(pm.centre_within_condition(sub))
    return fits, refs, sets, {"exposure": expo, "strength": strength}


def predictions(a, fits, refs, cond, cells):
    mx, mz = pm.Marginal(a["Rx"], a["Ry"]), pm.Marginal(a["Rz"], a["Ry"])
    out = {"independence": np.zeros((a["Rx"].shape[0], a["Ry"].shape[0])),
           "pf_within_measured": pm.transfer(a["Rx"], a["Ry"], fits["pf"]["B"], mx),
           "pf_within_latent": pm.transfer(a["Rz"], a["Ry"], fits["pf"]["B"], mz),
           "paired_closed_form": pm.transfer(a["Rx"], a["Ry"], refs["paired_closed_form"]["B"], mx),
           "reference_regression": a["Rx"] @ refs["reference_regression"]["W"],
           "transferred_correlation": refs["transferred_correlation"][cond]}
    for name, ch in fits.items():
        if name != "pf":
            out[f"{name}_measured"] = pm.transfer(a["Rx"], a["Ry"], ch["B"], mx)
    W_bench, _ = pm.est.benchmark(a["x"], a["y"], cells)
    out["benchmark"] = a["Rx"] @ W_bench
    xs, _ = pm.standardize(a["x"])
    ys, _ = pm.standardize(a["y"])
    out["replicate"] = xs.T @ ys / len(xs)
    return out


def run():
    t0 = time.time()
    d = load("train")
    x, y = pm.features(d)
    pop = pm.keys_of(d)
    keys = pm.eligible_training(pop, d["part"], MIN_TRAIN_HALF)
    rng = np.random.default_rng(DRAW_SEED)
    preds, diag = {}, []
    for part in ("test", "nt"):
        ad = load(f"{part}_adaptation")
        akeys = pm.keys_of(ad)
        eligible = [k for k in sorted(set(akeys)) if np.sum(akeys == k) >= MIN_ADAPT]
        folds = sorted({k.split("|")[0] for k in eligible}) if part == "test" else [None]
        for held in folds:
            fits, refs, sets, expo = fold_fits(d, x, y, pop, keys, held, rng)
            for key in [k for k in eligible if part == "nt" or k.split("|")[0] == held]:
                idx = np.flatnonzero(akeys == key)
                a = pm.group_inputs(ad["counts"][idx], ad["library"][idx], ad["y"][idx])
                cond = key.split("|")[1]
                for arm, C in predictions(a, fits, refs, cond, ad["cell"][idx]).items():
                    preds[f"{part}/{key}/{arm}"] = C
                VS = fits["pf"]["VS"]
                preds[f"{part}/{key}/S"] = VS
                diag.append({"part": part, "key": key, "target": key.split("|")[0], "replicate": cond,
                             "n_adapt": int(len(idx)), "fold": held, "dim_S": int(VS.shape[1]),
                             "scale": fits["pf"]["scale"], "diagnostics": pm.diagnostics(a, fits["pf"]),
                             "dim_S_ablation": {k: int(fits[f"abl_{k}"]["VS"].shape[1]) for k in sets},
                             "sets": {k: list(map(str, v)) for k, v in sets.items() if not k.startswith("random")},
                             "exposure": expo["exposure"]})
            print(part, held, f"({time.time() - t0:.0f}s)", flush=True)
    return preds, diag


def main():
    freeze = check_freeze()
    preds, diag = run()
    np.savez_compressed(HERE / "results/predictions.npz", **preds)
    (HERE / "results/digests.json").write_text(json.dumps({k: digest(v) for k, v in preds.items()}, indent=0) + "\n")
    (HERE / "results/diagnostics.json").write_text(json.dumps(diag, indent=1) + "\n")
    manifest = {n: sha(HERE / "results" / n) for n in RESULT_FILES}
    manifest.update(frozen_at=freeze["frozen_at"], note="written before the scoring files were opened")
    (HERE / "results/manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
