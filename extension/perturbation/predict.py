#!/usr/bin/env python3
"""Perturbation test, step 2: predictions for held-out and non-targeting groups from adaptation cells only.

Refuses to run unless PLAN.md, the code and the channel fits match
results/freeze.json. For every eligible group (at least MIN_ADAPT adaptation
cells) it computes each arm's predicted within-group gene-protein
cross-correlation from the group's RNA-only and protein-only statistics and the
training fits, plus descriptive diagnostics, and records the SHA-256 digest of
every prediction. Writes results/predictions.npz, results/digests.json,
results/diagnostics.json and results/manifest.json (hashes of all result files,
written before the scoring files are opened).
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
from threadpoolctl import threadpool_limits

import pmethods as pm
from freeze import check_freeze
from pdata import MIN_ADAPT, load, sha

PRIMARY = "pf_within_measured"
PAIRED = "paired_closed_form"
RESULT_FILES = ["predictions.npz", "digests.json", "diagnostics.json", "channels.npz", "channels.json", "seal.json",
                "freeze.json"]


def digest(M):
    return hashlib.sha256(np.ascontiguousarray(M, dtype=np.float64).tobytes()).hexdigest()


def load_fits():
    z = np.load(pm.HERE / "results/channels.npz")
    info = json.loads((pm.HERE / "results/channels.json").read_text())
    chans = {name: {"W": z[f"{name}/W"], "B": z[f"{name}/B"], "VS": z[f"{name}/VS"]} for name in info["channels"]}
    conds = sorted({k.split("/")[2] for k in z.files if k.startswith("paired/corr/")})
    paired = {"B": z["paired/B"], "W_ref": z["paired/W_ref"], "corr": {c: z[f"paired/corr/{c}"] for c in conds}}
    cond = {c: {k: z[f"condition/{c}/{k}"] for k in ("Rx", "Ry", "Rz")} for c in conds}
    for c in conds:
        cond[c]["marginal"] = pm.Marginal(cond[c]["Rx"], cond[c]["Ry"])
    return chans, paired, cond


def groups(d):
    keys = pm.keys_of(d)
    out = []
    for k in sorted(set(keys)):
        idx = np.flatnonzero(keys == k)
        if len(idx) >= MIN_ADAPT:
            out.append((k, idx))
    return out


def arm_predictions(a, cond, chans, paired, cells):
    zero = np.zeros((a["Rx"].shape[0], a["Ry"].shape[0]))
    mx, mz, mc = pm.Marginal(a["Rx"], a["Ry"]), pm.Marginal(a["Rz"], a["Ry"]), cond["marginal"]
    preds = {"independence": zero,
             "pf_within_measured": pm.transfer(a["Rx"], a["Ry"], chans["pf_within"]["B"], mx),
             "pf_within_latent": pm.transfer(a["Rz"], a["Ry"], chans["pf_within"]["B"], mz),
             "pf_within_regression": a["Rx"] @ chans["pf_within"]["W"].T,
             "pf_within_condition": pm.transfer(cond["Rx"], cond["Ry"], chans["pf_within"]["B"], mc),
             "pf_measured": pm.transfer(a["Rx"], a["Ry"], chans["pf"]["B"], mx),
             "pf_latent": pm.transfer(a["Rz"], a["Ry"], chans["pf"]["B"], mz),
             "paired_closed_form": pm.transfer(a["Rx"], a["Ry"], paired["B"], mx),
             "paired_closed_form_latent": pm.transfer(a["Rz"], a["Ry"], paired["B"], mz),
             "reference_regression": a["Rx"] @ paired["W_ref"],
             "transferred_correlation": paired["corr_c"]}
    for name, ch in chans.items():
        if name.startswith(("within_pseudo", "within_derange", "pseudo", "derange")):
            preds[f"{name}_measured"] = pm.transfer(a["Rx"], a["Ry"], ch["B"], mx)
    W_bench, bench = pm.est.benchmark(a["x"], a["y"], cells)
    preds["benchmark"] = a["Rx"] @ W_bench
    return preds, bench


def run(parts=("heldout", "nt"), fits=None):
    """Predictions and diagnostics for every eligible group; used by evaluate.py to recompute them."""
    chans, paired, cond = fits or load_fits()
    out, diag = {}, []
    for part in parts:
        d = load(f"{part}_adaptation")
        for key, idx in groups(d):
            c = key.split("|")[1]
            a = pm.group_inputs(d["counts"][idx], d["library"][idx], d["y"][idx])
            preds, bench = arm_predictions(a, cond[c], chans, dict(paired, corr_c=paired["corr"][c]), d["cell"][idx])
            for arm, C in preds.items():
                out[f"{part}/{key}/{arm}"] = C
            row = {"part": part, "key": key, "target": key.split("|")[0], "condition": c, "n_adapt": int(len(idx)),
                   "nu_mean": float(a["nu"].mean()), "benchmark_lambda": bench["lambda"]}
            for name in ("pf_within", "pf"):
                row[name] = pm.diagnostics(a, chans[name])
            diag.append(row)
        print(part, "groups", sum(1 for r in diag if r["part"] == part), flush=True)
    return out, diag


def main():
    freeze = check_freeze()
    preds, diag = run()
    np.savez_compressed(pm.HERE / "results/predictions.npz", **preds)
    (pm.HERE / "results/digests.json").write_text(json.dumps({k: digest(v) for k, v in preds.items()}, indent=0) + "\n")
    (pm.HERE / "results/diagnostics.json").write_text(json.dumps(diag, indent=1) + "\n")
    manifest = {n: sha(pm.HERE / "results" / n) for n in RESULT_FILES}
    manifest.update(frozen_at=freeze["frozen_at"], note="written before the scoring files were opened")
    (pm.HERE / "results/manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
