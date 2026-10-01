#!/usr/bin/env python3
"""Development on training targets only (exploratory): feasibility, noise levels and running time.

Every fourth training target in SHA-256 order (salt perturb-dev-v1) plays the
held-out role; its part-0 cells act as adaptation cells and its part-1 cells as
scoring cells (alternating sub-halves in cell-hash order). The other training
targets train the channels. No held-out or non-targeting cell is read.
Writes results/dev.json.
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

import pmethods as pm
from pdata import CELL_SALT, MIN_TRAIN_HALF, h, load, seal

sys.path.append(str(pm.HERE.parent / "recoverability"))
from stats import cluster_bootstrap  # noqa: E402

DEV_SALT = "perturb-dev-v1"


def main(closed_form):
    t0 = time.time()
    d = load("training")
    x, y = pm.features(d)
    pop = pm.keys_of(d)
    targets = seal()["training_targets"]
    order = sorted(targets, key=lambda t: h(DEV_SALT, t))
    dev_test = set(order[::4])
    train_cells = ~np.isin(d["target"], list(dev_test))
    keys = pm.eligible_training(pop[train_cells], d["part"][train_cells], MIN_TRAIN_HALF)
    xt, yt, pt, partt, condt = x[train_cells], y[train_cells], pop[train_cells], d["part"][train_cells], d["condition"][train_cells]
    raw = pm.moments(xt, yt, pt, partt, keys)
    chans = {"pf": pm.fit_channel(raw), "pf_within": pm.fit_channel(pm.centre_within_condition(raw))}
    print(f"channels: {len(keys)} populations; pf dim S {chans['pf']['VS'].shape[1]}, scale {chans['pf']['scale']}; "
          f"within dim S {chans['pf_within']['VS'].shape[1]} ({time.time() - t0:.0f}s)", flush=True)
    controls = pm.control_channels(xt, yt, pt, partt, keys, condt)
    controls.update({f"w{k}": v for k, v in pm.control_channels(xt, yt, pt, partt, keys, condt, centre=True).items()})
    print(f"controls done ({time.time() - t0:.0f}s): dims", {k: c["VS"].shape[1] for k, c in controls.items()}, flush=True)
    people = pm.paired_people(xt, yt, pt, keys)
    refs = pm.paired_references(people, closed_form=closed_form)
    print(f"paired references done ({time.time() - t0:.0f}s)", flush=True)
    cm = pm.condition_marginals(xt, yt, d["counts"][train_cells], d["library"][train_cells], pt, partt, keys)
    rows = []
    for key in sorted(set(pop[~train_cells])):
        m = pop == key
        ad = np.flatnonzero(m & (d["part"] == 0))
        sc = np.flatnonzero(m & (d["part"] == 1))
        sc = sc[np.argsort([h(CELL_SALT, d["cell"][i]) for i in sc])]
        sub = np.zeros(len(sc), int)
        sub[1::2] = 1
        if len(ad) < 40 or min(np.sum(sub == 0), np.sum(sub == 1)) < 20:
            continue
        a = pm.group_inputs(d["counts"][ad], d["library"][ad], d["y"][ad])
        t = pm.scoring_targets(d["counts"][sc], d["library"][sc], d["y"][sc], sub)
        cond = key.split("|")[1]
        preds = {"independence": np.zeros_like(t["T"]),
                 "pf_latent": pm.transfer(a["Rz"], a["Ry"], chans["pf"]["B"]),
                 "pf_measured": pm.transfer(a["Rx"], a["Ry"], chans["pf"]["B"]),
                 "pf_within_latent": pm.transfer(a["Rz"], a["Ry"], chans["pf_within"]["B"]),
                 "pf_within_measured": pm.transfer(a["Rx"], a["Ry"], chans["pf_within"]["B"]),
                 "pf_within_regression_latent": a["Rz"] @ chans["pf_within"]["W"].T,
                 "pf_within_condition_latent": pm.transfer(cm[cond]["Rz"], cm[cond]["Ry"], chans["pf_within"]["B"]),
                 "pf_regression_latent": a["Rz"] @ chans["pf"]["W"].T,
                 "pf_condition_latent": pm.transfer(cm[cond]["Rz"], cm[cond]["Ry"], chans["pf"]["B"]),
                 "reference_regression": a["Rx"] @ refs["reference_regression"]["W"],
                 "transferred_correlation": refs["transferred_correlation"][cond]}
        if closed_form:
            preds["paired_closed_form"] = pm.transfer(a["Rx"], a["Ry"], refs["paired_closed_form"]["B"])
        W_bench, _ = pm.est.benchmark(a["x"], a["y"], d["cell"][ad])
        preds["benchmark"] = a["Rx"] @ W_bench
        for name, ch in controls.items():
            preds[name] = pm.transfer(a["Rz"], a["Ry"], ch["B"])
        row = {"key": key, "target": key.split("|")[0], "condition": cond, "n_adapt": len(ad), "n_score": len(sc)}
        for name, C in preds.items():
            row[name] = pm.endpoint_terms(C, t)
        row["diag_pf"] = pm.diagnostics(a, chans["pf"])
        row["diag_pf_within"] = pm.diagnostics(a, chans["pf_within"])
        row["norm_T"] = float(np.linalg.norm(t["T"]))
        rows.append(row)
    print(f"groups scored: {len(rows)} ({time.time() - t0:.0f}s)", flush=True)
    methods = [k for k in rows[0] if isinstance(rows[0][k], tuple)]
    targets_of = np.array([r["target"] for r in rows])

    def rf(name, idx):
        num = sum(rows[i][name][0] for i in idx)
        den = sum(rows[i][name][1] for i in idx)
        return num / den

    summary = {}
    for name in methods:
        idx = np.arange(len(rows))
        boot = cluster_bootstrap(targets_of, lambda i: rf(name, i), 500, 1)
        summary[name] = {"rf": rf(name, idx), "ci": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]}
        for c in ("control", "ifng", "coculture"):
            ic = [i for i in idx if rows[i]["condition"] == c]
            summary[name][c] = rf(name, ic)
    for prefix in ("pseudo", "derange", "wpseudo", "wderange"):
        summary[f"{prefix}_mean"] = float(np.mean([summary[k]["rf"] for k in methods if k.startswith(prefix)
                                                   and k.rstrip("0123456789") == prefix]))
    den = sum(r["independence"][1] for r in rows)
    out = {"dev_test_targets": sorted(dev_test), "training_populations": len(keys), "groups": len(rows),
           "denominator_total": den, "mean_norm_T": float(np.mean([r["norm_T"] for r in rows])),
           "pf": {k: v for k, v in chans["pf"].items() if k in ("scale", "ridge", "at_edge", "n_train")},
           "pf_dim_S": int(chans["pf"]["VS"].shape[1]),
           "diag_pf_median": {k: float(np.median([r["diag_pf"][k] for r in rows])) for k in rows[0]["diag_pf"]},
           "diag_pf_within_median": {k: float(np.median([r["diag_pf_within"][k] for r in rows]))
                                     for k in rows[0]["diag_pf_within"]},
           "pf_within": {k: v for k, v in chans["pf_within"].items() if k in ("scale", "ridge", "at_edge", "n_train")},
           "summary": summary, "seconds": time.time() - t0}
    (pm.HERE / f"results/dev{'_cf' if closed_form else ''}.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "summary"}, indent=1))
    for name in methods:
        s = summary[name]
        if name.startswith(("pseudo", "derange", "wpseudo", "wderange")) and name[-2:] != "00":
            continue
        print(f"{name:28s} RF {s['rf']:+.3f} [{s['ci'][0]:+.3f}, {s['ci'][1]:+.3f}]  control {s['control']:+.3f} "
              f"ifng {s['ifng']:+.3f} coculture {s['coculture']:+.3f}")
    print({k: round(summary[f"{k}_mean"], 3) for k in ("pseudo", "derange", "wpseudo", "wderange")})


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main("--closed-form" in sys.argv)
