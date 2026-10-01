#!/usr/bin/env python3
"""Prospective test of the saving law (PLAN.md).

    python prun.py predict <name>     # every problem of a data set: predictions and the budget draws' estimates,
                                      # hashed in results/<name>_manifest.json; no statistic of truth cells
    python prun.py evaluate <name>    # checks the hashes, then truth-cell statistics, observed curves and savings
    python prun.py summary            # hypotheses H1-H6 across data sets -> results/summary.json

A problem is a data set with one X panel (100, 200 or 400 features by the dispersion rule, from unpaired pool cells).
Everything is fixed in PLAN.md and ptools.py, which were hashed before any new data set was downloaded
(results/freeze.json).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import ortho_group, spearmanr
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ptools as pt  # noqa: E402

DATA = Path("/home/claude/cbio/rawdata/predictability")
RES = HERE / "results"
NEW = ("sln111", "sln206", "pbmc10k", "malt10k", "bmcite", "fetal_cortex", "snare_cortex", "fafb_vpn", "banc_vpn",
       "human_gaba")
PANELS = (100, 200, 400)
DRAWS = 20
BUDGET_MAX = 12800
PILOT_MAX, PILOT_FRACTION = 200, 0.2
TARGETS = (0.5, 0.3, 0.7)                 # recovered fraction of the population's dependence; 0.5 primary
SEED = 20261073
PERMUTATIONS, BOOT, STAT_SEED = 10000, 2000, 20261074
H1_MAX, H4_MAX, H5_MAX = np.log2(1.25), np.log2(1.5), np.log2(1.5)
H3_RANGE, H6_SLACK = (0.8, 1.25), 1.25
MIN_UNCENSORED = 8


def check_freeze():
    rec = json.loads((RES / "freeze.json").read_text())
    bad = [f for f, dg in rec["files"].items() if pt.sha(HERE / f) != dg]
    assert not bad, f"changed since the freeze: {bad}"
    return rec


def dataset(name):
    return pt.load(DATA / f"{name}.npz", name)


def problems(d):
    """(tag, Problem) for every X panel; a panel identical to a smaller one is dropped."""
    seen = []
    for nx in PANELS:
        P = pt.Problem(d, n_x=nx)
        key = tuple(P.panels[0].tolist())
        if key in seen:
            continue
        seen.append(key)
        yield f"p{nx}", P


def budgets_for(P):
    N = len(P.Xr)
    m = min(PILOT_MAX, int(PILOT_FRACTION * N))
    avail = N - m
    return m, avail, [25 * 2 ** k for k in range(20) if 25 * 2 ** k <= min(BUDGET_MAX, avail)]


def collapse(wbar, nr, nc):
    """The frozen development profile (4 RNA x 2 protein blocks) mapped to a problem with nr x nc blocks: trailing
    RNA (protein) blocks are merged into the last one when the panel has fewer."""
    w = np.array(wbar, float)
    if nr < 4:
        w = np.vstack([w[:nr - 1], w[nr - 1:].sum(0, keepdims=True)])
    if nc < 2:
        w = w.sum(1, keepdims=True)
    return w.ravel() / w.sum()


def law_predictions(blocks, targets):
    """Proposition S14 with block moments: savings and budgets at each target (continuous closed form)."""
    w, pi = pt.shares(blocks)
    r2 = sum(max(b["r2"], 0.0) for b in blocks)
    tau = sum(b["tau"] for b in blocks)
    out = {}
    for tg in targets:
        eps = 1 - tg
        if r2 <= 0:
            out[str(tg)] = None
            continue
        S = pt.saving_closed(w, pi, eps)
        b1 = (tau / r2) * (1 - eps) / eps
        out[str(tg)] = {"saving": S, "cells_js": b1, "cells_blocks": b1 / S}
    return out


def predict(name, log):
    check_freeze()
    d = dataset(name)
    wbar = json.loads((RES / "transfer_profile.json").read_text())["w_bar"]
    store, msq, preds = {}, {}, {}
    for tag, P in problems(d):
        m, avail, budgets = budgets_for(P)
        rng = np.random.default_rng([SEED, sum(map(ord, name)), int(tag[1:])])
        Ur = ortho_group.rvs(P.U.shape[0], random_state=int(rng.integers(2 ** 31)))
        Vr = ortho_group.rvs(P.V.shape[0], random_state=int(rng.integers(2 ** 31))) if P.V.shape[0] > 1 else np.eye(1)
        rot_x, rot_y = P.U.T @ Ur, Vr.T @ P.V
        for B in budgets:
            acc = {"bjs2": 0.0, "js": 0.0, "random": 0.0}
            sq = {"bjs2": 0.0, "js": 0.0, "random": 0.0}
            for dr in range(DRAWS):
                r = np.random.default_rng([SEED, sum(map(ord, name)), int(tag[1:]), B, dr])
                idx = m + r.choice(avail, B, replace=False)
                X, Y = P.Xr[idx], P.Yr[idx]
                A, A1 = P.fit(X, Y)
                Ar = rot_x @ P.fit(X, Y, Ur, Vr)[0] @ rot_y
                for k, v in (("bjs2", A), ("js", A1), ("random", Ar)):
                    acc[k] = acc[k] + v / DRAWS
                    sq[k] += float(np.sum(v * v)) / DRAWS
            for k in acc:
                store[f"{tag}/{k}/{B}"] = np.asarray(acc[k], np.float64)
                msq[f"{tag}/{k}/{B}"] = sq[k]
        store[f"{tag}/U"], store[f"{tag}/V"] = P.U, P.V
        # predictions: never truth cells
        mom_pop = P.moments(P.Xr, P.Yr, ell=False)
        mom_pilot = P.moments(P.Xr[:m], P.Yr[:m], ell=False)
        mom_rand = P.moments(P.Xr, P.Yr, Ur, Vr, ell=False)
        un = P.unpaired_noise()
        tu = np.array([b["tau"] for b in un])
        pi_u = tu / tu.sum()
        nr, nc = len(P.rblocks), len(P.pblocks)
        wt = collapse(wbar, nr, nc)
        rec = {"n": P.n, "pilot": m, "budgets": budgets, "blocks": [nr, nc],
               "law": law_predictions(mom_pop, TARGETS), "pilot_law": law_predictions(mom_pilot, TARGETS),
               "random_law": law_predictions(mom_rand, TARGETS),
               "unpaired_bound": float(1 / pi_u.min()), "noise_shares_unpaired": pi_u.tolist(),
               "transfer": {str(tg): pt.saving_closed(wt, pi_u, 1 - tg) for tg in TARGETS},
               "moments_population": mom_pop, "moments_pilot": mom_pilot}
        c = pt.curves(mom_pop, budgets, "lin")
        rec["law_curves"] = None if c is None else {"bjs2": c[0].tolist(), "js": c[1].tolist()}
        c = pt.curves(mom_pilot, budgets, "lin")
        rec["pilot_curves"] = None if c is None else {"bjs2": c[0].tolist(), "js": c[1].tolist()}
        c = pt.boot_curves(P, P.Xr, P.Yr, budgets, draws=DRAWS, seed=SEED)
        rec["boot_curves"] = None if c is None else {"bjs2": c[0].tolist(), "js": c[1].tolist()}
        preds[tag] = rec
        log(f"{tag}: n {P.n}, pilot {m}, budgets {budgets[0]}-{budgets[-1]}, law saving at 0.5 "
            f"{(rec['law']['0.5'] or {}).get('saving', float('nan')):.2f}")
    RES.mkdir(exist_ok=True)
    np.savez_compressed(RES / f"{name}_estimates.npz", **store)
    (RES / f"{name}_msq.json").write_text(json.dumps(msq) + "\n")
    (RES / f"{name}_predictions.json").write_text(pt.jdump(preds) + "\n")
    man = {"estimates": pt.sha(RES / f"{name}_estimates.npz"), "msq": pt.sha(RES / f"{name}_msq.json"),
           "predictions": pt.sha(RES / f"{name}_predictions.json"), "freeze": pt.sha(RES / "freeze.json"),
           "data": pt.sha(DATA / f"{name}.npz"), "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "note": "written after all predictions and draws and before any statistic of truth cells"}
    (RES / f"{name}_manifest.json").write_text(json.dumps(man, indent=1) + "\n")
    log("predictions hashed")


def saving_obs(js, blocks, target, budgets):
    nj, rj = pt.needed(js, target, budgets)
    nk, rk = pt.needed(blocks, target, budgets)
    return {"saving": nj / nk, "cells_js": nj, "rel_js": rj, "cells_blocks": nk, "rel_blocks": rk,
            "estimable": rj == "=" and rk == "="}


def evaluate(name, log):
    check_freeze()
    man = json.loads((RES / f"{name}_manifest.json").read_text())
    for k, fn in (("estimates", f"{name}_estimates.npz"), ("msq", f"{name}_msq.json"),
                  ("predictions", f"{name}_predictions.json")):
        assert pt.sha(RES / fn) == man[k], f"{fn} does not match the manifest"
    assert pt.sha(DATA / f"{name}.npz") == man["data"]
    z = np.load(RES / f"{name}_estimates.npz")
    msq = json.loads((RES / f"{name}_msq.json").read_text())
    preds = json.loads((RES / f"{name}_predictions.json").read_text())
    d = dataset(name)
    out = {"dataset": name, "problems": {}}
    for tag, P in problems(d):
        assert np.array_equal(P.U, z[f"{tag}/U"]) and np.array_equal(P.V, z[f"{tag}/V"])
        P.score_setup()
        pr = preds[tag]
        b = pr["budgets"]
        curves = {k: [(2 * float(np.sum(z[f"{tag}/{k}/{B}"] * P.T)) - msq[f"{tag}/{k}/{B}"]) / P.den for B in b]
                  for k in ("bjs2", "js", "random")}
        rec = {"curves": curves, "budgets": b, "truth_cells": P.n["truth"], "theta_norm2": P.den, "targets": {}}
        for tg in TARGETS:
            o = saving_obs(curves["js"], curves["bjs2"], tg, b)
            ro = saving_obs(curves["js"], curves["random"], tg, b)
            rec["targets"][str(tg)] = {"observed": o, "random_observed": ro}
        out["problems"][tag] = rec
        log(f"{tag}: proposed {', '.join(f'{v:.2f}' for v in curves['bjs2'])}; saving at 0.5 "
            f"{rec['targets']['0.5']['observed']['saving']:.2f} ({'estimable' if rec['targets']['0.5']['observed']['estimable'] else 'censored'})")
    (RES / f"{name}.json").write_text(pt.jdump(out) + "\n")


def rows():
    """One row per problem and target with observed and predicted savings."""
    out = []
    for name in NEW:
        if not (RES / f"{name}.json").exists():
            continue
        ev = json.loads((RES / f"{name}.json").read_text())
        pr = json.loads((RES / f"{name}_predictions.json").read_text())
        for tag, rec in ev["problems"].items():
            p = pr[tag]
            for tg in TARGETS:
                t = str(tg)
                o = rec["targets"][t]["observed"]
                row = {"dataset": name, "problem": tag, "target": tg, "observed": o["saving"],
                       "estimable": o["estimable"], "rel_js": o["rel_js"], "rel_blocks": o["rel_blocks"],
                       "cells_blocks": o["cells_blocks"], "cells_js": o["cells_js"],
                       "random_observed": rec["targets"][t]["random_observed"]["saving"],
                       "random_estimable": rec["targets"][t]["random_observed"]["estimable"],
                       "law": (p["law"][t] or {}).get("saving"), "law_cells_blocks": (p["law"][t] or {}).get("cells_blocks"),
                       "law_cells_js": (p["law"][t] or {}).get("cells_js"),
                       "pilot": (p["pilot_law"][t] or {}).get("saving"),
                       "pilot_cells_blocks": (p["pilot_law"][t] or {}).get("cells_blocks"),
                       "random_law": (p["random_law"][t] or {}).get("saving"),
                       "transfer": p["transfer"][t], "bound": p["unpaired_bound"]}
                out.append(row)
    return out


def median_abs_log2(rs, key):
    v = [abs(np.log2(r["observed"] / r[key])) for r in rs if r[key]]
    return float(np.median(v)) if v else None, len(v)


def boot_ci(rs, key, rng):
    """95% interval of the median |log2(observed/predicted)| resampling data sets (with their problems)."""
    names = sorted({r["dataset"] for r in rs})
    vals = []
    for _ in range(BOOT):
        pick = rng.choice(names, len(names), replace=True)
        sample = [r for n in pick for r in rs if r["dataset"] == n]
        v = median_abs_log2(sample, key)[0]
        if v is not None:
            vals.append(v)
    return np.quantile(vals, [0.025, 0.975]).tolist() if vals else None


def summary():
    check_freeze()
    allrows = rows()
    rng = np.random.default_rng(STAT_SEED)
    res = {"datasets": sorted({r["dataset"] for r in allrows}), "problems": len({(r["dataset"], r["problem"]) for r in allrows}),
           "targets": {}}
    for tg in TARGETS:
        rs = [r for r in allrows if r["target"] == tg]
        est = [r for r in rs if r["estimable"]]
        t = {"problems": len(rs), "estimable": len(est)}
        for key in ("law", "pilot", "transfer"):
            med, n = median_abs_log2(est, key)
            t[key] = {"median_abs_log2": med, "n": n, "ci": boot_ci(est, key, rng) if n else None,
                      "within_25pct": float(np.mean([abs(np.log2(r["observed"] / r[key])) <= np.log2(1.25)
                                                     for r in est if r[key]])) if n else None,
                      "median_log2_bias": float(np.median([np.log2(r["observed"] / r[key]) for r in est if r[key]])) if n else None}
        x = np.array([r["law"] for r in est if r["law"]])
        y = np.array([r["observed"] for r in est if r["law"]])
        if len(x) >= 3:
            rho = float(spearmanr(x, y).correlation)
            null = np.array([spearmanr(rng.permutation(x), y).correlation for _ in range(PERMUTATIONS)])
            t["spearman_law"] = {"rho": rho, "p_one_sided": float((1 + np.sum(null >= rho)) / (1 + PERMUTATIONS))}
        rnd = [r["random_observed"] for r in rs if r["random_estimable"]]
        t["random_control"] = {"median_observed": float(np.median(rnd)) if rnd else None, "n": len(rnd),
                               "median_law": float(np.median([r["random_law"] for r in rs if r["random_law"]]))}
        t["bound_violations"] = [(r["dataset"], r["problem"]) for r in est
                                 if not (1 / H6_SLACK <= r["observed"] <= H6_SLACK * r["bound"])]
        cb = [abs(np.log2(r["cells_blocks"] / r["law_cells_blocks"])) for r in est if r["law_cells_blocks"]]
        cp = [abs(np.log2(r["cells_blocks"] / r["pilot_cells_blocks"])) for r in est if r["pilot_cells_blocks"]]
        t["budget_blocks"] = {"law_median_abs_log2": float(np.median(cb)) if cb else None,
                              "pilot_median_abs_log2": float(np.median(cp)) if cp else None}
        res["targets"][str(tg)] = t
    p = res["targets"]["0.5"]
    ok = p["estimable"] >= MIN_UNCENSORED
    res["verdicts"] = {
        "evaluable": ok,
        "H1_law_calibrated": bool(ok and p["law"]["median_abs_log2"] < H1_MAX),
        "H2_law_ranks": bool(ok and p.get("spearman_law", {}).get("p_one_sided", 1) < 0.05
                             and p["spearman_law"]["rho"] > 0),
        "H3_random_control_no_saving": bool(p["random_control"]["median_observed"] is not None
                                            and H3_RANGE[0] <= p["random_control"]["median_observed"] <= H3_RANGE[1]),
        "H4_pilot_calibrated": bool(ok and p["pilot"]["median_abs_log2"] is not None and p["pilot"]["median_abs_log2"] < H4_MAX),
        "H5_transfer_calibrated": bool(ok and p["transfer"]["median_abs_log2"] < H5_MAX),
        "H6_within_unpaired_bounds": bool(ok and not p["bound_violations"])}
    (RES / "summary.json").write_text(json.dumps(res, indent=1) + "\n")
    (RES / "rows.json").write_text(pt.jdump(allrows) + "\n")
    print(json.dumps(res["verdicts"], indent=1))
    for tg in TARGETS:
        t = res["targets"][str(tg)]
        print(tg, {k: (t[k]["median_abs_log2"], t[k]["n"]) for k in ("law", "pilot", "transfer")}, t.get("spearman_law"))


if __name__ == "__main__":
    stage = sys.argv[1]
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:6.0f}s] {sys.argv[2] if len(sys.argv) > 2 else ''}: {msg}", flush=True)
    with threadpool_limits(limits=1):
        if stage == "predict":
            predict(sys.argv[2], log)
        elif stage == "evaluate":
            evaluate(sys.argv[2], log)
        elif stage == "run":
            predict(sys.argv[2], log)
            evaluate(sys.argv[2], log)
        elif stage == "summary":
            summary()
