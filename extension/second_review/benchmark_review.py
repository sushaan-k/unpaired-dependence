#!/usr/bin/env python3
"""J of PLAN.md: comparisons at fixed accuracy in the benchmark and the external test.

    python benchmark_review.py run NAME     # checks results/freeze.json; NAME: a benchmark data set or 'external'
    python benchmark_review.py score        # results/benchmark_review.json

Curves are the published ones (recomputed from the hashed predictions with the tests' own scoring code, and checked
against the published values); bootstrap curves use the tests' own resamples of held-out units.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
EXT = HERE.parent
LEVELS = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40)
OUT = HERE / "results" / "benchmark"


def first_crossing(values, level, budgets, running_max):
    """Paired cells at which a curve first reaches level, by log-linear interpolation from the budget before;
    'below' if it starts at or above the level, 'above' if it never reaches it."""
    v = np.nan_to_num(np.asarray(values, float), nan=-np.inf)
    if running_max:
        v = np.maximum.accumulate(v)
    b = np.asarray(budgets, float)
    if v[0] >= level:
        return b[0], "below"
    hit = np.flatnonzero(v >= level)
    if hit.size == 0:
        return b[-1], "above"
    i = int(hit[0])
    lo, hi = v[i - 1], v[i]
    fr = (level - lo) / (hi - lo) if hi > lo else 1.0
    return float(np.exp(np.log(b[i - 1]) + fr * (np.log(b[i]) - np.log(b[i - 1])))), "="


def compare(prop, env, budgets, boot_prop=None, boot_env=None):
    out = {}
    for L in LEVELS:
        rec = {}
        for mode, rm in (("raw", False), ("running_max", True)):
            n_p, r_p = first_crossing(prop, L, budgets, rm)
            n_e, r_e = first_crossing(env, L, budgets, rm)
            if r_p == "=" and r_e == "=":
                kind = "estimable"
            elif r_p == "below" and r_e == "below":
                kind = "unresolved below"
            elif "above" in (r_p, r_e):
                kind = "unresolved above"
            else:
                kind = "one curve below"
            m = {"kind": kind, "estimator_cells": n_p, "estimator_relation": r_p, "paired_only_cells": n_e,
                 "paired_only_relation": r_e, "ratio": n_e / n_p}
            if kind == "estimable" and boot_prop is not None:
                bs = [first_crossing(be, L, budgets, rm)[0] / first_crossing(bp, L, budgets, rm)[0]
                      for bp, be in zip(boot_prop, boot_env)]
                m["ci"] = np.percentile(bs, [2.5, 97.5]).tolist()
            rec[mode] = m
        out[str(L)] = rec
    return out


def benchmark(name):
    sys.path.insert(0, str(EXT / "generality"))
    import grun
    res = grun.RES
    store = dict(np.load(res / f"{name}_predictions.npz"))
    msq = {int(f): m for f, m in json.loads((res / f"{name}_msq.json").read_text()).items()}
    infos = {int(f): i for f, i in json.loads((res / f"{name}_info.json").read_text()).items()}
    pub = json.loads((res / f"{name}.json").read_text())
    d = grun.load(grun.source(name))
    fold, part, half = grun.roles(grun.source(name), d)
    stats = {f: grun.heldout_stats(d, f, fold, half, np.array(infos[f]["x_panel_index"]),
                                   np.array(infos[f]["y_panel_index"]), infos[f]["conditions"])
             for f in range(grun.FOLDS)}
    budgets = {f: infos[f]["budgets"] for f in range(grun.FOLDS)}
    curves, _, common = grun.score(name, stats, store, msq, budgets)
    diff = max(float(np.max(np.abs(np.asarray(pub["rf"][a]) - curves[a][0]))) for a in pub["rf"])
    units = sorted(set(d["unit"]))
    rng = np.random.default_rng([grun.BOOT_SEED, (grun.PRIMARY + grun.SECONDARY).index(name)])
    draws = rng.integers(0, len(units), size=(grun.BOOT, len(units)))
    counts = np.stack([np.bincount(draws[r], minlength=len(units)) for r in range(grun.BOOT)])
    uidx = {u: i for i, u in enumerate(units)}
    arms = [grun.PROPOSED] + [a for a in grun.PAIRED_ONLY if a in curves]
    boot = {a: [] for a in arms}
    for s_ in range(0, grun.BOOT, 500):
        W = {f: counts[s_:s_ + 500][:, [uidx[u] for u in stats[f][next(iter(stats[f]))]["units"]]] for f in stats}
        cb, _, _ = grun.score(name, stats, store, msq, budgets, W)
        for a in arms:
            boot[a].append(cb[a])
    boot = {a: np.vstack(v) for a, v in boot.items()}
    prop = curves[grun.PROPOSED][0]
    env = np.nanmax(np.stack([curves[a][0] for a in arms[1:]]), 0)
    benv = np.nanmax(np.stack([boot[a] for a in arms[1:]]), 0)
    return {"dataset": name, "budgets": common, "max_abs_difference_from_published": diff,
            "estimator": prop.tolist(), "paired_only": env.tolist(),
            "levels": compare(prop, env, common, boot[grun.PROPOSED], benv)}


def external():
    sys.path.insert(0, str(EXT / "semipaired" / "external"))
    sys.path.insert(1, str(EXT / "semipaired"))
    import run_external as rx
    resd = rx.HERE / "results"
    info = json.loads((resd / "predict_info.json").read_text())
    parts = [rx.od.load(p, info["genes"]) for p in ("test_adaptation", "test_scoring")]
    cat = {k: np.concatenate([p[k] for p in parts]) for k in ("counts", "library", "y", "part", "target", "condition")}
    stats = rx.tg.group_stats(cat["counts"], cat["library"], cat["y"], rx.pm.keys_of(cat), cat["part"])
    z = np.load(resd / "predictions.npz")
    mean_pred, msq = {}, {}
    for key in z.files:
        arm, B, c = "/".join(key.split("/")[:2]), key.split("/")[2], key.split("/")[4]
        if B == "0":
            continue
        mean_pred.setdefault(arm, {}).setdefault(B, {}).setdefault(c, []).append(z[key])
    for arm in mean_pred:
        msq[arm] = {B: {c: float(np.mean([np.sum(C * C) for C in v])) for c, v in d.items()}
                    for B, d in mean_pred[arm].items()}
        mean_pred[arm] = {B: {c: np.mean(v, 0) for c, v in d.items()} for B, d in mean_pred[arm].items()}
    proposed = "bjs2/mapped"                  # the arm named in the frozen plan (Supplementary Note 5)
    arms = [proposed] + rx.PAIRED_ONLY
    use = {a: mean_pred[a] for a in arms}
    rf = rx.curves(stats, np.ones(len(stats)), use, msq)[0]
    pub = json.loads((resd / "overcite.json").read_text())
    diff = max(float(np.max(np.abs(np.array(pub["rf"][a]) - np.array(rf[a])))) for a in arms)
    orfs = sorted({r["target"] for r in stats})
    member = {t: [i for i, r in enumerate(stats) if r["target"] == t] for t in orfs}
    rng = np.random.default_rng(rx.BOOT_SEED)
    bp, be = [], []
    for _ in range(rx.BOOT):
        w = np.zeros(len(stats))
        for t in rng.choice(orfs, len(orfs), replace=True):
            w[member[t]] += 1
        b = rx.curves(stats, w, use, msq)[0]
        bp.append(np.array(b[proposed]))
        be.append(np.nanmax(np.array([b[a] for a in rx.PAIRED_ONLY]), 0))
    budgets = list(rx.BUDGETS)
    prop = np.array(rf[proposed])
    env = np.nanmax(np.array([rf[a] for a in rx.PAIRED_ONLY]), 0)
    return {"dataset": "external", "budgets": budgets, "max_abs_difference_from_published": diff,
            "estimator": prop.tolist(), "paired_only": env.tolist(), "levels": compare(prop, env, budgets, bp, be)}


def score():
    per = {p.stem: json.loads(p.read_text()) for p in sorted(OUT.glob("*.json"))}
    out = {"datasets": per, "levels": {}}
    for L in map(str, LEVELS):
        for mode in ("raw", "running_max"):
            recs = {n: r["levels"][L][mode] for n, r in per.items()}
            est = [v["ratio"] for v in recs.values() if v["kind"] == "estimable"]
            kinds = {}
            for v in recs.values():
                kinds[v["kind"]] = kinds.get(v["kind"], 0) + 1
            out["levels"].setdefault(L, {})[mode] = {
                "estimable": len(est), "geometric_mean": float(np.exp(np.mean(np.log(est)))) if est else None,
                "counts": kinds, "per_dataset": {n: [v["kind"], v["ratio"], v.get("ci")] for n, v in recs.items()}}
    (HERE / "results" / "benchmark_review.json").write_text(json.dumps(out, indent=1) + "\n")
    for L, v in out["levels"].items():
        print(L, {m: (w["estimable"], w["geometric_mean"], w["counts"]) for m, w in v.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("run", "score"))
    ap.add_argument("name", nargs="?")
    a = ap.parse_args()
    from validation_review import check_freeze
    check_freeze()
    OUT.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        if a.what == "run":
            rec = external() if a.name == "external" else benchmark(a.name)
            (OUT / f"{a.name}.json").write_text(json.dumps(rec) + "\n")
            print(a.name, "difference from published", rec["max_abs_difference_from_published"])
        else:
            score()


if __name__ == "__main__":
    main()
