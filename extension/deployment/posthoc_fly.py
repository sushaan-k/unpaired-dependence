#!/usr/bin/env python3
"""Post hoc (after the generality benchmark's secondary analysis was scored): the cross-animal fly analysis with
paired neurons centred and scaled on their own statistics.

The prespecified secondary analysis (extension/generality, banc_crossanimal) took the unpaired pool from other
animals (FAFB brain, MANC nerve cord) and, as every earlier analysis, centred and scaled the paired BANC neurons by
the pool's means and standard deviations. Here the same paired neurons (same folds, reservoir, budgets, draws and
seeds), the same unpaired pool and the same held-out cell types are used, but the paired neurons are centred on
their own condition means and scaled by their own standard deviations (drun.own_std), so that the other animals
contribute correlation structure only. Nothing else changes.

    python posthoc_fly.py     # writes results_posthoc/fly_own_standardization.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import drun  # noqa: E402

grun, cm, rs, es, pm = drun.grun, drun.cm, drun.rs, drun.es, drun.pm
NAME = "banc_crossanimal"
OUT = HERE / "results_posthoc"


def main():
    d = grun.load("banc")
    fold, part, half = grun.roles("banc", d)
    idx = len(grun.PRIMARY) + grun.SECONDARY.index(NAME)
    num, den = {}, 0.0
    info = {}
    t0 = time.time()
    for f in range(grun.FOLDS):
        train, x, y, pop, (xp, yp), counts = grun.prepare_crossanimal(d, f, fold, part)
        keys = pm.eligible_training(pop, train["part"], grun.MIN_TRAIN_HALF)
        pool, un, X_res, Y_res, cond_res = cm.setup_fold(train, x, y, pop, keys)
        ctx = rs.build_context(train, x, y, pop, keys, pool, un)
        use = np.isin(pop, keys)
        rmask = cm.hm.reservoir(train["cell"]) & use & np.isin(pop, pool.keys)
        xr, yr, popr, condr = x[rmask], y[rmask], pop[rmask], np.array([cm.cond_of(k) for k in pop[rmask]])
        assert len(xr) == len(X_res)
        conds = un.conds
        budgets = [B for B in grun.BUDGETS if B <= len(xr)]
        preds, msq = {}, {}
        for B in budgets:
            for dr in range(grun.DRAWS):
                rng = np.random.default_rng([grun.SEED, idx, f, B, dr])
                pick = rng.choice(len(xr), size=B, replace=False)
                o = drun.own_std(xr[pick], yr[pick], popr[pick])
                if o is None:
                    continue
                Xo, Yo, keep = o
                co = condr[pick][keep]
                arms = {}
                P = es.two_sided_js(Xo, Yo, ctx.Uz, ctx.Ry, ctx.blocks)[0]
                arms["proposed_own"] = {c: ctx.mapped(P, c) for c in conds}
                for a in drun.PAIRED_ONLY:
                    crng = np.random.default_rng([grun.SEED + 1, f, B, dr])
                    Pp = rs.denoise(a, Xo, Yo, ctx.Rx, ctx.Ry, ctx.n_x, ctx.n_y, ctx.Uz, ctx.blocks, crng)
                    arms[f"{a}/pooled_own"] = {c: Pp for c in conds}
                    arms[f"{a}/percond_own"] = {}
                    for c in conds:
                        m = co == c
                        arms[f"{a}/percond_own"][c] = (rs.denoise(a, Xo[m], Yo[m], ctx.Rx, ctx.Ry, ctx.n_x, ctx.n_y,
                                                                  ctx.Uz, ctx.blocks, crng)
                                                       if m.sum() >= grun.MIN_PERCOND else np.zeros_like(Pp))
                for a, byc in arms.items():
                    for c, Q in byc.items():
                        k = (a, B, c)
                        preds[k] = preds.get(k, 0.0) + Q / grun.DRAWS
                        msq[k] = msq.get(k, 0.0) + float(np.sum(Q * Q)) / grun.DRAWS
            print(f"fold {f} B={B} [{time.time() - t0:.0f}s]", flush=True)
        stats = grun.heldout_stats(d, f, fold, half, xp, yp, conds)
        for c in conds:
            rec = stats[c]
            T = grun.targets(rec, np.ones((1, len(rec["units"]))))
            den += float(np.einsum("rpq,rpq->r", T["TA"], T["TB"])[0])
            for (a, B, cc), Pm in preds.items():
                if cc != c:
                    continue
                num[(a, B)] = num.get((a, B), 0.0) + 2 * float(np.sum(Pm * T["T"][0])) - msq[(a, B, cc)]
        info[str(f)] = {"reservoir": int(len(xr)), "pool_keys": len(pool.keys), "conditions": conds}
    budgets = sorted({B for (_, B) in num})
    rf = {}
    for (a, B), v in num.items():
        rf.setdefault(a, {})[B] = v / den
    rf = {a: [rf[a].get(B, float("nan")) for B in budgets] for a in rf}
    po = [a for a in rf if a != "proposed_own"]
    rf["paired_only_own"] = list(np.nanmax(np.array([rf[a] for a in po]), axis=0))
    orig = json.loads((grun.RES / f"{NAME}.json").read_text())
    same = json.loads((grun.RES / "banc.json").read_text())
    env = lambda r, arms: list(np.nanmax(np.array([r[a] for a in arms]), axis=0))
    out = {"note": "post hoc; the prespecified secondary analysis centred and scaled paired neurons by the other "
                   "animals' statistics (generality/results/banc_crossanimal.json)",
           "budgets": budgets,
           "rf": {"proposed_own_standardization": rf["proposed_own"],
                  "paired_only_own_standardization": rf["paired_only_own"],
                  "proposed_pool_standardization_prespecified": orig["rf"][grun.PROPOSED],
                  "paired_only_pool_standardization_prespecified": env(orig["rf"], grun.PAIRED_ONLY),
                  "proposed_same_animal_pool": same["rf"][grun.PROPOSED],
                  "paired_only_same_animal_pool": env(same["rf"], grun.PAIRED_ONLY)},
           "arms_own": {a: rf[a] for a in rf},
           "folds": info, "written": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    OUT.mkdir(exist_ok=True)
    (OUT / "fly_own_standardization.json").write_text(json.dumps(out, indent=1))
    for k, v in out["rf"].items():
        print(f"{k:48s}", [round(100 * float(z), 1) for z in v])


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
