#!/usr/bin/env python3
"""After scoring: how robust is the validation's atlas result? (results_posthoc/robustness.json)

1. From the hashed predictions: the result per donor and per processing pool, and with each pool left out in turn
   (the target recomputed as half of the best recovery without that pool).
2. With the atlas reference changed and everything else as in the test (panel, folds, draws, seeds, paired-only
   arm): atlas reference size (cells of 4, 7, 15, 30 and 59 of the 118 patients, chosen in hash order), and
   mismatched cell-type composition (B cells and CD14 monocytes, or CD4 and CD8 T cells, from 15 patients only while
   the other types come from all 118; or one cell type absent, whose held-out cells then get the pooled estimate
   without a condition map). Each variant's atlas arm is scored against the frozen paired-only arm at the test's
   primary target (fixed at its frozen value). The unchanged atlas is recomputed first and must reproduce the hashed
   atlas predictions.

Intervals: 2,000 resamples of donors, seed 20261205.

    python posthoc_robustness.py      # about 15 minutes on two cores
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
import vrun  # noqa: E402  (frozen; imported unchanged)

drun, pc, es, grun = vrun.drun, vrun.pc, vrun.es, vrun.grun
CONDS = vrun.CONDS
BOOT, BOOT_SEED = 2000, 20261205
SIZES = (4, 7, 15, 30, 59)
FEW = 15
OUT = HERE / "results_posthoc"


def fold_stats(new, info, half, gidx, yidx):
    fds = {str(fd["fold"]): fd for fd in vrun.folds(new)}
    return {f: vrun.heldout(new, fds[f], half, gidx, yidx) for f in info}


def boot_weights(info):
    donors = sorted({fi["donor"] for fi in info.values()})
    folds_of = {d: [f for f, fi in info.items() if fi["donor"] == d] for d in donors}
    rng = np.random.default_rng(BOOT_SEED)
    out = []
    for _ in range(BOOT):
        w = {}
        for d in rng.choice(donors, size=len(donors), replace=True):
            for f in folds_of[d]:
                w[f] = w.get(f, 0.0) + 1.0
        out.append(w)
    return out


def arm_curve(terms, weights, arm, budgets):
    den = sum(w * terms[f][1] for f, w in weights.items())
    return np.array([sum(w * terms[f][0].get((arm, B), np.nan) for f, w in weights.items()) / den for B in budgets])


def stored_part(new, info, stats, budgets, target):
    store = dict(np.load(HERE / "results" / "predictions.npz"))
    terms = vrun.fold_terms(store, info, stats)
    out = {"per_donor": {}, "per_pool": {}, "leave_one_pool_out": {}}
    groups = {"per_donor": sorted({fi["donor"] for fi in info.values()}),
              "per_pool": sorted({fi["pool"] for fi in info.values()}, key=lambda p: int(p[2:]))}
    for key, vals in groups.items():
        field = "donor" if key == "per_donor" else "pool"
        for v in vals:
            w = {f: 1.0 for f, fi in info.items() if fi[field] == v}
            cv = vrun.curves(terms, w, budgets)
            na, npo, now = (grun.needed(cv[a], target, budgets) for a in ("atlas", "paired_only", "own"))
            out[key][v] = {"samples": len(w), "rf": {a: [float(x) for x in cv[a]] for a in ("paired_only", "atlas", "own")},
                           "atlas_cells": na, "paired_only_cells": npo, "own_cells": now,
                           "saving": npo[0] / na[0], "atlas_over_own": na[0] / now[0]}
    for P in groups["per_pool"]:
        w = {f: 1.0 for f, fi in info.items() if fi["pool"] != P}
        cv = vrun.curves(terms, w, budgets)
        level, tgt, n = vrun.comparisons(cv, budgets, 0.5)
        out["leave_one_pool_out"][P] = {"samples": len(w), "level": level, "target": tgt,
                                        "cells": {a: n[a] for a in ("paired_only", "atlas", "own", "other")},
                                        "saving": n["paired_only"][0] / n["atlas"][0],
                                        "atlas_over_own": n["atlas"][0] / n["own"][0]}
    return terms, out


def variants(atlas, part_a):
    """Rows of the atlas kept by each variant."""
    patients = sorted(set(atlas["unit"]), key=lambda u: vrun.h(f"validation-refsize-v1|{u}"))
    rank = {u: i for i, u in enumerate(patients)}
    r = np.array([rank[u] for u in atlas["unit"]])
    cond = atlas["cond"]
    out = {"full": ("all 118 patients", np.ones(len(r), bool))}
    for n in SIZES:
        out[f"patients_{n}"] = (f"cells of {n} of the {len(patients)} patients", r < n)
    out["few_B_mono"] = (f"B cells and CD14 monocytes from {FEW} patients, other types from all",
                         ~np.isin(cond, ["B", "CD14 Mono"]) | (r < FEW))
    out["few_T"] = (f"CD4 and CD8 T cells from {FEW} patients, other types from all",
                    ~np.isin(cond, ["CD4 T", "CD8 T"]) | (r < FEW))
    for c in CONDS:
        out[f"without_{c.replace(' ', '_')}"] = (f"no {c} cells", cond != c)
    return out


def variant_context(atlas, rows, part_a, gidx_a, yidx_a):
    x, C, lib = vrun.rna(atlas, rows, gidx_a)
    y = vrun.prot(atlas, rows, yidx_a)
    cells = grun.outside_reservoir("validation-atlas", atlas["cell"][rows])
    pool, ctx = drun.context(x, y, C, lib, vrun.popkey(atlas, rows), part_a[rows], cells, atlas["unit"][rows])
    return pool, ctx


def atlas_arm(new, ctx, info, part, res, gidx, yidx, budgets):
    """The atlas arm of vrun.predict with another reference: mean prediction and mean squared norm over draws."""
    store = {}
    fds = {str(fd["fold"]): fd for fd in vrun.folds(new)}
    for f, fi in info.items():
        fd = fds[f]
        rres = np.flatnonzero(np.isin(new["sample"], fd["paired"]) & res)
        xr, _, _ = vrun.rna(new, rres, gidx)
        yr = vrun.prot(new, rres, yidx)
        popr = vrun.popkey(new, rres)
        for B in budgets:
            acc = {}
            for dr in range(vrun.DRAWS):
                rng = np.random.default_rng([vrun.SEED, int(f), B, dr])
                pick = rng.choice(len(rres), size=B, replace=False)
                hc = pc.contrasts(xr[pick], yr[pick], popr[pick])
                if hc is None:
                    continue
                Pe = es.two_sided_js(hc[0], hc[1], ctx.Uz, ctx.Ry, ctx.blocks)[0]
                for c in CONDS:
                    Q = drun.mapped(ctx, Pe, c)
                    s, q = acc.get(c, (0.0, 0.0))
                    acc[c] = (s + Q / vrun.DRAWS, q + float(np.sum(Q * Q)) / vrun.DRAWS)
            for c, (s, q) in acc.items():
                store[f"P/{f}/atlas/{B}/{c}"] = np.asarray(s, np.float32)
                store[f"msq/{f}/atlas/{B}/{c}"] = np.array(q)
    return store


def main():
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:5.0f}s] {m}", flush=True)  # noqa: E731
    pinfo = json.loads((HERE / "results" / "predictions_info.json").read_text())
    info = pinfo["folds"]
    summ = json.loads((HERE / "results" / "summary.json").read_text())
    budgets = summ["budgets"]
    target = summ["targets"]["0.5"]["target"]
    new, atlas = vrun.load_new(), vrun.load_atlas()
    part, half, res = vrun.roles(new)
    part_a = np.array([int(vrun.h(f"validation-part-v1|{c}") >= 0.5) for c in atlas["cell"]])
    genes, gidx, gidx_a, yidx, yidx_a = vrun.panels(atlas, new, part_a)
    assert genes == pinfo["panels"]["genes"]
    stats = fold_stats(new, info, half, gidx, yidx)
    terms, out = stored_part(new, info, stats, budgets, target)
    log("stored predictions: per donor, per pool, leave one pool out")
    bw = boot_weights(info)
    po = vrun.curves(terms, {f: 1.0 for f in info}, budgets)["paired_only"]
    po_boot = [vrun.curves(terms, w, budgets)["paired_only"] for w in bw]
    stored = np.load(HERE / "results" / "predictions.npz")
    out["variants"] = {}
    for name, (desc, keep) in variants(atlas, part_a).items():
        rows = np.flatnonzero(keep)
        pool, ctx = variant_context(atlas, rows, part_a, gidx_a, yidx_a)
        vstore = atlas_arm(new, ctx, info, part, res, gidx, yidx, budgets)
        if name == "full":
            diff = max(float(np.max(np.abs(vstore[k].astype(np.float64) - stored[k].astype(np.float64))))
                       for k in vstore if k.startswith("P/"))
            assert diff <= 1e-5, f"full atlas does not reproduce the hashed predictions ({diff})"
            out["reproduces_hashed_atlas_arm"] = diff
        vterms = {}
        for f in info:
            num = {}
            den = 0.0
            for c in CONDS:
                T = stats[f][c]
                den += float(np.sum(T["TA"] * T["TB"]))
                for B in budgets:
                    k = f"{f}/atlas/{B}/{c}"
                    if f"P/{k}" in vstore:
                        num[("atlas", B)] = num.get(("atlas", B), 0.0) + 2 * float(np.sum(vstore[f"P/{k}"] * T["T"])) \
                            - float(vstore[f"msq/{k}"])
            vterms[f] = (num, den)
        cv = arm_curve(vterms, {f: 1.0 for f in info}, "atlas", budgets)
        na, npo = grun.needed(cv, target, budgets), grun.needed(po, target, budgets)
        sv, cn = [], []
        for w, pb in zip(bw, po_boot):
            cb = arm_curve(vterms, w, "atlas", budgets)
            a_, p_ = grun.needed(cb, target, budgets)[0], grun.needed(pb, target, budgets)[0]
            sv.append(p_ / a_)
            cn.append(a_)
        maps = sorted(getattr(ctx, "maps", {}).keys())
        out["variants"][name] = {"description": desc, "cells": int(len(rows)),
                                 "patients": int(len(set(atlas["unit"][rows]))), "populations": len(pool.keys),
                                 "maps": maps, "rf": [float(v) for v in cv],
                                 "atlas_cells": {"cells": na[0], "relation": na[1],
                                                 "ci": [float(np.percentile(cn, 2.5)), float(np.percentile(cn, 97.5))]},
                                 "saving": {"ratio": npo[0] / na[0], "numerator_relation": npo[1],
                                            "denominator_relation": na[1],
                                            "ci": [float(np.percentile(sv, 2.5)), float(np.percentile(sv, 97.5))]}}
        log(f"{name}: {len(rows)} cells, {len(pool.keys)} populations, maps {maps}; rf "
            + " ".join(f"{100 * v:.1f}" for v in cv) + f"; atlas cells {na[0]:.0f}{na[1]}; saving {npo[0] / na[0]:.2f}")
    out["target"] = target
    out["budgets"] = budgets
    out["written"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    OUT.mkdir(exist_ok=True)
    (OUT / "robustness.json").write_text(json.dumps(out, indent=1, default=float))
    log("written")


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        vrun.check_freeze()
        main()
