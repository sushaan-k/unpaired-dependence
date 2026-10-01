#!/usr/bin/env python3
"""D2 of PLAN.md: the validation's saving and the comparison with reference regression when the atlas's patients
are resampled.

    python atlas_bootstrap.py smoke                 # synthetic values: one replicate, folds 0 and 17, budgets 50-100
    python atlas_bootstrap.py run K [K ...]         # checks results/freeze.json; -1: every patient once, built as
                                                    # the replicates are; 0-9: bootstrap replicates
                                                    # -> results/atlas_bootstrap/rep<k>.npz
    python atlas_bootstrap.py score                 # results/atlas_bootstrap.json

A replicate draws the atlas's 118 patients with replacement; a patient drawn twice contributes two sets of
populations. The atlas context is rebuilt from those cells, and the estimator and the four non-pooled forms of
reference regression are refitted on the first paired draw of every fold (validation seeds). The paired-only
envelope on the same draw comes from validation_review.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
EXT = HERE.parent
sys.path.insert(0, str(HERE))
import validation_review as vr  # noqa: E402

vrun, drun, pc, es, rs, grun = vr.vrun, vr.drun, vr.pc, vr.es, vr.rs, vr.vrun.grun
CONDS, SEED, MIN_PERCOND = vr.CONDS, vr.SEED, vr.MIN_PERCOND
BUDGETS = (50, 100, 200, 400, 800)
REPLICATES = 10
FORMS = ("ridge_p/mapped", "ridge_p/percond", "ridge_u/mapped", "ridge_u/percond")
ARMS = ("atlas",) + FORMS
BOOT_PER_REPLICATE = 200
SEED_REP, SEED_BOOT = 20261306, 20261306 + 1
OUT = HERE / "results" / "atlas_bootstrap"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load(synthetic=False):
    new, atlas = vrun.load_new(synthetic), vrun.load_atlas(synthetic)
    part, half, res = vrun.roles(new)
    part_a = np.array([int(vrun.h(f"validation-part-v1|{c}") >= 0.5) for c in atlas["cell"]])
    genes, gidx, gidx_a, yidx, yidx_a = vrun.panels(atlas, new, part_a)
    # memory: keep only the panel genes' counts (vrun.rna reads the same values; library sizes are kept apart)
    for d, g in ((new, gidx), (atlas, gidx_a)):
        for k in ("x_data", "x_indices", "x_indptr"):
            d.pop(k, None)
        d["x_counts"] = d["x_counts"][:, g].tocsr()
    return new, atlas, part, half, res, part_a, np.arange(len(gidx)), np.arange(len(gidx_a)), yidx, yidx_a


def replicate_context(atlas, part_a, gidx_a, yidx_a, k):
    """The atlas context from the patients of replicate k (k < 0: every patient once)."""
    patients = sorted(set(atlas["unit"]))
    pick = patients if k < 0 else list(np.random.default_rng([SEED_REP, k]).choice(patients, len(patients)))
    rows, copy = [], []
    for j, p in enumerate(pick):
        r = np.flatnonzero(atlas["unit"] == p)
        rows.append(r)
        copy.append(np.full(len(r), j))
    rows, copy = np.concatenate(rows), np.concatenate(copy)
    x, C, lib = vrun.rna(atlas, rows, gidx_a)
    y = vrun.prot(atlas, rows, yidx_a)
    pop = np.array([f"{p}#{j}|{c}" for p, c, j in zip(atlas["pop"][rows], atlas["cond"][rows], copy)])
    unit = np.array([f"{u}#{j}" for u, j in zip(atlas["unit"][rows], copy)])
    cells = grun.outside_reservoir("validation-atlas", atlas["cell"][rows])
    _, ctx = drun.context(x, y, C, lib, pop, part_a[rows], cells, unit)
    return ctx, len(set(pick))


def run(replicates, synthetic=False, fold_ids=None, budgets=BUDGETS, out_dir=OUT):
    data = load(synthetic)
    for k in replicates:
        if not (out_dir / f"rep{k}.npz").exists():
            run_replicate(k, data, fold_ids, budgets, out_dir)


def run_replicate(k, data, fold_ids, budgets, out_dir):
    new, atlas, part, half, res, part_a, gidx, gidx_a, yidx, yidx_a = data
    t0 = time.time()
    A, n_patients = replicate_context(atlas, part_a, gidx_a, yidx_a, k)
    log(f"replicate {k}: {n_patients} distinct patients; context built [{time.time() - t0:.0f}s]")
    folds = [fd for fd in vrun.folds(new) if fold_ids is None or fd["fold"] in fold_ids]
    num = np.full((len(folds), len(budgets), len(ARMS)), np.nan)
    den = np.zeros(len(folds))
    donors = []
    for fi, fd in enumerate(folds):
        f = fd["fold"]
        rres = np.flatnonzero(np.isin(new["sample"], fd["paired"]) & res)
        xr, _, _ = vrun.rna(new, rres, gidx)
        yr = vrun.prot(new, rres, yidx)
        popr, condr = vrun.popkey(new, rres), new["cond"][rres]
        st = vrun.heldout(new, fd, half, gidx, yidx)
        den[fi] = sum(float(np.sum(st[c]["TA"] * st[c]["TB"])) for c in CONDS)
        donors.append(str(fd["donor"]))
        for bi, B in enumerate(budgets):
            if B > len(rres):
                continue
            pick = np.random.default_rng([SEED, f, B, 0]).choice(len(rres), size=B, replace=False)
            hc = pc.contrasts(xr[pick], yr[pick], popr[pick])
            if hc is None:
                continue
            Xh, Yh, ih = hc
            ch = condr[pick][ih]
            preds = {}
            Pe = es.two_sided_js(Xh, Yh, A.Uz, A.Ry, A.blocks)[0]
            for c in CONDS:
                preds[("atlas", c)] = drun.mapped(A, Pe, c)
            for meth in ("ridge_p", "ridge_u"):
                crng = np.random.default_rng([SEED + 1, f, B, 0])
                Pp = rs.denoise(meth, Xh, Yh, A.Rx, A.Ry, A.n_x, A.n_y, A.Uz, A.blocks, crng)
                for c in CONDS:
                    preds[(f"{meth}/mapped", c)] = drun.mapped(A, Pp, c)
                for c in CONDS:
                    m = ch == c
                    d = A.cond[c]
                    preds[(f"{meth}/percond", c)] = (rs.denoise(meth, Xh[m], Yh[m], d["Rx"], d["Ry"], d["n_x"],
                                                                d["n_y"], d["Uz"], A.blocks, crng)
                                                     if m.sum() >= MIN_PERCOND else np.zeros_like(Pp))
            for ai, a in enumerate(ARMS):
                num[fi, bi, ai] = sum(2 * float(np.sum(preds[(a, c)] * st[c]["T"])) - float(np.sum(preds[(a, c)] ** 2))
                                      for c in CONDS)
        log(f"replicate {k} fold {f} [{time.time() - t0:.0f}s]")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"rep{k}.npz", num=num, den=den, donors=np.array(donors),
                        folds=np.array([fd["fold"] for fd in folds]), budgets=np.array(budgets),
                        patients=np.array(n_patients))


def score(out_dir=OUT, review_dir=vr.OUT):
    reps = {int(p.stem[3:]): dict(np.load(p)) for p in sorted(out_dir.glob("rep*.npz"))}
    R = {int(r["fold"]): r for r in vr.load(review_dir)}
    target = json.loads((vr.VAL / "results" / "summary.json").read_text())["targets"]["0.5"]["target"]
    out = {"target": target, "replicates": {}}
    rng = np.random.default_rng(SEED_BOOT)
    pooled_v1, pooled_k2 = [], []
    for k, rp in sorted(reps.items()):
        folds = [int(f) for f in rp["folds"]]
        budgets = [int(b) for b in rp["budgets"]]
        rb = [list(R[folds[0]]["budgets"]).index(B) for B in budgets]
        po_idx = [vr.ARMS.index(a) for a in vr.PO_ARMS]
        # paired-only arms on the first draw, primary truth, summed over types: folds x budgets x arms
        po = np.stack([np.nansum(R[f]["score"][rb][:, 0][:, po_idx, :, 0, 0], -1) for f in folds])

        def ratios(w):
            d = float(w @ rp["den"])
            cur = np.einsum("f,fba->ab", w, rp["num"]) / d
            env_po = np.nanmax(np.einsum("f,fba->ab", w, po) / d, 0)
            n_at = vr.needed(cur[0], target, budgets)[0]
            return (vr.needed(env_po, target, budgets)[0] / n_at,
                    vr.needed(np.nanmax(cur[1:], 0), target, budgets)[0] / n_at, n_at)

        v1, k2, n_at = ratios(np.ones(len(folds)))
        donors = sorted(set(rp["donors"].tolist()))
        of = {d: np.flatnonzero(rp["donors"] == d) for d in donors}
        bv, bk = [], []
        for _ in range(BOOT_PER_REPLICATE):
            w = np.zeros(len(folds))
            for d in rng.choice(donors, len(donors)):
                w[of[d]] += 1
            a, b, _ = ratios(w)
            bv.append(a)
            bk.append(b)
        out["replicates"][str(k)] = {"patients": int(rp["patients"]), "V1": v1, "K2": k2, "atlas_cells": n_at}
        if k >= 0:
            pooled_v1 += bv
            pooled_k2 += bk
    boot = [v for k, v in out["replicates"].items() if int(k) >= 0]
    if boot:
        out["over_replicates"] = {
            "V1": {"median": float(np.median([v["V1"] for v in boot])), "range": [min(v["V1"] for v in boot),
                                                                            max(v["V1"] for v in boot)]},
            "K2": {"median": float(np.median([v["K2"] for v in boot])), "range": [min(v["K2"] for v in boot),
                                                                            max(v["K2"] for v in boot)],
                   "share_above_one": float(np.mean([v["K2"] > 1 for v in boot]))}}
        out["with_donor_resampling"] = {"V1_ci": np.percentile(pooled_v1, [2.5, 97.5]).tolist(),
                                        "K2_ci": np.percentile(pooled_k2, [2.5, 97.5]).tolist(),
                                        "K2_share_above_one": float(np.mean(np.array(pooled_k2) > 1))}
    path = HERE / ("results" if out_dir == OUT else "results_smoke") / "atlas_bootstrap.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("smoke", "run", "score"))
    ap.add_argument("replicates", nargs="*", type=int)
    a = ap.parse_args()
    with threadpool_limits(limits=2):          # as vrun.main: some comparators' tuning depends on it
        if a.what == "smoke":
            run([0], synthetic=True, fold_ids=[0, 17], budgets=(50, 100), out_dir=HERE / "results_smoke" / "atlas_bootstrap")
        elif a.what == "run":
            vr.check_freeze()
            run(a.replicates)
        else:
            vr.check_freeze()
            score()


if __name__ == "__main__":
    main()
