#!/usr/bin/env python3
"""Development comparison of the self-tuning variants (DEV_SELFTUNE.md) on the two development screens.

    python dev_selftune.py frangieh|papalexi

Same folds, reservoir, unpaired summaries, paired-cell draws (seeds of run_study.py) and scoring as the development
study; arms V0 (the fixed estimator, bjs2/mapped) and V1-V3 (selftune.py), each followed by the unchanged condition
map. V0 must reproduce the stored bjs2/mapped numbers. Writes results/selftune_<dataset>.json (per-group numerators
in the format of results/<dataset>.json) and prints the rule's criterion.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter

import numpy as np
from threadpoolctl import threadpool_limits

import common as cm
import run_study as rs
import selftune as st
import sp_estimators as es

pm = cm.pm
VARIANTS = ("V1", "V2", "V3")


def run_fold(dataset, fi, gs, train, x, y, pop, keys, masks, budgets, draws, log):
    pool, un, X_res, Y_res, cond_res = cm.setup_fold(train, x, y, pop, keys)
    ctx = rs.build_context(train, x, y, pop, keys, pool, un)
    T = np.array([g["T"] for g in gs])
    den = np.einsum("gpq,bpq->gb", np.array([g["TA"] * g["TB"] for g in gs]), masks)
    gcond = [g["cond"] for g in gs]
    V, _ = es.eigenbasis(ctx.Ry)
    out = {"den": den, "num": {}, "choices": {v: Counter() for v in VARIANTS}, "rules": {v: Counter() for v in VARIANTS},
           "seconds": {v: 0.0 for v in ("V0",) + VARIANTS}}

    def add(arm, B, C):
        num, _ = cm.scores(C, T, None, None, masks)
        out["num"].setdefault(arm, {}).setdefault(B, np.zeros_like(den))
        out["num"][arm][B] += num / draws

    for B in budgets:
        if B > len(X_res):
            continue
        for dr in range(draws):
            rng = np.random.default_rng([rs.DRAW_SEED, {"frangieh": 1, "papalexi": 2}[dataset], fi, B, dr])
            pick = rng.choice(len(X_res), size=B, replace=False)
            X, Y = X_res[pick], Y_res[pick]
            t0 = time.time()
            P = es.two_sided_js(X, Y, ctx.Uz, ctx.Ry, ctx.blocks)[0]
            out["seconds"]["V0"] += time.time() - t0
            add("V0", B, np.array([ctx.mapped(P, c) for c in gcond]))
            for v in VARIANTS:
                info = {}
                t0 = time.time()
                P = st.estimate(X, Y, ctx.Uz, V, v, info=info)
                out["seconds"][v] += time.time() - t0
                out["choices"][v][f"{tuple(info['rna_edges'])} x {tuple(info['prot_edges'])}"] += 1
                out["rules"][v].update(info["rules"])
                add(v, B, np.array([ctx.mapped(P, c) for c in gcond]))
        log(f"  B={B}: " + ", ".join(f"{a} {100 * out['num'][a][B][:, 0].sum() / den[:, 0].sum():.1f}"
                                      for a in ("V0",) + VARIANTS))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=sorted(rs.SETTINGS))
    args = ap.parse_args()
    st_ = rs.SETTINGS[args.dataset]
    budgets, draws = st_["budgets"], st_["draws"]
    t0 = time.time()

    def log(s):
        print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)
    mod, train, ev = cm.hr.load_dataset(args.dataset)
    x, y = pm.features(train)
    pop = pm.keys_of(train)
    keys = pm.eligible_training(pop, train["part"], mod.MIN_TRAIN_HALF)
    names, masks, defs = cm.hr.block_masks(train["genes"], train["proteins"], mod.ENCODING)
    groups_all = ev["test"]
    folds = [None] if args.dataset == "frangieh" else sorted({g["target"] for g in groups_all})
    rows, choices, rules = [], {v: Counter() for v in VARIANTS}, {v: Counter() for v in VARIANTS}
    seconds = Counter()
    for fi, held in enumerate(folds):
        gs = [g for g in groups_all if held is None or g["target"] == held]
        ks = [k for k in keys if held is None or k.split("|")[0] != held]
        log(f"fold {held}: {len(gs)} groups")
        out = run_fold(args.dataset, fi, gs, train, x, y, pop, ks, masks, budgets, draws, log)
        for v in VARIANTS:
            choices[v].update(out["choices"][v])
            rules[v].update(out["rules"][v])
        seconds.update(out["seconds"])
        for i, g in enumerate(gs):
            rows.append({"key": g["key"], "target": g["target"], "cond": g["cond"], "den": out["den"][i].tolist(),
                         "num": {a: {str(b): v[i].tolist() for b, v in d.items()} for a, d in out["num"].items()}})
    # check: V0 reproduces the development study's bjs2/mapped
    stored = json.loads((cm.HERE / "results" / f"{args.dataset}.json").read_text())
    sk = {r["key"]: r for r in stored["rows"]}
    dev = max(abs(a - b) for r in rows for B in r["num"]["V0"]
              for a, b in zip(r["num"]["V0"][B], sk[r["key"]]["num"]["bjs2/mapped"][B]))
    rf = {a: [sum(r["num"][a][str(B)][0] for r in rows) / sum(r["den"][0] for r in rows) for B in budgets]
          for a in ("V0",) + VARIANTS}
    res = {"dataset": args.dataset, "budgets": list(budgets), "draws": draws, "blocks": names, "rows": rows,
           "rf": rf, "criterion": {a: float(np.mean(v)) for a, v in rf.items()},
           "v0_max_abs_difference_from_stored_bjs2_mapped": dev,
           "partition_choices": {v: dict(choices[v].most_common()) for v in VARIANTS},
           "block_rules": {v: dict(rules[v]) for v in VARIANTS},
           "ms_per_fit": {a: 1000 * s / (len(budgets) * draws * len(folds)) for a, s in seconds.items()}}
    (cm.HERE / "results" / f"selftune_{args.dataset}.json").write_text(json.dumps(res) + "\n")
    log(f"V0 vs stored bjs2/mapped: max abs difference {dev:.2e}")
    for a, v in rf.items():
        log(f"{a}: " + " ".join(f"{100 * u:.1f}" for u in v) + f" | mean {100 * np.mean(v):.2f}")
    log("done")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
