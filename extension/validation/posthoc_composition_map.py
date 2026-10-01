#!/usr/bin/env python3
"""After scoring: does a skewed mix of cell types in the atlas fail through the axes or through the condition maps?
(results_posthoc/composition_map.json)

For the unchanged atlas and the two skewed variants of posthoc_robustness.py (B cells and CD14 monocytes, or CD4 and
CD8 T cells, from 15 patients only), the atlas arm is recomputed exactly as there but scored without the condition
map: every cell type receives the pooled estimate along the atlas's axes (eigenvectors). If the unmapped estimates of
the skewed variants recover about as much as the unmapped estimate of the unchanged atlas, the axes carry over and the
failure lies in the maps.

    python posthoc_composition_map.py     # about 10 minutes on two cores
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
import posthoc_robustness as pr  # noqa: E402
import vrun  # noqa: E402

pc, es = vrun.pc, vrun.es
VARIANTS = ("full", "few_B_mono", "few_T")


def unmapped_arm(new, ctx, info, res, gidx, yidx, budgets):
    store = {}
    fds = {str(fd["fold"]): fd for fd in vrun.folds(new)}
    for f in info:
        fd = fds[f]
        rres = np.flatnonzero(np.isin(new["sample"], fd["paired"]) & res)
        xr, _, _ = vrun.rna(new, rres, gidx)
        yr = vrun.prot(new, rres, yidx)
        popr = vrun.popkey(new, rres)
        for B in budgets:
            s, q = 0.0, 0.0
            for dr in range(vrun.DRAWS):
                rng = np.random.default_rng([vrun.SEED, int(f), B, dr])
                pick = rng.choice(len(rres), size=B, replace=False)
                hc = pc.contrasts(xr[pick], yr[pick], popr[pick])
                if hc is None:
                    continue
                Pe = es.two_sided_js(hc[0], hc[1], ctx.Uz, ctx.Ry, ctx.blocks)[0]
                s, q = s + Pe / vrun.DRAWS, q + float(np.sum(Pe * Pe)) / vrun.DRAWS
            for c in vrun.CONDS:
                store[f"P/{f}/atlas/{B}/{c}"] = np.asarray(s, np.float32)
                store[f"msq/{f}/atlas/{B}/{c}"] = np.array(q)
    return store


def main():
    t0 = time.time()
    pinfo = json.loads((HERE / "results" / "predictions_info.json").read_text())
    info = pinfo["folds"]
    budgets = json.loads((HERE / "results" / "summary.json").read_text())["budgets"]
    new, atlas = vrun.load_new(), vrun.load_atlas()
    part, half, res = vrun.roles(new)
    part_a = np.array([int(vrun.h(f"validation-part-v1|{c}") >= 0.5) for c in atlas["cell"]])
    genes, gidx, gidx_a, yidx, yidx_a = vrun.panels(atlas, new, part_a)
    stats = pr.fold_stats(new, info, half, gidx, yidx)
    rob = json.loads((HERE / "results_posthoc" / "robustness.json").read_text())
    out = {"budgets": budgets, "variants": {}}
    allv = pr.variants(atlas, part_a)
    for name in VARIANTS:
        desc, keep = allv[name]
        pool, ctx = pr.variant_context(atlas, np.flatnonzero(keep), part_a, gidx_a, yidx_a)
        vstore = unmapped_arm(new, ctx, info, res, gidx, yidx, budgets)
        terms = {}
        for f in info:
            num, den = {}, 0.0
            for c in vrun.CONDS:
                T = stats[f][c]
                den += float(np.sum(T["TA"] * T["TB"]))
                for B in budgets:
                    k = f"{f}/atlas/{B}/{c}"
                    num[("atlas", B)] = num.get(("atlas", B), 0.0) + 2 * float(np.sum(vstore[f"P/{k}"] * T["T"])) \
                        - float(vstore[f"msq/{k}"])
            terms[f] = (num, den)
        cv = pr.arm_curve(terms, {f: 1.0 for f in info}, "atlas", budgets)
        out["variants"][name] = {"description": desc, "unmapped_rf": [float(v) for v in cv],
                                 "mapped_rf": rob["variants"][name]["rf"]}
        print(f"[{time.time() - t0:5.0f}s] {name}: unmapped " + " ".join(f"{100 * v:.1f}" for v in cv)
              + " | mapped " + " ".join(f"{100 * v:.1f}" for v in rob["variants"][name]["rf"]), flush=True)
    out["written"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    (HERE / "results_posthoc" / "composition_map.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        vrun.check_freeze()
        main()
