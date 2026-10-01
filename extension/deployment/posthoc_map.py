#!/usr/bin/env python3
"""Post hoc (after the deployment test was scored): bases against condition maps for each source of unpaired cells.

With paired cells centred by within-population contrasts (posthoc_contrasts.py), the proposed estimator is fitted in
each pool's bases and scored with and without that pool's condition map, and with the bases of the nuclear pool
combined with the maps of the other-site CITE-seq pool. Folds, reservoir, pools, panels, budgets, draws and seeds are
those of the frozen design (drun.py, imported unchanged).

    python posthoc_map.py    # results_posthoc/map_by_source.json
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
import posthoc_contrasts as pc  # noqa: E402

es = drun.es
OUT = HERE / "results_posthoc"
ARMS = ("same_mapped", "same_unmapped", "other_mapped", "other_unmapped", "assay_mapped", "assay_unmapped",
        "assay_bases_other_map")


def main():
    cite, mult = drun.load("bmmc_cite"), drun.load("bmmc_multiome")
    part, half, res = drun.roles(cite)
    info = json.loads((drun.RES / "predictions_info.json").read_text())
    terms = {}
    t0 = time.time()
    for fd in drun.folds(cite, mult):
        f = fd["fold"]
        rr = np.flatnonzero(np.isin(cite["pop"], fd["other"]) & (part == 0))
        rp = np.flatnonzero(np.isin(cite["pop"], fd["other"]) & (part == 1))
        genes, yp = drun.panels(cite, mult, rr, rp)
        assert genes == info[str(f)]["genes"]
        P = drun.pools(cite, mult, fd, part, res, genes, yp)
        rres = np.flatnonzero(np.isin(cite["pop"], fd["paired"]) & res)
        xr, _, _ = drun.rna(cite, rres, genes)
        yr = drun.prot(cite, rres, yp)
        popr = np.array([f"{p}|{c}" for p, c in zip(cite["pop"][rres], cite["cond"][rres])])
        fb = [B for B in drun.BUDGETS if B <= len(rres)]
        stats = drun.heldout(cite, fd, half, genes, yp)
        den = sum(float(np.sum(stats[c]["TA"] * stats[c]["TB"])) for c in drun.CONDS)
        num = {}
        for B in fb:
            for dr in range(drun.DRAWS):
                rng = np.random.default_rng([drun.SEED, f, B, dr])
                pick = rng.choice(len(rres), size=B, replace=False)
                h = pc.contrasts(xr[pick], yr[pick], popr[pick])
                if h is None:
                    continue
                Xh, Yh, _ = h
                est = {}
                for key in ("same", "other", "assay"):
                    ctx = P[key][1]
                    est[key] = es.two_sided_js(Xh, Yh, ctx.Uz, ctx.Ry, ctx.blocks)[0]
                for c in drun.CONDS:
                    T = stats[c]["T"]
                    preds = {"same_mapped": drun.mapped(P["same"][1], est["same"], c),
                             "same_unmapped": est["same"],
                             "other_mapped": drun.mapped(P["other"][1], est["other"], c),
                             "other_unmapped": est["other"],
                             "assay_mapped": drun.mapped(P["assay"][1], est["assay"], c),
                             "assay_unmapped": est["assay"],
                             "assay_bases_other_map": drun.mapped(P["other"][1], est["assay"], c)}
                    for a, Q in preds.items():
                        num[(a, B)] = num.get((a, B), 0.0) + (2 * float(np.sum(Q * T)) - float(np.sum(Q * Q))) \
                            / drun.DRAWS
        terms[f] = (num, den, fb)
        print(f"fold {f} [{time.time() - t0:.0f}s]", flush=True)
    budgets = sorted(set.intersection(*[set(t[2]) for t in terms.values()]))
    D = sum(t[1] for t in terms.values())
    rf = {a: [sum(t[0][(a, B)] for t in terms.values()) / D for B in budgets] for a in ARMS}
    (OUT / "map_by_source.json").write_text(json.dumps({"budgets": budgets, "rf": rf,
                                                         "written": time.strftime("%Y-%m-%d %H:%M:%S %Z")},
                                                        indent=1))
    for a in ARMS:
        print(f"{a:24s} " + " ".join(f"{100 * v:7.1f}" for v in rf[a]))


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        drun.check_freeze()
        main()
