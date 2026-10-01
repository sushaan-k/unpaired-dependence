#!/usr/bin/env python3
"""Summary of the development runs (post hoc): observed against predicted savings and curves.

    python dev_summary.py      # prints tables; results/dev/summary.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ptools as pt  # noqa: E402

DEV = HERE / "results" / "dev"
TARGETS = (0.3, 0.5, 0.7)


def saving(curve_js, curve_k, target, budgets):
    nj, rj = pt.needed(curve_js, target, budgets)
    nk, rk = pt.needed(curve_k, target, budgets)
    cens = rj == ">" or rk == "<=" or rk == ">"
    return nj / nk, cens, (nj, rj, nk, rk)


def main():
    rows = []
    for p in sorted(DEV.glob("*.json")):
        if p.name == "summary.json" or p.name.endswith("_pilot.json"):
            continue
        r = json.loads(p.read_text())
        b = r["budgets"]
        o = r["observed"]
        for tg in TARGETS:
            s_obs, c_obs, det = saving(o["js"], o["bjs2"], tg, b)
            s_rnd, c_rnd, _ = saving(o["js"], o["random"], tg, b)
            row = {"dataset": r["dataset"], "target": tg, "obs": s_obs, "obs_cens": c_obs, "obs_detail": det,
                   "random_obs": s_rnd, "random_cens": c_rnd, "bound": r["pred"]["unpaired"]["bound"]}
            for src in ["population", "random_population"] + [k for k in r["pred"] if k.startswith("pilot")]:
                for law in ("lin", "sph"):
                    c = r["pred"][src][law]
                    if c is None:
                        row[f"{src}/{law}"] = None
                        continue
                    s, cen, _ = saving(c["js"], c["bjs2"], tg, b)
                    row[f"{src}/{law}"] = s
                    row[f"{src}/{law}/cens"] = cen
                w, pi = pt.shares(r["pred"][src]["blocks"])
                try:
                    row[f"{src}/closed"] = pt.saving_closed(w, pi, 1 - tg)
                except Exception:
                    row[f"{src}/closed"] = None
            rows.append(row)
    (DEV / "summary.json").write_text(pt.jdump(rows) + "\n")
    hdr = f"{'dataset':14s} tg   obs   c | pop.lin pop.sph closed | p50.lin p50.sph p100.sph p200.sph | rnd.obs rnd.pred | bound"
    print(hdr)
    for x in rows:
        f = lambda v: "   -  " if v is None else f"{v:6.2f}"
        print(f"{x['dataset']:14s} {x['target']:.1f} {x['obs']:5.2f} {'*' if x['obs_cens'] else ' '} | "
              f"{f(x['population/lin'])}  {f(x['population/sph'])}  {f(x['population/closed'])} | "
              f"{f(x.get('pilot50/lin'))}  {f(x.get('pilot50/sph'))}  {f(x.get('pilot100/sph'))}  {f(x.get('pilot200/sph'))} | "
              f"{f(x['random_obs'])}{'*' if x['random_cens'] else ' '} {f(x['random_population/sph'])} | {x['bound']:.1f}")
    # curve calibration of the population law
    print("\ncurve error (population law, mean |pred - obs| over budgets)")
    for p in sorted(DEV.glob("*.json")):
        if p.name == "summary.json" or p.name.endswith("_pilot.json"):
            continue
        r = json.loads(p.read_text())
        o = r["observed"]
        line = [r["dataset"]]
        for law in ("lin", "sph"):
            c = r["pred"]["population"][law]
            if c is None:
                continue
            for k in ("bjs2", "js"):
                line.append(f"{law}/{k} {np.mean(np.abs(np.array(c[k]) - np.array(o[k]))):.3f}")
        print("  ".join(line))


if __name__ == "__main__":
    main()
