#!/usr/bin/env python3
"""Summary of dev2 (post hoc): pilot predictors against observed savings."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ptools as pt  # noqa: E402
from dev_summary import saving  # noqa: E402
DEV = HERE / "results" / "dev"

rows = []
for p in sorted(DEV.glob("*_pilot.json")):
    q = json.loads(p.read_text())
    r = json.loads((DEV / p.name.replace("_pilot", "")).read_text())
    b, o = r["budgets"], r["observed"]
    for tg in (0.3, 0.5, 0.7):
        s_obs, c_obs, det = saving(o["js"], o["bjs2"], tg, b)
        row = {"dataset": r["dataset"], "target": tg, "obs": s_obs, "cens": c_obs}
        c = r["pred"]["population"]["lin"]
        row["pop_lin"] = saving(c["js"], c["bjs2"], tg, b)[0]
        c = q["boot_population"]
        row["pop_boot"] = saving(c["js"], c["bjs2"], tg, b)[0]
        for m in (50, 100, 200):
            for k in ("lin", "lin_thr", "lin_shrunk", "boot"):
                c = q[f"pilot{m}"][k]
                row[f"p{m}_{k}"] = None if c is None else saving(c["js"], c["bjs2"], tg, b)[0]
            vals = [saving(c["js"], c["bjs2"], tg, b)[0] for c in q[f"pilot{m}"]["lin_resampled"] if c is not None]
            row[f"p{m}_int"] = (float(np.quantile(vals, 0.1)), float(np.quantile(vals, 0.9))) if vals else None
        rows.append(row)
f = lambda v: "  -  " if v is None else f"{v:5.2f}"
print(f"{'dataset':13s} tg  obs  | pop.lin pop.boot | p100: lin  thr  shr  boot  [10-90%] | p200: lin thr shr boot")
for x in rows:
    i1 = x["p100_int"]
    print(f"{x['dataset']:13s} {x['target']:.1f} {x['obs']:5.2f}{'*' if x['cens'] else ' '}| {f(x['pop_lin'])} {f(x['pop_boot'])} | "
          f"{f(x['p100_lin'])} {f(x['p100_lin_thr'])} {f(x['p100_lin_shrunk'])} {f(x['p100_boot'])} [{f(i1[0]) if i1 else ''}-{f(i1[1]) if i1 else ''}] | "
          f"{f(x['p200_lin'])} {f(x['p200_lin_thr'])} {f(x['p200_lin_shrunk'])} {f(x['p200_boot'])}")
# error summary at 0.5 over uncensored
for key in ["pop_lin", "pop_boot"] + [f"p{m}_{k}" for m in (50, 100, 200) for k in ("lin", "lin_thr", "lin_shrunk", "boot")]:
    for tg in (0.3, 0.5, 0.7):
        e = [abs(np.log2(x[key] / x["obs"])) for x in rows if x["target"] == tg and not x["cens"] and x[key]]
        if e:
            print(f"{key:16s} {tg}: median |log2| {np.median(e):.2f}  max {max(e):.2f}  n={len(e)}")
