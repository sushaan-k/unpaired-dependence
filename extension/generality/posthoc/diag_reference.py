#!/usr/bin/env python3
"""Post hoc diagnostic (not part of the frozen analysis): per condition of one fold, the size of the held-out
target (||T||^2 and its noise-unbiased version <T_A, T_B>) against the fully paired reference prediction
(<C, T>, ||C||^2), from the hashed predictions.

    python posthoc/diag_reference.py <dataset> [fold]
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import grun as gr  # noqa: E402

name, f = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0
infos = {int(k): v for k, v in json.loads((gr.RES / f"{name}_info.json").read_text()).items()}
z = np.load(gr.RES / f"{name}_predictions.npz")
d = gr.load(gr.source(name))
fold, part, half = gr.roles(gr.source(name), d)
st = gr.heldout_stats(d, f, fold, half, np.array(infos[f]["x_panel_index"]), np.array(infos[f]["y_panel_index"]),
                      infos[f]["conditions"])
tot = np.zeros(4)
for c, rec in st.items():
    T = gr.targets(rec, np.ones((1, len(rec["units"]))))
    C = z[f"{f}/paired_all/pc/0/{c}"]
    P = z[f"{f}/bjs2/P/800"] if f"{f}/bjs2/P/800" in z.files else None
    v = np.array([np.sum(T["T"][0] ** 2), np.sum(T["TA"][0] * T["TB"][0]), np.sum(C * T["T"][0]), np.sum(C * C)])
    tot += v
    n = int(rec["T"]["n"].sum())
    print(f"{c:14s} cells {n:6d}  |T|^2 {v[0]:8.3f}  <TA,TB> {v[1]:8.3f}  <C,T> {v[2]:8.3f}  |C|^2 {v[3]:8.3f}  "
          f"RF_ref {(2 * v[2] - v[3]) / max(v[1], 1e-12):7.3f}")
print("total", np.round(tot, 3), "RF_ref", round((2 * tot[2] - tot[3]) / tot[1], 3))
