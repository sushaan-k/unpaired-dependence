#!/usr/bin/env python3
"""Fit every channel of the bone-marrow test on its panel (training.py; PLAN.md, "Channels").

Uses training data only (blood patients and colon adaptation cells); reads no
bone-marrow file except the seal record that lists the panel. Writes
results/bmmc_channels.npz and results/bmmc_channels.json (tuning summaries).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from training import all_channels

HERE = Path(__file__).resolve().parent


def main():
    chans = all_channels()
    arrays, info = {}, {}
    for name, c in chans.items():
        for k in ("W", "B", "G", "psi", "b", "R_train"):
            arrays[f"{name}/{k}"] = c[k]
        arrays[f"{name}/ridge"] = np.array(c["ridge"])
        info[name] = {"scale": float(c["scale"]), "at_edge": bool(c["at_edge"]), "n_train": int(c["n_train"])}
    np.savez_compressed(HERE / "results/bmmc_channels.npz", **arrays)
    (HERE / "results/bmmc_channels.json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps({k: v for k, v in info.items() if not k.startswith(("lc", "perm"))}, indent=1))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
