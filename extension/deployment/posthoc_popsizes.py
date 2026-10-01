#!/usr/bin/env python3
"""Post hoc: sizes of the populations (cell type x donor) among the drawn paired cells of the deployment test, for
the frozen folds, budgets, draws and seeds (drun.py, imported unchanged). No expression value is read.

    python posthoc_popsizes.py    # results_posthoc/population_sizes.json
"""

from __future__ import annotations

import collections
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import drun  # noqa: E402


def main():
    cite, mult = drun.load("bmmc_cite"), drun.load("bmmc_multiome")
    part, half, res = drun.roles(cite)
    acc = collections.defaultdict(list)
    for fd in drun.folds(cite, mult):
        f = fd["fold"]
        rres = np.flatnonzero(np.isin(cite["pop"], fd["paired"]) & res)
        popr = np.array([f"{p}|{c}" for p, c in zip(cite["pop"][rres], cite["cond"][rres])])
        for B in drun.BUDGETS:
            if B > len(rres):
                continue
            for dr in range(drun.DRAWS):
                rng = np.random.default_rng([drun.SEED, f, B, dr])
                pick = rng.choice(len(rres), size=B, replace=False)
                cnt = collections.Counter(popr[pick])
                n = np.array([cnt[k] for k in popr[pick]])
                acc[B].append({"single": float((n == 1).mean()), "at_most_three": float((n <= 3).mean()),
                               "populations": len(cnt), "contrasts": int(sum(v - 1 for v in cnt.values()))})
    out = {str(B): {k: float(np.mean([r[k] for r in v])) for k in v[0]} for B, v in sorted(acc.items())}
    (HERE / "results_posthoc" / "population_sizes.json").write_text(
        json.dumps({"budgets": out, "written": time.strftime("%Y-%m-%d %H:%M:%S %Z")}, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
