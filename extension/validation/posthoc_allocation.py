#!/usr/bin/env python3
"""After scoring: the sample-donor-pool allocation of the validation test and its donor separation.

For every fold (held-out sample) it lists the pool, the held-out donor and time point, the two paired samples and
the other-pool reference, and checks that no donor contributes cells to both sides of a fold:

* the held-out donor is not a donor of the paired samples, of the own-sample reference (the paired samples' other
  cells) or of the other-pool reference, at either time point;
* the atlas is another study (Stephenson et al. 2021), so it shares no donor with the validation study.

Reads only the sample, donor, pool and time-point labels (no expression value). Writes
results_posthoc/allocation.json.

    python posthoc_allocation.py
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

import vrun

HERE = Path(__file__).resolve().parent
OUT = HERE / "results_posthoc" / "allocation.json"


def main():
    d = vrun.load_new()
    part, half, res = vrun.roles(d)
    samples = sorted(set(d["sample"]))
    info = {}
    for s in samples:
        m = d["sample"] == s
        info[s] = {"donor": str(d["donor"][m][0]), "pool": str(d["pool"][m][0]),
                   "timepoint": str(d["timepoint"][m][0]), "cells": int(m.sum()),
                   "reservoir": int(np.sum(m & res))}
        assert len(set(d["donor"][m])) == 1 and len(set(d["pool"][m])) == 1 and len(set(d["timepoint"][m])) == 1
    donors = sorted({v["donor"] for v in info.values()}, key=lambda x: int(x[2:]))
    pools = sorted({v["pool"] for v in info.values()}, key=lambda p: int(p[2:]))
    per_pool = {P: Counter(v["donor"] for v in info.values() if v["pool"] == P) for P in pools}
    checks = {"one_sample_per_donor_and_pool": all(max(c.values()) == 1 for c in per_pool.values()),
              "two_timepoints_in_different_pools": all(
                  len({info[s]["pool"] for s in samples if info[s]["donor"] == dn})
                  == sum(1 for s in samples if info[s]["donor"] == dn) for dn in donors)}
    folds = []
    held_not_paired, held_not_other, paired_not_other, other_excludes_pool = [], [], [], []
    for fd in vrun.folds(d):
        hd = info[fd["heldout"]]["donor"]
        pd_ = [info[s]["donor"] for s in fd["paired"]]
        od = sorted({info[s]["donor"] for s in fd["other"]}, key=lambda x: int(x[2:]))
        pool_donors = set(per_pool[fd["pool"]])
        held_not_paired.append(hd not in pd_)
        held_not_other.append(hd not in od)
        paired_not_other.append(not set(pd_) & set(od))
        other_excludes_pool.append(not pool_donors & set(od))
        own_rows = np.isin(d["sample"], fd["paired"]) & ~res
        folds.append({"fold": fd["fold"], "pool": str(fd["pool"]), "heldout": str(fd["heldout"]), "donor": hd,
                      "timepoint": info[fd["heldout"]]["timepoint"],
                      "paired": [{"sample": str(s), "donor": info[s]["donor"], "timepoint": info[s]["timepoint"]}
                                 for s in fd["paired"]],
                      "own_reference_cells": int(own_rows.sum()),
                      "other": {"samples": len(fd["other"]), "donors": len(od),
                                "pools": sorted({info[s]["pool"] for s in fd["other"]}, key=lambda p: int(p[2:])),
                                "cells": int(np.isin(d["sample"], fd["other"]).sum())}})
    checks.update({"heldout_donor_not_paired": all(held_not_paired),
                   "heldout_donor_not_in_other_pool_reference": all(held_not_other),
                   "paired_donors_not_in_other_pool_reference": all(paired_not_other),
                   "other_pool_reference_excludes_every_donor_of_the_pool": all(other_excludes_pool),
                   "atlas": "another study (Stephenson et al. 2021, E-MTAB-10026); no shared donor"})
    roles = {dn: {"heldout": sum(f["donor"] == dn for f in folds),
                  "paired": sum(any(p["donor"] == dn for p in f["paired"]) for f in folds)} for dn in donors}
    rec = {"samples": info, "donors": donors, "pools": pools,
           "donors_by_pool": {P: sorted(c, key=lambda x: int(x[2:])) for P, c in per_pool.items()},
           "folds": folds, "roles": roles, "checks": checks,
           "donor_disjoint": all(v is True for k, v in checks.items() if k != "atlas"),
           "written": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(rec, indent=1))
    print(json.dumps(checks, indent=1))
    print("donor-disjoint:", rec["donor_disjoint"])
    print({P: rec["donors_by_pool"][P] for P in pools})
    print("held out twice:", sum(v["heldout"] == 2 for v in roles.values()),
          "once:", sum(v["heldout"] == 1 for v in roles.values()))


if __name__ == "__main__":
    main()
