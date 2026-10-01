#!/usr/bin/env python3
"""Hash the plan, the code and the extracted data files of the generality benchmark (results/freeze.json).

    python freeze.py

Run once, before any held-out statistic of these data sets is computed.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = Path("/home/claude/cbio/rawdata/generality")
FILES = ["PLAN.md", "gdata.py", "grun.py", "glaw.py", "freeze.py",
         "../semipaired/run_study.py", "../semipaired/common.py", "../semipaired/sp_estimators.py",
         "../semipaired/competitors.py", "../semipaired/selftune.py", "../semipaired/DEV_SELFTUNE.md",
         "../semipaired/bound_check.py", "../semipaired/results/frangieh.json", "../semipaired/results/papalexi.json",
         "../semipaired/external/results/overcite_plan_estimator.json", "../hybrid/hmethods.py", "../hybrid/run.py",
         "../perturbation/pmethods.py", "../perturbation/pdata.py", "../recoverability/noise.py",
         "../recoverability/compat.py", "../cross_study/estimators.py", "../cross_study/data.py",
         "../spectral_transfer/gaussian_transfer.py"]
DATASETS = ["hao", "stephenson", "bmmc_cite", "colon", "bmmc_multiome", "scala_m1", "gouwens_visp", "banc", "malecns",
            "fafb_brain", "manc_vnc"]


def sha(path):
    d = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            d.update(b)
    return d.hexdigest()


def main():
    out = HERE / "results" / "freeze.json"
    assert not out.exists(), "already frozen"
    rec = {"frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "files": {f: sha(HERE / f) for f in FILES},
           "data": {n: sha(DATA / f"{n}.npz") for n in DATASETS},
           "data_json": {n: sha(DATA / f"{n}.json") for n in DATASETS if (DATA / f"{n}.json").exists()}}
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({"frozen_at": rec["frozen_at"], "files": len(rec["files"]), "data": len(rec["data"])}))


if __name__ == "__main__":
    main()
