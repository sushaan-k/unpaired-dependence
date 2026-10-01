#!/usr/bin/env python3
"""Freeze records of the predictability study.

    python freeze.py plan     # results/freeze.json: plan, theory, development records and code, before extraction
    python freeze.py data     # results/data_freeze.json: extraction code and extracted files, before predictions
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ptools as pt  # noqa: E402

PLAN_FILES = ["PLAN.md", "DEV.md", "THEORY.md", "ptools.py", "prun.py", "freeze.py", "check_theory.py", "dev.py",
              "dev2.py", "dev_summary.py", "dev2_summary.py", "results/transfer_profile.json",
              "results/check_theory.json", "../generality/grun.py", "../semipaired/sp_estimators.py",
              "../semipaired/common.py", "../hybrid/hmethods.py", "../perturbation/pmethods.py",
              "../recoverability/noise.py", "../cross_study/data.py"]


def plan():
    out = HERE / "results" / "freeze.json"
    assert not out.exists(), "the plan was frozen already"
    dev = sorted((HERE / "results" / "dev").glob("*.json"))
    rec = {"frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "note": "written before any prospective data set was extracted and before any statistic of dependence "
                   "between its modalities was computed",
           "files": {f: pt.sha(HERE / f) for f in PLAN_FILES},
           "development_results": {str(p.relative_to(HERE)): pt.sha(p) for p in dev}}
    out.write_text(json.dumps(rec, indent=1) + "\n")
    print(rec["frozen_at"], len(rec["files"]), "files,", len(dev), "development results")


def data():
    import prun
    out = HERE / "results" / "data_freeze.json"
    rec = {"written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "note": "written after extraction and before any prediction; extraction used metadata and marginal "
                   "statistics of one modality at a time",
           "plan_freeze": pt.sha(HERE / "results" / "freeze.json"),
           "code": {f: pt.sha(HERE / f) for f in ("pdata.py",)},
           "data": {n: pt.sha(prun.DATA / f"{n}.npz") for n in prun.NEW if (prun.DATA / f"{n}.npz").exists()},
           "missing": [n for n in prun.NEW if not (prun.DATA / f"{n}.npz").exists()]}
    out.write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    {"plan": plan, "data": data}[sys.argv[1]]()
