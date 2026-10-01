#!/usr/bin/env python3
"""Hash the plan, the code (drun.py and every module it imports from extension/) and the data into
results/freeze.json, before any prediction of the deployment test."""
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT = HERE.parent
sys.path.insert(0, str(HERE))
import drun  # noqa: E402


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


files = {"deployment/PLAN.md", "deployment/drun.py", "deployment/freeze.py", "hybrid/run.py"}
for m in list(sys.modules.values()):
    f = getattr(m, "__file__", None)
    if f and Path(f).resolve().is_relative_to(EXT) and f.endswith(".py"):
        files.add(str(Path(f).resolve().relative_to(EXT)))
rec = {"frozen": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
       "files": {f: sha(EXT / f) for f in sorted(files)},
       "data": {n: drun.sha(drun.DATA / f"{n}.npz") for n in ("bmmc_cite", "bmmc_multiome")}}
(HERE / "results").mkdir(exist_ok=True)
out = HERE / "results" / "freeze.json"
assert not out.exists(), "already frozen"
out.write_text(json.dumps(rec, indent=1))
print(json.dumps(rec, indent=1))
