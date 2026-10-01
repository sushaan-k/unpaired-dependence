#!/usr/bin/env python3
"""Hash the plan, the code (vrun.py, vdata.py and every module they import from extension/) and the data into
results/freeze.json, before any prediction of the validation test."""
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT = HERE.parent
sys.path.insert(0, str(HERE))
import vdata  # noqa: E402
import vrun  # noqa: E402


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


files = {"validation/PLAN.md", "validation/vrun.py", "validation/vdata.py", "validation/freeze.py", "hybrid/run.py"}
for m in list(sys.modules.values()):
    f = getattr(m, "__file__", None)
    if f and Path(f).resolve().is_relative_to(EXT) and f.endswith(".py"):
        files.add(str(Path(f).resolve().relative_to(EXT)))
raw = sorted(p for p in vdata.RAW.iterdir() if p.is_file())
rec = {"frozen": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
       "files": {f: sha(EXT / f) for f in sorted(files)},
       "data": {"gse314416": [str(vrun.DATA), sha(vrun.DATA)], "stephenson": [str(vrun.ATLAS), sha(vrun.ATLAS)]},
       "raw": {p.name: sha(p) for p in raw},
       "checks": {"smoke": sha(HERE / "results_smoke" / "summary.json"),
                  "pools_check": sha(HERE / "results_check" / "pools_check.json")}}
(HERE / "results").mkdir(exist_ok=True)
out = HERE / "results" / "freeze.json"
assert not out.exists(), "already frozen"
out.write_text(json.dumps(rec, indent=1))
print(json.dumps({k: rec[k] for k in ("frozen", "files", "data")}, indent=1))
