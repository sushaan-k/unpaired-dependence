#!/usr/bin/env python3
"""Hash the plan, the runner, every module it imports from extension/ (including the validation's frozen vrun.py),
the data files, the validation's hashed predictions and the smoke run into results/freeze.json, before brun.py run
reads any held-out cell for this analysis."""
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT = HERE.parent.parent
sys.path.insert(0, str(HERE))
import brun  # noqa: E402


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


files = {"validation/readout/PLAN.md", "validation/readout/brun.py", "validation/readout/freeze.py"}
for m in list(sys.modules.values()):
    f = getattr(m, "__file__", None)
    if f and Path(f).resolve().is_relative_to(EXT) and f.endswith(".py"):
        files.add(str(Path(f).resolve().relative_to(EXT)))
VAL = brun.VAL
rec = {"frozen": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
       "files": {f: sha(EXT / f) for f in sorted(files)},
       "data": {"gse314416": [str(brun.vrun.DATA), sha(brun.vrun.DATA)],
                "stephenson": [str(brun.vrun.ATLAS), sha(brun.vrun.ATLAS)],
                "validation_predictions": [str(VAL / "results" / "predictions.npz"),
                                           sha(VAL / "results" / "predictions.npz")],
                "validation_manifest": [str(VAL / "results" / "manifest.json"), sha(VAL / "results" / "manifest.json")]},
       "checks": {"smoke": sha(HERE / "results_smoke" / "readout.json")}}
(HERE / "results").mkdir(exist_ok=True)
out = HERE / "results" / "freeze.json"
assert not out.exists(), "already frozen"
out.write_text(json.dumps(rec, indent=1))
print(json.dumps({k: rec[k] for k in ("frozen", "files")}, indent=1))
