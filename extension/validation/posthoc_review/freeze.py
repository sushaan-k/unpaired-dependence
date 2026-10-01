#!/usr/bin/env python3
"""Hash the plan, the runner, every module it imports from extension/ (including the validation's and the readout's
frozen code), Champollion's worker and settings, the data files, the validation's hashed predictions, the results
the plan names as known, the smoke run and the timing into results/freeze.json, before rrun.py run or champ reads
any held-out cell."""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT = HERE.parent.parent
sys.path.insert(0, str(HERE))
import rrun  # noqa: E402


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


files = {"validation/posthoc_review/PLAN.md", "validation/posthoc_review/rrun.py",
         "validation/posthoc_review/freeze.py", "semipaired/champ_worker.py", "semipaired/logs/champ_choice.json"}
for m in list(sys.modules.values()):
    f = getattr(m, "__file__", None)
    if f and Path(f).resolve().is_relative_to(EXT) and f.endswith(".py"):
        files.add(str(Path(f).resolve().relative_to(EXT)))
VAL = rrun.VAL
data = {"gse314416": rrun.vrun.DATA, "stephenson": rrun.vrun.ATLAS,
        "validation_predictions": VAL / "results" / "predictions.npz",
        "validation_manifest": VAL / "results" / "manifest.json",
        "validation_summary": VAL / "results" / "summary.json",
        "readout": VAL / "readout" / "results" / "readout.json",
        "readout_freeze": VAL / "readout" / "results" / "freeze.json",
        "allocation": VAL / "results_posthoc" / "allocation.json"}
env = subprocess.run([str(rrun.ct.VENV), "-c", "import sys, torch, importlib.metadata as m; "
                      "print(sys.version.split()[0], torch.__version__, m.version('champollion-omics'))"],
                     capture_output=True, text=True).stdout.split()
rec = {"frozen": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
       "files": {f: sha(EXT / f) for f in sorted(files)},
       "data": {k: [str(p), sha(p)] for k, p in data.items()},
       "checks": {"smoke": sha(HERE / "results_smoke" / "review.json"),
                  "timing": sha(HERE / "results" / "timing.json")},
       "champollion_environment": {"python": env[0] if env else None, "torch": env[1] if len(env) > 1 else None,
                                   "champollion-omics": env[2] if len(env) > 2 else None}}
out = HERE / "results" / "freeze.json"
assert not out.exists(), "already frozen"
out.write_text(json.dumps(rec, indent=1))
print(json.dumps({k: rec[k] for k in ("frozen", "champollion_environment")}, indent=1), len(rec["files"]), "files")
