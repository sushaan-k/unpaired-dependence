#!/usr/bin/env python3
"""Hash the plan, the scripts, every module they import from extension/, the data files and the smoke results into
results/freeze.json, before any script of this folder reads a held-out cell, a truth cell or a result of the plan."""

import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT = HERE.parent
sys.path.insert(0, str(HERE))
import atlas_bootstrap  # noqa: E402,F401
import benchmark_review  # noqa: E402,F401
import law_review  # noqa: E402
import noise_calibration  # noqa: E402
import readout_checks  # noqa: E402,F401
import validation_review  # noqa: E402,F401


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


files = {f"second_review/{n}" for n in ("PLAN.md", "freeze.py", "noise_calibration.py", "validation_review.py",
                                         "readout_checks.py", "law_review.py", "benchmark_review.py",
                                         "atlas_bootstrap.py")}
for m in list(sys.modules.values()):
    f = getattr(m, "__file__", None)
    if f and Path(f).resolve().is_relative_to(EXT) and f.endswith(".py"):
        files.add(str(Path(f).resolve().relative_to(EXT)))
vrun, drun, prun = noise_calibration.vrun, noise_calibration.drun, law_review.prun
data = {"gse314416": vrun.DATA, "stephenson": vrun.ATLAS,
        "bmmc_cite": drun.DATA / "bmmc_cite.npz", "bmmc_multiome": drun.DATA / "bmmc_multiome.npz",
        "validation_predictions": EXT / "validation/results/predictions.npz",
        "validation_summary": EXT / "validation/results/summary.json",
        "review": EXT / "validation/posthoc_review/results/review.json"}
for n in prun.NEW:
    data[f"law_{n}"] = prun.DATA / f"{n}.npz"
rec = {"frozen": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
       "files": {f: sha(EXT / f) for f in sorted(files)},
       "data": {k: [str(p), sha(p)] for k, p in data.items()},
       "smoke": {str(p.relative_to(HERE)): sha(p) for p in sorted((HERE / "results_smoke").rglob("*")) if p.is_file()}}
out = HERE / "results" / "freeze.json"
out.parent.mkdir(exist_ok=True)
assert not out.exists(), "already frozen"
out.write_text(json.dumps(rec, indent=1) + "\n")
print(rec["frozen"], len(rec["files"]), "files,", len(rec["data"]), "data files,", len(rec["smoke"]), "smoke results")
