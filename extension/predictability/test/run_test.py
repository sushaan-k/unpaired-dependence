#!/usr/bin/env python3
"""Harness test of prun.py on development data sets (not a result): predict, evaluate and summary on colon and
banc from extension/generality's extracted files, written to test/results."""
import json, shutil, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import prun  # noqa: E402

prun.DATA = Path("/home/claude/cbio/rawdata/generality")
prun.RES = HERE / "results"
prun.RES.mkdir(exist_ok=True)
shutil.copy(HERE.parent / "results" / "transfer_profile.json", prun.RES / "transfer_profile.json")
(prun.RES / "freeze.json").write_text(json.dumps({"files": {}}) + "\n")
names = sys.argv[1:] or ["colon", "banc"]
prun.NEW = tuple(names)
t0 = time.time()
log = lambda m: print(f"[{time.time() - t0:5.0f}s] {m}", flush=True)
from threadpoolctl import threadpool_limits
with threadpool_limits(limits=1):
    for n in names:
        prun.predict(n, log)
        prun.evaluate(n, log)
    prun.summary()
