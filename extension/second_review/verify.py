#!/usr/bin/env python3
"""Verify the analyses added in revision without the data.

1. The plan, the scripts, every module they import and the smoke results have the SHA-256 recorded in
   results/freeze.json, and every result record was written after the freeze.
2. Every scored record is recomputed from the stored per-fold, per-replicate or per-data-set records by the scripts'
   own `score` commands, run in a scratch copy of this folder, and compared with the one in results/ (byte for byte,
   or value by value to floating-point rounding on another machine).
3. The draw-scoring check (../checks) passed in every test it covers.

    python verify.py            # about ten minutes

The runs themselves (`run` commands) need the public data (../DATA.md); results/logs/ holds their logs.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT = HERE.parent
RES = HERE / "results"
SCORED = [("noise_calibration.py", "noise_calibration.json"), ("validation_review.py", "validation_review.json"),
          ("atlas_bootstrap.py", "atlas_bootstrap.json"), ("law_review.py", "law_review.json"),
          ("benchmark_review.py", "benchmark_review.json")]
RUN_RECORDS = ["noise", "validation", "atlas_bootstrap", "law", "benchmark", "readout_checks.json"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def check_hashes(rec):
    missing = [f for f in rec["files"] if not (EXT / f).exists()]
    bad = [f for f, d in rec["files"].items() if f not in missing and sha(EXT / f) != d]
    bad += [f"{f} (missing)" for f in missing]
    print(f"{len(rec['files'])} frozen files: {'all match' if not bad else 'changed: ' + ', '.join(bad)}")
    smoke = [f for f, d in rec["smoke"].items() if not (HERE / f).exists() or sha(HERE / f) != d]
    print(f"{len(rec['smoke'])} smoke results: "
          f"{'all match' if not smoke else 'changed or missing: ' + ', '.join(smoke)}")
    data = [(k, Path(p)) for k, (p, _) in rec["data"].items()]
    present = [(k, p) for k, p in data if p.exists()]
    wrong = [k for k, p in present if sha(p) != rec["data"][k][1]]
    print(f"{len(present)} of {len(data)} data files present here; "
          f"{'all match' if not wrong else 'changed: ' + ', '.join(wrong)}")
    return not bad and not smoke and not wrong


def check_order(rec):
    frozen = datetime.strptime(rec["frozen"][:19], "%Y-%m-%d %H:%M:%S").timestamp()
    files = [p for r in RUN_RECORDS for p in ([RES / r] if (RES / r).is_file() else sorted((RES / r).rglob("*")))
             if p.is_file()]
    early = [str(p.relative_to(HERE)) for p in files if p.stat().st_mtime < frozen]
    print(f"{len(files)} result records; written before the freeze: {early if early else 'none'}")
    return bool(files) and not early


def first_difference(a, b, path="", rtol=1e-9, atol=1e-12):
    """The first place where two parsed records differ beyond floating-point rounding, or None."""
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return f"{path or '/'}: keys differ"
        diffs = (first_difference(a[k], b[k], f"{path}/{k}", rtol, atol) for k in a)
        return next((d for d in diffs if d), None)
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return f"{path}: lengths differ"
        diffs = (first_difference(x, y, f"{path}/{k}", rtol, atol) for k, (x, y) in enumerate(zip(a, b)))
        return next((d for d in diffs if d), None)
    return None if same_value(a, b, rtol, atol) else f"{path}: {a!r} against {b!r}"


def same_value(a, b, rtol=1e-9, atol=1e-12):
    """Whether two leaf values of a parsed record agree. A boolean equals only the same boolean, never 0 or 1; a
    float may differ from another number by rounding only if both are finite; an infinity equals only itself, and
    NaN equals NaN; anything else must be of the same type and equal."""
    if isinstance(a, bool) or isinstance(b, bool) or not (isinstance(a, float) or isinstance(b, float)):
        return type(a) is type(b) and a == b
    if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
        return False
    try:
        x, y = float(a), float(b)
    except OverflowError:
        return a == b
    if math.isnan(x) or math.isnan(y):
        return math.isnan(x) and math.isnan(y)
    if math.isinf(x) or math.isinf(y):
        return x == y
    return abs(x - y) <= atol + rtol * max(abs(x), abs(y))


def rescore():
    """Run each script's score command in a scratch copy of this folder and compare the records: byte for byte, or,
    on another machine whose floating-point sums may round differently, value by value to a relative 1e-9."""
    scratch = EXT / f"second_review_verify_{os.getpid()}"
    ok = True
    try:
        shutil.copytree(HERE, scratch, ignore=shutil.ignore_patterns("__pycache__", "results_smoke", "logs"))
        for _, out in SCORED:
            (scratch / "results" / out).unlink(missing_ok=True)
        for script, out in SCORED:
            r = subprocess.run([sys.executable, script, "score"], cwd=scratch, capture_output=True, text=True)
            new, old = scratch / "results" / out, RES / out
            if r.returncode or not new.exists():
                ok = False
                print(f"{out}: FAILED: {r.stderr[-400:]}")
            elif new.read_bytes() == old.read_bytes():
                print(f"{out}: identical")
            else:
                diff = first_difference(json.loads(new.read_text()), json.loads(old.read_text()))
                ok &= diff is None
                print(f"{out}: {'equal to rounding' if diff is None else 'DIFFERENT at ' + diff}")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return ok


def check_draws():
    rec = json.loads((EXT / "checks" / "results" / "draw_scoring.json").read_text())
    rows = rec["tests"]
    failed = [r["test"] for r in rows if not r["passed"]]
    print(f"draw scoring: {len(rows)} tests, {'all passed' if not failed else 'failed: ' + ', '.join(failed)}")
    return not failed


def main():
    rec = json.loads((RES / "freeze.json").read_text())
    print(f"frozen {rec['frozen']}")
    results = [check_hashes(rec), check_order(rec), rescore(), check_draws()]
    print("ALL CHECKS PASSED" if all(results) else "SOME CHECKS FAILED")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
