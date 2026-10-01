#!/usr/bin/env python3
"""Record (python freeze.py) or check (check_freeze) the SHA-256 of PLAN.md and the test code.

The record, results/freeze.json, is written once, before any cross-assay or
within-assay statistic of the bone-marrow cells is computed; predict_bmmc.py,
bootstrap_bmmc.py and evaluate_bmmc.py refuse to run if any listed file differs.
An amendment (python freeze.py --amend "reason"), made before the scoring file
is opened and described under "Deviations" in PLAN.md, is written to a new file
results/freeze_amendment_N.json with the time, the reason, the files that
changed and all current digests; the original record is never rewritten, and
checks use the latest amendment.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILES = ["PLAN.md", "freeze.py", "bmmc.py", "compat.py", "diagnostics.py", "noise.py", "reference.py", "stats.py", "training.py",
         "fit_bmmc_channels.py", "predict_bmmc.py", "bootstrap_bmmc.py", "evaluate_bmmc.py", "extract_bmmc.py",
         "define_bmmc_units.py", "fit_failure_model.py", "results/failure_model.json", "results/bmmc_units.json",
         "results/bmmc_seal.json", "results/bmmc_channels.npz", "results/bmmc_channels.json"]


def digests():
    return {f: hashlib.sha256((HERE / f).read_bytes()).hexdigest() for f in FILES}


def amendments():
    return sorted((HERE / "results").glob("freeze_amendment_*.json"), key=lambda p: int(p.stem.rsplit("_", 1)[1]))


def latest():
    record = json.loads((HERE / "results/freeze.json").read_text())
    for path in amendments():
        record = dict(json.loads(path.read_text()), frozen_at=record["frozen_at"])
    return record


def check_freeze():
    record = latest()
    now = digests()
    changed = [f for f in FILES if record["files"].get(f) != now[f]]
    assert not changed, f"files changed since the freeze: {changed}"
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--amend")
    args = parser.parse_args()
    stamp = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    path = HERE / "results/freeze.json"
    if args.amend is None:
        assert not path.exists(), "freeze.json exists; the protocol is already frozen"
        record = {"frozen_at": stamp, "files": digests()}
    else:
        assert path.exists() and not (HERE / "results/bmmc_evaluation.json").exists()
        previous = latest()["files"]
        now = digests()
        record = {"amended_at": stamp, "reason": args.amend,
                  "changed": sorted(f for f in set(now) | set(previous) if now.get(f) != previous.get(f)), "files": now}
        path = HERE / f"results/freeze_amendment_{len(amendments()) + 1}.json"
    path.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
