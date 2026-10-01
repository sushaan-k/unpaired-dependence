#!/usr/bin/env python3
"""Record (python freeze.py) or check (check_freeze) the SHA-256 of PLAN.md and the code of the replication.

results/freeze.json is written once, after the split and seal and before any
statistic of test or non-targeting cells other than library sizes is computed;
predict.py and evaluate.py refuse to run if a listed file differs. Amendments
(python freeze.py --amend "reason"), before the scoring files are opened, are
written to results/freeze_amendment_N.json and described in PLAN.md.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILES = ["PLAN.md", "freeze.py", "rdata.py", "extract.py", "predict.py", "evaluate.py", "results/seal.json",
         "../perturbation/pmethods.py", "../perturbation/pdata.py", "../cross_study/estimators.py",
         "../cross_study/data.py", "../spectral_transfer/gaussian_transfer.py", "../recoverability/compat.py",
         "../recoverability/noise.py", "../recoverability/stats.py"]


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
        assert path.exists() and not (HERE / "results/evaluation.json").exists()
        previous = latest()["files"]
        now = digests()
        record = {"amended_at": stamp, "reason": args.amend,
                  "changed": sorted(f for f in set(now) | set(previous) if now.get(f) != previous.get(f)), "files": now}
        path = HERE / f"results/freeze_amendment_{len(amendments()) + 1}.json"
    path.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
