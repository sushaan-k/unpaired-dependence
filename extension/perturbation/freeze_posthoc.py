#!/usr/bin/env python3
"""Record the SHA-256 of PLAN_programs.md and the code of the program and design analyses before they run.

python freeze_posthoc.py writes results/freeze_posthoc.json once; programs.py and
design.py are run afterwards, and verify_perturbation.py checks the record.
"""

from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILES = ["PLAN_programs.md", "freeze_posthoc.py", "programs.py", "design.py", "pmethods.py", "pdata.py", "predict.py",
         "results/channels.npz", "results/predictions.npz", "results/seal.json", "results/evaluation.json"]


def digests():
    return {f: hashlib.sha256((HERE / f).read_bytes()).hexdigest() for f in FILES}


def check():
    record = json.loads((HERE / "results/freeze_posthoc.json").read_text())
    now = digests()
    changed = [f for f in FILES if record["files"][f] != now[f]]
    assert not changed, f"changed since the record: {changed}"
    return record


def main():
    path = HERE / "results/freeze_posthoc.json"
    assert not path.exists(), "already recorded"
    assert not (HERE / "results/programs.json").exists() and not (HERE / "results/design.json").exists()
    record = {"recorded_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"), "files": digests()}
    path.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
