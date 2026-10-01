#!/usr/bin/env python3
"""Freeze the external plan: SHA-256 of PLAN.md, the code and the fixed inputs, with the time read from the system
clock, written to results/freeze.json before any prediction is computed.

    python freeze.py          # refuses to overwrite an existing freeze
"""

from __future__ import annotations

import json
import time

import run_external as rx


def main():
    path = rx.HERE / "results/freeze.json"
    assert not path.exists(), "results/freeze.json exists; a plan is frozen once"
    rec = {"frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "clock": "system clock (America/New_York)",
           "files": {f: rx.sha(rx.HERE / f) for f in rx.FROZEN},
           "note": "written before any prediction was computed and before any held-out file was opened"}
    path.write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
