#!/usr/bin/env python3
"""Data record of the predictability study (Deviation 2: replaces freeze.py's data mode, which hashes a file named
pdata.py).

    python freeze_data.py      # results/data_freeze.json, after extraction and before any prediction
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prun  # noqa: E402
import ptools as pt  # noqa: E402

if __name__ == "__main__":
    out = HERE / "results" / "data_freeze.json"
    assert not out.exists(), "the data record exists already"
    prun.check_freeze()
    rec = {"written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "note": "written after extraction and before any prediction; extraction used metadata and marginal "
                   "statistics of one modality at a time (Deviations 1-2)",
           "plan_freeze": pt.sha(HERE / "results" / "freeze.json"),
           "code": {f: pt.sha(HERE / f) for f in ("pextract.py", "freeze_data.py", "DEVIATIONS.md")},
           "data": {n: pt.sha(prun.DATA / f"{n}.npz") for n in prun.NEW if (prun.DATA / f"{n}.npz").exists()},
           "missing": [n for n in prun.NEW if not (prun.DATA / f"{n}.npz").exists()]}
    out.write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec, indent=1))
