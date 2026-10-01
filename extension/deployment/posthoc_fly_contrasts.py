#!/usr/bin/env python3
"""Post hoc: the cross-animal fly analysis of posthoc_fly.py with the paired neurons centred within their populations
by Helmert contrasts (posthoc_contrasts.contrasts) instead of df-corrected own centring (drun.own_std). Everything
else is posthoc_fly.py, run unchanged.

    python posthoc_fly_contrasts.py    # results_posthoc/fly_contrasts.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import posthoc_contrasts as pc  # noqa: E402
import posthoc_fly as pf  # noqa: E402

if __name__ == "__main__":
    tmp = pf.OUT / "_fly_contrasts"
    pf.OUT = tmp
    pf.drun.own_std = pc.contrasts          # same interface: X, Y, index of a drawn cell per row
    with threadpool_limits(limits=1):
        pf.main()
    rec = json.loads((tmp / "fly_own_standardization.json").read_text())
    rec["note"] = ("post hoc; paired neurons centred within their populations by Helmert contrasts and scaled by the "
                   "contrasts' pooled standard deviations (keys named *_own_standardization refer to this centring); "
                   "the prespecified secondary analysis centred and scaled paired neurons by the other animals' "
                   "statistics (generality/results/banc_crossanimal.json)")
    (HERE / "results_posthoc" / "fly_contrasts.json").write_text(json.dumps(rec, indent=1))
    (tmp / "fly_own_standardization.json").unlink()
    tmp.rmdir()
