#!/usr/bin/env python3
"""Checks of the predictability study (extension/predictability).

    python verify_predictability.py [--data]

Without --data (results only): the plan, theory, development records and code are unchanged since the plan's
freeze; the data record was written after it; every data set's manifest was written after the data record, names
the plan freeze and matches its prediction files (where present); the summary and verdicts are recomputed from the
per-data-set results and predictions; the theory checks passed; the manuscript assets regenerate identically. With
--data (needs /home/claude/cbio/rawdata/predictability): the extracted files match the data record and two data
sets are re-scored from their hashed estimates.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
P = HERE / "predictability"
RES = P / "results"
SOURCE = HERE.parent / "revision" / "source"
sys.path.insert(0, str(P))
checks = []


def check(name, ok, detail=""):
    checks.append(bool(ok))
    print(f"[{'ok' if ok else 'FAIL'}] {name}{': ' + detail if detail else ''}", flush=True)


def wall_clock(stamp):
    """A recorded time such as '2026-09-27T18:07:08-0400' as the wall-clock time it names (all were recorded in EDT)."""
    return datetime.datetime.strptime(stamp[:19], "%Y-%m-%dT%H:%M:%S")


def file_clock(path):
    """A file's modification time as the host's wall-clock time. A zip archive stores and restores wall-clock times,
    so comparing these with recorded wall-clock times gives the same order in any time zone."""
    return datetime.datetime.fromtimestamp(os.path.getmtime(path))


def first_difference(a, b, path=""):
    """Where two parsed records first differ, or None. Finite numbers that are not both integers may differ by
    floating-point rounding (relative 1e-9, absolute 1e-12), since sums can round differently with another machine
    or library build; keys, lengths, strings, booleans, integers and non-finite numbers must match exactly."""
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return f"{path or '/'}: keys differ"
        return next((d for k in a if (d := first_difference(a[k], b[k], f"{path}/{k}"))), None)
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return f"{path}: lengths differ"
        return next((d for i, (x, y) in enumerate(zip(a, b)) if (d := first_difference(x, y, f"{path}/{i}"))), None)
    return None if same_value(a, b) else f"{path}: {a!r} against {b!r}"


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


def results_only():
    import ptools as pt
    import prun
    rec = json.loads((RES / "freeze.json").read_text())
    bad = [f for f, d in rec["files"].items() if pt.sha(P / f) != d]
    devbad = [f for f, d in rec["development_results"].items() if pt.sha(P / f) != d]
    check("freeze: plan, theory, development records and code unchanged", not bad and not devbad,
          rec["frozen_at"] + (f"; changed {bad + devbad}" if bad or devbad else ""))
    drec = json.loads((RES / "data_freeze.json").read_text())
    check("data record written after the plan's freeze and naming it",
          drec["written_at"] > rec["frozen_at"] and drec["plan_freeze"] == pt.sha(RES / "freeze.json")
          and drec["code"]["pextract.py"] == pt.sha(P / "pextract.py")
          and drec["code"]["DEVIATIONS.md"] == pt.sha(P / "DEVIATIONS.md"), drec["written_at"])
    for n in prun.NEW:
        mp = RES / f"{n}_manifest.json"
        if not mp.exists():
            check(f"{n}: manifest present", False)
            continue
        man = json.loads(mp.read_text())
        present = [k for k, fn in (("estimates", f"{n}_estimates.npz"), ("msq", f"{n}_msq.json"),
                                   ("predictions", f"{n}_predictions.json")) if (RES / fn).exists()]
        match = all(pt.sha(RES / {"estimates": f"{n}_estimates.npz", "msq": f"{n}_msq.json",
                                  "predictions": f"{n}_predictions.json"}[k]) == man[k] for k in present)
        ok = (man["written_at"] > drec["written_at"] and man["freeze"] == pt.sha(RES / "freeze.json")
              and man["data"] == drec["data"][n] and match and (RES / f"{n}.json").exists())
        check(f"{n}: predictions hashed after the data record and before scoring; files match", ok,
              f"{man['written_at']}; present: {', '.join(present)}")
    last_manifest = max(json.loads((RES / f"{n}_manifest.json").read_text())["written_at"] for n in prun.NEW)
    first_eval = min(file_clock(RES / f"{n}.json") for n in prun.NEW)
    check("every data set's predictions were hashed before the first evaluation",
          wall_clock(last_manifest) < first_eval,
          f"last manifest {last_manifest}; first evaluation {first_eval:%Y-%m-%dT%H:%M:%S}")
    saved = json.loads((RES / "summary.json").read_text())
    import shutil
    with tempfile.TemporaryDirectory() as tmp:
        for f in ["freeze.json"] + [f"{n}{s}" for n in prun.NEW for s in (".json", "_predictions.json")]:
            if (RES / f).exists():
                shutil.copy(RES / f, Path(tmp) / f)
        code = ("import sys; sys.path.insert(0, %r); import prun; from pathlib import Path; "
                "prun.RES = Path(%r); prun.summary()" % (str(P), tmp))
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        ok = r.returncode == 0 and first_difference(json.loads((Path(tmp) / "summary.json").read_text()),
                                                    saved) is None
    check("summary and verdicts recomputed from per-data-set results", ok,
          json.dumps(saved["verdicts"]) if ok else r.stderr[-300:])
    th = json.loads((RES / "check_theory.json").read_text())
    check("theory checks (Propositions S9-S11, stored under their development labels S13-S15: bound in every "
          "simulated case; bounds and limits of the law; pilot identity)",
          th["S13_summary"]["all_hold"] and th["S14"]["bounds_hold"] and th["S15"]["rel_diff"] < 0.01,
          f"S13 largest gap/bound {th['S13_summary']['max_gap_over_bound']:.2f}")
    with tempfile.TemporaryDirectory() as tmp:
        r = subprocess.run([sys.executable, "make_assets.py", "--out", tmp], cwd=P, capture_output=True, text=True)
        names = ["pred_macros.tex", "pred_table.tex"]
        diff = [n for n in names if r.returncode or (Path(tmp) / n).read_bytes() != (SOURCE / n).read_bytes()]
        check("assets: regenerated pred_*.tex equal revision/source", not diff, ", ".join(diff) or r.stderr[-200:])


def with_data():
    import numpy as np
    import ptools as pt
    import prun
    drec = json.loads((RES / "data_freeze.json").read_text())
    bad = [n for n, d in drec["data"].items() if pt.sha(prun.DATA / f"{n}.npz") != d]
    check("data: extracted files match the data record", not bad, ", ".join(bad))
    for n in ("pbmc10k", "fafb_vpn"):
        if not (RES / f"{n}_estimates.npz").exists():
            check(f"{n}: re-scored from hashed estimates", False, "estimates not in this package")
            continue
        stored = json.loads((RES / f"{n}.json").read_text())
        z = np.load(RES / f"{n}_estimates.npz")
        msq = json.loads((RES / f"{n}_msq.json").read_text())
        pr = json.loads((RES / f"{n}_predictions.json").read_text())
        d = prun.dataset(n)
        worst = 0.0
        for tag, Pb in prun.problems(d):
            Pb.score_setup()
            for k in ("bjs2", "js", "random"):
                for i, B in enumerate(pr[tag]["budgets"]):
                    v = (2 * float(np.sum(z[f"{tag}/{k}/{B}"] * Pb.T)) - msq[f"{tag}/{k}/{B}"]) / Pb.den
                    worst = max(worst, abs(v - stored["problems"][tag]["curves"][k][i]))
        check(f"{n}: re-scored from hashed estimates", worst < 1e-9, f"largest difference {worst:.1e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", action="store_true")
    a = ap.parse_args()
    results_only()
    if a.data:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            with_data()
    print(f"\n{sum(checks)} of {len(checks)} checks passed")
    sys.exit(0 if all(checks) else 1)
