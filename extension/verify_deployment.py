#!/usr/bin/env python3
"""Checks of the deployment test (extension/deployment) and of its post hoc analyses.

    python verify_deployment.py          # freeze, manifest, verdicts and post hoc records (results only)
    python verify_deployment.py --data   # also re-scores the test and the contrast reanalysis from the stored
                                         # predictions (needs the prediction files and the extracted data)

Results-only checks read the JSON records; the --data checks recompute results/summary.json and
results_posthoc/contrasts_summary.json in a temporary folder and compare them with the stored ones.
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEP = HERE / "deployment"
sys.path.insert(0, str(DEP))


def check(name, cond, detail=""):
    print(f"[{'ok' if cond else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))
    return bool(cond)


def strip(d):
    return {k: v for k, v in d.items() if k not in ("evaluated", "written")}


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


def main():
    data = "--data" in sys.argv
    ok = True
    freeze = json.loads((DEP / "results" / "freeze.json").read_text())
    import drun
    try:
        drun.check_freeze()
        ok &= check("plan, runner, imported modules and data match the freeze", True, freeze["frozen"])
    except (AssertionError, FileNotFoundError) as e:
        if isinstance(e, FileNotFoundError) and not data:
            print(f"[skip] data files not in this package ({Path(e.filename).name}); code hashes checked below")
            for rel, digest in freeze["files"].items():
                ok &= check(f"frozen file {rel}", drun.sha(HERE / rel) == digest)
        else:
            ok &= check("freeze", False, str(e))
    man = json.loads((DEP / "results" / "manifest.json").read_text())
    ok &= check("predictions hashed after the freeze", man["written"] > freeze["frozen"],
                f"{freeze['frozen']} < {man['written']}")
    pred = DEP / "results" / "predictions.npz"
    if pred.exists():
        ok &= check("predictions match the manifest", drun.sha(pred) == man["predictions"])
    else:
        print("[skip] predictions.npz not in this package (640 MB); its hash is in results/manifest.json")
    s = json.loads((DEP / "results" / "summary.json").read_text())
    h = s["hypotheses"]
    ok &= check("verdicts D1 met, D2 not met, D3 met", h == {"D1": True, "D2": False, "D3": True}, str(h))
    p = s["targets"]["0.5"]
    ok &= check("D1 lower bound above one", p["D1_other_vs_paired_only"]["ci"][0] > 1,
                f"{p['D1_other_vs_paired_only']['ratio']:.2f} ({p['D1_other_vs_paired_only']['ci'][0]:.2f}-"
                f"{p['D1_other_vs_paired_only']['ci'][1]:.2f})")
    ok &= check("D3 upper bound below 1.25", p["D3_other_over_same"]["ci"][1] < 1.25,
                f"{p['D3_other_over_same']['ratio']:.2f} ({p['D3_other_over_same']['ci'][0]:.2f}-"
                f"{p['D3_other_over_same']['ci'][1]:.2f})")
    post = DEP / "results_posthoc"
    for f in ("contrasts_summary.json", "map_by_source.json", "law_by_source.json", "fly_contrasts.json",
              "fly_own_standardization.json", "assay_structure.json", "population_sizes.json", "calibration.json"):
        ok &= check(f"post hoc record {f}", (post / f).exists())
    con = json.loads((post / "contrasts_summary.json").read_text())
    ok &= check("contrast reanalysis written after the test was scored", con["evaluated"] > s["evaluated"],
                f"{s['evaluated']} < {con['evaluated']}")
    cal = json.loads((post / "calibration.json").read_text())
    ok &= check("calibration of the noise term written after the test was scored", cal["written"] > s["evaluated"],
                f"{s['evaluated']} < {cal['written']}")
    rc = [v["ratio_estimated_to_actual"] for v in cal["calibration"]["contrasts"].values()]
    rd = [v["ratio_estimated_to_actual"] for v in cal["calibration"]["dfcentre"].values()]
    ok &= check("noise term calibrated under contrasts, too small under the test's own centring",
                all(abs(r - 1) < 0.02 for r in rc) and all(r < 0.95 for r in rd),
                "contrasts " + ", ".join(f"{r:.3f}" for r in rc)
                + "; own centring " + ", ".join(f"{r:.3f}" for r in rd))
    if data:
        import posthoc_contrasts as pc
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            for f in ("predictions.npz", "predictions_info.json", "manifest.json"):
                (t / f).symlink_to(DEP / "results" / f)
            drun.evaluate(out_dir=t, log=lambda m: None)
            new = json.loads((t / "summary.json").read_text())
            diff = first_difference(strip(new), strip(s))
            ok &= check("summary.json recomputed from the hashed predictions", diff is None, diff or "")
            pc.evaluate(log=lambda m: None, out_path=t / "contrasts_summary.json")
            new = json.loads((t / "contrasts_summary.json").read_text())
            diff = first_difference(strip(new), strip(con))
            ok &= check("contrasts_summary.json recomputed from the stored contrast predictions", diff is None,
                        diff or "")
    print("passed" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
