#!/usr/bin/env python3
"""Checks of the validation test (extension/validation).

    python verify_validation.py          # freeze, order of the steps, manifest and verdicts (results only)
    python verify_validation.py --data   # also re-scores the test from the hashed predictions and compares the result
                                         # with results/summary.json (needs predictions.npz and the extracted data)

Results-only checks read the JSON records and the hashes of the frozen code; the --data check recomputes
results/summary.json in a temporary folder from the hashed predictions and the held-out cells, and compares it with
the stored one.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
VAL = HERE / "validation"
DEP = HERE / "deployment"


def check(name, cond, detail=""):
    print(f"[{'ok' if cond else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))
    return bool(cond)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


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
    freeze = json.loads((VAL / "results" / "freeze.json").read_text())
    for rel, digest in freeze["files"].items():
        ok &= check(f"frozen file {rel}", sha(HERE / rel) == digest)
    for name, (path, digest) in freeze["data"].items():
        if Path(path).exists():
            ok &= check(f"frozen data {name}", sha(path) == digest)
        elif data:
            ok &= check(f"frozen data {name}", False, f"{path} missing")
        else:
            print(f"[skip] data file {Path(path).name} not in this package (raw counts are not redistributed)")
    for name, rel in (("smoke", "results_smoke/summary.json"), ("pools_check", "results_check/pools_check.json")):
        ok &= check(f"check before the freeze ({name}) unchanged", sha(VAL / rel) == freeze["checks"][name])
    man = json.loads((VAL / "results" / "manifest.json").read_text())
    ok &= check("predictions hashed after the freeze", man["written"] > freeze["frozen"],
                f"{freeze['frozen']} < {man['written']}")
    ok &= check("real data, not the synthetic smoke run", man["synthetic"] is False)
    for key, f in (("info", "predictions_info.json"), ("audit", "audit.json")):
        ok &= check(f"{f} matches the manifest", sha(VAL / "results" / f) == man[key])
    pred = VAL / "results" / "predictions.npz"
    if pred.exists():
        ok &= check("predictions.npz matches the manifest", sha(pred) == man["predictions"])
    else:
        print("[skip] predictions.npz not in this package (948 MB); its hash is in results/manifest.json")
    s = json.loads((VAL / "results" / "summary.json").read_text())
    ok &= check("scored after the predictions were hashed", s["evaluated"] > man["written"],
                f"{man['written']} < {s['evaluated']}")
    h = s["hypotheses"]
    ok &= check("verdicts V1-V3 met, V4 not met", h == {"V1": True, "V2": True, "V3": True, "V4": False}, str(h))
    p = s["targets"]["0.5"]
    for name, bound, side in (("V1_atlas_vs_paired_only", 1, "lower"), ("V3_other_vs_paired_only", 1, "lower"),
                              ("V4_atlas_over_own", 1.25, "upper")):
        r = p[name]
        detail = f"{r['ratio']:.2f} ({r['ci'][0]:.2f}-{r['ci'][1]:.2f})"
        if side == "lower":
            ok &= check(f"{name.split('_')[0]} lower bound above {bound}", r["ci"][0] > bound, detail)
        else:
            ok &= check(f"{name.split('_')[0]} upper bound not below {bound} (not met)", r["ci"][1] >= bound, detail)
    b = s["budgets"]
    lo50, lo100 = s["rf_ci"]["atlas"][b.index(50)][0], s["rf_ci"]["atlas"][b.index(100)][0]
    ok &= check("V2 lower bounds above zero at 50 and 100 paired cells", lo50 > 0 and lo100 > 0,
                f"{100 * lo50:.1f}%, {100 * lo100:.1f}%")
    ok &= check("paired-only estimation censored at 800 (saving is a lower bound)",
                p["cells"]["paired_only"]["relation"] == ">")
    cal = s["audit"]["calibration"]
    rc = [v["ratio_estimated_to_actual"] for v in cal["contrasts"].values()]
    ok &= check("noise term calibrated under contrasts (25-200 cells)", all(abs(r - 1) < 0.02 for r in rc),
                ", ".join(f"{r:.3f}" for r in rc))
    cbm = json.loads((DEP / "results_posthoc" / "calibration.json").read_text())
    dsum = json.loads((DEP / "results" / "summary.json").read_text())
    ok &= check("bone-marrow calibration written after that test was scored", cbm["written"] > dsum["evaluated"],
                f"{dsum['evaluated']} < {cbm['written']}")
    # biological readout: specified after scoring, frozen before its endpoints were computed
    rz = json.loads((VAL / "readout" / "results" / "freeze.json").read_text())
    for rel, digest in rz["files"].items():
        ok &= check(f"readout: frozen file {rel}", sha(HERE / rel) == digest)
    for name, (path, digest) in rz["data"].items():
        if Path(path).exists():
            ok &= check(f"readout: frozen input {name}", sha(path) == digest)
        else:
            print(f"[skip] readout input {Path(path).name} not in this package; its hash is in "
                  "readout/results/freeze.json")
    ok &= check("readout: smoke run unchanged",
                sha(VAL / "readout" / "results_smoke" / "readout.json") == rz["checks"]["smoke"])
    rd = json.loads((VAL / "readout" / "results" / "readout.json").read_text())
    ok &= check("readout: specified after the validation was scored and frozen before it ran",
                s["evaluated"] < rz["frozen"] < rd["written"], f"{s['evaluated']} < {rz['frozen']} < {rd['written']}")
    ok &= check("readout: real data", rd["synthetic"] is False)
    ig = rd["integrity"]
    n_compared = 34 * 6 * 11 * 5      # folds x frozen budgets x (8 paired-only + 3 reference arms) x cell types
    ok &= check("readout: regenerated predictions reproduce the hashed means",
                ig["max_abs_P"] <= 1e-5 and ig["max_rel_msq"] <= 1e-6 and ig["compared"] == n_compared,
                f"largest difference {ig['max_abs_P']:.1e}, {ig['compared']} predictions compared")
    ok &= check("readout: every hypothesis has a verdict", set(rd["hypotheses"]) == {"B1", "B2", "B3"},
                str(rd["hypotheses"]))
    rb = json.loads((VAL / "results_posthoc" / "robustness.json").read_text())
    ok &= check("robustness: written after scoring", rb["written"] > s["evaluated"],
                f"{s['evaluated']} < {rb['written']}")
    ok &= check("robustness: the unchanged atlas reproduces the hashed atlas arm",
                rb["reproduces_hashed_atlas_arm"] <= 1e-5, f"{rb['reproduces_hashed_atlas_arm']:.1e}")
    cm = json.loads((VAL / "results_posthoc" / "composition_map.json").read_text())
    ok &= check("composition diagnostic written after the robustness record", cm["written"] > rb["written"],
                f"{rb['written']} < {cm['written']}")
    # donor separation of the folds (sample labels only)
    al = json.loads((VAL / "results_posthoc" / "allocation.json").read_text())
    ok &= check("allocation: every fold donor-disjoint (held-out donor absent from paired samples and references)",
                al["donor_disjoint"] and all(v for k, v in al["checks"].items() if k != "atlas")
                and len(al["folds"]) == 34, f"{len(al['folds'])} folds, {len(al['donors'])} donors")
    # other methods given the atlas, and calibration of the readout: specified after scoring, frozen before they ran
    RV = VAL / "posthoc_review"
    qz = json.loads((RV / "results" / "freeze.json").read_text())
    for rel, digest in qz["files"].items():
        ok &= check(f"review: frozen file {rel}", sha(HERE / rel) == digest)
    for name, (path, digest) in qz["data"].items():
        if Path(path).exists():
            ok &= check(f"review: frozen input {name}", sha(path) == digest)
        else:
            print(f"[skip] review input {Path(path).name} not in this package; its hash is in "
                  "posthoc_review/results/freeze.json")
    ok &= check("review: smoke run and Champollion timing unchanged",
                sha(RV / "results_smoke" / "review.json") == qz["checks"]["smoke"]
                and sha(RV / "results" / "timing.json") == qz["checks"]["timing"])
    rv = json.loads((RV / "results" / "review.json").read_text())
    ok &= check("review: specified after the readout was written and frozen before it ran",
                rd["written"] < qz["frozen"] < rv["written"], f"{rd['written']} < {qz['frozen']} < {rv['written']}")
    ok &= check("review: real data, all folds", rv["smoke"] is False and rv["folds"] == 34
                and rv.get("champollion", {}).get("folds") == 34)
    ig = rv["integrity"]
    ok &= check("review: regenerated predictions reproduce the hashed means",
                ig["max_abs_P"] <= 1e-5 and ig["max_rel_msq"] <= 1e-6 and ig["compared"] == n_compared,
                f"largest difference {ig['max_abs_P']:.1e}, {ig['compared']} predictions compared")
    rp = rv["reproduction"]
    # the runs used one BLAS thread per process, the readout two: an estimate within rounding of zero can change sign
    ok &= check("review: reproduces the readout and the validation's recovered fractions",
                rp["readout_max_abs_difference"] <= 1e-5 and rp["rf_max_abs_difference"] <= 1e-6,
                f"{rp['readout_max_abs_difference']:.1e}, {rp['rf_max_abs_difference']:.1e}")
    alt = rv["alternatives"]
    for hid, key in (("K1", "K1_semicca_over_atlas"), ("K2", "K2_regression_over_atlas")):
        r = alt["ratios"][key]
        ok &= check(f"review: {hid} verdict follows from its interval", alt[hid] == (r["ci"][0] > 1),
                    f"{r['ratio']:.2f} ({r['ci'][0]:.2f}-{r['ci'][1]:.2f}), {'met' if alt[hid] else 'not met'}")
    if (RV / "results" / "folds").exists() and len(list((RV / "results" / "folds").glob("fold*.npz"))) == 34:
        sys.path.insert(0, str(RV))
        import rrun
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "review.json"
            rrun.score(RV / "results" / "folds", RV / "results" / "champ", out, VAL / "results" / "summary.json",
                       VAL / "readout" / "results" / "readout.json", log=lambda m: None)
            diff = first_difference(strip(json.loads(out.read_text())), strip(rv))
            ok &= check("review: review.json recomputed from the per-fold records", diff is None, diff or "")
    else:
        print("[skip] per-fold records of the review not in this package")
    if data:
        sys.path.insert(0, str(VAL))
        import vrun
        vrun.check_freeze()
        ok &= check("plan, runner, imported modules and data match the freeze (vrun.check_freeze)", True)
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            for f in ("predictions.npz", "predictions_info.json", "audit.json", "manifest.json"):
                (t / f).symlink_to(VAL / "results" / f)
            vrun.evaluate(out_dir=t, log=lambda m: None)
            new = json.loads((t / "summary.json").read_text())
            diff = first_difference(strip(new), strip(s))
            ok &= check("summary.json recomputed from the hashed predictions", diff is None, diff or "")
    print("passed" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
