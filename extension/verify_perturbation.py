#!/usr/bin/env python3
"""Checks of the perturbation analyses: perturbation/, perturbation_replication/ and hybrid/.

    python verify_perturbation.py

For each sealed analysis: the freeze record matches the plan and code, the
result files match the manifest written before unsealing, and the prespecified
verdicts follow from the stored intervals. For the program and design analyses:
the post hoc freeze matches. For the hybrid development comparison: the plan
records and the decision rule recomputed from the per-group results. Finally,
the manuscript assets regenerated from the results equal those in
revision/source. Reads results only; raw counts are not needed. The asset
scripts also rewrite their RESULTS.md files, with identical content.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "revision" / "source"
checks = []


def check(name, ok, detail=""):
    checks.append({"check": name, "ok": bool(ok), "detail": detail})
    print(f"[{'ok' if ok else 'FAIL'}] {name}{': ' + detail if detail else ''}", flush=True)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest(folder):
    m = json.loads((folder / "results/manifest.json").read_text())
    absent = [k for k in m if k not in ("frozen_at", "note") and not (folder / "results" / k).exists()]
    for k in absent:
        print(f"[skip] {folder.name}/results/{k} not in this package; its SHA-256 is in results/manifest.json")
    bad = [k for k, v in m.items() if k not in ("frozen_at", "note") and k not in absent
           and sha(folder / "results" / k) != v]
    check(f"{folder.name}: result files match the manifest written before unsealing", not bad, ", ".join(bad))


def perturbation():
    folder = HERE / "perturbation"
    fr = module(folder / "freeze.py", "freeze_perturbation")
    try:
        rec = fr.check_freeze()
        check("perturbation: plan, code and fits match the freeze", True, rec["frozen_at"])
    except AssertionError as e:
        check("perturbation: plan, code and fits match the freeze", False, str(e))
    manifest(folder)
    ph = json.loads((folder / "results/freeze_posthoc.json").read_text())
    bad = [f for f, d in ph["files"].items() if (folder / f).exists() and sha(folder / f) != d]
    check("perturbation: program and design analyses match their post hoc freeze", not bad,
          ph.get("frozen_at", "") + (" changed: " + ", ".join(bad) if bad else ""))
    ev = json.loads((folder / "results/evaluation.json").read_text())
    e = ev["heldout"]["estimates"]
    v = {"H1_recovery": e["pf_within_measured"]["ci"][0] > 0,
         "H2_beats_pseudo_populations": e["primary_minus_within_pseudo"]["ci"][0] > 0,
         "H2_beats_derangements": e["primary_minus_within_derange"]["ci"][0] > 0,
         "H3_nontargeting": ev["nt"]["pooled_ci"]["pf_within_measured"][0] > 0,
         "H4_most_of_paired": e["share_of_paired"]["ci"][0] > 0.5}
    check("perturbation: verdicts H1-H4 recomputed", v == ev["verdict"], json.dumps(v))
    pr = json.loads((folder / "results/programs.json").read_text())
    i = pr["heldout"]["intervals"]
    v = {"P1_ifn_share_above_half": i["ifn/share_paired"]["ci"][0] > 0.5,
         "P2_along_S_share_above_half": i["along_S/share_paired"]["ci"][0] > 0.5,
         "P3_ifn_above_complement": i["ifn_minus_complement/pf"]["ci"][0] > 0}
    check("perturbation: program verdicts P1-P3 recomputed", v == pr["heldout"]["verdict"], json.dumps(v))
    de = json.loads((folder / "results/design.json").read_text())
    v = {}
    for k in ("5", "10"):
        a, f = de["ablation"][k]["all"], de["ablation"][k]["ifn"]
        v[f"D1_remove{k}"] = a["random_minus_exposure"]["ci"][0] > 0 and a["remove_exposure"] < a["remove_random_q05"]
        v[f"D1_ifn_remove{k}"] = (f["random_minus_exposure"]["ci"][0] > 0
                                  and f["remove_exposure"] < f["remove_random_q05"])
    check("perturbation: design verdicts D1 recomputed", v == de["verdict"], json.dumps(v))


def replication():
    folder = HERE / "perturbation_replication"
    fr = module(folder / "freeze.py", "freeze_replication")
    try:
        rec = fr.check_freeze()
        check("replication: plan and code match the freeze", True, rec["frozen_at"])
    except AssertionError as e:
        check("replication: plan and code match the freeze", False, str(e))
    manifest(folder)
    ev = json.loads((folder / "results/evaluation.json").read_text())
    e, n = ev["test"]["intervals"], ev["nt"]["intervals"]
    v = {"R1_recovery": e["primary"]["ci"][0] > 0,
         "R2_beats_pseudo_populations": e["primary_minus_pseudo"]["ci"][0] > 0,
         "R2_beats_derangements": e["primary_minus_derange"]["ci"][0] > 0,
         "R3_ifn_share_above_half": e["ifn_share_paired"]["ci"][0] > 0.5,
         "R4_nontargeting": n["primary"]["ci"][0] > 0,
         "R5_exposure_targets_matter": e["random_minus_exposure_removal"]["ci"][0] > 0}
    check("replication: verdicts R1-R5 recomputed", v == ev["verdict"], json.dumps(v))


def hybrid():
    folder = HERE / "hybrid"
    orig = json.loads((folder / "results/plan.json").read_text())
    amended = json.loads((folder / "results/plan_amended.json").read_text())
    now = sha(folder / "PLAN.md")
    check("hybrid: PLAN.md matches its amended record (original record kept)", now == amended["PLAN.md"],
          f"original {orig['recorded_at']}, amended {amended['recorded_at']}")
    sys.path.insert(0, str(folder))
    summ = module(folder / "summarize.py", "summarize_hybrid")
    stored = json.loads((folder / "results/summary.json").read_text())
    ok, details = True, []
    for ds in ("frangieh", "papalexi"):
        s = summ.analyse(summ.load(ds))["parts"]["test"]
        for b, c in s["criteria"].items():
            st = stored[ds]["parts"]["test"]["criteria"][str(b)]
            if abs(c["diff"] - st["diff"]) > 1e-12 or c["best"] != st["best"]:
                ok = False
                details.append(f"{ds} {b}")
        details.append(f"{ds}: criteria {s['pass1']}/{s['pass2']}")
    decision = all(stored[ds]["parts"]["test"]["pass1"] and stored[ds]["parts"]["test"]["pass2"]
                   for ds in ("frangieh", "papalexi"))
    check("hybrid: decision rule recomputed from per-group results", ok and decision == stored["decision"],
          "; ".join(details) + f"; decision {'met' if decision else 'not met'}")


def assets():
    with tempfile.TemporaryDirectory() as tmp:
        for folder, names in (("perturbation", ("pert_macros.tex", "pert_table.tex", "pert_program_table.tex",
                                                 "pert_figure.tex")),
                              ("perturbation_replication", ("rep_macros.tex", "rep_table.tex"))):
            out = Path(tmp) / folder
            r = subprocess.run([sys.executable, "make_assets.py", "--out", str(out)], cwd=HERE / folder,
                               capture_output=True, text=True)
            if r.returncode:
                check(f"{folder}: assets regenerate", False, r.stderr[-300:])
                continue
            diff = [n for n in names if (out / n).read_bytes() != (SOURCE / n).read_bytes()]
            check(f"{folder}: regenerated assets equal revision/source", not diff, ", ".join(diff))


def main():
    perturbation()
    replication()
    hybrid()
    assets()
    failed = [c for c in checks if not c["ok"]]
    print(f"\n{len(checks) - len(failed)} of {len(checks)} checks passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
