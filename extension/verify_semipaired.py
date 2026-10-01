#!/usr/bin/env python3
"""Checks of the estimator's development and external test: semipaired/ and semipaired/external/.

    python verify_semipaired.py

External test: the freeze record matches the plan, code and fixed inputs; the prediction files match the
manifest written before the held-out files were opened, and the manifest postdates the freeze; the verdicts
follow from the stored intervals; the provenance of the plan's estimator (deviation 1) is intact. Theory: the
stored numerical checks of the risk bound cover the implemented estimator. Development: the savings and
differences in results/summary_<dataset>.json are recomputed from the per-group results. Finally, the manuscript
assets regenerated from the results equal those in revision/source. Reads results only; raw counts are not needed.
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
SP = HERE / "semipaired"
EXT = SP / "external"
SOURCE = HERE.parent / "revision" / "source"
checks = []


def check(name, ok, detail=""):
    checks.append({"check": name, "ok": bool(ok), "detail": detail})
    print(f"[{'ok' if ok else 'FAIL'}] {name}{': ' + detail if detail else ''}", flush=True)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def external():
    freeze = json.loads((EXT / "results/freeze.json").read_text())
    bad = [f for f, d in freeze["files"].items() if sha(EXT / f) != d]
    check("external: plan, code and fixed inputs match the freeze", not bad,
          freeze["frozen_at"] + (" changed: " + ", ".join(bad) if bad else ""))
    man = json.loads((EXT / "results/manifest.json").read_text())
    present = [f for f in ("predictions.npz", "predict_info.json") if (EXT / "results" / f).exists()]
    absent = sorted({"predictions.npz", "predict_info.json"} - set(present))
    bad = [f for f in present if sha(EXT / "results" / f) != man[f]]
    check("external: predictions match the manifest written before unsealing", not bad and
          man["freeze.json"] == sha(EXT / "results/freeze.json"),
          man.get("written_at", "")
          + (f"; absent from this package, not checked: {', '.join(absent)}" if absent else ""))
    check("external: manifest written after the freeze", man["written_at"] > freeze["frozen_at"],
          f"{freeze['frozen_at']} < {man['written_at']}")
    for name, label in (("overcite_plan_estimator.json", "plan's estimator bjs2/mapped"),
                        ("overcite.json", "frozen script's arm bjs/mapped (deviation 1)")):
        o = json.loads((EXT / "results" / name).read_text())
        t = str(o["target"])
        s = o["savings"]
        reached = s["paired_only"][t]["proposed_relation"] != ">"
        v = {"proposed_reaches_target": reached,
             "H1_vs_paired_only": reached and s["paired_only"][t]["ci"][0] > 1,
             "H2_vs_semicca": reached and s["semicca"][t]["ci"][0] > 1,
             "H3_noninferior_to_champollion": reached and s["champollion"][t]["ci"][0] > 0.8}
        check(f"external ({label}): verdicts recomputed from the stored intervals", v == o["verdicts"],
              json.dumps(v))
    plan = json.loads((EXT / "results/overcite_plan_estimator.json").read_text())
    frozen = json.loads((EXT / "results/overcite.json").read_text())
    same = plan["rf"] == frozen["rf"] and plan["loss_rel"] == frozen["loss_rel"]
    check("external: both scorings share every arm's recovered fraction and loss (only the proposed arm differs)",
          same and plan.get("proposed") == "bjs2/mapped")


def provenance():
    """Deviation 1: the plan names the two-sided estimator; its predictions were hashed before unsealing and are
    reproduced exactly by the frozen code (provenance_check.py, which needs the training file); both scoring
    outputs and the explanation are kept unchanged since the provenance record was written."""
    plan = (EXT / "PLAN.md").read_text()
    names = "Proposed estimator (`bjs2/mapped`)" in plan and "bjs/mapped" not in plan.replace("bjs2/mapped", "")
    check("provenance: the frozen plan names bjs2/mapped as the proposed estimator and never bjs/mapped", names)
    p = EXT / "results/provenance.json"
    if not p.exists():
        check("provenance: record present", False, "run external/provenance_check.py")
        return
    rec = json.loads(p.read_text())
    man = json.loads((EXT / "results/manifest.json").read_text())
    ok = rec["predictions_sha256"] == man["predictions.npz"] and rec["plan_sha256"] == sha(EXT / "PLAN.md")
    check("provenance: record matches the manifest and the frozen plan", ok, rec["manifest_written_at"])
    diff = rec["max_abs_difference_recomputed_vs_stored"]
    check("provenance: recomputed predictions equal the stored ones for both arms",
          all(v == 0.0 for v in diff.values()) and rec["arrays_per_arm"]["bjs2/mapped"] > 0,
          json.dumps(diff) + f"; {rec['arrays_per_arm']['bjs2/mapped']} arrays per arm")
    outs = rec["scoring_outputs"]
    kept = (outs["frozen_script (proposed = bjs/mapped)"] == sha(EXT / "results/overcite.json")
            and outs["plan_estimator (proposed = bjs2/mapped)"] == sha(EXT / "results/overcite_plan_estimator.json")
            and outs["DEVIATIONS.md"] == sha(EXT / "DEVIATIONS.md"))
    check("provenance: original output, corrected output and explanation unchanged since the record", kept)
    t = rec["file_modification_times"]
    order = ["PLAN.md", "results/freeze.json", "results/manifest.json", "results/overcite.json",
             "results/overcite_plan_estimator.json", "DEVIATIONS.md"]
    check("provenance: recorded order freeze < manifest < original scoring < corrected scoring < explanation",
          all(t[a] <= t[b] for a, b in zip(order[1:], order[2:])) and t["PLAN.md"] <= t["results/freeze.json"],
          ", ".join(t[f][11:19] for f in order))


def bound():
    """The risk bound of Proposition S6 (Supplementary Note 5) against the implemented estimator
    (semipaired/bound_check.py, written when it was numbered S10)."""
    for ds in ("overcite", "frangieh", "papalexi"):
        p = SP / "logs" / f"bound_check_{ds}.json"
        if not p.exists():
            check(f"bound ({ds}): check present", False)
            continue
        r = json.loads(p.read_text())
        ok = all(v["risk_implemented"] <= v["bound"] for v in r["budgets"].values()) and r["condition_tau_ge_4rho"]
        worst = min(v["bound"] / v["risk_implemented"] for v in r["budgets"].values())
        check(f"bound ({ds}): implemented estimator below the bound at every budget; tau >= 4 rho", ok,
              f"smallest bound/risk {worst:.2f}; budgets {', '.join(r['budgets'])}")


def development():
    sys.path.insert(0, str(SP))
    spec = importlib.util.spec_from_file_location("summarize_semipaired", SP / "summarize.py")
    summ = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(summ)
    for ds in ("frangieh", "papalexi"):
        res = summ.load(ds)
        arms = sorted({a for r in res["rows"] for a in r["num"]})
        comps = [a for a in arms if not a.startswith(("bjs", "abl/", "dose/")) and a not in ("paired_all",
                                                                                             "independence")]
        s = summ.analyse(res, comps)
        stored = json.loads((SP / "results" / f"summary_{ds}.json").read_text())
        ok = all(abs(s[k][t]["factor"] - stored[k][t]["factor"]) < 1e-9
                 for k in ("savings_vs_paired_only", "savings_vs_published", "savings_vs_best") for t in s[k])
        ok &= all(abs(a["diff"] - b["diff"]) < 1e-12 for a, b in zip(s["versus_best"], stored["versus_best"]))
        check(f"development ({ds}): savings and differences recomputed from per-group results", ok)


def assets():
    with tempfile.TemporaryDirectory() as tmp:
        r = subprocess.run([sys.executable, "make_assets.py", "--out", tmp], cwd=SP, capture_output=True, text=True)
        if r.returncode:
            check("semipaired: assets regenerate", False, r.stderr[-300:])
            return
        names = ["semi_macros.tex", "semi_figure.tex", "semi_dev_table.tex", "semi_table.tex"]
        diff = [n for n in names if (Path(tmp) / n).read_bytes() != (SOURCE / n).read_bytes()]
        check("semipaired: regenerated assets equal revision/source", not diff, ", ".join(diff))


def main():
    external()
    provenance()
    bound()
    development()
    assets()
    failed = [c for c in checks if not c["ok"]]
    print(f"\n{len(checks) - len(failed)} of {len(checks)} checks passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
