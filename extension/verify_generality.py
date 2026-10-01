#!/usr/bin/env python3
"""Checks of the generality benchmark (extension/generality).

    python verify_generality.py [--data]

Without --data (results only): the freeze record matches the plan and code; every data set's prediction manifest
was written after the freeze and names the frozen plan; prediction files, where present, match their manifest; the
cross-data-set summary (G1-G3) and the structural law (G4) are recomputed from the per-data-set results; the
manuscript assets regenerate identically. With --data (needs the extracted data files in
/home/claude/cbio/rawdata/generality): the data files match the freeze record and one data set per system is
re-scored from its hashed predictions, reproducing its stored curves.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
G = HERE / "generality"
RES = G / "results"
SOURCE = HERE.parent / "revision" / "source"
DATA = Path("/home/claude/cbio/rawdata/generality")
PRIMARY = ("hao", "stephenson", "bmmc_cite", "colon", "bmmc_multiome", "scala_m1", "gouwens_visp", "banc", "malecns")
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


def sha(path):
    d = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            d.update(b)
    return d.hexdigest()


def results_only():
    rec = json.loads((RES / "freeze.json").read_text())
    bad = [f for f, dg in rec["files"].items() if sha(G / f) != dg]
    check("freeze: plan and code unchanged since the freeze", not bad,
          rec["frozen_at"] + (f"; changed {bad}" if bad else ""))
    for n in PRIMARY + ("banc_crossanimal",):
        mp = RES / f"{n}_manifest.json"
        if not mp.exists():
            check(f"{n}: manifest present", False)
            continue
        man = json.loads(mp.read_text())
        ok = (man["written_at"] > rec["frozen_at"] and man["plan"] == rec["files"]["PLAN.md"]
              and man["freeze"] == sha(RES / "freeze.json"))
        files = {"predictions": f"{n}_predictions.npz", "msq": f"{n}_msq.json", "info": f"{n}_info.json"}
        present = [k for k, fn in files.items() if (RES / fn).exists()]
        match = all(sha(RES / files[k]) == man[k] for k in present)
        res_after = (RES / f"{n}.json").exists()
        check(f"{n}: predictions hashed after the freeze, before scoring; files match the manifest",
              ok and match and res_after, f"{man['written_at']}; present: {', '.join(present) or 'none'}")
    # G1-G3 recomputed from the per-data-set results
    summ = json.loads((RES / "summary.json").read_text())
    res = {n: json.loads((RES / f"{n}.json").read_text()) for n in PRIMARY}
    ok = True
    for label, key in (("G1_paired_only", "paired_only"), ("G2_semicca", "semicca")):
        logs = np.array([np.log(res[n]["savings"][key]["0.5"]["factor"]) for n in PRIMARY])
        boot = np.mean([res[n]["savings"][key]["0.5"]["log_boot"] for n in PRIMARY], axis=0)
        gm, ci = float(np.exp(logs.mean())), np.exp(np.quantile(boot, [.025, .975]))
        s = summ[f"{label}/0.5"]
        ok &= abs(gm - s["geometric_mean"]) < 1e-9 and np.allclose(ci, s["ci"], rtol=1e-9)
    g1 = summ["G1_paired_only/0.5"]
    check("summary: G1 and G2 recomputed from per-data-set results", ok,
          f"G1 {g1['geometric_mean']:.2f} ({g1['ci'][0]:.2f}-{g1['ci'][1]:.2f}); "
          f"verdicts {json.dumps(summ['verdicts'])}")
    law = json.loads((RES / "law.json").read_text())
    names = law["datasets"]
    rho = spearmanr([law["predicted"][n] for n in names], [law["observed"][n] for n in names]).correlation
    obs_ok = all(abs(law["observed"][n] - res[n]["savings"]["paired_only"]["0.5"]["factor"]) < 1e-12
                 for n in PRIMARY if n in law["observed"])
    check("law: Spearman correlation recomputed; observed savings equal the per-data-set results",
          abs(rho - law["spearman"]) < 1e-12 and obs_ok, f"rho {law['spearman']:.2f}, p {law['p_one_sided']:.4f}")
    # amendments 1-2: recorded before the data sets they apply to prospectively were scored; summaries recomputed
    rec_a = json.loads((RES / "amend1_freeze.json").read_text())
    ok = rec_a["amendment_2"]["grun_amend.py"] == sha(G / "grun_amend.py")
    later = [n for n in PRIMARY if n not in rec_a["amendment_2"]["scored_before"]]
    scored = {n: file_clock(RES / f"{n}.json") for n in PRIMARY}
    t2 = wall_clock(rec_a["amendment_2"]["written_at"])
    ok &= all(scored[n] > t2 for n in later)
    check("amendments: script unchanged since amendment 2; the other data sets were scored after it", ok,
          f"amendment 2 at {rec_a['amendment_2']['written_at']}; later: {', '.join(later)}")
    sa = json.loads((RES / "summary_amend1.json").read_text())
    am = {n: json.loads((RES / f"{n}_amend1.json").read_text()) for n in PRIMARY}
    inf = [n for n in PRIMARY if am[n]["level_ci"][0] > 0]
    ok = inf == sa["informative"] and all(am[n]["frozen_rule_reproduced_max_abs_difference"] < 1e-9 for n in PRIMARY)
    for scope, names in (("informative", inf), ("all", list(PRIMARY))):
        logs = np.array([np.log(am[n]["savings"]["paired_only"]["0.5"]["factor"]) for n in names])
        boot = np.mean([am[n]["savings"]["paired_only"]["0.5"]["log_boot"] for n in names], axis=0)
        v = sa[f"{scope}/G1_paired_only/0.5"]
        ok &= (abs(np.exp(logs.mean()) - v["geometric_mean"]) < 1e-9
               and np.allclose(np.exp(np.quantile(boot, [.025, .975])), v["ci"]))
    g = sa["informative/G1_paired_only/0.5"]
    check("amended summary: G1' recomputed; every amended run reproduced the frozen savings exactly", ok,
          f"G1' {g['geometric_mean']:.2f} ({g['ci'][0]:.2f}-{g['ci'][1]:.2f}) over {len(inf)} informative data sets")
    with tempfile.TemporaryDirectory() as tmp:
        r = subprocess.run([sys.executable, "make_assets.py", "--out", tmp, "--results-md", str(Path(tmp) / "R.md")],
                           cwd=G, capture_output=True, text=True)
        names = ["gen_macros.tex", "gen_table.tex", "gen_figure.tex"]
        diff = [n for n in names if r.returncode or (Path(tmp) / n).read_bytes() != (SOURCE / n).read_bytes()]
        check("assets: regenerated gen_*.tex equal revision/source", not diff, ", ".join(diff) or r.stderr[-200:])


def with_data():
    rec = json.loads((RES / "freeze.json").read_text())
    bad = [n for n, dg in rec["data"].items() if not (DATA / f"{n}.npz").exists() or sha(DATA / f"{n}.npz") != dg]
    check("data: extracted files match the freeze record", not bad, ", ".join(bad))
    sys.path.insert(0, str(G))
    import grun as gr
    for n in ("colon", "scala_m1", "banc"):
        stored = json.loads((RES / f"{n}.json").read_text())
        man = json.loads((RES / f"{n}_manifest.json").read_text())
        if not (RES / f"{n}_predictions.npz").exists():
            check(f"{n}: re-scored from hashed predictions", False, "predictions not in this package")
            continue
        z = np.load(RES / f"{n}_predictions.npz")
        store = {k: z[k] for k in z.files}
        msq = {int(f): m for f, m in json.loads((RES / f"{n}_msq.json").read_text()).items()}
        infos = {int(f): i for f, i in json.loads((RES / f"{n}_info.json").read_text()).items()}
        d = gr.load(n)
        fold, part, half = gr.roles(n, d)
        stats = {f: gr.heldout_stats(d, f, fold, half, np.array(infos[f]["x_panel_index"]),
                                     np.array(infos[f]["y_panel_index"]), infos[f]["conditions"])
                 for f in range(gr.FOLDS)}
        curves, ref, common = gr.score(n, stats, store, msq, {f: infos[f]["budgets"] for f in range(gr.FOLDS)})
        diff = max(abs(float(v[0][i]) - stored["rf"][a][i]) for a, v in curves.items() for i in range(len(common)))
        check(f"{n}: re-scored from hashed predictions ({man['written_at']})", diff < 1e-9,
              f"largest difference {diff:.1e}")


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
