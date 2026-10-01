"""Dry run of the replication's predict.run and evaluate on training halves only (no test cells)."""
import sys, json, types
sys.path.insert(0, "/home/claude/cbio/cbio-agent-handoff/extension/perturbation_replication")
sys.path.append("/home/claude/cbio/cbio-agent-handoff/extension/perturbation")
import numpy as np
import rdata
real = rdata.load
tr = real("train")
# pretend: targets ATF2 and CD86 are "test": adaptation = their part-0 cells, scoring = part-1 cells (A/B alternate)
tt = np.isin(tr["target"], ["ATF2", "CD86"])
def sub(mask, **over):
    out = {k: (v[mask] if isinstance(v, np.ndarray) and len(v) == len(mask) else v) for k, v in tr.items()}
    out.update(over); return out
ad = sub(tt & (tr["part"] == 0)); sc = sub(tt & (tr["part"] == 1))
sc["part"] = np.arange(len(sc["cell"])) % 2
nt = np.isin(tr["target"], ["JAK2"])
ntad = sub(nt & (tr["part"] == 0)); ntad["target"] = np.array(["NT"] * len(ntad["cell"]))
ntsc = sub(nt & (tr["part"] == 1)); ntsc["target"] = np.array(["NT"] * len(ntsc["cell"])); ntsc["part"] = np.arange(len(ntsc["cell"])) % 2
trn = sub(~tt & ~nt)
files = {"train": trn, "test_adaptation": ad, "test_scoring": sc, "nt_adaptation": ntad, "nt_scoring": ntsc}
import predict, evaluate
predict.load = lambda p: files[p]
evaluate.load = lambda p: files[p]
predict.DRAWS = 4
evaluate.DRAWS = 4
evaluate.BOOT = 20
preds, diag = predict.run()
print("predictions", len(preds), "groups", len(diag))
out = "/tmp/claude-0/-home-claude/fa2f5e47-7e0a-544e-ba47-35b58380f98c/scratchpad/dry_rep"
import os; os.makedirs(out + "/results", exist_ok=True)
np.savez_compressed(out + "/results/predictions.npz", **preds)
json.dump({k: predict.digest(v) for k, v in preds.items()}, open(out + "/results/digests.json", "w"))
json.dump(diag, open(out + "/results/diagnostics.json", "w"))
evaluate.HERE = __import__("pathlib").Path(out)
evaluate.check_manifest = lambda: {}
evaluate.main()
print("DRY RUN OK")
