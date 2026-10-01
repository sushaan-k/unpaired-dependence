"""Dry run of programs.py and design.py on training targets only (no held-out data)."""
import sys, json
import numpy as np
sys.path.insert(0, "/home/claude/cbio/cbio-agent-handoff/extension/perturbation")
exec(open("/home/claude/cbio/cbio-agent-handoff/extension/perturbation/dev_dryrun.py").read().split("preds, diag = predict.run(fits=fits)")[0])
preds, diag = predict.run(fits=fits)
import programs, design
real_load = np.load
class FakeZ(dict):
    files = property(lambda self: list(self.keys()))
def fake_np_load(path, *a, **k):
    p = str(path)
    if p.endswith("predictions.npz"):
        return preds
    if p.endswith("channels.npz"):
        return {"pf_within/VS": chans["pf_within"]["VS"] if chans["pf_within"]["VS"].shape[1] == 1 else chans["pf_within"]["VS"][:, :1]}
    return real_load(path, *a, **k)
programs.np.load = fake_np_load
programs.load = fake
design.load = fake
import pmethods
# design: shrink budgets and draws for the dry run
design.REMOVE = (5, 10)
design.DRAWS = 5
design.PERMUTATIONS = 200
programs.BOOT = 50
design.BOOT = 50
# redirect outputs
programs.pm.HERE = __import__("pathlib").Path("/tmp/claude-0/-home-claude/fa2f5e47-7e0a-544e-ba47-35b58380f98c/scratchpad/dry_out")
(programs.pm.HERE / "results").mkdir(parents=True, exist_ok=True)
# design uses training targets excluding dev-test ones in the dry run
orig_load_training = fake
def fake2(part):
    if part == "training":
        return subset(tr)
    return files[part]
design.load = fake2
design.main()
print("DRY RUN OK")
