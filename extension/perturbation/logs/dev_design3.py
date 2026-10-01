"""Development (training targets only): ablation design. Exploratory."""
exec(open("/tmp/claude-0/-home-claude/fa2f5e47-7e0a-544e-ba47-35b58380f98c/scratchpad/dev_design2.py").read().split("full = fit(targets)")[0])
gen_expo, gen_str = design.exposures(raw_tr)
expo["general"] = gen_expo
expo["strength"] = gen_str
full = fit(targets)
blk = {"ifnblock": blocks["ifn"], "ifn": progs["ifn"], "all": allb}
print("full", {p: round(rf_block(full, b, hidx), 3) for p, b in blk.items()}, flush=True)
for k in (5, 10, 20):
    rres = []
    for _ in range(15):
        drop = set(rng.choice(targets, k, replace=False))
        ch = fit([t for t in targets if t not in drop])
        rres.append({p: rf_block(ch, b, hidx) for p, b in blk.items()})
    print(f"k={k} random removal mean", {p: round(np.mean([r[p] for r in rres]), 3) for p in blk}, "q10", {p: round(np.quantile([r[p] for r in rres], .1), 3) for p in blk}, flush=True)
    for name in ("ifn", "general", "strength"):
        top = sorted(targets, key=lambda t: -expo[name][t])[:k]
        ch = fit([t for t in targets if t not in set(top)])
        print(f"   remove top[{name}] dimS {ch['VS'].shape[1]}", {p: round(rf_block(ch, b, hidx), 3) for p, b in blk.items()}, top[:8], flush=True)
