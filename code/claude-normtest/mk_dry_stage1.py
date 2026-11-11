# Synthetic Stage 1 dir for the scorer's dry run: every J_p IS the record J_l (symlink), so J_22 == J_6 and the gate MUST STOP (D == 0).
# Reads NO real Stage 1 file. n_valid values are invented.
import json, os, hashlib, torch
JD = "$JLENS_HOME/results-whole-swift-record"; OUT = "$JLENS_HOME/claude-normtest/dry-stage1-identical"; SAVE_L = [20, 24, 28, 32, 36, 40, 44, 48, 52, 56]
os.makedirs(OUT, exist_ok=True); md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
n = [230 + 3 * i for i in range(16)]
src_md5 = {l: md5(os.path.join(JD, f"J_l_{l}.pt")) for l in SAVE_L}
for i in range(16):
    d = os.path.join(OUT, f"doc_{i+1:02d}"); os.makedirs(d, exist_ok=True)
    for l in SAVE_L:
        p = os.path.join(d, f"J_p_{l}.pt")
        if not os.path.lexists(p): os.symlink(os.path.join(JD, f"J_l_{l}.pt"), p)
    json.dump({"doc": i + 1, "genre": f"synthetic-{i+1}", "n_valid": n[i], "ts": "dry", "J_p_md5": {str(l): src_md5[l] for l in SAVE_L}}, open(os.path.join(d, "meta.json"), "w"))
for l in SAVE_L:
    J = torch.load(os.path.join(JD, f"J_l_{l}.pt")).float(); torch.save((J * sum(n), sum(n)), os.path.join(OUT, f"acc_{l}.pt"))
print("dry dir ready", OUT, "sum n", sum(n))
